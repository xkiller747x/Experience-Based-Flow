"""Case retriever — lightweight text-similarity based RAG."""

from __future__ import annotations
import json
from dataclasses import dataclass, fields
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.rag.case_generator import Case


# Reroute-prone event types for solver injection strategy
SOLVER_INJECT_EVENTS = {
    "traffic_accident",
    "road_closed",
    "traffic_congestion",
    "road_narrowing",
    "bridge_weight_limit",
    "natural_disaster",
    "public_event",
}


EVENT_TYPE_HIERARCHY = {
    "traffic_congestion": ["traffic_accident", "road_closed"],
    "weather_delay": ["traffic_accident"],
    "order_modify": ["order_cancel"],
    "order_update": ["order_cancel"],
    "demand_surge": [],
    "demand_drop": [],
}


@dataclass
class RetrievedCase:
    case: Case
    score: float
    match_reason: str


class CaseRetriever:
    _case_fields: set[str] | None = None

    SOLVER_PATH = "D:/Code/logistic-ai/output/solver_labeled_cases.jsonl"

    def __init__(
        self,
        cases_path: str = "D:/Code/logistic-ai/data/cases/cases.jsonl",
        solver_cases_path: str | None = None,
    ):
        self.cases: list[Case] = []
        self.solver_cases: list[Case] = []
        self._load(cases_path, self.cases)

        # Build TF-IDF index on reasoning field (original pool)
        self._vectorizer = TfidfVectorizer()
        reasoning_docs = [c.reasoning for c in self.cases]
        self._tfidf_matrix = self._vectorizer.fit_transform(reasoning_docs)

        # Load solver pool if path provided
        if solver_cases_path is not None:
            self._load(solver_cases_path, self.solver_cases)
            self._solver_vectorizer = TfidfVectorizer()
            solver_docs = [c.reasoning for c in self.solver_cases]
            if solver_docs:
                self._solver_tfidf_matrix = self._solver_vectorizer.fit_transform(solver_docs)

    @classmethod
    def _get_case_fields(cls) -> set[str]:
        if cls._case_fields is None:
            cls._case_fields = {f.name for f in fields(Case)}
        return cls._case_fields

    def _load(self, path: str, target_list: list[Case] | None = None) -> None:
        valid_fields = self._get_case_fields()
        if target_list is None:
            target_list = self.cases
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    filtered = {k: v for k, v in d.items() if k in valid_fields}
                    filtered.setdefault("timestamp", 0.0)
                    target_list.append(Case(**filtered))

    def _get_event_type_score(self, query_type: str, case_type: str) -> float:
        """Return 3.0 for exact match, 2.0 for semantic hierarchy match, 0.0 otherwise."""
        if query_type == case_type:
            return 3.0
        fallback_types = EVENT_TYPE_HIERARCHY.get(query_type, [])
        if case_type in fallback_types:
            return 2.0
        return 0.0

    def _compute_tfidf_similarity(self, query_text: str) -> np.ndarray:
        """Compute cosine similarity between query and all case reasoning texts."""
        query_vec = self._vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        return scores

    def _retrieve_from_pool(
        self,
        pool_cases: list[Case],
        vectorizer: TfidfVectorizer,
        tfidf_matrix,
        event_type: str,
        severity: int,
        scenario: str,
        top_k: int = 5,
    ) -> list[RetrievedCase]:
        """Internal: retrieve top-k from a given pool (original or solver)."""
        candidates = pool_cases

        # Stage 1: Coarse ranking using structured scoring
        scored = []
        for case in candidates:
            score = 0.0
            reasons = []

            if case.event_type == event_type:
                score += 3.0
                reasons.append(f"事件类型相同({event_type})")

            sev_diff = abs(case.severity - severity)
            score += max(0, 2.0 - sev_diff * 0.5)

            if case.scenario == scenario:
                score += 1.0
                reasons.append(f"场景相同({scenario})")

            scored.append((score, reasons, case))

        scored.sort(key=lambda x: x[0], reverse=True)
        coarse_top15 = scored[:15]

        if not coarse_top15:
            return []

        # Stage 2: Reranking
        action_counter = Counter(case.action for _, _, case in coarse_top15)
        reference_action = action_counter.most_common(1)[0][0] if action_counter else None

        query_text = f"{event_type} severity={severity} scenario={scenario}"

        # Compute TF-IDF similarities
        query_vec = vectorizer.transform([query_text])
        tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        reranked = []
        for orig_score, reasons, case in coarse_top15:
            case_idx = pool_cases.index(case)
            event_type_score = self._get_event_type_score(event_type, case.event_type)

            action_outcome_score = 0.0
            if reference_action is not None:
                if case.action == reference_action and case.outcome == "success":
                    action_outcome_score = 2.0
                elif case.action == reference_action:
                    action_outcome_score = 1.0

            severity_proximity = max(0, 2.0 - abs(severity - case.severity) * 0.5)
            scenario_match = 1.0 if scenario == case.scenario else 0.0
            tfidf_sim = tfidf_scores[case_idx]

            total_rerank_score = (
                3.0 * event_type_score
                + 2.0 * action_outcome_score
                + 1.5 * severity_proximity
                + 1.0 * scenario_match
                + 1.0 * tfidf_sim
            )

            reranked.append((total_rerank_score, reasons, case))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedCase(case=c, score=s, match_reason="; ".join(r))
            for s, r, c in reranked[:top_k]
        ]

    def retrieve(
        self,
        event_type: str,
        severity: int,
        scenario: str,
        top_k: int = 5,
        require_outcome: str | None = None,  # "success" or "failure"
        inject_solver: bool = False,  # inject top-1 solver case for reroute-prone events
    ) -> list[RetrievedCase]:
        """检索最相似的 case。Stage 1 coarse ranking + Stage 2 reranking with TF-IDF.

        If inject_solver=True and the event_type matches SOLVER_INJECT_EVENTS,
        injects the top-1 solver-labeled case at position 0 for improved reroute accuracy.
        """
        # Filter by outcome if required
        pool = self.cases
        if require_outcome:
            pool = [c for c in self.cases if c.outcome == require_outcome]

        # Stage 1 + 2 from the original pool
        result = self._retrieve_from_pool(
            pool, self._vectorizer, self._tfidf_matrix,
            event_type, severity, scenario, top_k=top_k,
        )

        # If inject_solver requested and solver pool loaded and event_type matches
        if (
            inject_solver
            and self.solver_cases
            and event_type in SOLVER_INJECT_EVENTS
        ):
            # Get top-1 from solver pool
            solver_result = self._retrieve_from_pool(
                self.solver_cases, self._solver_vectorizer, self._solver_tfidf_matrix,
                event_type, severity, scenario, top_k=1,
            )
            if solver_result:
                # Remove the lowest-ranked case and inject solver case at position 0
                result = [solver_result[0]] + result[:top_k - 1]

        return result

    def retrieve_by_text(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedCase]:
        """纯文本检索，query 可以是任意描述。"""
        query_lower = query.lower()
        scored = []
        for case in self.cases:
            score = 0.0
            reasons = []

            if case.event_type in query_lower:
                score += 2.0
            if case.action in query_lower:
                score += 1.0
            if case.outcome in query_lower:
                score += 1.0
            if case.scenario in query_lower:
                score += 0.5
            if str(case.severity) in query_lower:
                score += 0.5

            if score > 0:
                reasons.append(f"关键词匹配(+{score})")
                scored.append((score, reasons, case))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedCase(case=c, score=s, match_reason="; ".join(r))
            for s, r, c in scored[:top_k]
        ]