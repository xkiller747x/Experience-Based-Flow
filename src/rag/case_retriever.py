"""Case retriever — lightweight text-similarity based RAG."""

from __future__ import annotations
import json
from dataclasses import dataclass
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.rag.case_generator import Case


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
    def __init__(self, cases_path: str = "D:/Code/logistic-ai/data/cases/cases.jsonl"):
        self.cases: list[Case] = []
        self._load(cases_path)

        # Build TF-IDF index on reasoning field
        self._vectorizer = TfidfVectorizer()
        reasoning_docs = [c.reasoning for c in self.cases]
        self._tfidf_matrix = self._vectorizer.fit_transform(reasoning_docs)

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self.cases.append(Case(**d))

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

    def retrieve(
        self,
        event_type: str,
        severity: int,
        scenario: str,
        top_k: int = 5,
        require_outcome: str | None = None,  # "success" or "failure"
    ) -> list[RetrievedCase]:
        """检索最相似的 case。Stage 1 coarse ranking + Stage 2 reranking with TF-IDF."""
        candidates = self.cases

        if require_outcome:
            candidates = [c for c in candidates if c.outcome == require_outcome]

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
        # Find the most frequent action in top 15 as reference action
        action_counter = Counter(case.action for _, _, case in coarse_top15)
        reference_action = action_counter.most_common(1)[0][0] if action_counter else None

        # Build query text
        query_text = f"{event_type} severity={severity} scenario={scenario}"

        # Compute TF-IDF similarities for all cases
        tfidf_scores = self._compute_tfidf_similarity(query_text)

        # Rerank top 15
        reranked = []
        for orig_score, reasons, case in coarse_top15:
            case_idx = self.cases.index(case)

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