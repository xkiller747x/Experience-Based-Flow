"""BM25-based case retriever for baseline comparison."""

from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.rag import Case
from src.rag.case_retriever import RetrievedCase
from src.rag.negative_aware_retriever import DEFAULT_CASES_PATH


class BM25Retriever:
    """BM25-based case retriever for baseline comparison."""

    def __init__(self, cases_path: str = DEFAULT_CASES_PATH):
        self.cases: list[Case] = []
        self._documents: list[str] = []
        self._load(cases_path)
        self._tokenized_documents = [
            self._tokenize(document) for document in self._documents
        ]
        self._bm25 = BM25Okapi(self._tokenized_documents)

    def _load(self, path: str) -> None:
        cases_path = Path(path)
        with cases_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                case = Case(**json.loads(line))
                self.cases.append(case)
                self._documents.append(self._case_document(case))

    def retrieve(
        self,
        event_type: str,
        severity: int,
        scenario: str,
        context: dict | None = None,
        top_k: int = 5,
    ) -> list[RetrievedCase]:
        """Retrieve top-k similar cases with BM25Okapi."""

        query = self._query_document(event_type, severity, scenario, context)
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]

        return [
            RetrievedCase(
                case=self.cases[index],
                score=float(scores[index]),
                match_reason=f"BM25 score={float(scores[index]):.4f}",
            )
            for index in ranked_indices
        ]

    def _case_document(self, case: Case) -> str:
        return (
            f"{case.event_type} {case.scenario} severity={case.severity} "
            f"load_rate={case.current_load_rate} action={case.action} "
            f"outcome={case.outcome} {case.reasoning or ''}"
        )

    def _query_document(
        self,
        event_type: str,
        severity: int,
        scenario: str,
        context: dict | None,
    ) -> str:
        parts = [f"{event_type} severity={severity} {scenario}"]
        if context:
            for key in (
                "current_load_rate",
                "load_rate",
                "time_window_pressure",
                "pressure",
                "avg_delay_minutes",
                "urgent_orders",
                "available_backup_vehicles",
                "customer_priority_mix",
            ):
                if key in context:
                    parts.append(f"{key}={context[key]}")
        return " ".join(parts)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.split()
