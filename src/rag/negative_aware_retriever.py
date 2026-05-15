"""Negative-aware dual-path retriever for logistics RAG."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from src.rag import Case
from src.rag.case_retriever import RetrievedCase
from src.rag.solver_ground_truth import SCENARIO_PROFILES


DEFAULT_CASES_PATH = "D:/Code/logistic-ai/data/cases/cases_30k.jsonl"


CONTEXT_WEIGHTS: dict[str, float] = {
    "severity": 3.0,
    "current_load_rate": 2.5,
    "available_backup_vehicles": 2.0,
    "time_window_pressure": 2.0,
    "avg_delay_minutes": 1.5,
    "customer_priority_mix": 1.0,
    "urgent_orders": 1.0,
}


@dataclass
class DualRetrievalResult:
    """Top success and failure cases for one anomaly query."""

    success_cases: list[RetrievedCase]
    failure_cases: list[RetrievedCase]
    context_similarity: float


class NegativeAwareRetriever:
    """
    Dual-path retriever that returns similar successful and failed cases.

    Candidate filtering is intentionally strict: only cases with the same
    ``(event_type, scenario)`` as the query enter ranking. Ranking then uses
    weighted Euclidean distance over normalized context features.
    """

    def __init__(self, cases_path: str = DEFAULT_CASES_PATH):
        self.cases: list[Case] = []
        self._success_index: dict[tuple[str, str], list[Case]] = {}
        self._failure_index: dict[tuple[str, str], list[Case]] = {}
        self._load(cases_path)

    def _load(self, path: str) -> None:
        cases_path = Path(path)
        with cases_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                case = Case(**json.loads(line))
                self.cases.append(case)
                key = (case.event_type, case.scenario)
                if case.outcome == "success":
                    self._success_index.setdefault(key, []).append(case)
                elif case.outcome == "failure":
                    self._failure_index.setdefault(key, []).append(case)

    def retrieve(
        self,
        event_type: str,
        severity: int,
        scenario: str,
        context: dict,
        top_success: int = 3,
        top_failure: int = 2,
    ) -> DualRetrievalResult:
        """
        Retrieve similar success and failure cases.

        ``context`` may contain current_load_rate, urgent_orders,
        available_backup_vehicles, avg_delay_minutes, affected_routes,
        time_window_pressure, and customer_priority_mix.
        """

        query_ctx = dict(context or {})
        query_ctx["severity"] = severity
        query_ctx["scenario"] = scenario

        key = (event_type, scenario)
        success_cases = self._rank_cases(
            query_ctx,
            self._success_index.get(key, []),
            top_success,
        )
        failure_cases = self._rank_cases(
            query_ctx,
            self._failure_index.get(key, []),
            top_failure,
        )

        all_distances = [
            1.0 / max(item.score, 1e-8) - 1.0
            for item in success_cases + failure_cases
        ]
        avg_distance = sum(all_distances) / len(all_distances) if all_distances else float("inf")
        context_similarity = 0.0 if math.isinf(avg_distance) else 1.0 / (1.0 + avg_distance)

        return DualRetrievalResult(
            success_cases=success_cases,
            failure_cases=failure_cases,
            context_similarity=context_similarity,
        )

    def _rank_cases(
        self,
        query_ctx: dict,
        candidates: list[Case],
        top_k: int,
    ) -> list[RetrievedCase]:
        ranked: list[tuple[float, Case]] = []
        for case in candidates:
            case_ctx = self._context_from_case(case)
            distance = self._context_distance(query_ctx, case_ctx)
            ranked.append((distance, case))

        ranked.sort(key=lambda item: item[0])
        results = []
        for distance, case in ranked[:top_k]:
            similarity = 1.0 / (1.0 + distance)
            reason = (
                f"same event_type/scenario; context_distance={distance:.4f}; "
                f"outcome={case.outcome}; action={case.action}"
            )
            results.append(
                RetrievedCase(case=case, score=similarity, match_reason=reason)
            )
        return results

    def _context_distance(self, query_ctx: dict, case_ctx: dict) -> float:
        """Return weighted Euclidean distance; smaller means more similar."""

        scenario = str(query_ctx.get("scenario") or case_ctx.get("scenario") or "medium")
        total = 0.0
        for feature, weight in CONTEXT_WEIGHTS.items():
            left = self._normalize(feature, query_ctx.get(feature, 0.0), scenario)
            right = self._normalize(feature, case_ctx.get(feature, 0.0), scenario)
            total += weight * (left - right) ** 2
        return math.sqrt(total)

    def _context_from_case(self, case: Case) -> dict:
        return {
            "scenario": case.scenario,
            "severity": case.severity,
            "current_load_rate": case.current_load_rate,
            "urgent_orders": case.urgent_orders,
            "available_backup_vehicles": case.available_backup_vehicles,
            "avg_delay_minutes": case.avg_delay_minutes,
            "affected_routes": case.affected_routes,
            "time_window_pressure": case.time_window_pressure,
            "customer_priority_mix": case.customer_priority_mix,
        }

    @staticmethod
    def _normalize(feature: str, value: object, scenario: str) -> float:
        profile = SCENARIO_PROFILES.get(scenario, SCENARIO_PROFILES["medium"])
        numeric = float(value or 0.0)

        if feature == "severity":
            return _clip01(numeric / 10.0)
        if feature in {"current_load_rate", "time_window_pressure", "customer_priority_mix"}:
            return _clip01(numeric)
        if feature == "urgent_orders":
            return _clip01(numeric / max(float(profile["orders"]), 1.0))
        if feature == "available_backup_vehicles":
            return _clip01(numeric / max(float(profile["vehicles"]), 1.0))
        if feature == "avg_delay_minutes":
            max_delay = max(60.0, 15.0 * 10.0)
            return _clip01(numeric / max_delay)
        return _clip01(numeric)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))
