"""Generate raw experiment data for paper figures and tables.

This script is intentionally standalone: it only reads project data/modules and
writes experiment outputs under ``paper_output/data`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag import Case, ExperienceModel, ExperienceModelV2, LearnedRetrieverV2  # noqa: E402
from src.rag.case_retriever import EVENT_TYPE_HIERARCHY  # noqa: E402
from src.rag.experience_model import ALL_ACTIONS, ALL_EVENT_TYPES, ALL_SCENARIOS  # noqa: E402
from src.rag.solver_ground_truth import SolverDrivenGroundTruth  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper_output" / "data"
DEFAULT_CASES_PATH = REPO_ROOT / "data" / "cases"
METHODS = ("baseline", "v1", "v2")
NO_RAG_METHOD = "no_rag"
BUSINESS_RULES_METHOD = "business_rules"
ORTOOLS_FULL_REPLAN_METHOD = "ortools_full_replan"
BM25_RAG_METHOD = "bm25_rag"
HYBRID_RAG_METHOD = "hybrid_rag"
RULE_PROMPT_METHOD = "rule_prompt"
LEARNED_RAG_METHOD = "learned_rag"
RUN_ALL_BASELINE_METHODS = (
    NO_RAG_METHOD,
    BUSINESS_RULES_METHOD,
    ORTOOLS_FULL_REPLAN_METHOD,
    BM25_RAG_METHOD,
    HYBRID_RAG_METHOD,
)
OPTIONAL_METHODS = RUN_ALL_BASELINE_METHODS + (RULE_PROMPT_METHOD, LEARNED_RAG_METHOD)
METHOD_LABELS = {
    "baseline": "Baseline",
    "v1": "V1 Experience Model",
    "v2": "V2 Experience Model",
    NO_RAG_METHOD: "No-RAG Zero-shot LLM",
    BUSINESS_RULES_METHOD: "Pure Business Rules",
    ORTOOLS_FULL_REPLAN_METHOD: "OR-Tools Full Replan",
    BM25_RAG_METHOD: "BM25 Keyword RAG",
    HYBRID_RAG_METHOD: "Hybrid BM25+BGE RAG",
    RULE_PROMPT_METHOD: "Rule-Prompt LLM",
    "learned_rag": "Learned-RAG (V2 Retriever + LLM)",
}
SCENARIO_ORDER = ("small", "medium", "large", "stress")
SEVERITY_LEVELS = (1, 2, 3, 4, 5)
SCENARIO_SIMULATION_HOURS = 8
SCENARIO_INTERMEDIATE_NODES = 10
SCENARIO_BENCHMARKS: dict[str, dict[str, Any]] = {
    "small": {
        "vehicles": 5,
        "orders": 20,
        "area_size_km": 30.0,
        "event_frequency": 0.1,
        "severe_event_probability": 0.2,
    },
    "medium": {
        "vehicles": 10,
        "orders": 50,
        "area_size_km": 50.0,
        "event_frequency": 0.1,
        "severe_event_probability": 0.2,
    },
    "large": {
        "vehicles": 20,
        "orders": 100,
        "area_size_km": 80.0,
        "event_frequency": 0.1,
        "severe_event_probability": 0.2,
    },
    "stress": {
        "vehicles": 30,
        "orders": 200,
        "area_size_km": 100.0,
        "event_frequency": 0.3,
        "severe_event_probability": 0.4,
    },
}
EVENT_BENCHMARK_WEIGHTS = (
    ("traffic_accident", 7),
    ("traffic_congestion", 10),
    ("road_closed", 6),
    ("road_narrowing", 4),
    ("bridge_weight_limit", 3),
    ("order_cancel", 7),
    ("order_modify", 6),
    ("order_update", 5),
    ("priority_order_urgent", 4),
    ("delivery_failure", 3),
    ("vehicle_breakdown", 8),
    ("vehicle_maintenance", 4),
    ("driver_unavailable", 5),
    ("fuel_shortage", 3),
    ("weather_delay", 5),
    ("natural_disaster", 2),
    ("public_event", 3),
    ("warehouse_delay", 6),
    ("inventory_stockout", 4),
    ("demand_surge", 2),
    ("demand_drop", 3),
)
TRAFFIC_EVENT_TYPES = {"traffic_accident", "traffic_congestion", "road_closed", "road_narrowing", "bridge_weight_limit"}
ORDER_EVENT_TYPES = {"order_cancel", "order_modify", "order_update", "priority_order_urgent", "delivery_failure"}
VEHICLE_EVENT_TYPES = {"vehicle_breakdown", "vehicle_maintenance", "driver_unavailable", "fuel_shortage"}
WAREHOUSE_EVENT_TYPES = {"warehouse_delay", "inventory_stockout"}
TRAFFIC_EVENT_TYPES = {
    "traffic_accident",
    "traffic_congestion",
    "road_closed",
    "road_narrowing",
    "bridge_weight_limit",
}
ORDER_EVENT_TYPES = {
    "order_cancel",
    "order_modify",
    "order_update",
    "priority_order_urgent",
    "delivery_failure",
}
VEHICLE_EVENT_TYPES = {"vehicle_breakdown", "vehicle_maintenance", "driver_unavailable"}
WAREHOUSE_EVENT_TYPES = {"warehouse_delay", "inventory_stockout"}


def ground_truth_action(event_type: str, severity: int) -> str:
    """Rule-based oracle used for comparable retrieval evaluation."""
    if event_type in ("vehicle_breakdown", "vehicle_maintenance", "driver_unavailable"):
        if severity >= 5:
            return "reassign_order"
        if severity >= 3:
            return "adjust_capacity"
        return "reroute"
    if event_type == "fuel_shortage":
        return "adjust_capacity" if severity >= 4 else "reroute"
    if event_type in ("traffic_accident", "traffic_congestion", "road_narrowing"):
        return "reroute" if severity >= 4 else "ignore"
    if event_type in ("road_closed", "bridge_weight_limit"):
        return "reroute" if severity >= 3 else "delay_tolerant"
    if event_type == "order_cancel":
        if severity >= 4:
            return "reroute"
        if severity >= 2:
            return "delay_tolerant"
        return "ignore"
    if event_type == "order_modify":
        return "reroute" if severity >= 4 else "delay_tolerant"
    if event_type == "order_update":
        return "ignore"
    if event_type in ("priority_order_urgent", "delivery_failure"):
        return "reassign_order"
    if event_type == "demand_surge":
        return "adjust_capacity" if severity >= 4 else "delay_tolerant"
    if event_type == "demand_drop":
        return "delay_tolerant"
    if event_type in ("weather_delay", "natural_disaster", "public_event"):
        return "delay_tolerant" if severity >= 4 else "reroute"
    if event_type in ("warehouse_delay", "inventory_stockout"):
        return "delay_tolerant"
    return "reroute"


def discover_case_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Case path does not exist: {path}")
    files = sorted(p for p in path.glob("*.jsonl") if p.stat().st_size > 0)
    if not files:
        raise FileNotFoundError(f"No non-empty JSONL case files found in: {path}")
    return files


def load_cases(
    cases_path: Path,
    max_cases: int,
    scenarios: set[str],
    seed: int,
) -> tuple[list[Case], dict[str, Any]]:
    """Load a deterministic reservoir sample of valid cases from JSONL files."""
    rng = random.Random(seed)
    files = discover_case_files(cases_path)
    cases: list[Case] = []
    valid_seen = 0
    raw_seen = 0
    skipped = 0

    allowed_events = set(ALL_EVENT_TYPES)
    allowed_actions = set(ALL_ACTIONS)
    allowed_scenarios = set(ALL_SCENARIOS).intersection(scenarios)

    for file_path in files:
        with file_path.open(encoding="utf-8") as fh:
            for line in fh:
                raw_seen += 1
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                if (
                    record.get("event_type") not in allowed_events
                    or record.get("action") not in allowed_actions
                    or record.get("scenario") not in allowed_scenarios
                    or int(record.get("severity", 0)) not in SEVERITY_LEVELS
                    or record.get("outcome") not in {"success", "failure"}
                ):
                    skipped += 1
                    continue

                case = Case(**record)
                valid_seen += 1
                if max_cases <= 0:
                    cases.append(case)
                elif len(cases) < max_cases:
                    cases.append(case)
                else:
                    replacement_index = rng.randrange(valid_seen)
                    if replacement_index < max_cases:
                        cases[replacement_index] = case

    cases.sort(key=lambda item: item.id)
    metadata = {
        "case_files": [str(path) for path in files],
        "raw_records_seen": raw_seen,
        "valid_records_seen": valid_seen,
        "records_skipped": skipped,
        "records_loaded": len(cases),
        "max_cases": max_cases,
    }
    if not cases:
        raise RuntimeError("No valid experiment cases were loaded; check --cases-path and filters.")
    return cases, metadata


def regenerate_cases(output_path: Path, total: int, workers: int, chunk_size: int) -> Path:
    """Generate solver-labeled cases with rich scenario context."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260428)
    solver_gt = SolverDrivenGroundTruth(seed=20260428)
    event_types, event_weights = zip(*EVENT_BENCHMARK_WEIGHTS)

    action_counts: Counter[str] = Counter()
    solved_count = 0
    start_ns = time.perf_counter_ns()
    with output_path.open("w", encoding="utf-8") as fh:
        for index in range(total):
            scenario = SCENARIO_ORDER[index % len(SCENARIO_ORDER)]
            event_type = rng.choices(event_types, weights=event_weights, k=1)[0]
            severe_probability = SCENARIO_BENCHMARKS[scenario]["severe_event_probability"]
            severity = rng.randint(4, 5) if rng.random() < severe_probability else rng.randint(1, 3)
            context = solver_gt.generate_context(scenario, event_type, severity, rng)
            action, details = solver_gt.best_action(scenario, event_type, severity, context)
            results = details.get("results", {})
            action_result = results.get(action, {})
            costs = details.get("costs", {})
            solved = bool(details.get("solved", False))
            solved_count += int(solved)
            action_counts[action] += 1

            benchmark = SCENARIO_BENCHMARKS[scenario]
            cost_before = float(action_result.get("cost_before", 0.0))
            cost_after = float(costs.get(action, action_result.get("cost", 0.0)))
            unassigned_after = int(action_result.get("unassigned", 0))
            outcome = "success" if solved and unassigned_after <= max(2, int(benchmark["orders"] * 0.08)) else "failure"
            case = Case(
                id=f"case_{index + 1:06d}",
                event_type=event_type,
                severity=severity,
                scenario=scenario,
                vehicles_count=benchmark["vehicles"],
                orders_count=benchmark["orders"],
                action=action,
                outcome=outcome,
                reasoning=(
                    f"Solver selected {action} from context-sensitive candidate costs. "
                    f"Load={context['current_load_rate']}, urgent={context['urgent_orders']}, "
                    f"backup={context['available_backup_vehicles']}, pressure={context['time_window_pressure']}."
                ),
                before_distance=round(cost_before, 2),
                after_distance=round(cost_after, 2),
                unassigned_before=0,
                unassigned_after=unassigned_after,
                timestamp=round(index / max(total, 1), 6),
                current_load_rate=context["current_load_rate"],
                urgent_orders=context["urgent_orders"],
                available_backup_vehicles=context["available_backup_vehicles"],
                avg_delay_minutes=context["avg_delay_minutes"],
                affected_routes=context["affected_routes"],
                time_window_pressure=context["time_window_pressure"],
                customer_priority_mix=context["customer_priority_mix"],
                cost_before=round(cost_before, 3),
                cost_after=round(cost_after, 3),
                candidate_costs={candidate: round(float(cost), 3) for candidate, cost in costs.items()},
                solved=solved,
            )
            fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")

    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
    print(
        "Generated solver-driven cases: "
        f"total={total}, solved={solved_count}/{max(total, 1)} "
        f"({100.0 * solved_count / max(total, 1):.1f}%), "
        f"avg_ms={elapsed_ms / max(total, 1):.1f}, "
        f"actions={dict(sorted(action_counts.items()))}"
    )
    return output_path


def coarse_score(case: Case, event_type: str, severity: int, scenario: str, context: dict | None = None) -> float:
    score = 0.0
    if case.event_type == event_type:
        score += 3.0
    elif case.event_type in EVENT_TYPE_HIERARCHY.get(event_type, []):
        score += 2.0
    score += max(0.0, 2.0 - abs(case.severity - severity) * 0.5)
    if case.scenario == scenario:
        score += 1.0
    if case.outcome == "success":
        score += 0.25
    if context is not None:
        score -= abs(case.current_load_rate - float(context.get("current_load_rate", 0.5))) * 0.5
        score -= abs(case.urgent_orders - int(context.get("urgent_orders", 0))) * 0.1
        score -= abs(case.available_backup_vehicles - int(context.get("available_backup_vehicles", 0))) * 0.2
        score -= abs(case.avg_delay_minutes - float(context.get("avg_delay_minutes", 0.0))) * 0.015
        score -= abs(case.affected_routes - int(context.get("affected_routes", 0))) * 0.08
        score -= abs(case.time_window_pressure - float(context.get("time_window_pressure", 0.5))) * 0.5
        score -= abs(case.customer_priority_mix - float(context.get("customer_priority_mix", 0.2))) * 0.4
    return score


def retrieve_baseline(cases: list[Case], event_type: str, severity: int, scenario: str, top_k: int, context: dict | None = None) -> list[tuple[float, Case]]:
    scored = [(coarse_score(case, event_type, severity, scenario, context), case) for case in cases]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def retrieve_v1(
    cases: list[Case],
    model: ExperienceModel,
    event_type: str,
    severity: int,
    scenario: str,
    top_k: int,
    context: dict | None = None,
) -> list[tuple[float, Case]]:
    prediction = model.predict(event_type, severity, scenario)
    expected_action = prediction["action"]
    confidence = float(prediction["action_confidence"])
    scored = []
    for case in cases:
        score = coarse_score(case, event_type, severity, scenario, context)
        if case.action == expected_action:
            score += 2.0 * confidence
        if case.outcome == prediction["outcome"]:
            score += 0.5 * float(prediction["outcome_confidence"])
        scored.append((score, case))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def retrieve_v2(
    cases: list[Case],
    model: ExperienceModelV2,
    event_type: str,
    severity: int,
    scenario: str,
    top_k: int,
    context: dict | None = None,
) -> list[tuple[float, Case]]:
    coarse = retrieve_baseline(cases, event_type, severity, scenario, top_k=max(30, top_k), context=context)
    coarse_cases = [case for _, case in coarse]
    coarse_scores = [score for score, _ in coarse]
    relevance_scores = model.predict_relevance(event_type, severity, scenario, coarse_cases)
    combined = [
        (coarse_scores[index] * 0.6 + relevance_scores[index] * 10.0 * 0.4, case)
        for index, case in enumerate(coarse_cases)
    ]
    combined.sort(key=lambda item: item[0], reverse=True)
    return combined[:top_k]


def case_to_search_document(case: Case) -> str:
    return " ".join(
        (
            case.event_type.replace("_", " "),
            f"event_type_{case.event_type}",
            f"severity_{case.severity}",
            f"scenario_{case.scenario}",
            f"action_{case.action}",
            f"outcome_{case.outcome}",
            f"load_rate_{case.current_load_rate:.2f}",
            f"urgent_orders_{case.urgent_orders}",
            f"backup_vehicles_{case.available_backup_vehicles}",
            f"avg_delay_{case.avg_delay_minutes:.0f}",
            f"affected_routes_{case.affected_routes}",
            f"time_pressure_{case.time_window_pressure:.2f}",
            f"priority_mix_{case.customer_priority_mix:.2f}",
            case.reasoning,
        )
    )


def query_to_search_document(event_type: str, severity: int, scenario: str, context: dict | None = None) -> str:
    context = context or {}
    return " ".join(
        (
            event_type.replace("_", " "),
            f"event_type_{event_type}",
            f"severity_{severity}",
            f"scenario_{scenario}",
            f"load_rate_{float(context.get('current_load_rate', 0.5)):.2f}",
            f"urgent_orders_{int(context.get('urgent_orders', 0))}",
            f"backup_vehicles_{int(context.get('available_backup_vehicles', 0))}",
            f"avg_delay_{float(context.get('avg_delay_minutes', 0.0)):.0f}",
            f"affected_routes_{int(context.get('affected_routes', 0))}",
            f"time_pressure_{float(context.get('time_window_pressure', 0.5)):.2f}",
            f"priority_mix_{float(context.get('customer_priority_mix', 0.2)):.2f}",
            "logistics anomaly dispatch route planning decision",
        )
    )


def tokenize_search_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def tokenize_search_text(text: str) -> list[str]:
    """Simple whitespace/lowercase tokenizer for search text."""
    return text.lower().replace("_", " ").split()


def case_to_search_document(case: Case) -> str:
    """Convert a case to a searchable text document for BM25/embedding indexing."""
    return (
        f"event_type: {case.event_type} severity: {case.severity} scenario: {case.scenario} "
        f"action: {case.action} outcome: {case.outcome} vehicles: {case.vehicles_count} orders: {case.orders_count} "
        f"load_rate: {case.current_load_rate:.2f} urgent_orders: {case.urgent_orders} "
        f"backup_vehicles: {case.available_backup_vehicles} avg_delay: {case.avg_delay_minutes:.0f} "
        f"affected_routes: {case.affected_routes} time_pressure: {case.time_window_pressure:.2f} "
        f"priority_mix: {case.customer_priority_mix:.2f} "
        f"reasoning: {case.reasoning}"
    )


def query_to_search_document(event_type: str, severity: int, scenario: str, context: dict | None = None) -> str:
    """Convert a query triple to a searchable text document."""
    context = context or {}
    return (
        f"event_type: {event_type} severity: {severity} scenario: {scenario} "
        f"load_rate: {float(context.get('current_load_rate', 0.5)):.2f} "
        f"urgent_orders: {int(context.get('urgent_orders', 0))} "
        f"backup_vehicles: {int(context.get('available_backup_vehicles', 0))} "
        f"avg_delay: {float(context.get('avg_delay_minutes', 0.0)):.0f} "
        f"affected_routes: {int(context.get('affected_routes', 0))} "
        f"time_pressure: {float(context.get('time_window_pressure', 0.5)):.2f} "
        f"priority_mix: {float(context.get('customer_priority_mix', 0.2)):.2f}"
    )


class BM25CaseRetriever:
    """Zero-training keyword RAG baseline with a plain BM25 index."""

    def __init__(self, cases: list[Case], k1: float = 1.5, b: float = 0.75) -> None:
        self.cases = cases
        self.k1 = k1
        self.b = b
        self.documents = [tokenize_search_text(case_to_search_document(case)) for case in cases]
        self.doc_lengths = [len(document) for document in self.documents]
        self.avg_doc_length = mean(self.doc_lengths)
        self.term_frequencies = [Counter(document) for document in self.documents]
        document_frequencies: Counter[str] = Counter()
        for document in self.documents:
            document_frequencies.update(set(document))
        total_documents = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (total_documents - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def score_documents(self, query: str) -> list[float]:
        query_terms = tokenize_search_text(query)
        if not query_terms or not self.documents:
            return [0.0 for _ in self.documents]
        scores: list[float] = []
        for term_frequency, doc_length in zip(self.term_frequencies, self.doc_lengths):
            score = 0.0
            for term in query_terms:
                frequency = term_frequency.get(term, 0)
                if frequency <= 0:
                    continue
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * doc_length / max(self.avg_doc_length, 1.0))
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / denominator
            scores.append(score)
        return scores

    def retrieve(self, event_type: str, severity: int, scenario: str, top_k: int, context: dict | None = None) -> list[tuple[float, Case]]:
        query = query_to_search_document(event_type, severity, scenario, context)
        scores = self.score_documents(query)
        ranked = [(score + 0.15 * coarse_score(case, event_type, severity, scenario, context), case) for score, case in zip(scores, self.cases)]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:top_k]


class HybridCaseRetriever:
    """Zero-training hybrid BM25 + public pretrained BGE embedding retriever."""

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self, cases: list[Case], bm25_retriever: BM25CaseRetriever) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Hybrid RAG baseline requires sentence-transformers. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.cases = cases
        self.bm25_retriever = bm25_retriever
        self.encoder = SentenceTransformer(self.MODEL_NAME)
        documents = [case_to_search_document(case) for case in cases]
        self.embeddings = self.encoder.encode(documents, normalize_embeddings=True, show_progress_bar=False)

    @staticmethod
    def normalize_scores(scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if math.isclose(max_score, min_score):
            return [0.0 for _ in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def retrieve(self, event_type: str, severity: int, scenario: str, top_k: int, context: dict | None = None) -> list[tuple[float, Case]]:
        query = query_to_search_document(event_type, severity, scenario, context)
        bm25_scores = self.normalize_scores(self.bm25_retriever.score_documents(query))
        query_embedding = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        dense_scores = [float(embedding @ query_embedding) for embedding in self.embeddings]
        dense_scores = self.normalize_scores(dense_scores)
        structured_scores = self.normalize_scores([coarse_score(case, event_type, severity, scenario, context) for case in self.cases])
        ranked = [
            (0.45 * bm25_score + 0.45 * dense_score + 0.10 * structured_score, case)
            for bm25_score, dense_score, structured_score, case in zip(
                bm25_scores,
                dense_scores,
                structured_scores,
                self.cases,
            )
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:top_k]


class ORToolsFullReplanBaseline:
    """Zero-training optimization baseline that fully replans each scenario."""

    def __init__(self, seed: int = 20260428) -> None:
        self.seed = seed
        self._scenario_cache: dict[str, dict[str, float]] = {}

    def _solve_scenario(self, scenario: str) -> dict[str, float]:
        if scenario in self._scenario_cache:
            return self._scenario_cache[scenario]
        from src.optimizer.vrp_solver import ORToolsSolver
        from src.simulation.scenarios import build_scenario
        from src.simulation.simulator import LogisticsSimulator

        config = build_scenario(scenario)
        config.seed = self.seed + SCENARIO_ORDER.index(scenario)
        simulator = LogisticsSimulator(config).generate()
        solver = ORToolsSolver(time_limit_seconds=1)
        plan = solver.solve(simulator.vehicles, simulator.orders, simulator.road_network)
        result = {
            "unassigned_ratio": len(plan.unassigned_order_ids) / max(len(simulator.orders), 1),
            "distance_per_order": plan.total_distance / max(len(simulator.orders), 1),
        }
        self._scenario_cache[scenario] = result
        return result

    def decide(self, event_type: str, severity: int, scenario: str) -> str:
        result = self._solve_scenario(scenario)
        if event_type in {"order_update", "demand_drop"} and severity <= 2:
            return "ignore"
        if result["unassigned_ratio"] > 0.15:
            return "adjust_capacity"
        if event_type in VEHICLE_EVENT_TYPES and severity >= 4:
            return "reassign_order"
        if event_type in {"warehouse_delay", "inventory_stockout", "demand_surge"} and severity >= 4:
            return "adjust_capacity"
        if severity <= 2 and event_type not in {"road_closed", "bridge_weight_limit"}:
            return "delay_tolerant"
        return "reroute"


def relevance(case: Case, target_action: str) -> int:
    if case.action == target_action and case.outcome == "success":
        return 2
    if case.action == target_action:
        return 1
    return 0


def dcg(values: list[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(values))


def ndcg_at_k(ranked_cases: list[Case], all_cases: list[Case], target_action: str, k: int) -> float:
    observed = [relevance(case, target_action) for case in ranked_cases[:k]]
    ideal = sorted((relevance(case, target_action) for case in all_cases), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(observed) / ideal_dcg if ideal_dcg > 0 else 0.0


def precision_recall_at_k(ranked_cases: list[Case], all_cases: list[Case], target_action: str, k: int) -> tuple[float, float, float]:
    retrieved = ranked_cases[:k]
    hits = sum(1 for case in retrieved if case.action == target_action and case.outcome == "success")
    relevant_total = sum(1 for case in all_cases if case.action == target_action and case.outcome == "success")
    precision = hits / k if k > 0 else 0.0
    recall = hits / relevant_total if relevant_total > 0 else 0.0
    hit_rate = 1.0 if hits > 0 else 0.0
    return precision, recall, hit_rate


def build_no_rag_decision_prompt(event_type: str, severity: int, scenario: str) -> tuple[str, str]:
    benchmark = SCENARIO_BENCHMARKS[scenario]
    system_prompt = (
        "You are a logistics dispatch decision model. Make a zero-shot decision "
        "using only the event and scenario base information provided. Do not use "
        "retrieved historical cases, examples, or business rules. Output JSON only."
    )
    user_prompt = (
        "Select the single best logistics response action.\n\n"
        "## Event\n"
        f"  event_type: {event_type}\n"
        f"  severity: {severity}\n\n"
        "## Scenario Base Information\n"
        f"  scenario: {scenario}\n"
        f"  vehicles: {benchmark['vehicles']}\n"
        f"  orders: {benchmark['orders']}\n"
        f"  area_size_km: {benchmark['area_size_km']}\n"
        f"  event_frequency: {benchmark['event_frequency']}\n"
        f"  severe_event_probability: {benchmark['severe_event_probability']}\n\n"
        f"Allowed actions: {', '.join(ALL_ACTIONS)}\n"
        "Respond with JSON only: {\"action\": \"reroute|ignore|adjust_capacity|reassign_order|delay_tolerant\"}"
    )
    return system_prompt, user_prompt


def extract_json_object(text: str) -> dict[str, Any]:
    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]}")

    depth = 0
    for index, char in enumerate(text[brace_start:], brace_start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[brace_start : index + 1])
    raise ValueError(f"Unclosed JSON object in LLM response: {text[brace_start:300]}")


def decide_no_rag_action(llm: Any, event_type: str, severity: int, scenario: str) -> str:
    system_prompt, user_prompt = build_no_rag_decision_prompt(event_type, severity, scenario)
    raw_response = llm.generate(user_prompt, system_prompt=system_prompt)
    action = str(extract_json_object(raw_response).get("action", "")).lower().strip()
    if action not in ALL_ACTIONS:
        raise ValueError(f"No-RAG LLM returned invalid action '{action}'. Must be one of {ALL_ACTIONS}")
    return action


def business_rule_action(event_type: str, severity: int, scenario: str) -> str:
    benchmark = SCENARIO_BENCHMARKS[scenario]
    if event_type in VEHICLE_EVENT_TYPES:
        if severity >= 5 or benchmark["vehicles"] <= 5:
            return "reassign_order"
        if severity >= 3:
            return "adjust_capacity"
        return "reroute"
    if event_type == "fuel_shortage":
        return "adjust_capacity" if severity >= 4 else "reroute"
    if event_type in TRAFFIC_EVENT_TYPES:
        if severity <= 2 and event_type not in {"road_closed", "bridge_weight_limit"}:
            return "ignore"
        if severity >= 3 or event_type in {"road_closed", "bridge_weight_limit"}:
            return "reroute"
        return "delay_tolerant"
    if event_type == "order_cancel":
        if severity >= 4:
            return "reroute"
        if severity >= 2:
            return "delay_tolerant"
        return "ignore"
    if event_type == "order_modify":
        return "reroute" if severity >= 4 else "delay_tolerant"
    if event_type == "order_update":
        return "ignore"
    if event_type in {"priority_order_urgent", "delivery_failure"}:
        return "reassign_order"
    if event_type == "demand_surge":
        return "adjust_capacity" if severity >= 4 or benchmark["orders"] >= 100 else "delay_tolerant"
    if event_type == "demand_drop":
        return "delay_tolerant"
    if event_type in {"weather_delay", "natural_disaster", "public_event"}:
        return "delay_tolerant" if severity >= 4 else "reroute"
    if event_type in WAREHOUSE_EVENT_TYPES:
        return "delay_tolerant"
    return "reroute"


def build_rule_prompt_decision_prompt(event_type: str, severity: int, scenario: str) -> tuple[str, str]:
    benchmark = SCENARIO_BENCHMARKS[scenario]
    system_prompt = (
        "You are a logistics dispatch decision model. Make a zero-training decision "
        "using only the provided hard-coded business rules and event context. Output JSON only."
    )
    rule_lines = [
        "Vehicle or driver disruptions: severity 5 -> reassign_order; severity 3-4 -> adjust_capacity; otherwise reroute.",
        "Fuel shortage: severity >=4 -> adjust_capacity; otherwise reroute.",
        "Traffic accidents, congestion, and road narrowing: severity >=4 -> reroute; otherwise ignore.",
        "Road closure or bridge weight limit: severity >=3 -> reroute; otherwise delay_tolerant.",
        "Order cancellation: severity >=4 -> reroute; severity 2-3 -> delay_tolerant; severity 1 -> ignore.",
        "Order modification: severity >=4 -> reroute; otherwise delay_tolerant. Order update -> ignore.",
        "Urgent priority orders and delivery failures -> reassign_order.",
        "Demand surge: severity >=4 -> adjust_capacity; otherwise delay_tolerant. Demand drop -> delay_tolerant.",
        "Weather, disaster, and public events: severity >=4 -> delay_tolerant; otherwise reroute.",
        "Warehouse delay and inventory stockout -> delay_tolerant.",
    ]
    user_prompt = (
        "Select the single best logistics response action.\n\n"
        "## Hard-coded Business Rules\n"
        + "\n".join(f"- {line}" for line in rule_lines)
        + "\n\n## Event\n"
        f"  event_type: {event_type}\n"
        f"  severity: {severity}\n\n"
        "## Scenario Base Information\n"
        f"  scenario: {scenario}\n"
        f"  vehicles: {benchmark['vehicles']}\n"
        f"  orders: {benchmark['orders']}\n"
        f"  area_size_km: {benchmark['area_size_km']}\n\n"
        f"Allowed actions: {', '.join(ALL_ACTIONS)}\n"
        "Respond with JSON only: {\"action\": \"reroute|ignore|adjust_capacity|reassign_order|delay_tolerant\"}"
    )
    return system_prompt, user_prompt


def decide_rule_prompt_action(llm: Any, event_type: str, severity: int, scenario: str) -> str:
    system_prompt, user_prompt = build_rule_prompt_decision_prompt(event_type, severity, scenario)
    raw_response = llm.generate(user_prompt, system_prompt=system_prompt)
    action = str(extract_json_object(raw_response).get("action", "")).lower().strip()
    if action not in ALL_ACTIONS:
        raise ValueError(f"Rule-prompt LLM returned invalid action '{action}'. Must be one of {ALL_ACTIONS}")
    return action


def build_learned_rag_decision_prompt(
    event_type: str,
    severity: int,
    scenario: str,
    context: dict,
    retrieved_results: list[dict],
) -> tuple[str, str]:
    benchmark = SCENARIO_BENCHMARKS[scenario]
    system_prompt = (
        "You are a logistics dispatch decision assistant. You will be given the current logistics event and a set of "
        "retrieved historical cases with similar context pressure patterns.\n\n"
        "These cases were selected by a learned retriever that matches not just surface-level event types, but deeper "
        "contextual similarity -- including load rate, time-window pressure, urgency, and resource availability. Even "
        "if the retrieved cases have different event types, they faced similar operational pressure.\n\n"
        "Use these historical cases as references to make your decision. Prioritize cases with higher similarity scores "
        "and \"highly reusable\" context analysis. If cases conflict, weight those with closer context similarity higher.\n\n"
        "Output JSON only."
    )

    case_blocks = []
    for index, result in enumerate(retrieved_results, start=1):
        case = result["case"]
        score = float(result["score"])
        context_diff = result["context_diff"]
        load_diff = float(context_diff["load_rate"]["diff"])
        urgent_diff = context_diff["urgent_orders"]["diff"]
        pressure_diff = float(context_diff["time_window_pressure"]["diff"])
        distance_ratio = case.after_distance / case.before_distance if case.before_distance else 0.0
        unassigned_delta = case.unassigned_after - case.unassigned_before
        case_blocks.append(
            f"[Case {index}] Score: {score:.4f} | Match: {result['match_reason']}\n"
            f"  Event: {case.event_type}, Severity: {case.severity}, Scenario: {case.scenario}\n"
            f"  Context Similarity: {context_diff['overall_similarity']}\n"
            f"  Insight: {context_diff['key_insight']}\n"
            f"  Context Diff: load_rate {load_diff:.4f}, urgent_orders diff={urgent_diff}, "
            f"time_pressure diff={pressure_diff:.4f}\n"
            f"  Decision: {case.action} -> {case.outcome}\n"
            f"  Outcome: distance_ratio={distance_ratio:.3f}, unassigned_delta={unassigned_delta}"
        )

    user_prompt = (
        "Select the single best logistics response action.\n\n"
        "## Current Event\n"
        f"  event_type: {event_type}\n"
        f"  severity: {severity}\n\n"
        "## Scenario\n"
        f"  scenario: {scenario}\n"
        f"  vehicles: {benchmark['vehicles']}\n"
        f"  orders: {benchmark['orders']}\n\n"
        "## Current Context\n"
        f"  load_rate: {context['current_load_rate']}\n"
        f"  urgent_orders: {context['urgent_orders']}\n"
        f"  available_backup_vehicles: {context['available_backup_vehicles']}\n"
        f"  avg_delay_minutes: {context['avg_delay_minutes']}\n"
        f"  time_window_pressure: {context['time_window_pressure']}\n"
        f"  customer_priority_mix: {context['customer_priority_mix']}\n\n"
        "## Retrieved Historical Cases (sorted by relevance score)\n\n"
        + "\n\n".join(case_blocks)
        + "\n\nAllowed actions: reroute, ignore, adjust_capacity, reassign_order, delay_tolerant\n\n"
        "Respond with JSON only: "
        "{\"action\": \"reroute|ignore|adjust_capacity|reassign_order|delay_tolerant\", \"reasoning\": \"...\"}"
    )
    return system_prompt, user_prompt


def decide_learned_rag_action(
    llm: Any,
    retriever: LearnedRetrieverV2,
    pool_cases: list[Case],
    event_type: str,
    severity: int,
    scenario: str,
    context: dict,
    top_k: int = 5,
) -> tuple[str, list[tuple[float, Case]]]:
    """Retrieve similar cases via LearnedRetrieverV2, build prompt, call LLM."""
    query_context = {
        "event_type": event_type,
        "severity": severity,
        "scenario": scenario,
        "vehicles_count": SCENARIO_BENCHMARKS[scenario]["vehicles"],
        "orders_count": SCENARIO_BENCHMARKS[scenario]["orders"],
        **context,
    }
    results = retriever.retrieve(query_context, pool_cases, top_k=top_k)
    system_prompt, user_prompt = build_learned_rag_decision_prompt(
        event_type, severity, scenario, context, results
    )
    raw_response = llm.generate(user_prompt, system_prompt=system_prompt)
    action = str(extract_json_object(raw_response).get("action", "")).lower().strip()
    if action not in ALL_ACTIONS:
        raise ValueError(f"Learned-RAG LLM returned invalid action '{action}'")
    ranked = [(r["score"], r["case"]) for r in results]
    return action, ranked


def selected_methods(run_flags: dict[str, bool]) -> tuple[str, ...]:
    return METHODS + tuple(method for method in OPTIONAL_METHODS if run_flags.get(method, False))


def case_context(case: Case) -> dict[str, Any]:
    return {
        "current_load_rate": case.current_load_rate,
        "urgent_orders": case.urgent_orders,
        "available_backup_vehicles": case.available_backup_vehicles,
        "avg_delay_minutes": case.avg_delay_minutes,
        "affected_routes": case.affected_routes,
        "time_window_pressure": case.time_window_pressure,
        "customer_priority_mix": case.customer_priority_mix,
    }


def build_queries(cases: list[Case], scenarios: list[str], query_limit: int, seed: int) -> list[tuple[str, int, str, dict]]:
    rng = random.Random(seed)
    available_cases = [
        case
        for case in cases
        if case.scenario in scenarios and case.event_type in ALL_EVENT_TYPES and case.severity in SEVERITY_LEVELS
    ]
    if available_cases:
        source_cases = available_cases
        if query_limit > 0 and query_limit < len(source_cases):
            source_cases = rng.sample(source_cases, query_limit)
        queries = [
            (case.event_type, case.severity, case.scenario, case_context(case))
            for case in source_cases
        ]
        queries.sort(key=lambda item: (item[0], item[1], item[2], json.dumps(item[3], sort_keys=True)))
        return queries

    event_types = sorted({case.event_type for case in cases})
    available_scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in scenarios and scenario in {case.scenario for case in cases}]
    solver_gt = SolverDrivenGroundTruth(seed=seed)
    queries = []
    for event_type in event_types:
        for severity in SEVERITY_LEVELS:
            for scenario in available_scenarios:
                queries.append((event_type, severity, scenario, solver_gt.generate_context(scenario, event_type, severity, rng)))
    if query_limit > 0 and query_limit < len(queries):
        queries = rng.sample(queries, query_limit)
        queries.sort(key=lambda item: (item[0], item[1], item[2], json.dumps(item[3], sort_keys=True)))
    return queries


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def benchmark_generated_events(scenario: str, seed: int) -> int:
    benchmark = SCENARIO_BENCHMARKS[scenario]
    event_types, event_weights = zip(*EVENT_BENCHMARK_WEIGHTS)
    rng = random.Random(seed + 3)
    generated_events = 0
    node_count = 1 + benchmark["orders"] + SCENARIO_INTERMEDIATE_NODES
    edge_count = node_count * (node_count - 1)

    for _ in range(SCENARIO_SIMULATION_HOURS):
        if rng.random() >= benchmark["event_frequency"]:
            continue
        generated_events += 1
        event_type = rng.choices(event_types, weights=event_weights, k=1)[0]
        if event_type in TRAFFIC_EVENT_TYPES and edge_count:
            rng.randrange(edge_count)
        elif event_type in ORDER_EVENT_TYPES and benchmark["orders"]:
            rng.randrange(benchmark["orders"])
        elif event_type in VEHICLE_EVENT_TYPES and benchmark["vehicles"] > 0:
            rng.randint(0, benchmark["vehicles"] - 1)
        elif event_type not in WAREHOUSE_EVENT_TYPES:
            rng.randrange(node_count)
        if rng.random() < benchmark["severe_event_probability"]:
            rng.randint(4, 5)
        else:
            rng.randint(1, 3)

    return generated_events


def scenario_metadata(scenarios: list[str], seed: int) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for offset, scenario in enumerate(scenarios):
        benchmark = SCENARIO_BENCHMARKS[scenario]
        metadata[scenario] = {
            "vehicles": benchmark["vehicles"],
            "orders": benchmark["orders"],
            "area_size_km": benchmark["area_size_km"],
            "generated_events": benchmark_generated_events(scenario, seed + offset),
            "event_frequency": benchmark["event_frequency"],
            "severe_event_probability": benchmark["severe_event_probability"],
        }
    return metadata


def evaluate_experiments(
    cases: list[Case],
    train_cases: list[Case],
    queries: list[tuple[str, int, str, dict]],
    top_k_values: list[int],
    run_flags: dict[str, bool] | None = None,
    seed: int = 20260428,
) -> dict[str, Any]:
    run_flags = run_flags or {}
    max_k = max(top_k_values)
    model_v1 = ExperienceModel()
    model_v1.train(train_cases)
    model_v2 = ExperienceModelV2()
    model_v2.train(train_cases)
    methods = selected_methods(run_flags)
    shared_llm = None
    no_rag_llm = None
    rule_prompt_llm = None
    if (
        run_flags.get(NO_RAG_METHOD, False)
        or run_flags.get(RULE_PROMPT_METHOD, False)
        or run_flags.get(LEARNED_RAG_METHOD, False)
    ):
        from src.agent.llm import LLMGateway

        shared_llm = LLMGateway()
        no_rag_llm = shared_llm
        rule_prompt_llm = shared_llm
    learned_retriever = None
    if LEARNED_RAG_METHOD in methods:
        learned_retriever = LearnedRetrieverV2(random_state=seed)
        learned_retriever.train(train_cases)
    bm25_retriever = BM25CaseRetriever(cases) if run_flags.get(BM25_RAG_METHOD, False) or run_flags.get(HYBRID_RAG_METHOD, False) else None
    hybrid_retriever = HybridCaseRetriever(cases, bm25_retriever) if run_flags.get(HYBRID_RAG_METHOD, False) and bm25_retriever else None
    ortools_replanner = ORToolsFullReplanBaseline(seed) if run_flags.get(ORTOOLS_FULL_REPLAN_METHOD, False) else None
    solver_gt = SolverDrivenGroundTruth(seed)

    model_rows: dict[str, dict[str, list[float]]] = {
        method: {"accuracy": [], "ndcg": [], "latency_ms": []} for method in methods
    }
    scenario_rows: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"accuracy": [], "ndcg": [], "latency_ms": []})
    anomaly_rows: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"accuracy": [], "ndcg": []})
    quality_rows: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: {"precision": [], "recall": [], "hit_rate": []})
    raw_rows: list[dict[str, Any]] = []

    for event_type, severity, scenario, context in queries:
        target_action, target_details = solver_gt.best_action(scenario, event_type, severity, context)
        for method in methods:
            start_ns = time.perf_counter_ns()
            if method == "baseline":
                ranked = retrieve_baseline(cases, event_type, severity, scenario, max_k, context=context)
            elif method == "v1":
                ranked = retrieve_v1(cases, model_v1, event_type, severity, scenario, max_k, context=context)
            elif method == "v2":
                ranked = retrieve_v2(cases, model_v2, event_type, severity, scenario, max_k, context=context)
            elif method == NO_RAG_METHOD:
                ranked = []
                top1_action = decide_no_rag_action(no_rag_llm, event_type, severity, scenario)
            elif method == BUSINESS_RULES_METHOD:
                ranked = []
                top1_action = business_rule_action(event_type, severity, scenario)
            elif method == ORTOOLS_FULL_REPLAN_METHOD:
                ranked = []
                top1_action = ortools_replanner.decide(event_type, severity, scenario)
            elif method == BM25_RAG_METHOD:
                ranked = bm25_retriever.retrieve(event_type, severity, scenario, max_k, context=context)
            elif method == HYBRID_RAG_METHOD:
                ranked = hybrid_retriever.retrieve(event_type, severity, scenario, max_k, context=context)
            elif method == RULE_PROMPT_METHOD:
                ranked = []
                top1_action = decide_rule_prompt_action(rule_prompt_llm, event_type, severity, scenario)
            elif method == LEARNED_RAG_METHOD:
                top1_action, ranked = decide_learned_rag_action(
                    shared_llm, learned_retriever, cases,
                    event_type, severity, scenario, context, max_k,
                )
            else:
                raise ValueError(f"Unknown experiment method: {method}")
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

            ranked_cases = [case for _, case in ranked]
            if ranked_cases and method != LEARNED_RAG_METHOD:
                top1_action = ranked_cases[0].action if ranked_cases else ""
            action_correct = 1.0 if top1_action == target_action else 0.0
            ndcg5 = ndcg_at_k(ranked_cases, cases, target_action, min(5, max_k))

            model_rows[method]["accuracy"].append(action_correct)
            model_rows[method]["ndcg"].append(ndcg5)
            model_rows[method]["latency_ms"].append(latency_ms)
            scenario_rows[(scenario, method)]["accuracy"].append(action_correct)
            scenario_rows[(scenario, method)]["ndcg"].append(ndcg5)
            scenario_rows[(scenario, method)]["latency_ms"].append(latency_ms)
            anomaly_rows[(event_type, method)]["accuracy"].append(action_correct)
            anomaly_rows[(event_type, method)]["ndcg"].append(ndcg5)

            quality_values: dict[int, tuple[float, float, float]] = {}
            for k in top_k_values:
                quality = precision_recall_at_k(ranked_cases, cases, target_action, k)
                quality_values[k] = quality
                quality_rows[(method, k)]["precision"].append(quality[0])
                quality_rows[(method, k)]["recall"].append(quality[1])
                quality_rows[(method, k)]["hit_rate"].append(quality[2])

            raw_row = {
                "event_type": event_type,
                "severity": severity,
                "scenario": scenario,
                "current_load_rate": context.get("current_load_rate", 0.0),
                "urgent_orders": context.get("urgent_orders", 0),
                "available_backup_vehicles": context.get("available_backup_vehicles", 0),
                "avg_delay_minutes": context.get("avg_delay_minutes", 0.0),
                "affected_routes": context.get("affected_routes", 0),
                "time_window_pressure": context.get("time_window_pressure", 0.0),
                "customer_priority_mix": context.get("customer_priority_mix", 0.0),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "ground_truth_action": target_action,
                "ground_truth_solved": bool(target_details.get("solved", False)),
                "top1_action": top1_action,
                "action_correct": int(action_correct),
                "ndcg_at_5": round(ndcg5, 6),
                "latency_ms": round(latency_ms, 6),
                "top_case_ids": ";".join(case.id for case in ranked_cases),
            }
            for k, (precision, recall, hit_rate) in quality_values.items():
                raw_row[f"precision_at_{k}"] = round(precision, 6)
                raw_row[f"recall_at_{k}"] = round(recall, 6)
                raw_row[f"hit_rate_at_{k}"] = round(hit_rate, 6)
            raw_rows.append(raw_row)

    model_summary = [
        {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "action_accuracy_pct": round(100.0 * mean(values["accuracy"]), 3),
            "ndcg_at_5_pct": round(100.0 * mean(values["ndcg"]), 3),
            "avg_latency_ms": round(mean(values["latency_ms"]), 6),
            "query_count": len(values["accuracy"]),
        }
        for method, values in model_rows.items()
    ]
    scenario_summary = [
        {
            "scenario": scenario,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "action_accuracy_pct": round(100.0 * mean(values["accuracy"]), 3),
            "ndcg_at_5_pct": round(100.0 * mean(values["ndcg"]), 3),
            "avg_latency_ms": round(mean(values["latency_ms"]), 6),
            "query_count": len(values["accuracy"]),
        }
        for (scenario, method), values in sorted(scenario_rows.items())
    ]
    anomaly_summary = [
        {
            "event_type": event_type,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "action_accuracy_pct": round(100.0 * mean(values["accuracy"]), 3),
            "ndcg_at_5_pct": round(100.0 * mean(values["ndcg"]), 3),
            "query_count": len(values["accuracy"]),
        }
        for (event_type, method), values in sorted(anomaly_rows.items())
    ]
    quality_summary = [
        {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "k": k,
            "precision_at_k_pct": round(100.0 * mean(values["precision"]), 3),
            "recall_at_k_pct": round(100.0 * mean(values["recall"]), 3),
            "hit_rate_at_k_pct": round(100.0 * mean(values["hit_rate"]), 3),
            "query_count": len(values["precision"]),
        }
        for (method, k), values in sorted(quality_rows.items())
    ]

    return {
        "model_comparison": model_summary,
        "scenario_performance": scenario_summary,
        "anomaly_accuracy": anomaly_summary,
        "retrieval_quality": quality_summary,
        "raw_query_results": raw_rows,
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper experiments and export raw CSV/JSON data.")
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES_PATH, help="Case JSONL file or directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV/JSON outputs.")
    parser.add_argument("--max-cases", type=int, default=3000, help="Reservoir sample size; <=0 loads all valid cases.")
    parser.add_argument("--train-cases", type=int, default=800, help="Cases used to train V1/V2 models.")
    parser.add_argument("--query-limit", type=int, default=0, help="Maximum query triples; <=0 evaluates all generated queries.")
    parser.add_argument("--top-k-values", default="1,3,5,10", help="Comma-separated k values for retrieval quality.")
    parser.add_argument("--scenarios", default=",".join(SCENARIO_ORDER), help="Comma-separated scenario names.")
    parser.add_argument("--seed", type=int, default=20260428, help="Random seed for sampling and scenario generation.")
    parser.add_argument("--regenerate-cases", type=int, default=0, help="If >0, run simulator-backed case generation for this many fresh cases before experiments; default 0 keeps simulation disabled.")
    parser.add_argument("--generated-cases-path", type=Path, default=DEFAULT_OUTPUT_DIR / "generated_cases.jsonl")
    parser.add_argument("--case-workers", type=int, default=1, help="Workers for optional case regeneration.")
    parser.add_argument("--case-chunk-size", type=int, default=1000, help="Chunk size for optional case regeneration.")
    parser.add_argument("--run-no-rag", action="store_true", help="Run the No-RAG zero-shot LLM blank-control group.")
    parser.add_argument("--run-all-baselines", action="store_true", help="Run all eight comparison groups: original methods plus zero-training baselines.")
    parser.add_argument("--run-business-rules", action="store_true", help="Run the pure hard-coded business-rule baseline.")
    parser.add_argument("--run-ortools-full-replan", action="store_true", help="Run the pure OR-Tools full-replanning baseline.")
    parser.add_argument("--run-bm25-rag", action="store_true", help="Run the zero-training BM25 keyword RAG baseline.")
    parser.add_argument("--run-hybrid-rag", action="store_true", help="Run the zero-training hybrid BM25 + pretrained BGE RAG baseline.")
    parser.add_argument("--run-rule-prompt", action="store_true", help="Run the zero-training LLM prompt baseline with hard-coded business rules injected.")
    parser.add_argument("--run-learned-rag", action="store_true", help="Run the Learned-RAG LLM method with LearnedRetrieverV2 retrieval.")
    return parser.parse_args()


def build_run_flags(args: argparse.Namespace) -> dict[str, bool]:
    flags = {
        NO_RAG_METHOD: args.run_no_rag,
        BUSINESS_RULES_METHOD: args.run_business_rules,
        ORTOOLS_FULL_REPLAN_METHOD: args.run_ortools_full_replan,
        BM25_RAG_METHOD: args.run_bm25_rag,
        HYBRID_RAG_METHOD: args.run_hybrid_rag,
        RULE_PROMPT_METHOD: args.run_rule_prompt,
        LEARNED_RAG_METHOD: args.run_learned_rag,
    }
    if args.run_all_baselines:
        for method in RUN_ALL_BASELINE_METHODS:
            flags[method] = True
    return flags


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    invalid_scenarios = sorted(set(scenarios) - set(SCENARIO_ORDER))
    if invalid_scenarios:
        raise ValueError(f"Unknown scenarios: {invalid_scenarios}; available: {list(SCENARIO_ORDER)}")
    top_k_values = sorted({int(item.strip()) for item in args.top_k_values.split(",") if item.strip()})
    if not top_k_values or min(top_k_values) <= 0:
        raise ValueError("--top-k-values must contain positive integers")

    cases_path = args.cases_path.resolve()
    if args.regenerate_cases > 0:
        cases_path = regenerate_cases(
            args.generated_cases_path.resolve(),
            total=args.regenerate_cases,
            workers=args.case_workers,
            chunk_size=args.case_chunk_size,
        )

    cases, load_metadata = load_cases(cases_path, args.max_cases, set(scenarios), args.seed)
    train_size = min(args.train_cases, len(cases)) if args.train_cases > 0 else len(cases)
    train_cases = random.Random(args.seed).sample(cases, train_size)
    queries = build_queries(cases, scenarios, args.query_limit, args.seed)
    if not queries:
        raise RuntimeError("No experiment queries were generated from loaded cases.")

    scenario_info = scenario_metadata(scenarios, args.seed)
    run_flags = build_run_flags(args)
    experiments = evaluate_experiments(cases, train_cases, queries, top_k_values, run_flags=run_flags, seed=args.seed)
    for row in experiments["scenario_performance"]:
        row.update(scenario_info.get(row["scenario"], {}))

    metadata = {
        "project_root": str(REPO_ROOT),
        "output_dir": str(output_dir),
        "cases_path": str(cases_path),
        "scenarios": scenarios,
        "top_k_values": top_k_values,
        "seed": args.seed,
        "train_cases": train_size,
        "query_count": len(queries),
        "case_loading": load_metadata,
        "scenario_metadata": scenario_info,
    }
    enabled_optional_methods = [method for method in OPTIONAL_METHODS if run_flags.get(method, False)]
    if enabled_optional_methods:
        metadata["enabled_optional_methods"] = enabled_optional_methods
    if args.run_no_rag:
        metadata["run_no_rag"] = True
    if args.run_all_baselines:
        metadata["run_all_baselines"] = True
    payload = {"metadata": metadata, **experiments}

    save_json(output_dir / "paper_experiment_data.json", payload)
    save_json(output_dir / "retrieval_model_comparison.json", experiments["model_comparison"])
    save_json(output_dir / "scenario_performance.json", experiments["scenario_performance"])
    save_json(output_dir / "anomaly_type_accuracy.json", experiments["anomaly_accuracy"])
    save_json(output_dir / "retrieval_quality.json", experiments["retrieval_quality"])
    save_json(output_dir / "manifest.json", metadata)

    save_csv(output_dir / "retrieval_model_comparison.csv", experiments["model_comparison"])
    save_csv(output_dir / "scenario_performance.csv", experiments["scenario_performance"])
    save_csv(output_dir / "anomaly_type_accuracy.csv", experiments["anomaly_accuracy"])
    save_csv(output_dir / "retrieval_quality.csv", experiments["retrieval_quality"])
    save_csv(output_dir / "raw_query_results.csv", experiments["raw_query_results"])

    print(f"Loaded {len(cases)} valid cases; trained on {train_size}; evaluated {len(queries)} queries.")
    print(f"Raw paper data written to: {output_dir}")


if __name__ == "__main__":
    main()
