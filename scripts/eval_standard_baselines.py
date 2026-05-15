"""Evaluate standard RAG baselines on solver-driven logistics queries."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.llm import LLMGateway  # noqa: E402
from src.rag import Case  # noqa: E402
from src.rag.bm25_retriever import BM25Retriever  # noqa: E402
from src.rag.case_retriever import RetrievedCase  # noqa: E402
from src.rag.solver_ground_truth import SolverDrivenGroundTruth  # noqa: E402


SCENARIOS = ["small", "medium", "large", "stress"]
ALL_EVENT_TYPES = [
    "vehicle_breakdown",
    "traffic_accident",
    "road_closed",
    "order_cancel",
    "vehicle_maintenance",
    "fuel_shortage",
    "driver_unavailable",
    "traffic_congestion",
    "road_narrowing",
    "bridge_weight_limit",
    "order_modify",
    "order_update",
    "priority_order_urgent",
    "delivery_failure",
    "demand_surge",
    "demand_drop",
    "weather_delay",
    "natural_disaster",
    "public_event",
    "warehouse_delay",
    "inventory_stockout",
]
METHODS = ["no_rag", "bm25_rag", "tfidf_rag", "dense_rag"]
VALID_ACTIONS = {
    "reroute",
    "ignore",
    "adjust_capacity",
    "reassign_order",
    "delay_tolerant",
}
CONTEXT_KEYS = [
    "current_load_rate",
    "available_backup_vehicles",
    "time_window_pressure",
    "avg_delay_minutes",
    "urgent_orders",
    "customer_priority_mix",
]
RAW_FIELDNAMES = [
    "query_id",
    "method",
    "event_type",
    "severity",
    "scenario",
    "gt_action",
    "gt_actions",
    "pred_action",
    "correct_single",
    "correct_multi",
    "latency_ms",
    "error",
    "raw_response",
]

ENGLISH_SYSTEM_PROMPT = (
    "You are a logistics dispatch advisor. Given an anomaly and similar historical cases,\n"
    "recommend the single best action.\n"
    'Actions: "reroute", "ignore", "adjust_capacity", "reassign_order", "delay_tolerant"\n'
    'CRITICAL: Output ONLY valid JSON. Do NOT include any explanation, analysis, or extra text.\n'
    'Format: {"action": "...", "reasoning": "...", "reroute_needed": true/false}'
)


def get_top_gt_actions(gt_action: str, details: dict[str, Any], top_k: int = 3) -> list[str]:
    costs = details.get("costs", {})
    sorted_actions = sorted(
        [action for action, cost in costs.items() if cost < float("inf")],
        key=lambda action: costs[action],
    )
    return sorted_actions[:top_k] or [gt_action]


def generate_queries(query_count: int = 200, seed: int = 20260428) -> list[dict[str, Any]]:
    solver = SolverDrivenGroundTruth(seed=seed)
    rng = random.Random(seed)
    queries = []
    for query_id in range(query_count):
        scenario = rng.choice(SCENARIOS)
        event_type = rng.choice(ALL_EVENT_TYPES)
        severity = rng.randint(1, 8)
        context = solver.generate_context(scenario, event_type, severity, rng)
        gt_action, details = solver.best_action(scenario, event_type, severity, context)
        gt_actions = get_top_gt_actions(gt_action, details, top_k=3)
        queries.append(
            {
                "query_id": query_id,
                "event_type": event_type,
                "severity": severity,
                "scenario": scenario,
                "context": context,
                "gt_action": gt_action,
                "gt_actions": gt_actions,
            }
        )
    return queries


def load_cases(cases_path: str | Path) -> list[Case]:
    valid_fields = {field.name for field in fields(Case)}
    cases: list[Case] = []
    with Path(cases_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            filtered = {key: value for key, value in data.items() if key in valid_fields}
            filtered.setdefault("timestamp", 0.0)
            cases.append(Case(**filtered))
    return cases


def case_to_json(case: Case) -> str:
    return json.dumps(case.to_dict(), ensure_ascii=False)


class TfidfRetriever:
    def __init__(self, cases_path: str | Path):
        self.cases = load_cases(cases_path)
        self.vectorizer = TfidfVectorizer()
        self.case_docs = [self._to_doc(case) for case in self.cases]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.case_docs)

    def retrieve(
        self,
        event_type: str,
        severity: int,
        scenario: str,
        context: dict[str, Any] | None,
        top_k: int = 5,
    ) -> list[RetrievedCase]:
        query_doc = self._to_query_doc(event_type, severity, scenario, context)
        query_vec = self.vectorizer.transform([query_doc])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        return [
            RetrievedCase(
                case=self.cases[index],
                score=float(scores[index]),
                match_reason="tfidf",
            )
            for index in top_indices
        ]

    def _to_doc(self, case: Case) -> str:
        return (
            f"{case.event_type} {case.scenario} severity={case.severity} "
            f"load_rate={case.current_load_rate} action={case.action} "
            f"outcome={case.outcome} {case.reasoning or ''}"
        )

    def _to_query_doc(
        self,
        event_type: str,
        severity: int,
        scenario: str,
        context: dict[str, Any] | None,
    ) -> str:
        parts = [f"{event_type} severity={severity} {scenario}"]
        if context:
            for key in CONTEXT_KEYS:
                value = context.get(key, 0)
                if value != 0:
                    parts.append(f"{key}={value}")
        return " ".join(parts)


class DenseRetriever(TfidfRetriever):
    def __init__(
        self,
        cases_path: str | Path,
        emb_path: str | Path,
        cases_cache_path: str | Path,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cases = load_cases(cases_path)
        self.case_docs = [self._to_doc(case) for case in self.cases]
        self.emb_path = Path(emb_path)
        self.cases_cache_path = Path(cases_cache_path)

        if self.emb_path.exists():
            self.embeddings = np.load(self.emb_path)
            if self.embeddings.shape[0] != len(self.cases):
                print(
                    "Dense embedding cache size does not match cases; rebuilding.",
                    flush=True,
                )
                self.embeddings = self._build_embeddings()
        else:
            self.embeddings = self._build_embeddings()

    def _build_embeddings(self) -> np.ndarray:
        print("Building dense embeddings with all-MiniLM-L6-v2...", flush=True)
        embeddings = self.model.encode(
            self.case_docs,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = normalize_rows(embeddings)
        self.emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.emb_path, embeddings)
        with self.cases_cache_path.open("w", encoding="utf-8") as handle:
            for case in self.cases:
                handle.write(case_to_json(case) + "\n")
        return embeddings

    def retrieve(
        self,
        event_type: str,
        severity: int,
        scenario: str,
        context: dict[str, Any] | None,
        top_k: int = 5,
    ) -> list[RetrievedCase]:
        query_doc = self._to_query_doc(event_type, severity, scenario, context)
        query_emb = self.model.encode([query_doc], convert_to_numpy=True)
        query_emb = normalize_rows(query_emb)
        scores = (self.embeddings @ query_emb.T).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        return [
            RetrievedCase(
                case=self.cases[index],
                score=float(scores[index]),
                match_reason="dense",
            )
            for index in top_indices
        ]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return values / norms


def format_cases(retrieved_cases: list[RetrievedCase]) -> str:
    if not retrieved_cases:
        return "  (none)"

    lines = []
    for index, item in enumerate(retrieved_cases, start=1):
        case = item.case
        lines.append(
            f"  Case {index}: event={case.event_type}, severity={case.severity}, "
            f"scenario={case.scenario}, outcome={case.outcome}, action={case.action}, "
            f"score={item.score:.4f}, load_rate={case.current_load_rate}, "
            f"backup={case.available_backup_vehicles}, pressure={case.time_window_pressure}, "
            f"delay={case.avg_delay_minutes}, urgent={case.urgent_orders}, "
            f"priority_mix={case.customer_priority_mix}, reasoning={case.reasoning}"
        )
    return "\n".join(lines)


def build_prompt(
    event_type: str,
    severity: int,
    scenario: str,
    context: dict[str, Any],
    retrieved_cases: list[RetrievedCase] | None,
) -> tuple[str, str]:
    cases_block = format_cases(retrieved_cases or [])
    user_prompt = (
        "## Current Anomaly\n"
        f"  event_type: {event_type}\n"
        f"  severity: {severity}\n"
        f"  scenario: {scenario}\n"
        f"  current_load_rate: {context.get('current_load_rate', 0)}\n"
        f"  available_backup_vehicles: {context.get('available_backup_vehicles', 0)}\n"
        f"  time_window_pressure: {context.get('time_window_pressure', 0)}\n"
        f"  avg_delay_minutes: {context.get('avg_delay_minutes', 0)}\n"
        f"  urgent_orders: {context.get('urgent_orders', 0)}\n"
        f"  customer_priority_mix: {context.get('customer_priority_mix', 0)}\n\n"
        "## Similar Historical Cases\n"
        f"{cases_block}\n\n"
        "Recommend the best action."
    )
    return ENGLISH_SYSTEM_PROMPT, user_prompt


def retrieve_for_method(
    method: str,
    query: dict[str, Any],
    retrievers: dict[str, Any],
) -> tuple[str, str]:
    event_type = query["event_type"]
    severity = query["severity"]
    scenario = query["scenario"]
    context = query["context"]

    if method == "no_rag":
        return build_prompt(event_type, severity, scenario, context, [])
    if method == "bm25_rag":
        retrieved = retrievers["bm25"].retrieve(
            event_type,
            severity,
            scenario,
            context=context,
            top_k=5,
        )
        return build_prompt(event_type, severity, scenario, context, retrieved)
    if method == "tfidf_rag":
        retrieved = retrievers["tfidf"].retrieve(
            event_type,
            severity,
            scenario,
            context=context,
            top_k=5,
        )
        return build_prompt(event_type, severity, scenario, context, retrieved)
    if method == "dense_rag":
        retrieved = retrievers["dense"].retrieve(
            event_type,
            severity,
            scenario,
            context=context,
            top_k=5,
        )
        return build_prompt(event_type, severity, scenario, context, retrieved)
    raise ValueError(f"Unknown method: {method}")


def extract_action(text: str) -> str | None:
    json_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            action = str(data.get("action", "")).lower().strip()
            if action in VALID_ACTIONS:
                return action
        except json.JSONDecodeError:
            pass

    action_match = re.search(r'"action"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if action_match:
        action = action_match.group(1).lower().strip()
        if action in VALID_ACTIONS:
            return action
    return None


def call_llm_with_retries(
    llm: LLMGateway,
    user_prompt: str,
    system_prompt: str,
    *,
    retries: int = 2,
    retry_sleep_seconds: float = 5.0,
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return llm.generate(user_prompt, system_prompt=system_prompt, max_tokens=1024)
        except Exception as exc:  # noqa: BLE001 - checkpoint row records final API error.
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep_seconds)
    assert last_error is not None
    raise last_error


def parse_gt_actions(value: Any, fallback: str) -> list[str]:
    if isinstance(value, list):
        return [str(action) for action in value]
    if isinstance(value, str) and value:
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list):
                return [str(action) for action in loaded]
        except json.JSONDecodeError:
            pass
    return [fallback]


def normalize_checkpoint_row(
    row: dict[str, Any],
    query: dict[str, Any] | None,
) -> dict[str, Any]:
    if query is not None:
        gt_actions = (
            parse_gt_actions(row.get("gt_actions"), query["gt_action"])
            if row.get("gt_actions")
            else list(query["gt_actions"])
        )
        row["gt_action"] = query["gt_action"]
        row["gt_actions"] = json.dumps(gt_actions, ensure_ascii=False)
        row["correct_single"] = int(row.get("pred_action") == query["gt_action"])
        row["correct_multi"] = int(row.get("pred_action") in gt_actions)
    else:
        row["gt_actions"] = json.dumps(
            parse_gt_actions(row.get("gt_actions"), row.get("gt_action", "")),
            ensure_ascii=False,
        )
        row["correct_single"] = int(row.get("correct_single", 0))
        row["correct_multi"] = int(row.get("correct_multi", 0))
    row["error"] = bool(row.get("error", False))
    row["latency_ms"] = float(row.get("latency_ms", 0.0))
    row["raw_response"] = str(row.get("raw_response", ""))
    return {field: row.get(field, "") for field in RAW_FIELDNAMES}


def load_checkpoint(
    checkpoint_path: Path,
    queries_by_id: dict[int, dict[str, Any]],
    methods: list[str],
) -> dict[tuple[int, str], dict[str, Any]]:
    rows_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    if not checkpoint_path.exists():
        return rows_by_key

    with checkpoint_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed checkpoint line {line_number}", flush=True)
                continue
            method = row.get("method")
            query_id = row.get("query_id")
            if method not in methods or not isinstance(query_id, int):
                continue
            query = queries_by_id.get(query_id)
            rows_by_key[(query_id, method)] = normalize_checkpoint_row(row, query)
    return rows_by_key


def make_result_row(
    query: dict[str, Any],
    method: str,
    pred_action: str,
    raw_response: str,
    latency_ms: float,
    error: bool,
) -> dict[str, Any]:
    return {
        "query_id": query["query_id"],
        "method": method,
        "event_type": query["event_type"],
        "severity": query["severity"],
        "scenario": query["scenario"],
        "gt_action": query["gt_action"],
        "gt_actions": json.dumps(query["gt_actions"], ensure_ascii=False),
        "pred_action": pred_action,
        "correct_single": int(pred_action == query["gt_action"]),
        "correct_multi": int(pred_action in query["gt_actions"]),
        "latency_ms": round(latency_ms, 2),
        "error": bool(error),
        "raw_response": raw_response,
    }


def sorted_rows(
    rows_by_key: dict[tuple[int, str], dict[str, Any]],
    methods: list[str],
) -> list[dict[str, Any]]:
    method_order = {method: index for index, method in enumerate(methods)}
    return [
        row
        for _, row in sorted(
            rows_by_key.items(),
            key=lambda item: (item[0][0], method_order.get(item[0][1], 999)),
        )
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(methods: list[str], raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows = []
    for method in methods:
        rows = [row for row in raw_rows if row["method"] == method]
        total = len(rows)
        correct_single = sum(int(row["correct_single"]) for row in rows)
        correct_multi = sum(int(row["correct_multi"]) for row in rows)
        summary_rows.append(
            {
                "method": method,
                "correct_single": correct_single,
                "correct_multi": correct_multi,
                "total": total,
                "accuracy_single_pct": round(correct_single / max(1, total) * 100, 2),
                "accuracy_multi_pct": round(correct_multi / max(1, total) * 100, 2),
                "avg_latency_ms": round(
                    sum(float(row["latency_ms"]) for row in rows) / max(1, total),
                    2,
                ),
                "error_count": sum(1 for row in rows if row["error"]),
            }
        )
    return summary_rows


def print_summary(methods: list[str], raw_rows: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 60}")
    print("Overall Results")
    print(f"{'=' * 60}")
    for method in methods:
        rows = [row for row in raw_rows if row["method"] == method]
        total = len(rows)
        correct_single = sum(int(row["correct_single"]) for row in rows)
        correct_multi = sum(int(row["correct_multi"]) for row in rows)
        latency = sum(float(row["latency_ms"]) for row in rows) / max(1, total)
        errors = sum(1 for row in rows if row["error"])
        print(
            f"  {method:12s} (single): {correct_single}/{total} = "
            f"{correct_single / max(1, total) * 100:.1f}%  "
            f"({latency:.0f}ms, errors={errors})"
        )
        print(
            f"  {method:12s} (multi):  {correct_multi}/{total} = "
            f"{correct_multi / max(1, total) * 100:.1f}%  "
            f"({latency:.0f}ms, errors={errors})"
        )


def print_action_breakdown(methods: list[str], raw_rows: list[dict[str, Any]]) -> None:
    gt_actions = sorted(
        {
            action
            for row in raw_rows
            for action in parse_gt_actions(row.get("gt_actions"), row["gt_action"])
        }
    )
    print(f"\n{'=' * 60}")
    print("Per-Action Breakdown (multi-GT acceptable actions)")
    print(f"{'=' * 60}")
    for action in gt_actions:
        print(f"\n  {action}:")
        for method in methods:
            rows = [
                row
                for row in raw_rows
                if row["method"] == method
                and action in parse_gt_actions(row.get("gt_actions"), row["gt_action"])
            ]
            total = len(rows)
            correct = sum(int(row["correct_multi"]) for row in rows)
            print(f"    {method:12s} {correct}/{total} = {correct / max(1, total) * 100:.1f}%")

    prediction_counts: dict[str, Counter[str]] = {
        method: Counter(
            row["pred_action"] for row in raw_rows if row["method"] == method
        )
        for method in methods
    }
    print(f"\n{'=' * 60}")
    print("Prediction Distribution")
    print(f"{'=' * 60}")
    for method in methods:
        print(f"  {method:12s} {dict(prediction_counts[method])}")


def parse_methods(values: list[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(item.strip() for item in value.split(",") if item.strip())
    if not tokens or tokens == ["all"]:
        return list(METHODS)
    unknown = [token for token in tokens if token not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return list(dict.fromkeys(tokens))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--cases-path",
        type=Path,
        default=ROOT / "data" / "cases" / "cases_30k.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "standard_baselines_eval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.jsonl"

    if args.force and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Removed checkpoint: {checkpoint_path}", flush=True)

    print(f"Generating {args.query_count} queries (seed={args.seed})", flush=True)
    queries = generate_queries(args.query_count, args.seed)
    queries_by_id = {query["query_id"]: query for query in queries}

    rows_by_key = load_checkpoint(checkpoint_path, queries_by_id, methods)
    complete_queries = sum(
        1
        for query in queries
        if all((query["query_id"], method) in rows_by_key for method in methods)
    )
    if rows_by_key:
        print(
            f"Resuming from checkpoint: {len(rows_by_key)} method rows, "
            f"{complete_queries}/{len(queries)} complete queries",
            flush=True,
        )
    else:
        print("No checkpoint rows found; starting fresh", flush=True)

    print(f"Loading retrievers from {args.cases_path}", flush=True)
    retrievers: dict[str, Any] = {}
    if "bm25_rag" in methods:
        retrievers["bm25"] = BM25Retriever(str(args.cases_path))
    if "tfidf_rag" in methods:
        retrievers["tfidf"] = TfidfRetriever(args.cases_path)
    if "dense_rag" in methods:
        retrievers["dense"] = DenseRetriever(
            args.cases_path,
            emb_path=output_dir / "dense_embeddings.npy",
            cases_cache_path=output_dir / "dense_cases_cache.jsonl",
        )

    print("Initializing LLM gateway", flush=True)
    llm = LLMGateway(timeout=120)

    for query_index, query in enumerate(queries, start=1):
        missing_methods = [
            method
            for method in methods
            if (query["query_id"], method) not in rows_by_key
        ]
        if not missing_methods:
            continue

        for method in missing_methods:
            started = time.perf_counter()
            pred_action = "error"
            raw_response = ""
            error = False

            try:
                system_prompt, user_prompt = retrieve_for_method(method, query, retrievers)
                raw_response = call_llm_with_retries(
                    llm,
                    user_prompt,
                    system_prompt,
                    retries=2,
                    retry_sleep_seconds=5.0,
                )
                parsed_action = extract_action(raw_response)
                if parsed_action is None:
                    error = True
                    raw_response = (
                        "action_parse_failed\n"
                        f"{raw_response}"
                    )
                else:
                    pred_action = parsed_action
            except Exception as exc:  # noqa: BLE001 - row captures final failure and resumes.
                error = True
                raw_response = f"{type(exc).__name__}: {exc}"

            latency_ms = (time.perf_counter() - started) * 1000.0
            row = make_result_row(
                query,
                method,
                pred_action,
                raw_response,
                latency_ms,
                error,
            )
            rows_by_key[(query["query_id"], method)] = row
            with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
                checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")

        if query_index % 20 == 0 or query_index == len(queries):
            complete_queries = sum(
                1
                for item in queries
                if all((item["query_id"], method) in rows_by_key for method in methods)
            )
            print(
                f"Evaluated through query {query_index}/{len(queries)} "
                f"({complete_queries} complete)",
                flush=True,
            )

    raw_rows = sorted_rows(rows_by_key, methods)
    write_csv(output_dir / "raw_results.csv", raw_rows, RAW_FIELDNAMES)
    summary_rows = build_summary_rows(methods, raw_rows)
    write_csv(
        output_dir / "overall_summary.csv",
        summary_rows,
        [
            "method",
            "correct_single",
            "correct_multi",
            "total",
            "accuracy_single_pct",
            "accuracy_multi_pct",
            "avg_latency_ms",
            "error_count",
        ],
    )

    print_summary(methods, raw_rows)
    print_action_breakdown(methods, raw_rows)
    print(f"\nSaved outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
