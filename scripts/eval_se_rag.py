"""SE-RAG end-to-end evaluation.

Runs the same 200-query test as eval_full_comparison but includes SE-RAG
as an additional method.  Can run standalone or alongside the original.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.llm import LLMGateway
from src.rag.case_retriever import CaseRetriever
from src.rag.causal_extractor import CausalRelationExtractor
from src.rag.negative_aware_retriever import DEFAULT_CASES_PATH
from src.rag.rule_miner import DecisionRuleMiner, MinedRuleBase
from src.rag.se_rag_retriever import (
    SE_RAG_SYSTEM_PROMPT,
    RuleMatcher,
    build_se_rag_prompt,
)
from src.rag.solver_ground_truth import SolverDrivenGroundTruth

SCENARIOS = ["small", "medium", "large", "stress"]
ALL_EVENT_TYPES = [
    "vehicle_breakdown", "traffic_accident", "road_closed", "order_cancel",
    "vehicle_maintenance", "fuel_shortage", "driver_unavailable", "traffic_congestion",
    "road_narrowing", "bridge_weight_limit", "order_modify", "order_update",
    "priority_order_urgent", "delivery_failure", "demand_surge", "demand_drop",
    "weather_delay", "natural_disaster", "public_event", "warehouse_delay",
    "inventory_stockout",
]
VALID_ACTIONS = {"reroute", "ignore", "adjust_capacity", "reassign_order", "delay_tolerant"}

# Simplified No-RAG prompt for baseline comparison
NO_RAG_SYSTEM_PROMPT = (
    "You are a logistics dispatch advisor. Given an anomaly, "
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


def method_label(method: str) -> str:
    return {
        "no_rag": "No-RAG",
        "se_rag": "SE-RAG",
    }.get(method, method)


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


def generate_queries(query_count: int, seed: int) -> list[dict[str, Any]]:
    solver_gt = SolverDrivenGroundTruth(seed=seed)
    rng = random.Random(seed)
    queries = []
    for query_id in range(query_count):
        scenario = rng.choice(SCENARIOS)
        event_type = rng.choice(ALL_EVENT_TYPES)
        severity = rng.randint(1, 8)
        context = solver_gt.generate_context(scenario, event_type, severity, rng)
        gt_action, details = solver_gt.best_action(scenario, event_type, severity, context)
        gt_actions = get_top_gt_actions(gt_action, details)
        queries.append({
            "query_id": query_id,
            "event_type": event_type,
            "severity": severity,
            "scenario": scenario,
            "context": context,
            "gt_action": gt_action,
            "gt_actions": gt_actions,
        })
    return queries


def build_no_rag_prompt(query: dict) -> tuple[str, str]:
    ctx = query["context"]
    user = (
        "## Current Anomaly\n"
        f"  event_type: {query['event_type']}\n"
        f"  severity: {query['severity']}\n"
        f"  scenario: {query['scenario']}\n"
        f"  current_load_rate: {ctx.get('current_load_rate', 0)}\n"
        f"  available_backup_vehicles: {ctx.get('available_backup_vehicles', 0)}\n"
        f"  time_window_pressure: {ctx.get('time_window_pressure', 0)}\n"
        f"  avg_delay_minutes: {ctx.get('avg_delay_minutes', 0)}\n"
        f"  urgent_orders: {ctx.get('urgent_orders', 0)}\n"
        f"  customer_priority_mix: {ctx.get('customer_priority_mix', 0)}\n\n"
        "Recommend the best action."
    )
    return NO_RAG_SYSTEM_PROMPT, user


def build_fallback_prompt(query: dict, similar_cases: list[Any]) -> tuple[str, str]:
    ctx = query["context"]
    parts = [
        "## Current Anomaly",
        f"  event_type: {query['event_type']}",
        f"  severity: {query['severity']}",
        f"  scenario: {query['scenario']}",
        f"  current_load_rate: {ctx.get('current_load_rate', 0)}",
        f"  available_backup_vehicles: {ctx.get('available_backup_vehicles', 0)}",
        f"  time_window_pressure: {ctx.get('time_window_pressure', 0)}",
        f"  avg_delay_minutes: {ctx.get('avg_delay_minutes', 0)}",
        f"  urgent_orders: {ctx.get('urgent_orders', 0)}",
        f"  customer_priority_mix: {ctx.get('customer_priority_mix', 0)}",
        "",
        "## Fallback Historical Cases",
        "No exact rules matched, but these similar historical cases were found:",
    ]

    if not similar_cases:
        parts.append("  (none)")

    for i, item in enumerate(similar_cases, 1):
        case = item.case
        parts.append(
            f"  Case {i}: event={case.event_type}, severity={case.severity}, "
            f"scenario={case.scenario}, outcome={case.outcome}, action={case.action}, "
            f"score={item.score:.4f}, load_rate={case.current_load_rate}, "
            f"backup={case.available_backup_vehicles}, pressure={case.time_window_pressure}, "
            f"avg_delay={case.avg_delay_minutes}, urgent_orders={case.urgent_orders}, "
            f"priority_mix={case.customer_priority_mix}"
        )
        if getattr(item, "match_reason", ""):
            parts.append(f"    match_reason: {item.match_reason}")
        if case.reasoning:
            parts.append(f"    reasoning: {case.reasoning[:240]}")

    parts.append("")
    parts.append("Recommend the best action based on the current anomaly and fallback historical cases.")
    return SE_RAG_SYSTEM_PROMPT, "\n".join(parts)


def retrieve_fallback_cases(
    fallback_retriever: CaseRetriever,
    query: dict[str, Any],
    top_k: int = 5,
) -> list[Any]:
    retrieve_params = inspect.signature(fallback_retriever.retrieve).parameters
    if "context" in retrieve_params:
        return fallback_retriever.retrieve(
            query["event_type"],
            query["severity"],
            query["scenario"],
            query["context"],
            top_k=top_k,
        )
    return fallback_retriever.retrieve(
        query["event_type"],
        query["severity"],
        query["scenario"],
        top_k=top_k,
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "se_rag_eval")
    parser.add_argument("--cases-path", type=Path, default=Path(DEFAULT_CASES_PATH))
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1 + 2: Offline mining
    rule_base_path = output_dir / "rule_base.json"
    causal_graph_path = output_dir / "causal_graph.json"

    if rule_base_path.exists():
        print(f"Loading existing rule base from {rule_base_path}")
        rule_base = MinedRuleBase.load(rule_base_path)
    else:
        print("Mining decision rules...")
        miner = DecisionRuleMiner(str(args.cases_path))
        rule_base = miner.mine()
        rule_base.save(rule_base_path)
        print(f"Mined {len(rule_base.rules)} rules → {rule_base_path}")

    if causal_graph_path.exists():
        from src.rag.causal_extractor import CausalGraph
        print(f"Loading existing causal graph from {causal_graph_path}")
        causal_graph = CausalGraph.load(causal_graph_path)
    else:
        print("Extracting causal relations...")
        extractor = CausalRelationExtractor(str(args.cases_path))
        causal_graph = extractor.extract()
        causal_graph.save(causal_graph_path)
        print(f"Found {len(causal_graph.edges)} edges → {causal_graph_path}")

    # Phase 3: Online matcher
    matcher = RuleMatcher(rule_base, causal_graph)
    fallback_retriever = CaseRetriever(str(args.cases_path))
    build_index = getattr(fallback_retriever, "build_index", None)
    if callable(build_index):
        build_index()

    # Generate queries
    print(f"Generating {args.query_count} queries (seed={args.seed})")
    queries = generate_queries(args.query_count, args.seed)
    queries_by_id = {q["query_id"]: q for q in queries}

    # --- Checkpoint / resume logic ---
    methods = ["no_rag", "se_rag"]
    checkpoint_path = output_dir / "checkpoint.jsonl"
    completed: set[int] = set()
    checkpoint_rows: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as cf:
            raw_text = cf.read()
        # Robust parser: extract JSON objects by brace-counting.
        # Handles cases where two records are on the same physical line.
        i = 0
        while i < len(raw_text):
            c = raw_text[i]
            if c == "{":
                depth = 0
                start = i
                in_string = False
                escaped = False
                while i < len(raw_text):
                    ch = raw_text[i]
                    if in_string:
                        if escaped:
                            escaped = False
                        elif ch == "\\":
                            escaped = True
                        elif ch == '"':
                            in_string = False
                    else:
                        if ch == '"':
                            in_string = True
                        elif ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                i += 1
                                obj_text = raw_text[start:i]
                                try:
                                    row = json.loads(obj_text)
                                except json.JSONDecodeError:
                                    break
                                query = queries_by_id.get(row["query_id"])
                                if query:
                                    gt_actions = parse_gt_actions(row.get("gt_actions"), query["gt_action"])
                                    if "gt_actions" not in row:
                                        gt_actions = query["gt_actions"]
                                    row["gt_actions"] = json.dumps(gt_actions, ensure_ascii=False)
                                    row["correct_single"] = int(row.get(
                                        "correct_single",
                                        row.get("pred_action") == query["gt_action"],
                                    ))
                                    row["correct_multi"] = int(row.get(
                                        "correct_multi",
                                        row.get("pred_action") in gt_actions,
                                    ))
                                checkpoint_rows.append(row)
                                completed.add(row["query_id"])
                                break
                    i += 1
            else:
                i += 1
        # A query is "done" when we have both no_rag + se_rag rows
        from collections import Counter as _Counter
        method_counts = _Counter(r["query_id"] for r in checkpoint_rows)
        completed = {qid for qid, cnt in method_counts.items() if cnt >= len(methods)}
        print(f"Resuming from checkpoint: {len(completed)}/{args.query_count} queries already done")
    else:
        print("No checkpoint found, starting fresh")

    # Run evaluation
    llm = LLMGateway()
    raw_rows: list[dict[str, Any]] = checkpoint_rows  # start from checkpoint

    for qi, query in enumerate(queries, 1):
        if query["query_id"] in completed:
            continue  # already done

        for method in methods:
            started = time.perf_counter()
            pred_action = "error"
            raw_response = ""
            error = ""

            try:
                if method == "no_rag":
                    sys_prompt, user_prompt = build_no_rag_prompt(query)
                else:  # se_rag
                    evidence = matcher.match(
                        query["event_type"],
                        query["severity"],
                        query["scenario"],
                        query["context"],
                    )
                    # Save evidence for analysis
                    evidence_path = output_dir / f"evidence_{query['query_id']}.json"
                    if qi <= 5:  # Only save first 5 for inspection
                        with evidence_path.open("w", encoding="utf-8") as f:
                            json.dump(evidence.to_dict(), f, ensure_ascii=False, indent=2)

                    if not evidence.triggered_rules:
                        similar = retrieve_fallback_cases(fallback_retriever, query, top_k=5)
                        sys_prompt, user_prompt = build_fallback_prompt(query, similar)
                    else:
                        sys_prompt, user_prompt = build_se_rag_prompt(evidence)

                raw_response = llm.generate(user_prompt, system_prompt=sys_prompt, max_tokens=1024)
                parsed = extract_action(raw_response)
                if parsed is None:
                    error = "action_parse_failed"
                else:
                    pred_action = parsed
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            correct_single = pred_action == query["gt_action"]
            correct_multi = pred_action in query["gt_actions"]

            row = {
                "query_id": query["query_id"],
                "method": method,
                "event_type": query["event_type"],
                "severity": query["severity"],
                "scenario": query["scenario"],
                "gt_action": query["gt_action"],
                "gt_actions": json.dumps(query["gt_actions"], ensure_ascii=False),
                "pred_action": pred_action,
                "correct_single": int(correct_single),
                "correct_multi": int(correct_multi),
                "latency_ms": latency_ms,
                "error": error,
                "raw_response": raw_response,
            }
            raw_rows.append(row)

            # Incremental save to checkpoint
            with checkpoint_path.open("a", encoding="utf-8") as cf:
                cf.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Mark as completed for this query
        completed.add(query["query_id"])

        if qi % 20 == 0 or qi == len(queries):
            remaining = args.query_count - len(completed)
            print(f"  Evaluated {qi}/{len(queries)} queries ({remaining} remaining / {len(completed)} done)", flush=True)

    # Save results
    write_csv(output_dir / "raw_results.csv", raw_rows)

    # Summary
    print_summary(methods, raw_rows)

    # Per-action breakdown
    print_action_breakdown(methods, raw_rows)

    # Save summary
    summary_rows = []
    for method in methods:
        rows = [r for r in raw_rows if r["method"] == method]
        correct_single = sum(int(r["correct_single"]) for r in rows)
        correct_multi = sum(int(r["correct_multi"]) for r in rows)
        total = len(rows)
        summary_rows.append({
            "method": method,
            "correct_single": correct_single,
            "correct_multi": correct_multi,
            "total": total,
            "accuracy_single_pct": round(correct_single / max(1, total) * 100, 2),
            "accuracy_multi_pct": round(correct_multi / max(1, total) * 100, 2),
            "avg_latency_ms": round(sum(float(r["latency_ms"]) for r in rows) / max(1, total), 2),
            "error_count": sum(1 for r in rows if r["error"]),
        })
    write_csv(output_dir / "overall_summary.csv", summary_rows)

    print(f"\nSaved results to {output_dir}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(methods: list[str], raw_rows: list[dict]) -> None:
    print(f"\n{'='*60}")
    print("Overall Results")
    print(f"{'='*60}")
    for method in methods:
        rows = [r for r in raw_rows if r["method"] == method]
        correct_single = sum(int(r["correct_single"]) for r in rows)
        correct_multi = sum(int(r["correct_multi"]) for r in rows)
        total = len(rows)
        latency = sum(float(r["latency_ms"]) for r in rows) / max(1, total)
        label = method_label(method)
        print(f"  {label:20s} (single): {correct_single}/{total} = {correct_single/max(1,total)*100:.1f}%  ({latency:.0f}ms)")
        print(f"  {label:20s} (multi):  {correct_multi}/{total} = {correct_multi/max(1,total)*100:.1f}%  ({latency:.0f}ms)")


def print_action_breakdown(methods: list[str], raw_rows: list[dict]) -> None:
    gt_actions = sorted({
        action
        for r in raw_rows
        for action in parse_gt_actions(r.get("gt_actions"), r["gt_action"])
    })
    print(f"\n{'='*60}")
    print("Per-Action Breakdown (multi-GT acceptable actions)")
    print(f"{'='*60}")
    for action in gt_actions:
        print(f"\n  {action}:")
        for method in methods:
            rows = [
                r for r in raw_rows
                if r["method"] == method
                and action in parse_gt_actions(r.get("gt_actions"), r["gt_action"])
            ]
            correct = sum(int(r["correct_multi"]) for r in rows)
            total = len(rows)
            print(f"    {method:20s} {correct}/{total} = {correct/max(1,total)*100:.1f}%")

    # Hard actions analysis
    gt_counter = Counter(
        action
        for r in raw_rows
        if r["method"] == methods[0]
        for action in parse_gt_actions(r.get("gt_actions"), r["gt_action"])
    )
    dominant = gt_counter.most_common(1)[0][0]
    hard_actions = [a for a in gt_actions if a != dominant]

    print(f"\n{'='*60}")
    print(f"Hard Actions (non-dominant, excluding '{dominant}')")
    print(f"{'='*60}")
    for method in methods:
        rows = [
            r for r in raw_rows
            if r["method"] == method
            and any(action in hard_actions for action in parse_gt_actions(r.get("gt_actions"), r["gt_action"]))
        ]
        correct = sum(int(r["correct_multi"]) for r in rows)
        total = len(rows)
        print(f"  {method:20s} {correct}/{total} = {correct/max(1,total)*100:.1f}%")


if __name__ == "__main__":
    main()
