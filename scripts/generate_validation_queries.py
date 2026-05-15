"""Sample validation queries from the generated logistics case bank."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


NUMERIC_FEATURES = [
    "severity",
    "vehicles_count",
    "orders_count",
    "current_load_rate",
    "urgent_orders",
    "available_backup_vehicles",
    "avg_delay_minutes",
    "affected_routes",
    "time_window_pressure",
    "customer_priority_mix",
    "cost_before",
    "before_distance",
    "unassigned_before",
]


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    cases = []
    with cases_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def case_to_query(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case["id"],
        "event_type": case["event_type"],
        "features": [to_float(case.get(name)) for name in NUMERIC_FEATURES],
        "gt_action": case["action"],
    }


def print_summary(queries: list[dict[str, Any]]) -> None:
    action_counts = Counter(query["gt_action"] for query in queries)
    event_counts = Counter(query["event_type"] for query in queries)

    print(f"Total queries: {len(queries)}")
    print(f"Total queries per action type: {dict(sorted(action_counts.items()))}")
    print(f"Event type distribution: {dict(sorted(event_counts.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample validation queries from a case JSONL file.")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument(
        "--cases-path",
        type=Path,
        default=ROOT / "data" / "cases" / "cases_30k.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "queries" / "validation_1000.jsonl",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases_path)
    rng = random.Random(args.seed)
    sampled = rng.sample(cases, min(args.count, len(cases)))
    queries = [case_to_query(case) for case in sampled]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query, ensure_ascii=False) + "\n")

    print_summary(queries)


if __name__ == "__main__":
    main()
