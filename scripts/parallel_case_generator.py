"""Parallel solver-driven case generator. Splits total into N workers, merges at the end."""
import sys
import json
import time
import random
import hashlib
import multiprocessing as mp
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.rag.solver_ground_truth import SolverDrivenGroundTruth
from src.rag.experience_model import ALL_EVENT_TYPES, ALL_SCENARIOS
from src.utils.paper_data_generator import (
    SCENARIO_ORDER, SCENARIO_BENCHMARKS, EVENT_BENCHMARK_WEIGHTS, Case,
)

VALID_ACTIONS = {"reroute", "ignore", "adjust_capacity", "reassign_order", "delay_tolerant", "normal"}


def worker(args):
    worker_id, total, seed = args
    rng = random.Random(seed + worker_id * 100000)
    solver_gt = SolverDrivenGroundTruth(seed=seed + worker_id * 100000)
    event_types, event_weights = zip(*EVENT_BENCHMARK_WEIGHTS)
    cases = []
    start = time.perf_counter()

    for index in range(total):
        global_index = worker_id * total + index
        scenario = SCENARIO_ORDER[global_index % len(SCENARIO_ORDER)]
        event_type = rng.choices(event_types, weights=event_weights, k=1)[0]
        severe_prob = SCENARIO_BENCHMARKS[scenario]["severe_event_probability"]
        severity = rng.randint(4, 5) if rng.random() < severe_prob else rng.randint(1, 3)
        context = solver_gt.generate_context(scenario, event_type, severity, rng)
        action, details = solver_gt.best_action(scenario, event_type, severity, context)
        results = details.get("results", {})
        action_result = results.get(action, {})
        costs = details.get("costs", {})
        solved = bool(details.get("solved", False))
        benchmark = SCENARIO_BENCHMARKS[scenario]
        cost_before = float(action_result.get("cost_before", 0.0))
        cost_after = float(costs.get(action, action_result.get("cost", 0.0)))
        unassigned_after = int(action_result.get("unassigned", 0))
        outcome = "success" if solved and unassigned_after <= max(2, int(benchmark["orders"] * 0.08)) else "failure"

        case = Case(
            id=f"case_{global_index + 1:07d}",
            event_type=event_type, severity=severity, scenario=scenario,
            vehicles_count=benchmark["vehicles"], orders_count=benchmark["orders"],
            action=action, outcome=outcome,
            reasoning=(
                f"Solver selected {action} from context-sensitive candidate costs. "
                f"Load={context['current_load_rate']:.2f}, urgent={context['urgent_orders']}, "
                f"backup={context['available_backup_vehicles']}, pressure={context['time_window_pressure']:.2f}."
            ),
            before_distance=round(cost_before, 2), after_distance=round(cost_after, 2),
            unassigned_before=0, unassigned_after=unassigned_after,
            timestamp=round(global_index / 1000000, 6),
            current_load_rate=context["current_load_rate"],
            urgent_orders=context["urgent_orders"],
            available_backup_vehicles=context["available_backup_vehicles"],
            avg_delay_minutes=context["avg_delay_minutes"],
            affected_routes=context["affected_routes"],
            time_window_pressure=context["time_window_pressure"],
            customer_priority_mix=context["customer_priority_mix"],
            cost_before=round(cost_before, 3), cost_after=round(cost_after, 3),
            candidate_costs={c: round(float(v), 3) for c, v in costs.items()},
            solved=solved,
        )
        cases.append(case.to_dict())

    elapsed = time.perf_counter() - start
    print(f"Worker {worker_id}: {total} cases in {elapsed:.1f}s ({elapsed*1000/max(total,1):.1f}ms/case)")
    return cases


def main():
    argparse = __import__("argparse")
    parser = argparse.ArgumentParser(description="Generate solver-driven logistics cases in parallel.")
    parser.add_argument("--count", type=int, default=30000, help="number of cases to generate")
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "cases" / "cases_30k.jsonl",
    )
    cli_args = parser.parse_args()

    total = cli_args.count
    num_workers = cli_args.workers
    per_worker = total // num_workers
    seed = cli_args.seed
    output_path = cli_args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {total} solver-driven cases with {num_workers} workers...")
    start = time.perf_counter()

    args = [(i, per_worker, seed) for i in range(num_workers)]
    with mp.Pool(num_workers) as pool:
        results = pool.map(worker, args)

    all_cases = []
    for batch in results:
        all_cases.extend(batch)
    all_cases.sort(key=lambda c: c["id"])

    action_counts = Counter(c["action"] for c in all_cases)
    solved_count = sum(1 for c in all_cases if c.get("solved"))

    with output_path.open("w", encoding="utf-8") as f:
        for case in all_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - start
    print(f"\nDone! {len(all_cases)} cases written to {output_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed*1000/max(len(all_cases),1):.1f}ms/case)")
    print(f"Solved: {solved_count}/{len(all_cases)} ({100*solved_count/max(len(all_cases),1):.1f}%)")
    print(f"Action distribution: {dict(sorted(action_counts.items()))}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
