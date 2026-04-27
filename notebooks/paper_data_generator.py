from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag import CaseRetriever, ExperienceModel, ExperienceModelV2
from src.rag.case_retriever import EVENT_TYPE_HIERARCHY
from src.rag.experience_model import ALL_EVENT_TYPES


OUTPUT_DIR = REPO_ROOT / "notebooks" / "paper"
SCENARIO_ORDER = ["small", "medium", "large", "stress"]
ACTION_ORDER = ["reroute", "ignore", "adjust_capacity", "reassign_order", "delay_tolerant"]
SEVERITY_LEVELS = [1, 2, 3, 4, 5]
METHOD_LABELS = {
    "baseline": "Baseline",
    "v1": "V1 Experience Model",
    "v2": "V2 Experience Model",
}
METHOD_COLORS = {
    "baseline": "#4C78A8",
    "v1": "#F58518",
    "v2": "#54A24B",
}


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
        }
    )


def ground_truth_action(event_type: str, severity: int) -> str:
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


def coarse_score(case, event_type: str, severity: int, scenario: str) -> float:
    score = 0.0
    if case.event_type == event_type:
        score += 3.0
    else:
        fallback = EVENT_TYPE_HIERARCHY.get(event_type, [])
        if case.event_type in fallback:
            score += 2.0
    score += max(0.0, 2.0 - abs(case.severity - severity) * 0.5)
    if case.scenario == scenario:
        score += 1.0
    return score


def dcg_at_k(relevance: list[int], k: int = 5) -> float:
    return sum(rel / max(1.0, math.log2(index + 2)) for index, rel in enumerate(relevance[:k]))


def ndcg_at_k(sorted_cases: list, gt_action: str, k: int = 5) -> float:
    relevance = [1 if case.action == gt_action else 0 for case in sorted_cases[:k]]
    ideal = sorted(relevance, reverse=True)
    dcg = dcg_at_k(relevance, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def retrieve_baseline(cases: list, event_type: str, severity: int, scenario: str, top_k: int = 5) -> list[tuple[float, object]]:
    scored = [(coarse_score(case, event_type, severity, scenario), case) for case in cases]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def retrieve_v1(
    cases: list,
    exp_model: ExperienceModel,
    event_type: str,
    severity: int,
    scenario: str,
    top_k: int = 5,
) -> list[tuple[float, object]]:
    prediction = exp_model.predict(event_type, severity, scenario)
    exp_action = prediction["action"]
    exp_confidence = prediction["action_confidence"]

    scored = []
    for case in cases:
        base = coarse_score(case, event_type, severity, scenario)
        bonus = 2.0 * exp_confidence if case.action == exp_action else 0.0
        scored.append((base + bonus, case))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def retrieve_v2(
    cases: list,
    exp_v2: ExperienceModelV2,
    event_type: str,
    severity: int,
    scenario: str,
    top_k: int = 5,
) -> list[tuple[float, object]]:
    coarse = retrieve_baseline(cases, event_type, severity, scenario, top_k=30)
    coarse_cases = [case for _, case in coarse]
    coarse_scores = [score for score, _ in coarse]
    relevance_scores = exp_v2.predict_relevance(event_type, severity, scenario, coarse_cases)

    combined = []
    for index, case in enumerate(coarse_cases):
        total_score = coarse_scores[index] * 0.6 + relevance_scores[index] * 10.0 * 0.4
        combined.append((total_score, case))
    combined.sort(key=lambda item: item[0], reverse=True)
    return combined[:top_k]


def build_case_distribution(cases: list) -> dict:
    event_types = sorted({case.event_type for case in cases})
    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in {case.scenario for case in cases}]
    actions = [action for action in ACTION_ORDER if action in {case.action for case in cases}]

    event_scenario_counter: dict[str, Counter] = defaultdict(Counter)
    event_action_counter: dict[str, Counter] = defaultdict(Counter)
    outcome_counter: Counter = Counter()
    action_counter: Counter = Counter()

    for case in cases:
        event_scenario_counter[case.event_type][case.scenario] += 1
        event_action_counter[case.event_type][case.action] += 1
        outcome_counter[case.outcome] += 1
        action_counter[case.action] += 1

    event_scenario_matrix = [
        [event_scenario_counter[event_type][scenario] for scenario in scenarios]
        for event_type in event_types
    ]
    event_action_matrix = [
        [event_action_counter[event_type][action] for action in actions]
        for event_type in event_types
    ]

    return {
        "summary": {
            "total_cases": len(cases),
            "event_type_count": len(event_types),
            "scenario_count": len(scenarios),
            "action_count": len(actions),
        },
        "labels": {
            "event_types": event_types,
            "scenarios": scenarios,
            "actions": actions,
            "outcomes": ["success", "failure"],
        },
        "event_scenario_matrix": event_scenario_matrix,
        "event_action_matrix": event_action_matrix,
        "event_scenario_counts": {
            event_type: {scenario: event_scenario_counter[event_type][scenario] for scenario in scenarios}
            for event_type in event_types
        },
        "event_action_counts": {
            event_type: {action: event_action_counter[event_type][action] for action in actions}
            for event_type in event_types
        },
        "action_counts": {action: action_counter[action] for action in actions},
        "outcome_counts": {
            "success": outcome_counter["success"],
            "failure": outcome_counter["failure"],
        },
    }


def evaluate_retrieval(cases: list) -> dict:
    exp_v1 = ExperienceModel()
    exp_v1.train(cases)

    exp_v2 = ExperienceModelV2()
    exp_v2.train(cases)

    event_types = sorted({case.event_type for case in cases})
    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in {case.scenario for case in cases}]

    query_details = []
    metrics = {
        "baseline": {"acc": [], "ndcg": [], "time_ms": []},
        "v1": {"acc": [], "ndcg": [], "time_ms": []},
        "v2": {"acc": [], "ndcg": [], "time_ms": []},
    }

    for event_type in event_types:
        for severity in SEVERITY_LEVELS:
            for scenario in scenarios:
                gt_action = ground_truth_action(event_type, severity)
                query_record = {
                    "event_type": event_type,
                    "severity": severity,
                    "scenario": scenario,
                    "ground_truth_action": gt_action,
                }

                for method in ("baseline", "v1", "v2"):
                    start = time.perf_counter_ns()
                    if method == "baseline":
                        ranked = retrieve_baseline(cases, event_type, severity, scenario)
                    elif method == "v1":
                        ranked = retrieve_v1(cases, exp_v1, event_type, severity, scenario)
                    else:
                        ranked = retrieve_v2(cases, exp_v2, event_type, severity, scenario)
                    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

                    ranked_cases = [case for _, case in ranked]
                    top1_action = ranked_cases[0].action if ranked_cases else None
                    metrics[method]["acc"].append(1 if top1_action == gt_action else 0)
                    metrics[method]["ndcg"].append(ndcg_at_k(ranked_cases, gt_action))
                    metrics[method]["time_ms"].append(elapsed_ms)

                    query_record[method] = {
                        "top1_action": top1_action,
                        "top5_case_ids": [case.id for case in ranked_cases],
                        "action_correct": top1_action == gt_action,
                        "ndcg_at_5": ndcg_at_k(ranked_cases, gt_action),
                        "time_ms": elapsed_ms,
                    }

                query_details.append(query_record)

    summary = {
        method: {
            "action_accuracy_pct": round(100.0 * float(np.mean(values["acc"])), 3),
            "ndcg_at_5_pct": round(100.0 * float(np.mean(values["ndcg"])), 3),
            "avg_time_ms": round(float(np.mean(values["time_ms"])), 6),
        }
        for method, values in metrics.items()
    }

    stability = evaluate_stability(cases, exp_v1, exp_v2, event_types, scenarios)

    return {
        "summary": {
            "total_queries": len(query_details),
            "severity_levels": SEVERITY_LEVELS,
            "scenarios": scenarios,
            "event_types": event_types,
        },
        "methods": summary,
        "stability": stability,
        "details": query_details,
    }


def evaluate_stability(
    cases: list,
    exp_v1: ExperienceModel,
    exp_v2: ExperienceModelV2,
    event_types: list[str],
    scenarios: list[str],
    repeats: int = 10,
) -> dict:
    rng = random.Random(20260421)
    query_sample = []
    for event_type in event_types:
        for severity in SEVERITY_LEVELS:
            for scenario in scenarios:
                query_sample.append((event_type, severity, scenario))

    metrics = {
        "baseline": {"top1_action": [], "top5_exact": []},
        "v1": {"top1_action": [], "top5_exact": []},
        "v2": {"top1_action": [], "top5_exact": []},
    }

    for event_type, severity, scenario in query_sample:
        repeated_results = {method: {"top1_action": [], "top5_exact": []} for method in metrics}

        for _ in range(repeats):
            shuffled_cases = list(cases)
            rng.shuffle(shuffled_cases)

            baseline_ranked = retrieve_baseline(shuffled_cases, event_type, severity, scenario)
            v1_ranked = retrieve_v1(shuffled_cases, exp_v1, event_type, severity, scenario)
            v2_ranked = retrieve_v2(shuffled_cases, exp_v2, event_type, severity, scenario)

            method_results = {
                "baseline": baseline_ranked,
                "v1": v1_ranked,
                "v2": v2_ranked,
            }

            for method, ranked in method_results.items():
                ranked_cases = [case for _, case in ranked]
                repeated_results[method]["top1_action"].append(ranked_cases[0].action if ranked_cases else None)
                repeated_results[method]["top5_exact"].append(tuple(case.id for case in ranked_cases))

        for method in metrics:
            top1_counts = Counter(repeated_results[method]["top1_action"])
            top5_counts = Counter(repeated_results[method]["top5_exact"])
            metrics[method]["top1_action"].append(max(top1_counts.values()) / repeats)
            metrics[method]["top5_exact"].append(max(top5_counts.values()) / repeats)

    return {
        "repeats_per_query": repeats,
        "methods": {
            method: {
                "top1_action_consistency_pct": round(100.0 * float(np.mean(values["top1_action"])), 3),
                "top5_exact_consistency_pct": round(100.0 * float(np.mean(values["top5_exact"])), 3),
            }
            for method, values in metrics.items()
        },
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def add_bar_labels(ax, values: list[float], fmt: str = "{:.1f}") -> None:
    ymax = max(values) if values else 0.0
    offset = ymax * 0.02 if ymax > 0 else 0.02
    for index, value in enumerate(values):
        ax.text(index, value + offset, fmt.format(value), ha="center", va="bottom", fontsize=8)


def plot_case_distribution(case_distribution: dict, output_path: Path) -> None:
    labels = case_distribution["labels"]
    matrix = np.array(case_distribution["event_scenario_matrix"])
    action_counts = [case_distribution["action_counts"][action] for action in labels["actions"]]
    outcome_counts = [
        case_distribution["outcome_counts"]["success"],
        case_distribution["outcome_counts"]["failure"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)

    heatmap = axes[0].imshow(matrix, cmap="YlGnBu", aspect="auto")
    axes[0].set_title("1a Event Type × Scenario Distribution")
    axes[0].set_xlabel("Scenario")
    axes[0].set_ylabel("Event Type")
    axes[0].set_xticks(range(len(labels["scenarios"])))
    axes[0].set_xticklabels(labels["scenarios"], rotation=20)
    axes[0].set_yticks(range(len(labels["event_types"])))
    axes[0].set_yticklabels(labels["event_types"])
    fig.colorbar(heatmap, ax=axes[0], shrink=0.8, label="Case Count")

    axes[1].bar(labels["actions"], action_counts, color="#4C78A8")
    axes[1].set_title("1b Action Distribution")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=25)
    add_bar_labels(axes[1], action_counts, "{:.0f}")

    axes[2].pie(
        outcome_counts,
        labels=["success", "failure"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#54A24B", "#E45756"],
    )
    axes[2].set_title("1c Outcome Distribution")

    fig.suptitle("Case Statistics Analysis", fontsize=14)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_retrieval_comparison(retrieval_data: dict, output_path: Path) -> None:
    methods = ["baseline", "v1", "v2"]
    labels = [METHOD_LABELS[method] for method in methods]
    accuracy_values = [retrieval_data["methods"][method]["action_accuracy_pct"] for method in methods]
    ndcg_values = [retrieval_data["methods"][method]["ndcg_at_5_pct"] for method in methods]
    time_values = [retrieval_data["methods"][method]["avg_time_ms"] for method in methods]
    colors = [METHOD_COLORS[method] for method in methods]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    axes[0].bar(labels, accuracy_values, color=colors)
    axes[0].set_title("Action Accuracy")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_ylim(0, max(100, max(accuracy_values) * 1.15))
    add_bar_labels(axes[0], accuracy_values)

    axes[1].bar(labels, ndcg_values, color=colors)
    axes[1].set_title("NDCG@5")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_ylim(0, max(100, max(ndcg_values) * 1.15))
    add_bar_labels(axes[1], ndcg_values)

    axes[2].bar(labels, time_values, color=colors)
    axes[2].set_title("Decision Time")
    axes[2].set_ylabel("Milliseconds (log)")
    axes[2].set_yscale("log")
    add_bar_labels(axes[2], time_values, "{:.4f}")

    fig.suptitle("Retrieval Experiment Comparison", fontsize=14)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_retrieval_stability(retrieval_data: dict, output_path: Path) -> None:
    methods = ["baseline", "v1", "v2"]
    labels = [METHOD_LABELS[method] for method in methods]
    top1_values = [
        retrieval_data["stability"]["methods"][method]["top1_action_consistency_pct"]
        for method in methods
    ]
    top5_values = [
        retrieval_data["stability"]["methods"][method]["top5_exact_consistency_pct"]
        for method in methods
    ]

    x = np.arange(len(methods))
    width = 0.35
    colors = [METHOD_COLORS[method] for method in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].bar(labels, top1_values, color=colors)
    axes[0].set_title("Top-1 Action Consistency")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_ylim(0, 105)
    add_bar_labels(axes[0], top1_values)

    bars_left = axes[1].bar(x - width / 2, top1_values, width=width, label="Top-1 Action", color="#4C78A8")
    bars_right = axes[1].bar(x + width / 2, top5_values, width=width, label="Top-5 Exact Match", color="#F58518")
    axes[1].set_title("Repeated Query Result Consistency")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    for bar in list(bars_left) + list(bars_right):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Retrieval Stability Analysis", fontsize=14)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cases...")
    retriever = CaseRetriever()
    cases = retriever.cases
    # Filter out invalid cases with event_type not in ALL_EVENT_TYPES
    cases = [c for c in cases if c.event_type in ALL_EVENT_TYPES]
    print(f"Loaded {len(cases)} valid cases (filtered out invalid event_type cases).")

    print("Building case distribution data...")
    case_distribution = build_case_distribution(cases)
    save_json(OUTPUT_DIR / "case_distribution.json", case_distribution)

    print("Running retrieval experiments...")
    retrieval_comparison = evaluate_retrieval(cases)
    save_json(OUTPUT_DIR / "retrieval_comparison.json", retrieval_comparison)

    print("Rendering figures...")
    plot_case_distribution(case_distribution, OUTPUT_DIR / "fig_case_distribution.png")
    plot_retrieval_comparison(retrieval_comparison, OUTPUT_DIR / "fig_retrieval_comparison.png")
    plot_retrieval_stability(retrieval_comparison, OUTPUT_DIR / "fig_retrieval_stability.png")

    paper_data = {
        "metadata": {
            "project_root": str(REPO_ROOT),
            "output_dir": str(OUTPUT_DIR),
        },
        "case_distribution": case_distribution,
        "retrieval_comparison": retrieval_comparison,
    }
    save_json(OUTPUT_DIR / "paper_data.json", paper_data)

    print("Done.")
    print(f"Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
