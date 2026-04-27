"""
Experiment: 三种检索方案对比

1. Baseline（五维打分 rerank）
2. V1 经验模型（预测 action，action 一致性奖励）
3. V2 经验模型（预测 query-case 相关性，综合排序）

衡量：
- Action 准确率（top1 action 与业务规则一致）
- NDCG@5（综合排序质量）
- 决策时间
"""

from __future__ import annotations

import json, os, sys, time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import CaseRetriever, ExperienceModel, ExperienceModelV2
from src.rag.case_retriever import EVENT_TYPE_HIERARCHY


def ground_truth_action(event_type: str, severity: int) -> str:
    if event_type in ("vehicle_breakdown", "vehicle_maintenance", "driver_unavailable"):
        if severity >= 5: return "reassign_order"
        elif severity >= 3: return "adjust_capacity"
        else: return "reroute"
    if event_type == "fuel_shortage":
        return "adjust_capacity" if severity >= 4 else "reroute"
    if event_type in ("traffic_accident", "traffic_congestion", "road_narrowing"):
        return "reroute" if severity >= 4 else "ignore"
    if event_type in ("road_closed", "bridge_weight_limit"):
        return "reroute" if severity >= 3 else "delay_tolerant"
    if event_type == "order_cancel":
        if severity >= 4: return "reroute"
        elif severity >= 2: return "delay_tolerant"
        else: return "ignore"
    if event_type == "order_modify":
        return "reroute" if severity >= 4 else "delay_tolerant"
    if event_type == "order_update": return "ignore"
    if event_type in ("priority_order_urgent", "delivery_failure"): return "reassign_order"
    if event_type == "demand_surge": return "adjust_capacity" if severity >= 4 else "delay_tolerant"
    if event_type == "demand_drop": return "delay_tolerant"
    if event_type in ("weather_delay", "natural_disaster", "public_event"):
        return "delay_tolerant" if severity >= 4 else "reroute"
    if event_type in ("warehouse_delay", "inventory_stockout"): return "delay_tolerant"
    return "reroute"


def coarse_score(case, ev, sev, sc):
    score = 0.0
    if case.event_type == ev:
        score += 3.0
    else:
        fb = EVENT_TYPE_HIERARCHY.get(ev, [])
        if case.event_type in fb:
            score += 2.0
    score += max(0, 2.0 - abs(case.severity - sev) * 0.5)
    if case.scenario == sc:
        score += 1.0
    return score


def retrieve_baseline(retriever, ev, sev, sc, top_k=5):
    scored = [(coarse_score(c, ev, sev, sc), c) for c in retriever.cases]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def retrieve_v1(retriever, exp_model, ev, sev, sc, top_k=5):
    exp = exp_model.predict(ev, sev, sc)
    exp_action = exp["action"]
    exp_conf = exp["action_confidence"]
    scored = []
    for c in retriever.cases:
        base = coarse_score(c, ev, sev, sc)
        bonus = 2.0 * exp_conf if c.action == exp_action else 0.0
        scored.append((base + bonus, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def retrieve_v2(retriever, exp_v2, ev, sev, sc, top_k=5):
    coarse = retrieve_baseline(retriever, ev, sev, sc, top_k=30)
    cases = [c for _, c in coarse]
    coarse_scores = [s for s, _ in coarse]
    exp_scores = exp_v2.predict_relevance(ev, sev, sc, cases)
    combined = [
        (coarse_scores[i] * 0.6 + exp_scores[i] * 10.0 * 0.4, cases[i])
        for i in range(len(cases))
    ]
    combined.sort(key=lambda x: x[0], reverse=True)
    return combined[:top_k]


def dcg_at_k(relevance, k=5):
    import math
    return sum(rel / max(1, math.log2(i + 2)) for i, rel in enumerate(relevance[:k]))


def ndcg_at_k(sorted_cases, gt_action, k=5):
    relevance = [1 if c.action == gt_action else 0 for c in sorted_cases[:k]]
    ideal = sorted(relevance, reverse=True)
    dcg = dcg_at_k(relevance, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def run():
    import math
    print("=" * 70)
    print("  V2 Experience Model Comparison")
    print("=" * 70)

    retriever = CaseRetriever()
    print(f"Cases: {len(retriever.cases)}")

    exp_v1 = ExperienceModel()
    exp_v1.train(retriever.cases)

    exp_v2 = ExperienceModelV2()
    exp_v2.train(retriever.cases)

    evs = sorted(set(c.event_type for c in retriever.cases))
    sevs = [1, 2, 3, 4, 5]
    sc = "small"

    total = len(evs) * len(sevs)
    print(f"Queries: {total} ({len(evs)} types x 5 severities)")

    metrics = {
        "baseline": {"acc": [], "ndcg": [], "time_ms": []},
        "v1": {"acc": [], "ndcg": [], "time_ms": []},
        "v2": {"acc": [], "ndcg": [], "time_ms": []},
    }

    for ev in evs:
        for sev in sevs:
            gt = ground_truth_action(ev, sev)

            t0 = time.perf_counter()
            b_res = retrieve_baseline(retriever, ev, sev, sc)
            t1 = time.perf_counter()
            metrics["baseline"]["time_ms"].append((t1 - t0) * 1000)
            metrics["baseline"]["acc"].append(1 if b_res[0][1].action == gt else 0)
            metrics["baseline"]["ndcg"].append(ndcg_at_k([c for _, c in b_res], gt))

            t2 = time.perf_counter()
            v1_res = retrieve_v1(retriever, exp_v1, ev, sev, sc)
            t3 = time.perf_counter()
            metrics["v1"]["time_ms"].append((t3 - t2) * 1000)
            metrics["v1"]["acc"].append(1 if v1_res[0][1].action == gt else 0)
            metrics["v1"]["ndcg"].append(ndcg_at_k([c for _, c in v1_res], gt))

            t4 = time.perf_counter()
            v2_res = retrieve_v2(retriever, exp_v2, ev, sev, sc)
            t5 = time.perf_counter()
            metrics["v2"]["time_ms"].append((t5 - t4) * 1000)
            metrics["v2"]["acc"].append(1 if v2_res[0][1].action == gt else 0)
            metrics["v2"]["ndcg"].append(ndcg_at_k([c for _, c in v2_res], gt))

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    for name, m in metrics.items():
        avg_acc = 100 * sum(m["acc"]) / total
        avg_ndcg = 100 * sum(m["ndcg"]) / total
        avg_time = sum(m["time_ms"]) / total
        print(f"  {name.upper()}:")
        print(f"    Action Accuracy: {avg_acc:.1f}%")
        print(f"    NDCG@5: {avg_ndcg:.1f}%")
        print(f"    Avg Time: {avg_time:.3f} ms")

    out = Path(__file__).parent / "exp_v2_comparison.json"
    with open(out, "w") as f:
        json.dump({k: {kk: sum(vv)/total if kk != "time_ms" else sum(vv)/total
                       for kk, vv in v.items()} for k, v in metrics.items()}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run()
