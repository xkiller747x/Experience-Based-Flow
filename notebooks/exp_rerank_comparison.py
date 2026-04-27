"""
Experiment: Rerank with vs without Experience Model

对比两种检索策略：
1. 无经验模型：只用五维打分 rerank
2. 有经验模型：五维打分 + 经验先验信号

衡量指标：
- Action 预测准确率（top1 action 与业务规则一致的比例）
- Outcome 预测准确率
- 检索差异率（有/无经验模型时 top1 不同的比例）
- 决策时间（毫秒）
- Rerank 稳定性
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import CaseRetriever, ExperienceModel
from src.rag.case_retriever import EVENT_TYPE_HIERARCHY


# ── Business Rule Ground Truth ─────────────────────────────────────────

def ground_truth_action(event_type: str, severity: int) -> str:
    """业务规则：给定的 event_type + severity 应该对应什么 action（用于计算准确率）。"""
    if event_type in ("vehicle_breakdown", "vehicle_maintenance", "driver_unavailable"):
        if severity >= 5:
            return "reassign_order"
        elif severity >= 3:
            return "adjust_capacity"
        else:
            return "reroute"
    if event_type == "fuel_shortage":
        if severity >= 4:
            return "adjust_capacity"
        else:
            return "reroute"
    if event_type in ("traffic_accident", "traffic_congestion", "road_narrowing"):
        if severity >= 4:
            return "reroute"
        else:
            return "ignore"
    if event_type in ("road_closed", "bridge_weight_limit"):
        if severity >= 3:
            return "reroute"
        else:
            return "delay_tolerant"
    if event_type == "order_cancel":
        if severity >= 4:
            return "reroute"
        elif severity >= 2:
            return "delay_tolerant"
        else:
            return "ignore"
    if event_type == "order_modify":
        if severity >= 4:
            return "reroute"
        else:
            return "delay_tolerant"
    if event_type == "order_update":
        return "ignore"
    if event_type == "priority_order_urgent":
        return "reassign_order"
    if event_type == "delivery_failure":
        return "reassign_order"
    if event_type == "demand_surge":
        if severity >= 4:
            return "adjust_capacity"
        else:
            return "delay_tolerant"
    if event_type == "demand_drop":
        return "delay_tolerant"
    if event_type in ("weather_delay", "natural_disaster", "public_event"):
        if severity >= 4:
            return "delay_tolerant"
        else:
            return "reroute"
    if event_type in ("warehouse_delay", "inventory_stockout"):
        return "delay_tolerant"
    return "reroute"


# ── Coarse-only retriever (simulate without experience model) ─────────

def coarse_score(case, event_type: str, severity: int, scenario: str) -> float:
    """五维打分，不含经验模型先验。"""
    score = 0.0
    if case.event_type == event_type:
        score += 3.0
    else:
        fallback = EVENT_TYPE_HIERARCHY.get(event_type, [])
        if case.event_type in fallback:
            score += 2.0
    score += max(0, 2.0 - abs(case.severity - severity) * 0.5)
    if case.scenario == scenario:
        score += 1.0
    return score


def retrieve_without_experience(retriever, event_type: str, severity: int, scenario: str, top_k: int = 5) -> list:
    """只用粗排 + 五维打分，无经验模型。"""
    candidates = retriever.cases
    scored = []
    for c in candidates:
        s = coarse_score(c, event_type, severity, scenario)
        scored.append((s, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def retrieve_with_experience(retriever, exp_model, event_type: str, severity: int, scenario: str, top_k: int = 5) -> list:
    """粗排 + 五维打分 + 经验先验奖励。"""
    exp = exp_model.predict(event_type, severity, scenario)
    exp_action = exp["action"]
    exp_conf = exp["action_confidence"]

    candidates = retriever.cases
    scored = []
    for c in candidates:
        # 五维基础分
        base = coarse_score(c, event_type, severity, scenario)
        # 经验先验奖励：action 一致 + confidence 加权
        bonus = 0.0
        if c.action == exp_action:
            bonus = 2.0 * exp_conf
        total = base + bonus
        scored.append((total, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ── Main Experiment ──────────────────────────────────────────────────

def run_comparison():
    print("=" * 70)
    print("  Rerank Comparison: With vs Without Experience Model")
    print("=" * 70)

    # Load data
    retriever = CaseRetriever()
    print(f"Cases loaded: {len(retriever.cases)}")

    exp_model = ExperienceModel()
    exp_model.train(retriever.cases)
    print("Experience model trained.")

    # Query set: all event types × severity 1-5 × scenario small
    event_types = list(set(c.event_type for c in retriever.cases))
    severities = [1, 2, 3, 4, 5]
    scenario = "small"

    total_queries = len(event_types) * len(severities)
    print(f"\nRunning {total_queries} queries across {len(event_types)} event types × 5 severity levels")

    # Metrics
    action_acc_with = []
    action_acc_without = []
    diff_count = 0
    time_with_ms = []
    time_without_ms = []

    results_detail = []

    for ev in sorted(event_types):
        for sev in severities:
            gt_action = ground_truth_action(ev, sev)

            # Without experience model
            t0 = time.perf_counter()
            without = retrieve_without_experience(retriever, ev, sev, scenario, top_k=5)
            t1 = time.perf_counter()
            time_without_ms.append((t1 - t0) * 1000)
            top1_without = without[0][1].action if without else None
            acc_without = 1 if top1_without == gt_action else 0
            action_acc_without.append(acc_without)

            # With experience model
            t2 = time.perf_counter()
            with_exp = retrieve_with_experience(retriever, exp_model, ev, sev, scenario, top_k=5)
            t3 = time.perf_counter()
            time_with_ms.append((t3 - t2) * 1000)
            top1_with = with_exp[0][1].action if with_exp else None
            acc_with = 1 if top1_with == gt_action else 0
            action_acc_with.append(acc_with)

            # Difference
            is_diff = top1_with != top1_without
            if is_diff:
                diff_count += 1

            results_detail.append({
                "event_type": ev,
                "severity": sev,
                "gt_action": gt_action,
                "top1_without": top1_without,
                "top1_with": top1_with,
                "acc_without": acc_without,
                "acc_with": acc_with,
                "diff": is_diff,
            })

    # Summary
    avg_time_without = sum(time_without_ms) / len(time_without_ms)
    avg_time_with = sum(time_with_ms) / len(time_with_ms)

    action_acc_without_pct = 100 * sum(action_acc_without) / len(action_acc_without)
    action_acc_with_pct = 100 * sum(action_acc_with) / len(action_acc_with)
    diff_rate = 100 * diff_count / total_queries

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total queries: {total_queries}")
    print(f"")
    print(f"  Action Accuracy (top1 matches ground truth):")
    print(f"    Without Experience Model: {action_acc_without_pct:.1f}% ({sum(action_acc_without)}/{total_queries})")
    print(f"    With Experience Model:     {action_acc_with_pct:.1f}% ({sum(action_acc_with)}/{total_queries})")
    print(f"")
    print(f"  Retrieval Difference Rate:")
    print(f"    Queries where top1 changed: {diff_count}/{total_queries} ({diff_rate:.1f}%)")
    print(f"")
    print(f"  Decision Time (ms per query):")
    print(f"    Without Experience Model: {avg_time_without:.4f} ms")
    print(f"    With Experience Model:     {avg_time_with:.4f} ms")
    print(f"")
    print(f"  Cases by event type:")
    type_acc_without = {}
    type_acc_with = {}
    for r in results_detail:
        ev = r["event_type"]
        if ev not in type_acc_without:
            type_acc_without[ev] = []
            type_acc_with[ev] = []
        type_acc_without[ev].append(r["acc_without"])
        type_acc_with[ev].append(r["acc_with"])
    print(f"  {'Event Type':<30} {'Without':>10} {'With':>10} {'Diff':>10}")
    print(f"  {'-'*60}")
    for ev in sorted(type_acc_without.keys()):
        avg_w = 100 * sum(type_acc_without[ev]) / len(type_acc_without[ev])
        avg_e = 100 * sum(type_acc_with[ev]) / len(type_acc_with[ev])
        diff = sum(1 for r in results_detail if r["event_type"] == ev and r["diff"])
        print(f"  {ev:<30} {avg_w:>9.1f}% {avg_e:>9.1f}% {diff:>10}")

    # Save results
    out_path = Path(__file__).parent / "exp_rerank_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_queries": total_queries,
                "action_acc_without": action_acc_without_pct,
                "action_acc_with": action_acc_with_pct,
                "diff_count": diff_count,
                "diff_rate": diff_rate,
                "avg_time_without_ms": avg_time_without,
                "avg_time_with_ms": avg_time_with,
            },
            "detail": results_detail,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")
    return action_acc_with_pct, action_acc_without_pct


if __name__ == "__main__":
    run_comparison()
