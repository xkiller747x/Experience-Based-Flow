"""
实验脚本：验证 RAG 增强对决策系统的效果
D:\Code\logistic-ai\notebooks\experiments.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag import CaseRetriever, RuleRetriever
from src.simulation import SimulationConfig, LogisticsSimulator, Event, EventType
from src.optimizer.vrp_solver import ORToolsSolver
from src.agent import AnomalyDetector, DecisionMaker, LLMGateway, LLMError


# ── Config ────────────────────────────────────────────────────────────

LLM = LLMGateway()

CASE_RETRIEVER = CaseRetriever()
RULE_RETRIEVER = RuleRetriever()

EVENT_TYPES = ["vehicle_breakdown", "order_cancel", "traffic_accident", "road_closed"]
SEVERITIES = [1, 2, 3, 4, 5]
SCENARIOS = ["small", "medium", "large", "stress"]


# ── Experiment 1: Case Retrieval Coverage ───────────────────────────

def exp1_case_coverage():
    """
    统计每个 (event_type, severity) 的 case 检索成功率。
    使用两阶段检索（含 rerank）。
    """
    print("\n" + "=" * 60)
    print("  Experiment 1: Case Retrieval Coverage")
    print("=" * 60)

    results = []
    for ev_type in EVENT_TYPES:
        for sev in SEVERITIES:
            cases = CASE_RETRIEVER.retrieve(
                event_type=ev_type,
                severity=sev,
                scenario="small",
                top_k=5,
            )
            # 检查是否有有效匹配（非空结果）
            has_match = len(cases) > 0 and cases[0].score > 0
            results.append({
                "event_type": ev_type,
                "severity": sev,
                "num_cases": len(cases),
                "top_score": float(cases[0].score) if cases else 0.0,
                "top_action": cases[0].case.action if cases else None,
                "has_match": bool(has_match),
            })

    # 打印结果
    header = f"{'Event Type':<20} {'Sev':>3}  {'#Cases':>6}  {'Top Score':>8}  {'Top Action':<20}  {'Match':>5}"
    print(header)
    print("-" * 80)
    for r in results:
        match_str = "YES" if r["has_match"] else "NO"
        print(f"{r['event_type']:<20} {r['severity']:>3}  {r['num_cases']:>6}  {r['top_score']:>8.2f}  {str(r['top_action'] or ''):<20}  {match_str:>5}")

    total = len(results)
    matched = sum(1 for r in results if r["has_match"])
    print(f"\nCoverage: {matched}/{total} ({100*matched/total:.1f}%)")

    # 保存 JSON
    out_path = os.path.join(os.path.dirname(__file__), "exp1_case_coverage.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}")


# ── Experiment 2: Rule Hit Rate ─────────────────────────────────────

def exp2_rule_hit_rate():
    """
    对 55 个历史 case，逐个查询规则，统计命中率和命中规则分布。
    """
    print("\n" + "=" * 60)
    print("  Experiment 2: Rule Hit Rate")
    print("=" * 60)

    case_retriever = CASE_RETRIEVER
    rule_retriever = RULE_RETRIEVER

    hit_count = 0
    category_hits = Counter()
    priority_hits = []

    for case in case_retriever.cases:
        rules = rule_retriever.retrieve(
            event_type=case.event_type,
            severity=case.severity,
            scenario=case.scenario,
            top_k=3,
        )
        if rules:
            hit_count += 1
            for rr in rules:
                category_hits[rr.rule.category] += 1
                priority_hits.append(rr.rule.priority)

    total = len(case_retriever.cases)
    print(f"Rule hit rate: {hit_count}/{total} ({100*hit_count/total:.1f}%)")
    print(f"\nCategory distribution:")
    for cat, cnt in category_hits.most_common():
        print(f"  {cat}: {cnt}")
    if priority_hits:
        print(f"\nHit priority: min={min(priority_hits)}, max={max(priority_hits)}, avg={sum(priority_hits)/len(priority_hits):.1f}")

    out_path = os.path.join(os.path.dirname(__file__), "exp2_rule_hits.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"hit_rate": hit_count/total, "total": total, "category_hits": dict(category_hits)}, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}")


# ── Experiment 3: Rerank Quality (Stability) ─────────────────────────

def exp3_rerank_stability():
    """
    验证两阶段 rerank 的稳定性：同一查询多次执行，顺序是否一致。
    """
    print("\n" + "=" * 60)
    print("  Experiment 3: Rerank Stability")
    print("=" * 60)

    query = {"event_type": "vehicle_breakdown", "severity": 4, "scenario": "small"}

    runs = []
    for i in range(5):
        cases = CASE_RETRIEVER.retrieve(**query, top_k=5)
        runs.append([c.case.id for c in cases])

    print(f"Query: {query}")
    for i, run in enumerate(runs):
        print(f"  Run {i+1}: {run}")

    # Check if all runs are identical
    all_same = all(r == runs[0] for r in runs)
    print(f"\nStable across 5 runs: {'YES' if all_same else 'NO'}")


# ── Experiment 4: Rerank vs No-Rerank Comparison ─────────────────────

def exp4_rerank_vs_basenline():
    """
    对比粗排 top5 vs 两阶段 rerank top5 的差异。
    看看 rerank 是否改变了 top5 的排序（尤其 action 和 outcome）。
    """
    print("\n" + "=" * 60)
    print("  Experiment 4: Rerank vs Baseline (coarse-only)")
    print("=" * 60)

    query = {"event_type": "vehicle_breakdown", "severity": 4, "scenario": "small", "top_k": 5}

    # Get reranked results (current implementation already uses rerank)
    reranked = CASE_RETRIEVER.retrieve(**query)
    reranked_ids = [c.case.id for c in reranked]
    reranked_actions = [c.case.action for c in reranked]

    print(f"Reranked top5:")
    for c in reranked:
        print(f"  {c.case.id}: score={float(c.score):.2f}, action={c.case.action}, outcome={c.case.outcome}")

    # Simulate coarse-only by taking only event_type+severity+scenario score
    coarse_only = []
    for case in CASE_RETRIEVER.cases:
        score = 0.0
        if case.event_type == query["event_type"]:
            score += 3.0
        score += max(0, 2.0 - abs(case.severity - query["severity"]) * 0.5)
        if case.scenario == query["scenario"]:
            score += 1.0
        coarse_only.append((score, case))
    coarse_only.sort(key=lambda x: x[0], reverse=True)
    coarse_top5 = [c for _, c in coarse_only[:5]]

    print(f"\nCoarse-only top5 (simulated):")
    for c in coarse_top5:
        print(f"  {c.id}: action={c.action}, outcome={c.outcome}")

    # Compare
    coarse_ids = [c.id for c in coarse_top5]
    same_order = coarse_ids == reranked_ids
    same_actions = [c.action for c in coarse_top5] == reranked_actions

    print(f"\nSame order as coarse: {'YES' if same_order else 'NO (rerank changed order)'}")
    print(f"Same top action: {'YES' if same_actions else 'NO (rerank changed action distribution)'}")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Logistics AI — Experiment Suite")
    print(f"Using {len(CASE_RETRIEVER.cases)} cases, {len(RULE_RETRIEVER.rules)} rules")

    exp1_case_coverage()
    exp2_rule_hit_rate()
    exp3_rerank_stability()
    exp4_rerank_vs_basenline()

    print("\n" + "=" * 60)
    print("  All experiments complete!")
    print("=" * 60)
