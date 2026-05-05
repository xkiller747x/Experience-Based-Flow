"""Evaluate LearnedRetrieverV2 on negative/anomaly queries.

In logistics, most records are normal. The key question is:
when an anomaly (failure) query comes in, can the retriever find
similar anomaly cases from the pool, rather than just matching
surface-level normal cases?
"""
import json, random, math, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.case_generator import Case
from src.rag.case_retriever import CaseRetriever
from src.rag.experience_model import LearnedRetrieverV2
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

print("Loading cases...", flush=True)
cases = []
with open(Path(__file__).resolve().parents[1] / "data" / "cases" / "cases_30k.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            cases.append(Case(**json.loads(line)))
print(f"Loaded {len(cases)} cases", flush=True)

rng = random.Random(20260428)

# Separate success and failure cases
success_cases = [c for c in cases if c.outcome == "success"]
failure_cases = [c for c in cases if c.outcome == "failure"]
print(f"Success: {len(success_cases)}, Failure: {len(failure_cases)}", flush=True)

# Pool = mix of success and failure (reflecting real distribution)
pool = rng.sample(cases, 5000)

# Queries = ONLY failure cases (anomaly queries)
failure_queries = [c for c in failure_cases if id(c) not in {id(p) for p in pool}]
if len(failure_queries) > 500:
    failure_queries = rng.sample(failure_queries, 500)
print(f"Pool: {len(pool)} (success pool: {sum(1 for c in pool if c.outcome=='success')}, "
      f"failure pool: {sum(1 for c in pool if c.outcome=='failure')})", flush=True)
print(f"Anomaly queries (failure): {len(failure_queries)}", flush=True)

print("\nTraining LearnedRetrieverV2...", flush=True)
learned = LearnedRetrieverV2(random_state=20260428)
learned.train(pool)

print("\nBuilding CaseRetriever baseline...", flush=True)
baseline = CaseRetriever.__new__(CaseRetriever)
baseline.cases = pool
baseline._vectorizer = TfidfVectorizer()
docs = [c.reasoning or "empty" for c in pool]
baseline._tfidf_matrix = baseline._vectorizer.fit_transform(docs)


def ctx_from_case(q):
    return {
        "event_type": q.event_type, "severity": q.severity, "scenario": q.scenario,
        "vehicles_count": q.vehicles_count, "orders_count": q.orders_count,
        "current_load_rate": q.current_load_rate, "urgent_orders": q.urgent_orders,
        "available_backup_vehicles": q.available_backup_vehicles,
        "avg_delay_minutes": q.avg_delay_minutes, "affected_routes": q.affected_routes,
        "time_window_pressure": q.time_window_pressure, "customer_priority_mix": q.customer_priority_mix,
    }


def eval_anomaly_retrieval(model_name, get_ranked, pool, queries, k=5):
    """
    For each anomaly (failure) query, evaluate:
    1. How many failure cases in Top-K (anomaly recall)
    2. How many same-event-type cases in Top-K
    3. Average position of first failure case
    4. Failure ratio in Top-K vs pool ratio
    """
    pool_failure_ratio = sum(1 for c in pool if c.outcome == "failure") / len(pool)
    
    totals = {
        "failure_in_topk": 0,       # failure cases found in Top-K
        "total_topk_slots": 0,       # total slots = queries * k
        "same_event_in_topk": 0,     # same event_type in Top-K
        "same_scenario_in_topk": 0,  # same scenario in Top-K
        "context_similar_in_topk": 0, # context pressure similar in Top-K
        "first_failure_pos_sum": 0.0,
        "first_failure_count": 0,
        "relevance_p": 0.0,          # precision by relevance label
        "relevance_ndcg": 0.0,
        "queries_with_failure": 0,    # queries that found at least 1 failure
        "count": 0,
    }
    
    for i, q in enumerate(queries, 1):
        ranked = get_ranked(q, pool, k)
        
        # Count failure cases in top-k
        failures_in_k = sum(1 for c in ranked if c.outcome == "failure")
        totals["failure_in_topk"] += failures_in_k
        totals["total_topk_slots"] += k
        
        # Same event_type
        totals["same_event_in_topk"] += sum(1 for c in ranked if c.event_type == q.event_type)
        
        # Same scenario
        totals["same_scenario_in_topk"] += sum(1 for c in ranked if c.scenario == q.scenario)
        
        # Context pressure similar
        for c in ranked:
            q_load = q.current_load_rate or 0
            c_load = c.current_load_rate or 0
            q_pres = q.time_window_pressure or 0
            c_pres = c.time_window_pressure or 0
            sev_diff = abs(q.severity - c.severity)
            if abs(q_load - c_load) < 0.2 and abs(q_pres - c_pres) < 0.2 and sev_diff < 3:
                totals["context_similar_in_topk"] += 1
        
        # First failure position
        for idx, c in enumerate(ranked, 1):
            if c.outcome == "failure":
                totals["first_failure_pos_sum"] += idx
                totals["first_failure_count"] += 1
                break
        
        if failures_in_k > 0:
            totals["queries_with_failure"] += 1
        
        # Relevance metrics (using LearnedRetrieverV2 labels)
        cands = [c for c in pool if c is not q]
        total_rel = sum(LearnedRetrieverV2.relevance_label(q, c) for c in cands)
        if total_rel > 0:
            labels = [LearnedRetrieverV2.relevance_label(q, c) for c in ranked]
            hits = sum(labels)
            totals["relevance_p"] += hits / k
            dcg = sum(1.0 / math.log2(idx + 1) for idx, l in enumerate(labels, 1) if l)
            idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, min(total_rel, k) + 1))
            totals["relevance_ndcg"] += dcg / idcg if idcg else 0
        
        totals["count"] += 1
        if i % 100 == 0:
            print(f"  {model_name} {i}/{len(queries)}", flush=True)
    
    n = max(totals["count"], 1)
    return {
        "failure_rate_topk": totals["failure_in_topk"] / max(totals["total_topk_slots"], 1),
        "pool_failure_rate": pool_failure_ratio,
        "enrichment": (totals["failure_in_topk"] / max(totals["total_topk_slots"], 1)) / max(pool_failure_ratio, 1e-8),
        "same_event_rate": totals["same_event_in_topk"] / max(totals["total_topk_slots"], 1),
        "same_scenario_rate": totals["same_scenario_in_topk"] / max(totals["total_topk_slots"], 1),
        "context_similar_rate": totals["context_similar_in_topk"] / max(totals["total_topk_slots"], 1),
        "avg_first_failure_pos": totals["first_failure_pos_sum"] / max(totals["first_failure_count"], 1),
        "first_failure_found_rate": totals["first_failure_count"] / n,
        "queries_with_failure_rate": totals["queries_with_failure"] / n,
        "relevance_p@5": totals["relevance_p"] / n,
        "relevance_ndcg@5": totals["relevance_ndcg"] / n,
        "queries": n,
    }


def eval_learned(q, pool, k):
    cands = [c for c in pool if c is not q]
    return [r["case"] for r in learned.retrieve(ctx_from_case(q), cands, top_k=k)]


def eval_baseline(q, pool, k):
    return [r.case for r in baseline.retrieve(q.event_type, q.severity, q.scenario, top_k=k)]


print("\n=== Anomaly Query Evaluation ===\n", flush=True)
r_learned = eval_anomaly_retrieval("LearnedRetrieverV2", eval_learned, pool, failure_queries)
r_baseline = eval_anomaly_retrieval("CaseRetriever", eval_baseline, pool, failure_queries)


def print_result(name, r):
    print(f"\n  {name}:", flush=True)
    print(f"    Failure rate in Top-5:      {r['failure_rate_topk']:.4f}  (pool baseline: {r['pool_failure_rate']:.4f})", flush=True)
    print(f"    Failure enrichment factor:  {r['enrichment']:.2f}x", flush=True)
    print(f"    Same event_type rate:       {r['same_event_rate']:.4f}", flush=True)
    print(f"    Same scenario rate:         {r['same_scenario_rate']:.4f}", flush=True)
    print(f"    Context similar rate:       {r['context_similar_rate']:.4f}", flush=True)
    print(f"    Avg first failure position: {r['avg_first_failure_pos']:.2f}", flush=True)
    print(f"    Queries with failure (Top-5): {r['queries_with_failure_rate']:.4f}", flush=True)
    print(f"    Relevance P@5:             {r['relevance_p@5']:.4f}", flush=True)
    print(f"    Relevance NDCG@5:          {r['relevance_ndcg@5']:.4f}", flush=True)
    print(f"    Total queries:             {r['queries']}", flush=True)


print_result("LearnedRetrieverV2", r_learned)
print_result("CaseRetriever (Baseline)", r_baseline)
