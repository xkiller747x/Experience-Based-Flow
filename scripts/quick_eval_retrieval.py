"""Quick eval: LearnedRetrieverV2 vs CaseRetriever (Baseline)."""
import json, random, math, sys
from pathlib import Path

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
pool = rng.sample(cases, 5000)
pool_ids = {id(c) for c in pool}
remaining = [c for c in cases if id(c) not in pool_ids]
queries = rng.sample(remaining, min(500, len(remaining)))
print(f"Pool: {len(pool)}, Queries: {len(queries)}", flush=True)

print("\nTraining LearnedRetrieverV2...", flush=True)
learned = LearnedRetrieverV2(random_state=20260428)
learned.train(pool)

print("\nBuilding CaseRetriever baseline...", flush=True)
baseline = CaseRetriever.__new__(CaseRetriever)
baseline.cases = pool
baseline._vectorizer = TfidfVectorizer()
docs = [c.reasoning or "empty" for c in pool]
baseline._tfidf_matrix = baseline._vectorizer.fit_transform(docs)


def evaluate(model_name, eval_fn, pool, queries, k=5):
    totals = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "mrr": 0.0, "count": 0.0}
    for i, q in enumerate(queries, 1):
        ranked = eval_fn(q, pool, k)
        cands = [c for c in pool if c is not q]
        total_rel = sum(LearnedRetrieverV2.relevance_label(q, c) for c in cands)
        if total_rel == 0:
            continue
        labels = [LearnedRetrieverV2.relevance_label(q, c) for c in ranked[:k]]
        hits = sum(labels)
        totals["precision"] += hits / k
        totals["recall"] += hits / total_rel
        dcg = sum(1.0 / math.log2(idx + 1) for idx, l in enumerate(labels, 1) if l)
        idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, min(total_rel, k) + 1))
        totals["ndcg"] += dcg / idcg if idcg else 0
        for idx, l in enumerate(labels, 1):
            if l:
                totals["mrr"] += 1.0 / idx
                break
        totals["count"] += 1
        if i % 100 == 0:
            print(f"  {model_name} {i}/{len(queries)}", flush=True)
    n = max(int(totals["count"]), 1)
    return {
        "P@5": totals["precision"] / n,
        "R@5": totals["recall"] / n,
        "NDCG@5": totals["ndcg"] / n,
        "MRR": totals["mrr"] / n,
        "Q": totals["count"],
    }


def eval_learned(q, pool, k):
    cands = [c for c in pool if c is not q]
    ctx = {
        "event_type": q.event_type, "severity": q.severity, "scenario": q.scenario,
        "vehicles_count": q.vehicles_count, "orders_count": q.orders_count,
        "current_load_rate": q.current_load_rate, "urgent_orders": q.urgent_orders,
        "available_backup_vehicles": q.available_backup_vehicles,
        "avg_delay_minutes": q.avg_delay_minutes, "affected_routes": q.affected_routes,
        "time_window_pressure": q.time_window_pressure, "customer_priority_mix": q.customer_priority_mix,
    }
    return [r["case"] for r in learned.retrieve(ctx, cands, top_k=k)]


def eval_baseline(q, pool, k):
    return [r.case for r in baseline.retrieve(q.event_type, q.severity, q.scenario, top_k=k)]


print()
r_learned = evaluate("LearnedRetrieverV2", eval_learned, pool, queries)
r_baseline = evaluate("CaseRetriever", eval_baseline, pool, queries)

print()
header = f"{'Model':28s} {'P@5':>8s} {'R@5':>8s} {'NDCG@5':>8s} {'MRR':>8s} {'Queries':>8s}"
print(header)
print("-" * len(header))
for name, r in [("LearnedRetrieverV2", r_learned), ("CaseRetriever(Baseline)", r_baseline)]:
    print(f"{name:28s} {r['P@5']:8.4f} {r['R@5']:8.4f} {r['NDCG@5']:8.4f} {r['MRR']:8.4f} {r['Q']:8.0f}")
print("-" * len(header))
