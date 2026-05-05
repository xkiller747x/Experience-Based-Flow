"""Offline retrieval evaluation for learned and baseline RAG retrievers."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.case_generator import Case  # noqa: E402
from src.rag.case_retriever import CaseRetriever  # noqa: E402
from src.rag.experience_model import (  # noqa: E402
    ExperienceModelV2Legacy,
    LearnedRetrieverV2,
)


def load_cases(path: Path, max_cases: int | None = None) -> list[Case]:
    cases: list[Case] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cases.append(Case(**json.loads(line)))
            if max_cases is not None and len(cases) >= max_cases:
                break
    return cases


def sample_pool_and_queries(
    cases: list[Case],
    pool_size: int,
    query_size: int,
    seed: int,
) -> tuple[list[Case], list[Case]]:
    rng = random.Random(seed)
    pool_size = min(pool_size, len(cases))
    pool = rng.sample(cases, pool_size)

    pool_ids = {id(case) for case in pool}
    remaining = [case for case in cases if id(case) not in pool_ids]
    if len(remaining) >= query_size:
        queries = rng.sample(remaining, query_size)
    else:
        queries = rng.sample(cases, min(query_size, len(cases)))
    return pool, queries


def case_to_query_context(case: Case) -> dict:
    return {
        "event_type": case.event_type,
        "severity": case.severity,
        "scenario": case.scenario,
        "vehicles_count": case.vehicles_count,
        "orders_count": case.orders_count,
        "current_load_rate": case.current_load_rate,
        "urgent_orders": case.urgent_orders,
        "available_backup_vehicles": case.available_backup_vehicles,
        "avg_delay_minutes": case.avg_delay_minutes,
        "affected_routes": case.affected_routes,
        "time_window_pressure": case.time_window_pressure,
        "customer_priority_mix": case.customer_priority_mix,
    }


def build_baseline_retriever(pool: list[Case]) -> CaseRetriever:
    retriever = CaseRetriever.__new__(CaseRetriever)
    retriever.cases = pool
    retriever._vectorizer = TfidfVectorizer()
    docs = [case.reasoning or "empty" for case in pool]
    retriever._tfidf_matrix = retriever._vectorizer.fit_transform(docs)
    return retriever


def candidate_pool(pool: list[Case], query: Case) -> list[Case]:
    if any(case is query for case in pool):
        return [case for case in pool if case is not query]
    return pool


def relevance_labels(query: Case, ranked_cases: Iterable[Case]) -> list[int]:
    return [LearnedRetrieverV2.relevance_label(query, case) for case in ranked_cases]


def count_relevant(query: Case, candidates: list[Case]) -> int:
    return sum(LearnedRetrieverV2.relevance_label(query, case) for case in candidates)


def update_metrics(
    totals: dict[str, float],
    query: Case,
    ranked_cases: list[Case],
    candidates: list[Case],
    k: int,
) -> bool:
    total_relevant = count_relevant(query, candidates)
    if total_relevant == 0:
        return False

    labels = relevance_labels(query, ranked_cases[:k])
    hits = sum(labels)
    totals["precision"] += hits / k
    totals["recall"] += hits / total_relevant

    dcg = 0.0
    for idx, label in enumerate(labels, start=1):
        if label:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(total_relevant, k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    totals["ndcg"] += dcg / idcg if idcg else 0.0

    mrr = 0.0
    for idx, label in enumerate(labels, start=1):
        if label:
            mrr = 1.0 / idx
            break
    totals["mrr"] += mrr
    totals["count"] += 1
    return True


def average_metrics(totals: dict[str, float]) -> dict[str, float]:
    count = max(int(totals["count"]), 1)
    return {
        "Precision@5": totals["precision"] / count,
        "Recall@5": totals["recall"] / count,
        "NDCG@5": totals["ndcg"] / count,
        "MRR": totals["mrr"] / count,
        "Queries": totals["count"],
    }


def evaluate_learned(
    model: LearnedRetrieverV2,
    pool: list[Case],
    queries: list[Case],
    k: int,
    progress_every: int,
) -> dict[str, float]:
    totals = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "mrr": 0.0, "count": 0.0}
    for idx, query in enumerate(queries, start=1):
        candidates = candidate_pool(pool, query)
        results = model.retrieve(case_to_query_context(query), candidates, top_k=k)
        ranked_cases = [item["case"] for item in results]
        update_metrics(totals, query, ranked_cases, candidates, k)
        if progress_every and idx % progress_every == 0:
            print(f"  LearnedRetrieverV2 evaluated {idx}/{len(queries)} queries")
    return average_metrics(totals)


def evaluate_baseline(
    retriever: CaseRetriever,
    pool: list[Case],
    queries: list[Case],
    k: int,
    progress_every: int,
) -> dict[str, float]:
    totals = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "mrr": 0.0, "count": 0.0}
    for idx, query in enumerate(queries, start=1):
        candidates = pool
        results = retriever.retrieve(
            query.event_type,
            query.severity,
            query.scenario,
            top_k=k,
        )
        ranked_cases = [item.case for item in results]
        update_metrics(totals, query, ranked_cases, candidates, k)
        if progress_every and idx % progress_every == 0:
            print(f"  CaseRetriever evaluated {idx}/{len(queries)} queries")
    return average_metrics(totals)


def evaluate_legacy(
    model: ExperienceModelV2Legacy,
    pool: list[Case],
    queries: list[Case],
    k: int,
    progress_every: int,
) -> dict[str, float]:
    totals = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "mrr": 0.0, "count": 0.0}
    for idx, query in enumerate(queries, start=1):
        candidates = candidate_pool(pool, query)
        context = case_to_query_context(query)
        scores = model.predict_relevance(
            query.event_type,
            query.severity,
            query.scenario,
            candidates,
            context=context,
        )
        ranked_cases = [
            case for case, _ in sorted(
                zip(candidates, scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )[:k]
        ]
        update_metrics(totals, query, ranked_cases, candidates, k)
        if progress_every and idx % progress_every == 0:
            print(f"  ExperienceModelV2Legacy evaluated {idx}/{len(queries)} queries")
    return average_metrics(totals)


def print_results(results: dict[str, dict[str, float]]) -> None:
    print("\nRetrieval evaluation results")
    print("-" * 76)
    print(f"{'Model':28s} {'Precision@5':>12s} {'Recall@5':>10s} {'NDCG@5':>10s} {'MRR':>10s}")
    print("-" * 76)
    for name, metrics in results.items():
        print(
            f"{name:28s} "
            f"{metrics['Precision@5']:12.4f} "
            f"{metrics['Recall@5']:10.4f} "
            f"{metrics['NDCG@5']:10.4f} "
            f"{metrics['MRR']:10.4f}"
        )
    print("-" * 76)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-path", type=Path, default=ROOT / "data" / "cases" / "cases.jsonl")
    parser.add_argument("--pool-size", type=int, default=30_000)
    parser.add_argument("--query-size", type=int, default=3_000)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--legacy-train-size", type=int, default=3_000)
    parser.add_argument("--learned-max-train-queries", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"Loading cases from {args.cases_path}")
    cases = load_cases(args.cases_path, max_cases=args.max_cases)
    print(f"Loaded {len(cases)} cases")

    pool, queries = sample_pool_and_queries(
        cases,
        pool_size=args.pool_size,
        query_size=args.query_size,
        seed=args.seed,
    )
    print(f"Pool size: {len(pool)}")
    print(f"Query size: {len(queries)}")

    print("\nTraining LearnedRetrieverV2")
    learned = LearnedRetrieverV2(
        random_state=args.seed,
        max_train_queries=args.learned_max_train_queries,
    )
    learned.train(pool)

    print("\nBuilding CaseRetriever baseline index")
    baseline = build_baseline_retriever(pool)

    print("\nTraining ExperienceModelV2Legacy")
    legacy = ExperienceModelV2Legacy()
    legacy_train_size = min(args.legacy_train_size, len(pool))
    legacy_train_cases = rng.sample(pool, legacy_train_size)
    print(f"  Legacy train size: {legacy_train_size}")
    legacy.train(legacy_train_cases)

    results = {}
    print("\nEvaluating LearnedRetrieverV2")
    results["LearnedRetrieverV2"] = evaluate_learned(
        learned, pool, queries, args.top_k, args.progress_every
    )

    print("\nEvaluating CaseRetriever")
    results["CaseRetriever"] = evaluate_baseline(
        baseline, pool, queries, args.top_k, args.progress_every
    )

    print("\nEvaluating ExperienceModelV2Legacy")
    results["ExperienceModelV2Legacy"] = evaluate_legacy(
        legacy, pool, queries, args.top_k, args.progress_every
    )

    print_results(results)


if __name__ == "__main__":
    main()
