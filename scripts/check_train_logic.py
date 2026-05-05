import sys, json, random
sys.path.insert(0, r"D:\code\logistic-ai")
from src.rag.solver_ground_truth import SolverDrivenGroundTruth
from src.rag.case_retriever import Case

solver = SolverDrivenGroundTruth(seed=20260428)
random.seed(20260428)

# Load first 850 cases, sample 800 like paper_data_generator does
cases = []
with open(r"D:\code\logistic-ai\data\cases\cases_30k.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if len(cases) >= 850:
            break
        d = json.loads(line)
        cases.append(Case(**d))

train_cases = random.sample(cases, 800)

# Simulate V2 train logic
pos_total = 0
neg_total = 0
max_neg_per_query = 10

for qi, query_case in enumerate(train_cases[:5]):  # just first 5 for speed
    ctx = {
        "current_load_rate": query_case.current_load_rate,
        "urgent_orders": query_case.urgent_orders,
        "available_backup_vehicles": query_case.available_backup_vehicles,
        "avg_delay_minutes": query_case.avg_delay_minutes,
        "affected_routes": query_case.affected_routes,
        "time_window_pressure": query_case.time_window_pressure,
        "customer_priority_mix": query_case.customer_priority_mix,
    }
    gt_action, _ = solver.best_action(query_case.scenario, query_case.event_type, query_case.severity, ctx)
    
    pos = 0
    neg = 0
    for cand_case in train_cases:
        if cand_case is query_case:
            continue
        if cand_case.action == gt_action and cand_case.outcome == "success":
            pos += 1
        else:
            neg += 1
    pos_total += pos
    neg_total += min(neg, max_neg_per_query)
    print(f"  query {qi}: gt={gt_action}, case.action={query_case.action}, match={gt_action==query_case.action}, pos={pos}, neg_candidates={neg}")

print(f"\nTotal pos={pos_total}, neg={neg_total} (first 5 queries only)")
