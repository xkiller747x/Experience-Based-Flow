import sys, json
sys.path.insert(0, r"D:\code\logistic-ai")
from src.rag.solver_ground_truth import SolverDrivenGroundTruth
from src.rag.case_retriever import Case

solver = SolverDrivenGroundTruth(seed=20260428)
match = 0
total = 0

with open(r"D:\code\logistic-ai\data\cases\cases_30k.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 500:
            break
        d = json.loads(line)
        c = Case(**d)
        ctx = {
            "current_load_rate": c.current_load_rate,
            "urgent_orders": c.urgent_orders,
            "available_backup_vehicles": c.available_backup_vehicles,
            "avg_delay_minutes": c.avg_delay_minutes,
            "affected_routes": c.affected_routes,
            "time_window_pressure": c.time_window_pressure,
            "customer_priority_mix": c.customer_priority_mix,
        }
        gt, _ = solver.best_action(c.scenario, c.event_type, c.severity, ctx)
        if gt == c.action:
            match += 1
        total += 1

print(f"{match}/{total} = {match/total*100:.1f}% match (30k solver-driven cases)")
