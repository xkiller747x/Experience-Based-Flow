import sys
sys.path.insert(0, r"D:\code\logistic-ai")
from src.rag.case_retriever import CaseRetriever
from src.rag.solver_ground_truth import SolverDrivenGroundTruth

r = CaseRetriever()
cases = r.cases
solver = SolverDrivenGroundTruth(seed=20260428)
match = 0
total = min(200, len(cases))
for c in cases[:total]:
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

print(f"{match}/{total} = {match/total*100:.1f}% match between case.action and solver gt_action")
