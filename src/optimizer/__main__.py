"""Run a small-scenario smoke test for the OR-Tools VRP solver."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.optimizer import ORToolsSolver
from src.simulation.scenarios import small
from src.simulation.simulator import LogisticsSimulator


def main() -> None:
    simulator = LogisticsSimulator(small()).generate()
    if simulator.road_network is None:
        raise RuntimeError("small scenario did not generate a road network")

    solver = ORToolsSolver(time_limit_seconds=5)
    plan = solver.solve(
        vehicles=simulator.get_vehicle_states(),
        orders=simulator.get_active_orders(),
        road_network=simulator.road_network,
    )

    used_routes = {
        vehicle_id: stops for vehicle_id, stops in plan.routes.items() if stops
    }
    assigned_order_ids = {
        stop.order_id
        for stops in used_routes.values()
        for stop in stops
        if stop.stop_type == "pickup" and stop.order_id is not None
    }

    print("Small scenario VRP result")
    print(f"vehicles: {len(simulator.vehicles)}")
    print(f"orders: {len(simulator.get_active_orders())}")
    print(f"assigned_orders: {len(assigned_order_ids)}")
    print(f"unassigned_orders: {len(plan.unassigned_order_ids)}")
    print(f"total_distance_km: {plan.total_distance:.2f}")
    print(f"total_travel_time_min: {plan.total_travel_time:.2f}")

    for vehicle_id, stops in used_routes.items():
        print(f"\n{vehicle_id}:")
        for stop in stops:
            print(
                "  "
                f"{stop.stop_type:<8} "
                f"order={stop.order_id:<8} "
                f"node={stop.node_id:<12} "
                f"arrival={stop.arrival_time:.2f}h "
                f"leg={stop.travel_distance:.2f}km/{stop.travel_time:.1f}min"
            )

    if plan.unassigned_order_ids:
        print("\nunassigned_order_ids:")
        print("  " + ", ".join(plan.unassigned_order_ids))


if __name__ == "__main__":
    main()
