"""Solver-driven labels for logistics RAG cases.

The labels in this module are intentionally context-sensitive: the same
event type and severity can produce different optimal actions when load,
backup capacity, urgency, and time-window pressure change.
"""

from __future__ import annotations

import copy
import hashlib
import random
from typing import Any

from src.optimizer.vrp_solver import ORToolsSolver, RoutePlan
from src.simulation.models import Vehicle
from src.simulation.scenarios import build_scenario
from src.simulation.simulator import LogisticsSimulator


CANDIDATE_ACTIONS = ("reroute", "ignore", "adjust_capacity", "reassign_order", "delay_tolerant")
TRAFFIC_EVENT_TYPES = {"traffic_accident", "traffic_congestion", "road_closed", "road_narrowing", "bridge_weight_limit"}
ORDER_EVENT_TYPES = {"order_cancel", "order_modify", "order_update", "priority_order_urgent", "delivery_failure"}
VEHICLE_EVENT_TYPES = {"vehicle_breakdown", "vehicle_maintenance", "driver_unavailable", "fuel_shortage"}
WAREHOUSE_EVENT_TYPES = {"warehouse_delay", "inventory_stockout"}
SCENARIO_PROFILES: dict[str, dict[str, float]] = {
    "small": {"vehicles": 5, "orders": 20, "max_solver_orders": 12},
    "medium": {"vehicles": 10, "orders": 50, "max_solver_orders": 15},
    "large": {"vehicles": 20, "orders": 100, "max_solver_orders": 18},
    "stress": {"vehicles": 30, "orders": 200, "max_solver_orders": 22},
}


def _fallback_ground_truth_action(event_type: str, severity: int) -> str:
    if event_type in ("vehicle_breakdown", "vehicle_maintenance", "driver_unavailable"):
        if severity >= 5:
            return "reassign_order"
        if severity >= 3:
            return "adjust_capacity"
        return "reroute"
    if event_type == "fuel_shortage":
        return "adjust_capacity" if severity >= 4 else "reroute"
    if event_type in ("traffic_accident", "traffic_congestion", "road_narrowing"):
        return "reroute" if severity >= 4 else "ignore"
    if event_type in ("road_closed", "bridge_weight_limit"):
        return "reroute" if severity >= 3 else "delay_tolerant"
    if event_type == "order_cancel":
        if severity >= 4:
            return "reroute"
        if severity >= 2:
            return "delay_tolerant"
        return "ignore"
    if event_type == "order_modify":
        return "reroute" if severity >= 4 else "delay_tolerant"
    if event_type == "order_update":
        return "ignore"
    if event_type in ("priority_order_urgent", "delivery_failure"):
        return "reassign_order"
    if event_type == "demand_surge":
        return "adjust_capacity" if severity >= 4 else "delay_tolerant"
    if event_type == "demand_drop":
        return "delay_tolerant"
    if event_type in ("weather_delay", "natural_disaster", "public_event"):
        return "delay_tolerant" if severity >= 4 else "reroute"
    if event_type in ("warehouse_delay", "inventory_stockout"):
        return "delay_tolerant"
    return "reroute"


class SolverDrivenGroundTruth:
    """Generate ground truth actions by comparing OR-Tools-priced candidates."""

    def __init__(self, seed: int = 20260428):
        self.seed = seed
        self._cost_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._best_cache: dict[tuple[Any, ...], tuple[str, dict[str, Any]]] = {}

    def generate_context(self, scenario: str, event_type: str, severity: int, rng) -> dict:
        """Generate deterministic, realistic context features for an event."""
        profile = SCENARIO_PROFILES[scenario]
        vehicles = int(profile["vehicles"])
        orders = int(profile["orders"])
        severity_ratio = severity / 5.0

        load_low = 0.35 if scenario in {"small", "medium"} else 0.45
        load_high = 0.92 if scenario != "stress" else 0.98
        current_load_rate = min(0.99, max(0.15, rng.uniform(load_low, load_high) + 0.04 * (severity - 3)))

        urgent_base = int(orders * rng.uniform(0.00, 0.035))
        if event_type in {"priority_order_urgent", "delivery_failure", "order_modify", "warehouse_delay"}:
            urgent_base += max(1, int(orders * rng.uniform(0.03, 0.10)))
        urgent_orders = urgent_base + int(rng.random() < 0.35 + severity_ratio * 0.25) * rng.randint(0, max(1, int(orders * 0.035)))
        urgent_orders = min(urgent_orders, max(1, int(orders * 0.18)))

        spare_cap = max(1, int(round(vehicles * (0.30 if scenario != "stress" else 0.18))))
        backup_bias = 1.0 - current_load_rate
        available_backup_vehicles = min(spare_cap, int(round(rng.uniform(0, spare_cap) * (0.55 + backup_bias))))

        delay_base = severity * rng.uniform(6.0, 15.0)
        if event_type in TRAFFIC_EVENT_TYPES | {"weather_delay", "natural_disaster", "public_event"}:
            delay_base *= 1.25
        avg_delay_minutes = max(0.0, delay_base + current_load_rate * rng.uniform(3.0, 18.0))

        affected_routes = max(1, int(round(severity_ratio * vehicles * rng.uniform(0.25, 0.75))))
        if event_type in VEHICLE_EVENT_TYPES:
            affected_routes = max(affected_routes, min(vehicles, severity))
        affected_routes = min(vehicles, affected_routes)

        time_window_pressure = min(
            1.0,
            max(0.0, rng.uniform(0.15, 0.65) + 0.10 * severity + 0.16 * current_load_rate),
        )
        customer_priority_mix = min(
            1.0,
            max(
                0.0,
                rng.uniform(0.05, 0.30)
                + (0.20 if event_type in {"priority_order_urgent", "delivery_failure"} else 0.0)
                + 0.025 * severity,
            ),
        )

        return {
            "current_load_rate": round(current_load_rate, 3),
            "urgent_orders": int(urgent_orders),
            "available_backup_vehicles": int(available_backup_vehicles),
            "avg_delay_minutes": round(avg_delay_minutes, 2),
            "affected_routes": int(affected_routes),
            "time_window_pressure": round(time_window_pressure, 3),
            "customer_priority_mix": round(customer_priority_mix, 3),
        }

    def solve_action_cost(
        self,
        scenario: str,
        event_type: str,
        severity: int,
        action: str,
        context: dict,
    ) -> dict:
        """Simulate an action and return comparable cost details."""
        cache_key = self._cache_key(scenario, event_type, severity, action, context)
        if cache_key in self._cost_cache:
            return dict(self._cost_cache[cache_key])

        try:
            base_plan, cost_before, disrupted_simulator, solver = self._prepare_solver_state(
                scenario,
                event_type,
                severity,
                context,
            )
            result = self._solve_action_from_state(
                base_plan,
                cost_before,
                copy.deepcopy(disrupted_simulator),
                solver,
                event_type,
                severity,
                action,
                context,
            )

            self._cost_cache[cache_key] = dict(result)
            return result
        except Exception as exc:
            result = {
                "action": action,
                "cost": float("inf"),
                "distance": 0.0,
                "unassigned": 0,
                "cost_before": 0.0,
                "solved": False,
                "error": str(exc),
            }
            self._cost_cache[cache_key] = dict(result)
            return result

    def best_action(
        self,
        scenario: str,
        event_type: str,
        severity: int,
        context: dict,
    ) -> tuple[str, dict]:
        """Solve all candidate actions and return the cheapest one plus costs."""
        cache_key = self._cache_key(scenario, event_type, severity, "best", context)
        if cache_key in self._best_cache:
            action, details = self._best_cache[cache_key]
            return action, copy.deepcopy(details)

        try:
            base_plan, cost_before, disrupted_simulator, solver = self._prepare_solver_state(
                scenario,
                event_type,
                severity,
                context,
            )
            results = {
                action: self._solve_action_from_state(
                    base_plan,
                    cost_before,
                    copy.deepcopy(disrupted_simulator),
                    solver,
                    event_type,
                    severity,
                    action,
                    context,
                )
                for action in CANDIDATE_ACTIONS
            }
            for action, result in results.items():
                self._cost_cache[self._cache_key(scenario, event_type, severity, action, context)] = dict(result)
        except Exception as exc:
            results = {
                action: {
                    "action": action,
                    "cost": float("inf"),
                    "distance": 0.0,
                    "unassigned": 0,
                    "cost_before": 0.0,
                    "solved": False,
                    "error": str(exc),
                }
                for action in CANDIDATE_ACTIONS
            }
        solved_results = {action: data for action, data in results.items() if data.get("solved") and data["cost"] < float("inf")}
        if not solved_results:
            fallback = _fallback_ground_truth_action(event_type, severity)
            details = {
                "costs": {action: data["cost"] for action, data in results.items()},
                "results": results,
                "solved": False,
                "fallback": True,
            }
            self._best_cache[cache_key] = (fallback, copy.deepcopy(details))
            return fallback, details

        best = min(
            solved_results,
            key=lambda item: (solved_results[item]["cost"], CANDIDATE_ACTIONS.index(item)),
        )
        details = {
            "costs": {action: data["cost"] for action, data in results.items()},
            "results": results,
            "solved": True,
            "fallback": False,
        }
        self._best_cache[cache_key] = (best, copy.deepcopy(details))
        return best, details

    def _prepare_solver_state(
        self,
        scenario: str,
        event_type: str,
        severity: int,
        context: dict,
    ) -> tuple[RoutePlan, float, LogisticsSimulator, ORToolsSolver]:
        simulator = self._build_simulator(scenario, event_type, severity, context)
        solver = ORToolsSolver(time_limit_seconds=0.025)
        base_simulator = copy.deepcopy(simulator)
        base_plan = solver.solve(base_simulator.vehicles, base_simulator.orders, base_simulator.road_network)
        cost_before = self._score_plan(base_plan, context, action="before", event_type=event_type, severity=severity)
        self._apply_event(simulator, event_type, severity, context)
        return base_plan, cost_before, simulator, solver

    def _solve_action_from_state(
        self,
        base_plan: RoutePlan,
        cost_before: float,
        simulator: LogisticsSimulator,
        solver: ORToolsSolver,
        event_type: str,
        severity: int,
        action: str,
        context: dict,
    ) -> dict:
        if action == "ignore":
            return self._score_ignore(base_plan, cost_before, event_type, severity, context)
        self._apply_action(simulator, action, context)
        plan = solver.solve(simulator.vehicles, simulator.orders, simulator.road_network)
        return self._score_solved_plan(plan, cost_before, event_type, severity, action, context)

    def _build_simulator(self, scenario: str, event_type: str, severity: int, context: dict) -> LogisticsSimulator:
        config = build_scenario(scenario)
        config.num_orders = min(config.num_orders, int(SCENARIO_PROFILES[scenario]["max_solver_orders"]))
        config.seed = self.seed + self._stable_int(scenario, event_type, severity, self._context_signature(context)) % 100_000
        simulator = LogisticsSimulator(config).generate()
        self._shape_base_state(simulator, scenario, context)
        return simulator

    def _shape_base_state(self, simulator: LogisticsSimulator, scenario: str, context: dict) -> None:
        rng = random.Random(self.seed + self._stable_int("shape", scenario, self._context_signature(context)))
        active_target = max(1, int(round(len(simulator.orders) * float(context.get("current_load_rate", 0.7)))))
        active_target = min(active_target, int(SCENARIO_PROFILES[scenario]["max_solver_orders"]))
        selected = set(rng.sample([order.id for order in simulator.orders], active_target))
        for order in simulator.orders:
            order.cancelled = order.id not in selected

        active_orders = [order for order in simulator.orders if not order.cancelled]
        high_priority_count = min(
            len(active_orders),
            max(int(context.get("urgent_orders", 0)), int(round(len(active_orders) * float(context.get("customer_priority_mix", 0.0))))),
        )
        for order in rng.sample(active_orders, high_priority_count) if high_priority_count else []:
            order.priority = 5
            if float(context.get("time_window_pressure", 0.0)) > 0.55:
                order.deliverywindow.endhour = max(order.deliverywindow.starthour + 0.25, order.deliverywindow.endhour - 0.25)

        depot = simulator.road_network.depot_id if simulator.road_network else simulator.vehicles[0].currentlocation
        existing = simulator.vehicles[:]
        avg_weight = sum(vehicle.capacityweight for vehicle in existing) / max(len(existing), 1)
        avg_volume = sum(vehicle.capacityvolume for vehicle in existing) / max(len(existing), 1)
        avg_speed = sum(vehicle.speed for vehicle in existing) / max(len(existing), 1)
        avg_cost = sum(vehicle.costper_km for vehicle in existing) / max(len(existing), 1)
        for index in range(int(context.get("available_backup_vehicles", 0))):
            simulator.vehicles.append(
                Vehicle(
                    id=f"backup_{index}",
                    capacityweight=avg_weight,
                    capacityvolume=avg_volume,
                    currentlocation=depot,
                    speed=avg_speed,
                    costper_km=avg_cost * 1.08,
                    available=False,
                )
            )

    def _apply_event(self, simulator: LogisticsSimulator, event_type: str, severity: int, context: dict) -> None:
        rng = random.Random(self.seed + self._stable_int("event", event_type, severity, self._context_signature(context)))
        affected_routes = int(context.get("affected_routes", max(1, severity)))

        if event_type in VEHICLE_EVENT_TYPES:
            active = [vehicle for vehicle in simulator.vehicles if vehicle.available]
            impacted = rng.sample(active, min(len(active), max(1, affected_routes)))
            for vehicle in impacted:
                if event_type == "fuel_shortage":
                    vehicle.speed = max(5.0, vehicle.speed * (1.0 - 0.09 * severity))
                else:
                    vehicle.available = False

        if event_type in TRAFFIC_EVENT_TYPES | {"weather_delay", "natural_disaster", "public_event"} and simulator.road_network:
            edge_count = min(len(simulator.road_network.edges), max(1, affected_routes * 6))
            for edge in rng.sample(simulator.road_network.edges, edge_count):
                if event_type in {"road_closed", "bridge_weight_limit", "natural_disaster"} and severity >= 3:
                    edge.congestion = 1.0
                    edge.travel_time *= 2.0 + 0.25 * severity
                else:
                    edge.congestion = min(1.0, edge.congestion + 0.10 * severity)

        active_orders = [order for order in simulator.orders if not order.cancelled]
        if event_type == "order_cancel":
            for order in rng.sample(active_orders, min(len(active_orders), max(1, severity))):
                order.cancelled = True
        elif event_type in {"order_modify", "order_update", "priority_order_urgent", "delivery_failure"}:
            count = min(len(active_orders), max(1, int(context.get("urgent_orders", severity))))
            for order in rng.sample(active_orders, count):
                order.priority = 5
                order.deliverywindow.endhour = max(order.deliverywindow.starthour + 0.2, order.deliverywindow.endhour - 0.15 * severity)
        elif event_type == "demand_surge":
            self._add_surge_orders(simulator, rng, severity, context)
        elif event_type == "demand_drop":
            for order in rng.sample(active_orders, min(len(active_orders), max(1, severity * 2))):
                order.cancelled = True
        elif event_type in WAREHOUSE_EVENT_TYPES:
            for order in active_orders:
                order.pickupwindow.starthour += 0.08 * severity
                order.pickupwindow.endhour += 0.08 * severity

    def _apply_action(self, simulator: LogisticsSimulator, action: str, context: dict) -> None:
        if action == "adjust_capacity":
            for vehicle in simulator.vehicles:
                if vehicle.id.startswith("backup_"):
                    vehicle.available = True
            for vehicle in simulator.vehicles:
                if vehicle.available:
                    vehicle.capacityweight *= 1.0 + 0.04 * float(context.get("current_load_rate", 0.7))
                    vehicle.capacityvolume *= 1.0 + 0.04 * float(context.get("current_load_rate", 0.7))
        elif action == "reassign_order":
            active_orders = [order for order in simulator.orders if not order.cancelled]
            urgent_count = min(len(active_orders), max(1, int(context.get("urgent_orders", 0))))
            prioritized = sorted(active_orders, key=lambda order: (order.priority, -order.created_at), reverse=True)[:urgent_count]
            for order in prioritized:
                order.priority = 5
                order.deliverywindow.endhour += 0.15
        elif action == "delay_tolerant":
            pressure = float(context.get("time_window_pressure", 0.5))
            for order in simulator.orders:
                if not order.cancelled:
                    relax = 0.35 + 0.55 * pressure
                    order.pickupwindow.endhour += relax
                    order.deliverywindow.endhour += relax

    def _add_surge_orders(self, simulator: LogisticsSimulator, rng: random.Random, severity: int, context: dict) -> None:
        active_orders = [order for order in simulator.orders if not order.cancelled]
        if not active_orders:
            return
        count = min(max(1, severity * 2), max(1, len(active_orders) // 5))
        for index, source in enumerate(rng.sample(active_orders, min(len(active_orders), count))):
            clone = copy.deepcopy(source)
            clone.id = f"surge_{index}_{source.id}"
            clone.priority = 4 if float(context.get("customer_priority_mix", 0.0)) > 0.3 else source.priority
            clone.created_at = max(0.0, source.created_at + 0.05)
            clone.cancelled = False
            simulator.orders.append(clone)

    def _score_ignore(
        self,
        base_plan: RoutePlan,
        cost_before: float,
        event_type: str,
        severity: int,
        context: dict,
    ) -> dict:
        penalty = (
            severity * 90.0
            + float(context.get("avg_delay_minutes", 0.0)) * (2.0 + float(context.get("time_window_pressure", 0.0)) * 5.0)
            + int(context.get("urgent_orders", 0)) * 80.0
            + int(context.get("affected_routes", 0)) * 45.0
            + float(context.get("customer_priority_mix", 0.0)) * 300.0
        )
        if event_type in {"order_update", "demand_drop"} and severity <= 2:
            penalty *= 0.30
        if event_type in {"road_closed", "vehicle_breakdown", "natural_disaster"}:
            penalty *= 1.45
        cost = cost_before + penalty
        return {
            "action": "ignore",
            "cost": round(cost, 3),
            "distance": round(base_plan.total_distance, 3),
            "unassigned": len(base_plan.unassigned_order_ids),
            "cost_before": round(cost_before, 3),
            "solved": True,
        }

    def _score_solved_plan(
        self,
        plan: RoutePlan,
        cost_before: float,
        event_type: str,
        severity: int,
        action: str,
        context: dict,
    ) -> dict:
        cost = self._score_plan(plan, context, action, event_type, severity)
        if action == "adjust_capacity":
            cost += max(1, int(context.get("available_backup_vehicles", 0))) * 110.0
            if int(context.get("available_backup_vehicles", 0)) <= 0:
                cost += 300.0
            cost -= (
                float(context.get("current_load_rate", 0.0)) * 280.0
                + int(context.get("available_backup_vehicles", 0)) * 90.0
                + (180.0 if event_type in VEHICLE_EVENT_TYPES | {"demand_surge", "warehouse_delay"} else 0.0)
            )
        elif action == "reassign_order":
            cost += 160.0
            cost -= min(850.0, int(context.get("urgent_orders", 0)) * 85.0 + float(context.get("customer_priority_mix", 0.0)) * 320.0)
        elif action == "delay_tolerant":
            planned_orders = self._planned_order_count(plan)
            priority_mix = float(context.get("customer_priority_mix", 0.0))
            cost += (
                220.0
                + planned_orders * (35.0 + priority_mix * 95.0)
                + int(context.get("urgent_orders", 0)) * 90.0
                + priority_mix * 420.0
            )
            cost -= float(context.get("time_window_pressure", 0.0)) * 160.0
            cost = max(cost, cost_before * (0.84 + priority_mix * 0.35))
        elif action == "reroute":
            cost += 60.0 + int(context.get("affected_routes", 0)) * 10.0
            if event_type in TRAFFIC_EVENT_TYPES:
                cost -= 420.0 + severity * 35.0
        return {
            "action": action,
            "cost": round(max(0.0, cost), 3),
            "distance": round(plan.total_distance, 3),
            "unassigned": len(plan.unassigned_order_ids),
            "cost_before": round(cost_before, 3),
            "solved": bool(plan.routes or not plan.unassigned_order_ids),
        }

    def _planned_order_count(self, plan: RoutePlan) -> int:
        assigned = {
            stop.order_id
            for stops in plan.routes.values()
            for stop in stops
            if stop.order_id is not None
        }
        return len(assigned) + len(plan.unassigned_order_ids)

    def _score_plan(self, plan: RoutePlan, context: dict, action: str, event_type: str, severity: int) -> float:
        urgent = int(context.get("urgent_orders", 0))
        pressure = float(context.get("time_window_pressure", 0.0))
        priority_mix = float(context.get("customer_priority_mix", 0.0))
        unassigned_penalty = 900.0 + 500.0 * priority_mix + 80.0 * severity
        delay_penalty = float(context.get("avg_delay_minutes", 0.0)) * (0.8 + pressure * 2.4)
        return (
            plan.total_distance
            + len(plan.unassigned_order_ids) * unassigned_penalty
            + urgent * 35.0
            + delay_penalty
        )

    def _cache_key(self, scenario: str, event_type: str, severity: int, action: str, context: dict) -> tuple[Any, ...]:
        return (scenario, event_type, int(severity), action, self._context_signature(context))

    def _context_signature(self, context: dict) -> tuple[Any, ...]:
        return (
            round(float(context.get("current_load_rate", 0.0)), 3),
            int(context.get("urgent_orders", 0)),
            int(context.get("available_backup_vehicles", 0)),
            round(float(context.get("avg_delay_minutes", 0.0)), 2),
            int(context.get("affected_routes", 0)),
            round(float(context.get("time_window_pressure", 0.0)), 3),
            round(float(context.get("customer_priority_mix", 0.0)), 3),
        )

    def _stable_int(self, *parts: Any) -> int:
        raw = "|".join(str(part) for part in parts).encode("utf-8")
        return int(hashlib.sha256(raw).hexdigest()[:12], 16)
