"""Historical case generator for the logistics RAG system.

This module creates synthetic case data by running the logistics simulator
and VRP solver, then injecting events and recording the outcomes.
"""

from __future__ import annotations

import json
import gc
import random
import sys
from pathlib import Path
from typing import Sequence

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag import Case
from src.simulation.config import SimulationConfig
from src.simulation.scenarios import build_scenario
from src.simulation.generator import generate_events
from src.simulation.models import Event, EventType
from src.simulation.simulator import LogisticsSimulator
from src.optimizer.vrp_solver import ORToolsSolver, RoutePlan


# ---------------------------------------------------------------------------
# Scenario recipe definitions
# ---------------------------------------------------------------------------

SCENARIO_RECIPES: dict[str, dict] = {
    "small":  {"n": 325, "seed_offset_base": 0,    "solver_secs": 0.1},
    "medium": {"n": 325, "seed_offset_base": 500,  "solver_secs": 0.2},
    "large":  {"n": 325, "seed_offset_base": 1000, "solver_secs": 0.3},
    "stress": {"n": 325, "seed_offset_base": 1500, "solver_secs": 0.5},
}

EVENT_TYPE_MAP = {
    EventType.VEHICLEBREAKDOWN: "vehicle_breakdown",
    EventType.TRAFFICACCIDENT: "traffic_accident",
    EventType.ROADCLOSED: "road_closed",
    EventType.ORDERCANCEL: "order_cancel",
    EventType.VEHICLEMAINTENANCE: "vehicle_maintenance",
    EventType.FUELSHORTAGE: "fuel_shortage",
    EventType.DRIVERUNAVAILABLE: "driver_unavailable",
    EventType.TRAFFICCONGESTION: "traffic_congestion",
    EventType.ROADNARROWING: "road_narrowing",
    EventType.BRIDGEWEIGHTLIMIT: "bridge_weight_limit",
    EventType.ORDERMODIFY: "order_modify",
    EventType.ORDERUPDATE: "order_update",
    EventType.PRIORITYORDERURGENT: "priority_order_urgent",
    EventType.DELIVERYFAILURE: "delivery_failure",
    EventType.DEMANDSURGE: "demand_surge",
    EventType.DEMANDDROP: "demand_drop",
    EventType.WEATHERDELAY: "weather_delay",
    EventType.NATURALDISASTER: "natural_disaster",
    EventType.PUBLICEVENT: "public_event",
    EventType.WAREHOUSEDELAY: "warehouse_delay",
    EventType.INVENTORYSTOCKOUT: "inventory_stockout",
}

ACTION_POOL = [
    "reroute",
    "ignore",
    "adjust_capacity",
    "reassign_order",
    "delay_tolerant",
    "normal",  # New: normal operating state, no intervention needed
]

FAILURE_RATE = 0.20
SAVE_EVERY = 10_000
SCENARIOS = ("small", "medium", "large", "stress")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seeded_config(scenario: str, index: int) -> SimulationConfig:
    cfg = build_scenario(scenario)
    base = SCENARIO_RECIPES[scenario]["seed_offset_base"]
    cfg.seed = 42 + base + index
    return cfg


def _flush_cases(fh, buffered_cases: list[str]) -> int:
    """Write buffered JSONL cases to disk and release Python-held memory."""
    if not buffered_cases:
        return 0

    fh.write("".join(buffered_cases))
    fh.flush()
    flushed = len(buffered_cases)
    buffered_cases.clear()
    gc.collect()
    return flushed


def _force_failure(case: Case) -> Case:
    return Case(
        id=case.id,
        event_type=case.event_type,
        severity=case.severity,
        scenario=case.scenario,
        vehicles_count=case.vehicles_count,
        orders_count=case.orders_count,
        action=case.action,
        outcome="failure",
        reasoning="Action insufficient to handle disruption. " + case.reasoning,
        before_distance=case.before_distance,
        after_distance=case.after_distance,
        unassigned_before=case.unassigned_before,
        unassigned_after=case.unassigned_after + 1,
        timestamp=case.timestamp,
    )


def _solve_and_measure(
    sim: LogisticsSimulator,
    solver: ORToolsSolver,
) -> tuple[RoutePlan, int, float]:
    plan = solver.solve(
        vehicles=sim.vehicles,
        orders=sim.orders,
        road_network=sim.road_network,
    )
    return plan, len(plan.unassigned_order_ids), plan.total_distance


def _inject_breakdown(sim: LogisticsSimulator) -> tuple[Event, list[str]]:
    available = [v for v in sim.vehicles if v.available]
    vehicle = random.choice(available) if available else sim.vehicles[0]
    event = Event(
        timestamp=sim.current_time + random.uniform(0.5, 3.0),
        type=EventType.VEHICLEBREAKDOWN,
        location=vehicle.id,
        severity=random.randint(1, 5),
        description="Vehicle breakdown reported",
    )
    sim.inject_event(event)
    return event, [vehicle.id] if vehicle.available else []


def _select_or_inject_event(sim: LogisticsSimulator) -> tuple[Event, Sequence[str]]:
    future = [e for e in sim.events if e.timestamp >= sim.current_time]
    if future:
        event = random.choice(future)
        return event, event.affected_orders
    return _inject_breakdown(sim)


def _select_action(event_type: EventType, severity: int) -> str:
    """
    Deterministic action selection based on business rules.
    Core change: action derived from rules, not random.
    """
    # Vehicle-type anomalies
    if event_type in (EventType.VEHICLEBREAKDOWN, EventType.VEHICLEMAINTENANCE, EventType.DRIVERUNAVAILABLE):
        if severity >= 5:
            return "reassign_order"
        elif severity >= 3:
            return "adjust_capacity"
        else:
            return "reroute"

    # Fuel-type anomalies
    if event_type == EventType.FUELSHORTAGE:
        if severity >= 4:
            return "adjust_capacity"
        else:
            return "reroute"

    # Traffic-type anomalies
    if event_type in (EventType.TRAFFICACCIDENT, EventType.TRAFFICCONGESTION, EventType.ROADNARROWING):
        if severity >= 4:
            return "reroute"
        else:
            return "ignore"

    # Road closure-type
    if event_type in (EventType.ROADCLOSED, EventType.BRIDGEWEIGHTLIMIT):
        if severity >= 3:
            return "reroute"
        else:
            return "delay_tolerant"

    # Order-type anomalies
    if event_type == EventType.ORDERCANCEL:
        if severity >= 4:
            return "reroute"
        elif severity >= 2:
            return "delay_tolerant"
        else:
            return "ignore"

    if event_type == EventType.ORDERMODIFY:
        if severity >= 4:
            return "reroute"
        else:
            return "delay_tolerant"

    if event_type == EventType.ORDERUPDATE:
        return "ignore"

    if event_type == EventType.PRIORITYORDERURGENT:
        return "reassign_order"

    if event_type == EventType.DELIVERYFAILURE:
        return "reassign_order"

    # Demand-type anomalies
    if event_type == EventType.DEMANDSURGE:
        if severity >= 4:
            return "adjust_capacity"
        else:
            return "delay_tolerant"

    if event_type == EventType.DEMANDDROP:
        return "delay_tolerant"

    # External-type anomalies
    if event_type in (EventType.WEATHERDELAY, EventType.NATURALDISASTER, EventType.PUBLICEVENT):
        if severity >= 4:
            return "delay_tolerant"
        else:
            return "reroute"

    # Warehouse-type anomalies
    if event_type in (EventType.WAREHOUSEDELAY, EventType.INVENTORYSTOCKOUT):
        return "delay_tolerant"

    # Default fallback
    return "reroute"


def _simulate_after_event(
    sim: LogisticsSimulator,
    event: Event,
    plan_before: RoutePlan,
) -> tuple[int, float]:
    unassigned_before = len(plan_before.unassigned_order_ids)
    dist_before = plan_before.total_distance

    if event.type in (EventType.VEHICLEBREAKDOWN, EventType.VEHICLEMAINTENANCE, EventType.DRIVERUNAVAILABLE):
        broken_id = event.location
        for v in sim.vehicles:
            if v.id == broken_id:
                v.available = False
                break
        n_affected = max(1, len(sim.orders) // len(sim.vehicles))
        reassignable = random.randint(0, max(0, n_affected - 1))
        unassigned_after = n_affected - reassignable
        after_dist = dist_before + n_affected * random.uniform(3.0, 8.0)

    elif event.type == EventType.FUELSHORTAGE:
        for v in sim.vehicles:
            if random.random() < 0.3:
                v.speed = v.speed * 0.5
        overhead = 1.0 + (event.severity / 5.0) * 0.15
        unassigned_after = unassigned_before
        after_dist = dist_before * overhead

    elif event.type in (EventType.TRAFFICACCIDENT, EventType.TRAFFICCONGESTION, EventType.ROADNARROWING):
        sim._update_edge_congestion(event.location, increment=0.1 * event.severity)
        overhead = 1.0 + (event.severity / 5.0) * 0.35
        unassigned_after = unassigned_before + random.randint(0, 1) if event.severity >= 4 else unassigned_before
        after_dist = dist_before * overhead

    elif event.type in (EventType.ROADCLOSED, EventType.BRIDGEWEIGHTLIMIT):
        sim._set_edge_closed(event.location)
        overhead = 1.0 + (event.severity / 5.0) * 0.5
        unassigned_after = unassigned_before + random.randint(0, 2) if event.severity >= 3 else unassigned_before
        after_dist = dist_before * overhead

    elif event.type in (EventType.ORDERCANCEL, EventType.ORDERMODIFY, EventType.ORDERUPDATE):
        for oid in event.affected_orders:
            for o in sim.orders:
                if o.id == oid:
                    o.cancelled = True
        unassigned_after = max(0, unassigned_before - len(event.affected_orders))
        after_dist = max(0, dist_before - len(event.affected_orders) * random.uniform(2.0, 5.0))

    elif event.type == EventType.PRIORITYORDERURGENT:
        for oid in event.affected_orders:
            for o in sim.orders:
                if o.id == oid:
                    o.priority = max(o.priority + 2, 10)
        unassigned_after = unassigned_before
        after_dist = dist_before * (1.0 + event.severity * 0.05)

    elif event.type == EventType.DELIVERYFAILURE:
        n_affected = max(1, len(event.affected_orders))
        unassigned_after = unassigned_before + n_affected
        after_dist = dist_before + n_affected * random.uniform(5.0, 10.0)

    elif event.type in (EventType.DEMANDSURGE, EventType.DEMANDDROP):
        extra_orders = random.randint(2, 5) if event.type == EventType.DEMANDSURGE else 0
        unassigned_after = unassigned_before + extra_orders
        overhead = 1.1 if event.type == EventType.DEMANDSURGE else 0.95
        after_dist = dist_before * overhead

    elif event.type in (EventType.WEATHERDELAY, EventType.NATURALDISASTER, EventType.PUBLICEVENT):
        for v in sim.vehicles:
            v.speed = v.speed * max(0.3, 1.0 - (event.severity / 5.0) * 0.5)
        overhead = 1.0 + (event.severity / 5.0) * 0.6
        unassigned_after = unassigned_before + random.randint(0, 2) if event.severity >= 4 else unassigned_before
        after_dist = dist_before * overhead

    elif event.type in (EventType.WAREHOUSEDELAY, EventType.INVENTORYSTOCKOUT):
        unassigned_after = unassigned_before + random.randint(1, 3)
        after_dist = dist_before

    else:
        unassigned_after = unassigned_before
        after_dist = dist_before

    return unassigned_after, after_dist


def _estimate_outcome(
    event_type: EventType,
    event: Event,
    plan_before: RoutePlan,
    unassigned_after: int,
    after_dist: float,
    affected_orders: Sequence[str],
) -> tuple[str, str]:
    unassigned_before = len(plan_before.unassigned_order_ids)

    if unassigned_after > unassigned_before:
        outcome = "failure"
    elif unassigned_after == unassigned_before and after_dist > plan_before.total_distance * 1.5:
        outcome = "failure"
    else:
        outcome = "success"

    ev_str = EVENT_TYPE_MAP.get(event_type, str(event_type.value))
    sev = event.severity

    if event.type == EventType.VEHICLEBREAKDOWN:
        reasoning = (
            f"Vehicle {event.location} broke down (severity {sev}). "
            f"Action: reassign orders to available vehicles. "
            f"Unassigned: {unassigned_before}->{unassigned_after}, "
            f"Distance: {plan_before.total_distance:.1f}km->{after_dist:.1f}km."
        )
    elif event.type == EventType.TRAFFICACCIDENT:
        reasoning = (
            f"Traffic accident near {event.location} (severity {sev}). "
            f"Action: reroute affected vehicles. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.ROADCLOSED:
        reasoning = (
            f"Road closure on segment {event.location} (severity {sev}). "
            f"Action: reroute or delay-tolerant. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.ORDERCANCEL:
        oid = affected_orders[0] if affected_orders else "N/A"
        reasoning = (
            f"Order {oid} cancelled (severity {sev}). "
            f"Action: ignore or delay-tolerant. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.VEHICLEMAINTENANCE:
        reasoning = (
            f"Vehicle {event.location} scheduled maintenance (severity {sev}). "
            f"Action: reassign or adjust capacity. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.FUELSHORTAGE:
        reasoning = (
            f"Fuel shortage reported near {event.location} (severity {sev}). "
            f"Action: adjust capacity or reroute. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.DRIVERUNAVAILABLE:
        reasoning = (
            f"Driver unavailable for vehicle {event.location} (severity {sev}). "
            f"Action: reassign or reroute. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.TRAFFICCONGESTION:
        reasoning = (
            f"Traffic congestion on segment {event.location} (severity {sev}). "
            f"Action: reroute or delay-tolerant. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.ROADNARROWING:
        reasoning = (
            f"Road narrowing on segment {event.location} (severity {sev}). "
            f"Action: reroute. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.BRIDGEWEIGHTLIMIT:
        reasoning = (
            f"Bridge weight limit imposed on segment {event.location} (severity {sev}). "
            f"Action: reroute or delay-tolerant. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.ORDERMODIFY:
        oid = affected_orders[0] if affected_orders else "N/A"
        reasoning = (
            f"Order {oid} modified (severity {sev}). "
            f"Action: delay-tolerant or reroute. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.ORDERUPDATE:
        oid = affected_orders[0] if affected_orders else "N/A"
        reasoning = (
            f"Order {oid} updated (severity {sev}). "
            f"Action: ignore or delay-tolerant. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.PRIORITYORDERURGENT:
        oid = affected_orders[0] if affected_orders else "N/A"
        reasoning = (
            f"Urgent priority order {oid} flagged (severity {sev}). "
            f"Action: reassign or reroute. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.DELIVERYFAILURE:
        oid = affected_orders[0] if affected_orders else "N/A"
        reasoning = (
            f"Delivery failure for order {oid} (severity {sev}). "
            f"Action: reassign. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.DEMANDSURGE:
        reasoning = (
            f"Demand surge detected (severity {sev}). "
            f"Action: adjust capacity. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.DEMANDDROP:
        reasoning = (
            f"Demand drop detected (severity {sev}). "
            f"Action: delay-tolerant. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.WEATHERDELAY:
        reasoning = (
            f"Weather delay reported near {event.location} (severity {sev}). "
            f"Action: delay-tolerant. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.NATURALDISASTER:
        reasoning = (
            f"Natural disaster near {event.location} (severity {sev}). "
            f"Action: delay-tolerant or reroute. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.PUBLICEVENT:
        reasoning = (
            f"Public event at {event.location} causing disruption (severity {sev}). "
            f"Action: reroute or delay-tolerant. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.WAREHOUSEDELAY:
        reasoning = (
            f"Warehouse delay at {event.location} (severity {sev}). "
            f"Action: delay-tolerant. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    elif event.type == EventType.INVENTORYSTOCKOUT:
        reasoning = (
            f"Inventory stockout at {event.location} (severity {sev}). "
            f"Action: delay-tolerant or ignore. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )
    else:
        reasoning = (
            f"Event type={ev_str}, severity={sev}. "
            f"Unassigned: {unassigned_before}->{unassigned_after}."
        )

    return outcome, reasoning


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------

def generate_normal_case(scenario: str, index: int) -> Case | None:
    """Generate a normal-state case with no anomalies, capture steady-state snapshot."""
    recipe = SCENARIO_RECIPES[scenario]
    solver_secs = recipe["solver_secs"]

    try:
        solver = ORToolsSolver(time_limit_seconds=solver_secs)
    except Exception:
        return None

    cfg = _seeded_config(scenario, index)
    sim = LogisticsSimulator(cfg)
    sim.generate()
    solver = ORToolsSolver(time_limit_seconds=solver_secs)
    plan = solver.solve(sim.vehicles, sim.orders, sim.road_network)

    return Case(
        id=f"case_normal_{scenario}_{index:04d}",
        event_type="none",
        severity=0,
        scenario=scenario,
        vehicles_count=len(sim.vehicles),
        orders_count=len(sim.orders),
        action="normal",
        outcome="success",
        reasoning="System operating normally. No disruption detected. No action required.",
        before_distance=round(plan.total_distance, 2),
        after_distance=round(plan.total_distance, 2),
        unassigned_before=len(plan.unassigned_order_ids),
        unassigned_after=len(plan.unassigned_order_ids),
        timestamp=round(cfg.simulation_hours, 2),
    )


def generate_one_case(case_id: str, scenario: str, index: int) -> Case | None:
    recipe = SCENARIO_RECIPES[scenario]
    solver_secs = recipe["solver_secs"]

    try:
        solver = ORToolsSolver(time_limit_seconds=solver_secs)
    except Exception:
        return None

    cfg = _seeded_config(scenario, index)
    sim = LogisticsSimulator(cfg)
    sim.generate()

    # Initial solve
    plan_before, unassigned_before, distance_before = _solve_and_measure(sim, solver)

    # Select and inject event
    event, affected_orders = _select_or_inject_event(sim)
    event_type_str = EVENT_TYPE_MAP.get(event.type, str(event.type.value))

    # Select action
    action = _select_action(event.type, event.severity)

    # Simulate post-event state
    unassigned_after, distance_after = _simulate_after_event(sim, event, plan_before)

    # Determine outcome
    outcome, reasoning = _estimate_outcome(
        event.type,
        event,
        plan_before,
        unassigned_after,
        distance_after,
        affected_orders,
    )

    return Case(
        id=case_id,
        event_type=event_type_str,
        severity=event.severity,
        scenario=scenario,
        vehicles_count=len(sim.vehicles),
        orders_count=len(sim.orders),
        action=action,
        outcome=outcome,
        reasoning=reasoning,
        before_distance=round(distance_before, 2),
        after_distance=round(distance_after, 2),
        unassigned_before=unassigned_before,
        unassigned_after=unassigned_after,
        timestamp=round(event.timestamp, 2),
    )


def generate_part(worker_id: int, start_id: int, count: int, output_file: str) -> int:
    """Generate a part of cases for parallel processing, return actual generated count"""
    # Calculate counts for this part: 90% normal, 10% anomaly
    normal_count = int(count * 0.9)
    anomaly_count = count - normal_count
    normal_per_scenario = normal_count // 4
    anomaly_per_scenario = anomaly_count //4

    current_id = start_id
    total_generated = 0
    current_failures = 0

    with open(output_file, "w", encoding="utf-8") as fh:
        # Generate normal cases
        for scenario in ["small", "medium", "large", "stress"]:
            for i in range(normal_per_scenario):
                case = generate_normal_case(scenario, current_id)
                if case is None:
                    continue
                case.id = f"case_{current_id:06d}"
                fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
                current_id += 1
                total_generated += 1
                # Progress per 10k per worker
                if total_generated % 10000 == 0:
                    print(f"[Worker {worker_id}] Generated {total_generated}/{count}...")

        # Generate anomaly cases
        for scenario in ["small", "medium", "large", "stress"]:
            for i in range(anomaly_per_scenario):
                case = generate_one_case(f"case_{current_id:06d}", scenario, i)
                if case is None:
                    continue

                # Enforce ~20% failure rate
                current_total = current_id - 1
                max_failures = int(round(FAILURE_RATE * current_total)) if current_total > 0 else 0

                if case.outcome == "success" and current_failures < max_failures:
                    case = Case(
                        id=case.id,
                        event_type=case.event_type,
                        severity=case.severity,
                        scenario=case.scenario,
                        vehicles_count=case.vehicles_count,
                        orders_count=case.orders_count,
                        action=case.action,
                        outcome="failure",
                        reasoning="Action insufficient to handle disruption. " + case.reasoning,
                        before_distance=case.before_distance,
                        after_distance=case.after_distance,
                        unassigned_before=case.unassigned_before,
                        unassigned_after=case.unassigned_after + 1,
                        timestamp=case.timestamp,
                    )

                if case.outcome == "failure":
                    current_failures +=1
                fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
                current_id += 1
                total_generated += 1
                # Progress per 10k per worker
                if total_generated % 10000 == 0:
                    print(f"[Worker {worker_id}] Generated {total_generated}/{count}...")

    print(f"[Worker {worker_id}] Completed! Generated {total_generated} cases")
    return total_generated


def generate_chunk(
    worker_id: int,
    start_id: int,
    count: int,
    total: int,
    output_file: str,
    flush_every: int = SAVE_EVERY,
) -> int:
    """Generate one bounded chunk and flush it to disk before the worker exits."""
    normal_count = int(total * 0.9)
    current_failures = 0
    total_generated = 0
    buffered_cases: list[str] = []

    with open(output_file, "w", encoding="utf-8") as fh:
        for offset in range(count):
            case_number = start_id + offset
            case_index = case_number - 1
            scenario = SCENARIOS[case_index % len(SCENARIOS)]

            if case_index < normal_count:
                case = generate_normal_case(scenario, case_index)
            else:
                case = generate_one_case(f"case_{case_number:06d}", scenario, case_index)
                current_total = total_generated + 1
                max_failures = int(round(FAILURE_RATE * current_total))
                if (
                    case is not None
                    and case.outcome == "success"
                    and current_failures < max_failures
                ):
                    case = _force_failure(case)

            if case is None:
                continue

            case.id = f"case_{case_number:06d}"
            if case.outcome == "failure":
                current_failures += 1

            buffered_cases.append(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
            total_generated += 1

            if len(buffered_cases) >= flush_every:
                _flush_cases(fh, buffered_cases)
                print(f"[Worker {worker_id}] Saved {total_generated}/{count} cases...")

        _flush_cases(fh, buffered_cases)

    gc.collect()
    print(f"[Worker {worker_id}] Completed chunk! Generated {total_generated} cases")
    return total_generated


def _build_chunk_tasks(
    total: int,
    output_dir: Path,
    chunk_size: int,
) -> tuple[list[tuple[int, int, int, int, str, int]], list[Path]]:
    tasks = []
    temp_files = []
    next_id = 1

    for chunk_id in range((total + chunk_size - 1) // chunk_size):
        count = min(chunk_size, total - (next_id - 1))
        temp_file = output_dir / f"temp_chunk_{chunk_id:06d}.jsonl"
        temp_files.append(temp_file)
        tasks.append((chunk_id, next_id, count, total, str(temp_file), chunk_size))
        next_id += count

    return tasks, temp_files


def _merge_temp_files(output_path: Path, temp_files: Sequence[Path]) -> int:
    total_generated = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for temp_file in temp_files:
            if not temp_file.exists():
                continue
            with temp_file.open(encoding="utf-8") as in_fh:
                for line in in_fh:
                    out_fh.write(line)
                    total_generated += 1
            temp_file.unlink()

    gc.collect()
    return total_generated

def generate_all(
    output_path: str | Path | None = None,
    total: int = 1000000,
    num_workers: int = 8,
    chunk_size: int = SAVE_EVERY,
) -> None:
    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / "data" / "cases" / "cases.jsonl"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if total <= 0:
        output_path.write_text("", encoding="utf-8")
        print(f"No cases requested. Empty file saved to {output_path}")
        return
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    worker_tasks, temp_files = _build_chunk_tasks(total, output_path.parent, chunk_size)

    print(
        f"Starting {num_workers} workers, total {total} cases, "
        f"saving every {chunk_size} cases..."
    )

    import multiprocessing as mp
    with mp.Pool(num_workers, maxtasksperchild=1) as pool:
        results = pool.starmap(generate_chunk, worker_tasks)

    print(f"Merging {len(temp_files)} chunks into final file...")
    total_generated = _merge_temp_files(output_path, temp_files)
    reported_total = sum(results)

    if total_generated != reported_total:
        print(f"Warning: merged {total_generated} lines, workers reported {reported_total} cases")

    print(f"All done! Total {total_generated} cases saved to {output_path}")
    print(f"File size: {round(output_path.stat().st_size / 1024 / 1024, 2)} MB")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows multiprocessing protection
    import multiprocessing as mp
    mp.freeze_support()

    output = Path(__file__).parent.parent.parent / "data" / "cases" / "cases.jsonl"
    generate_all(output, total=500000, num_workers=8, chunk_size=SAVE_EVERY)

