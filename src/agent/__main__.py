"""Test entry point for the LLM Agent module.

This script runs a complete end-to-end demonstration using the LogisticsSimulator
to generate a small scenario, inject a VEHICLE_BREAKDOWN event, run anomaly detection
and decision making, then print the results.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure src/ is on the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.agent import (
    AnomalyDetector,
    DecisionMaker,
    LLMGateway,
    LLMError,
)
from src.rag import CaseRetriever
from src.agent.anomaly import AnomalyResult
from src.agent.decision import DecisionResult
from src.optimizer.vrp_solver import ORToolsSolver
from src.simulation import SimulationConfig, LogisticsSimulator, Event, EventType


def pretty_print(title: str, obj: object) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    if isinstance(obj, (AnomalyResult, DecisionResult)):
        for key, val in obj.to_dict().items():
            print(f"  {key}: {val}")
    elif isinstance(obj, dict):
        for key, val in obj.items():
            print(f"  {key}: {val}")
    else:
        print(f"  {obj}")


def main() -> None:
    print("=" * 60)
    print("  LLM Agent Module — Integration Test")
    print("=" * 60)

    # ── 1. Build a small simulation scenario ────────────────────────────
    print("\n[1] Generating small simulation scenario...")
    config = SimulationConfig(
        num_vehicles=5,
        num_orders=10,
        area_size_km=30.0,
        seed=42,
    )
    simulator = LogisticsSimulator(config)
    simulator.generate()

    print(f"    Road network: depot={simulator.road_network.depot_id}")
    print(f"    Vehicles: {len(simulator.vehicles)}")
    print(f"    Orders: {len(simulator.orders)}")

    # ── 2. Build initial route plan ──────────────────────────────────
    print("\n[2] Building initial route plan with ORToolsSolver...")
    solver = ORToolsSolver(time_limit_seconds=5)
    route_plan = solver.solve(
        vehicles=simulator.vehicles,
        orders=simulator.orders,
        road_network=simulator.road_network,
    )
    print(
        f"    total_distance={route_plan.total_distance:.1f} km, "
        f"total_travel_time={route_plan.total_travel_time:.1f} h, "
        f"unassigned={route_plan.unassigned_order_ids}"
    )

    # ── 3. Inject a vehicle breakdown event ──────────────────────────
    print("\n[3] Injecting VEHICLE_BREAKDOWN event...")
    breakdown_vehicle_id = simulator.vehicles[0].id
    breakdown_location = simulator.vehicles[0].currentlocation
    event = Event(
        timestamp=simulator.current_time + 0.5,
        type=EventType.VEHICLEBREAKDOWN,
        location=breakdown_location,
        severity=4,
        affected_orders=[
            o.id
            for o in simulator.orders[:3]  # first few orders assigned to this vehicle
        ],
        description=f"Vehicle {breakdown_vehicle_id} suffered engine failure",
    )
    simulator.inject_event(event)
    print(f"    Event: type={event.type.value}, location={event.location}, severity={event.severity}")
    print(f"    Affected orders: {event.affected_orders}")

    # Verify the vehicle was marked unavailable
    for v in simulator.vehicles:
        if v.id == breakdown_vehicle_id:
            print(f"    Vehicle {v.id} available={v.available} (expected False)")

    # ── 4. Initialize LLM gateway ─────────────────────────────────────
    print("\n[4] Initializing LLM gateway...")
    try:
        llm = LLMGateway()
    except ValueError as exc:
        print(f"    ERROR: {exc}")
        print("    Copy config/llm.local.example.json to config/llm.local.json and fill in your settings.")
        sys.exit(1)
    print(f"    Provider: {llm.provider}")

    # ── 5. Anomaly detection ──────────────────────────────────────────
    print("\n[5] Running AnomalyDetector on the breakdown event...")
    detector = AnomalyDetector(llm)

    active_orders = simulator.get_active_orders()
    current_state = {
        "current_time": simulator.current_time,
        "vehicles": simulator.vehicles,
        "active_order_count": len(active_orders),
        "total_order_count": len(simulator.orders),
    }

    try:
        anomaly_result = detector.detect(event, current_state)
        pretty_print("Anomaly Detection Result", anomaly_result)
    except LLMError as exc:
        print(f"\n    LLM call failed: {exc}")
        print("    Check network connectivity and API key, then retry.")
        sys.exit(1)

    # ── 6a. Case retrieval (RAG) ────────────────────────────────
    print("\n[6a] Retrieving similar cases from historical case base...")
    retriever = CaseRetriever()
    retrieved = retriever.retrieve(
        event_type=event.type.value,
        severity=event.severity,
        scenario="small",
        top_k=3,
    )
    print(f"    Retrieved {len(retrieved)} similar case(s):")
    for rc in retrieved:
        print(f"    {rc.case.id}: score={rc.score:.2f}, outcome={rc.case.outcome}, action={rc.case.action}")

    # ── 6b. Rule retrieval (knowledge base) ──────────────────────────
    print("\n[6b] Retrieving relevant business rules...")
    from src.rag import RuleRetriever
    rule_retriever = RuleRetriever()
    retrieved_rules = rule_retriever.retrieve(
        event_type=event.type.value,
        severity=event.severity,
        scenario="small",  # derive from config
        top_k=3,
    )
    print(f"    Retrieved {len(retrieved_rules)} rule(s):")
    for rr in retrieved_rules:
        print(f"    [{rr.rule.category}] {rr.rule.explanation[:50]}...")

    # ── 6. Decision making ───────────────────────────────────────────
    print("\n[6c] Running DecisionMaker based on anomaly result...")
    maker = DecisionMaker(llm)

    # Prepare anomaly dict for decision maker
    anomaly_dict = anomaly_result.to_dict()
    anomaly_dict["affected_orders"] = event.affected_orders

    try:
        decision_result = maker.recommend(
            anomaly_result=anomaly_dict,
            route_plan=route_plan,
            vehicles=simulator.vehicles,
            retrieved_cases=retrieved,
            retrieved_rules=retrieved_rules,
        )
        pretty_print("Decision Recommendation", decision_result)
    except LLMError as exc:
        print(f"\n    LLM call failed: {exc}")
        sys.exit(1)

    # ── 7. Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Scenario: {config.num_vehicles} vehicles, {config.num_orders} orders")
    print(f"  Injected event: {event.type.value} at {event.location}")
    print(f"  Anomaly: is_anomaly={anomaly_result.is_anomaly}, severity={anomaly_result.severity}")
    print(f"  Recommended action: {decision_result.action}")
    print(f"  Action reasoning: {decision_result.reasoning}")
    print(f"  Reroute needed: {decision_result.reroute_needed}")
    print("=" * 60)
    print("\n  Test complete.")


if __name__ == "__main__":
    main()
