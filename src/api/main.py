"""FastAPI service for the Logistics AI system."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    HealthResponse,
    SimulateRequest, SimulateResponse,
    DetectRequest, DetectResponse,
    DecideRequest, DecideResponse,
    RerouteRequest, RerouteResponse,
    CaseQueryRequest,
    RuleQueryRequest,
    VehicleModel, OrderModel, RoadNetworkModel, RoutePlanModel,
    RetrievedCaseModel, RetrievedRuleModel, RouteModel,
)
from src.simulation import SimulationConfig, LogisticsSimulator, Event, EventType
from src.optimizer.vrp_solver import ORToolsSolver
from src.agent import AnomalyDetector, DecisionMaker, LLMGateway, LLMError
from src.rag import CaseRetriever, RuleRetriever, RetrievedCase, RetrievedRule


app = FastAPI(
    title="Logistics AI API",
    description="LLM-driven logistics anomaly detection and decision system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────

def _make_llm() -> LLMGateway:
    try:
        return LLMGateway()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _event_type_from_str(type_str: str) -> EventType:
    mapping = {
        "vehicle_breakdown": EventType.VEHICLEBREAKDOWN,
        "order_cancel": EventType.ORDER_CANCEL,
        "traffic_accident": EventType.TRAFFIC_ACCIDENT,
        "road_closed": EventType.ROAD_CLOSED,
    }
    if type_str not in mapping:
        raise HTTPException(status_code=400, detail=f"Unknown event type: {type_str}")
    return mapping[type_str]


# ── /health ───────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version="1.0.0")


# ── /simulate ────────────────────────────────────────────────────────

@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    config = SimulationConfig(
        num_vehicles=req.num_vehicles,
        num_orders=req.num_orders,
        area_size_km=req.area_size_km,
        seed=req.seed or 42,
    )
    simulator = LogisticsSimulator(config)
    simulator.generate()

    solver = ORToolsSolver(time_limit_seconds=5)
    route_plan = solver.solve(
        vehicles=simulator.vehicles,
        orders=simulator.orders,
        road_network=simulator.road_network,
    )

    return SimulateResponse(
        scenario=req.scenario,
        vehicles=[VehicleModel(**vars(v)) for v in simulator.vehicles],
        orders=[OrderModel(**vars(o)) for o in simulator.orders],
        depot_id=simulator.road_network.depot_id,
        route_plan=RoutePlanModel(
            total_distance=route_plan.total_distance,
            total_travel_time=route_plan.total_travel_time,
            unassigned_order_ids=list(route_plan.unassigned_order_ids),
        ),
    )


# ── /detect ──────────────────────────────────────────────────────────

@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    llm = _make_llm()
    detector = AnomalyDetector(llm)

    event = Event(
        timestamp=0.0,
        type=_event_type_from_str(req.event.type),
        location=req.event.location,
        severity=req.event.severity,
        affected_orders=req.event.affected_orders,
        description=req.event.description,
    )

    current_state = {
        "current_time": req.current_state.current_time,
        "vehicles": [],  # not needed for detect
        "active_order_count": req.current_state.active_order_count,
        "total_order_count": req.current_state.total_order_count,
    }

    try:
        result = detector.detect(event, current_state)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    return DetectResponse(
        is_anomaly=result.is_anomaly,
        severity=result.severity,
        reason=result.reason,
    )


# ── /decide ──────────────────────────────────────────────────────────

@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest):
    llm = _make_llm()
    maker = DecisionMaker(llm)

    # Case retrieval
    retriever = CaseRetriever()
    retrieved_cases = retriever.retrieve(
        event_type=req.anomaly_result.reason.split()[0] if req.anomaly_result.reason else "vehicle_breakdown",
        severity=req.anomaly_result.severity,
        scenario=req.scenario,
        top_k=5,
    )

    # Rule retrieval
    rule_retriever = RuleRetriever()
    retrieved_rules = rule_retriever.retrieve(
        event_type=req.anomaly_result.reason.split()[0] if req.anomaly_result.reason else "vehicle_breakdown",
        severity=req.anomaly_result.severity,
        scenario=req.scenario,
        top_k=3,
    )

    # Build route_plan duck type
    route_plan = type("RoutePlan", (), {
        "total_distance": req.route_plan.total_distance,
        "total_travel_time": req.route_plan.total_travel_time,
        "unassigned_order_ids": req.route_plan.unassigned_order_ids,
    })()

    anomaly_dict = req.anomaly_result.model_dump()
    anomaly_dict["affected_orders"] = []

    try:
        result = maker.recommend(
            anomaly_result=anomaly_dict,
            route_plan=route_plan,
            vehicles=[],  # simplified
            retrieved_cases=retrieved_cases,
            retrieved_rules=retrieved_rules,
        )
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    return DecideResponse(
        action=result.action,
        reasoning=result.reasoning,
        reroute_needed=result.reroute_needed,
        retrieved_cases=[
            RetrievedCaseModel(
                case_id=rc.case.id,
                event_type=rc.case.event_type,
                severity=rc.case.severity,
                action=rc.case.action,
                outcome=rc.case.outcome,
                reasoning=rc.case.reasoning,
                score=rc.score,
                match_reason=rc.match_reason,
            )
            for rc in retrieved_cases
        ],
        retrieved_rules=[
            RetrievedRuleModel(
                rule_id=rr.rule.id,
                category=rr.rule.category,
                action=rr.rule.action,
                explanation=rr.rule.explanation,
                priority=rr.rule.priority,
                match_reason=rr.match_reason,
            )
            for rr in retrieved_rules
        ],
    )


# ── /reroute ────────────────────────────────────────────────────────

@app.post("/reroute", response_model=RerouteResponse)
def reroute(req: RerouteRequest):
    # Reconstruct simulation objects (simplified)
    from src.simulation.models import Vehicle, Order, RoadNetwork

    vehicles = [Vehicle(**vars(v)) for v in req.vehicles]
    orders = [Order(**vars(o)) for o in req.orders]
    road_network = RoadNetwork(depot_id=req.road_network.depot_id)

    solver = ORToolsSolver(time_limit_seconds=10)
    route_plan = solver.solve(
        vehicles=vehicles,
        orders=orders,
        road_network=road_network,
    )

    return RerouteResponse(
        total_distance=route_plan.total_distance,
        total_travel_time=route_plan.total_travel_time,
        unassigned_order_ids=list(route_plan.unassigned_order_ids),
        routes=[],  # TODO: fill from route_plan if available
    )


# ── /rag/cases ───────────────────────────────────────────────────────

@app.post("/rag/cases")
def rag_cases(req: CaseQueryRequest):
    retriever = CaseRetriever()
    results = retriever.retrieve(
        event_type=req.event_type,
        severity=req.severity,
        scenario=req.scenario,
        top_k=req.top_k,
    )
    return {
        "query": req.model_dump(),
        "results": [
            {
                "case_id": rc.case.id,
                "event_type": rc.case.event_type,
                "severity": rc.case.severity,
                "action": rc.case.action,
                "outcome": rc.case.outcome,
                "reasoning": rc.case.reasoning,
                "score": rc.score,
                "match_reason": rc.match_reason,
            }
            for rc in results
        ],
    }


# ── /rag/rules ───────────────────────────────────────────────────────

@app.post("/rag/rules")
def rag_rules(req: RuleQueryRequest):
    retriever = RuleRetriever()
    results = retriever.retrieve(
        event_type=req.event_type,
        severity=req.severity,
        scenario=req.scenario,
        top_k=req.top_k,
        categories=req.categories,
    )
    return {
        "query": req.model_dump(),
        "results": [
            {
                "rule_id": rr.rule.id,
                "category": rr.rule.category,
                "action": rr.rule.action,
                "explanation": rr.rule.explanation,
                "priority": rr.rule.priority,
                "match_reason": rr.match_reason,
            }
            for rr in results
        ],
    }


# ── Run locally ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
