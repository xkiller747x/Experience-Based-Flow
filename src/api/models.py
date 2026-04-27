"""Pydantic models for API request/response validation."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


# ── Vehicle / Order / RoadNetwork（仿真数据模型）─────────────────────

class VehicleModel(BaseModel):
    id: str
    capacityweight: float
    capacityvolume: float
    currentlocation: str
    speed: float = 1.0
    costper_km: float = 1.0
    available: bool = True


class OrderModel(BaseModel):
    id: str
    weight: float
    volume: float
    pickup: str
    delivery: str
    pickupwindow_start: float = 0.0
    pickupwindow_end: float = 100.0
    deliverywindow_start: float = 0.0
    deliverywindow_end: float = 100.0
    priority: int = 1
    created_at: float = 0.0
    cancelled: bool = False
    assignedvehicle: str | None = None


class RoadNetworkModel(BaseModel):
    depot_id: str
    nodes: list = Field(default_factory=list)
    edges: list = Field(default_factory=list)


class RoutePlanModel(BaseModel):
    total_distance: float
    total_travel_time: float
    unassigned_order_ids: list[str]


# ── Event ────────────────────────────────────────────────────────────

class EventModel(BaseModel):
    type: str
    location: str
    severity: int = Field(ge=1, le=10)
    affected_orders: list[str] = Field(default_factory=list)
    description: str = ""


# ── /simulate ────────────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    num_vehicles: int = 5
    num_orders: int = 10
    area_size_km: float = 30.0
    scenario: str = "small"
    seed: int | None = None


class SimulateResponse(BaseModel):
    scenario: str
    vehicles: list[VehicleModel]
    orders: list[OrderModel]
    depot_id: str
    route_plan: RoutePlanModel


# ── /detect ──────────────────────────────────────────────────────────

class CurrentStateModel(BaseModel):
    current_time: float
    active_order_count: int
    total_order_count: int


class DetectRequest(BaseModel):
    event: EventModel
    current_state: CurrentStateModel


class DetectResponse(BaseModel):
    is_anomaly: bool
    severity: int
    reason: str


# ── /decide ──────────────────────────────────────────────────────────

class DecideRequest(BaseModel):
    anomaly_result: DetectResponse
    route_plan: RoutePlanModel
    vehicles: list[VehicleModel]
    scenario: str = "small"


class RetrievedCaseModel(BaseModel):
    case_id: str
    event_type: str
    severity: int
    action: str
    outcome: str
    reasoning: str
    score: float
    match_reason: str


class RetrievedRuleModel(BaseModel):
    rule_id: str
    category: str
    action: str
    explanation: str
    priority: int
    match_reason: str


class DecideResponse(BaseModel):
    action: str
    reasoning: str
    reroute_needed: bool
    retrieved_cases: list[RetrievedCaseModel]
    retrieved_rules: list[RetrievedRuleModel]


# ── /reroute ─────────────────────────────────────────────────────────

class RerouteRequest(BaseModel):
    vehicles: list[VehicleModel]
    orders: list[OrderModel]
    road_network: RoadNetworkModel
    event: EventModel


class RouteModel(BaseModel):
    vehicle_id: str
    order_sequence: list[str]
    total_distance: float


class RerouteResponse(BaseModel):
    total_distance: float
    total_travel_time: float
    unassigned_order_ids: list[str]
    routes: list[RouteModel]


# ── /rag/cases ───────────────────────────────────────────────────────

class CaseQueryRequest(BaseModel):
    event_type: str
    severity: int
    scenario: str
    top_k: int = 5


# ── /rag/rules ───────────────────────────────────────────────────────

class RuleQueryRequest(BaseModel):
    event_type: str
    severity: int
    scenario: str
    top_k: int = 3
    categories: list[str] | None = None


# ── Health ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
