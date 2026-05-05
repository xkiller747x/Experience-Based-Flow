"""RAG data utilities for the logistics scheduling environment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Case:
    """A historical logistics scheduling case for RAG retrieval."""

    id: str
    event_type: str
    severity: int
    scenario: str
    vehicles_count: int
    orders_count: int
    action: str
    outcome: str
    reasoning: str
    before_distance: float
    after_distance: float
    unassigned_before: int
    unassigned_after: int
    timestamp: float
    current_load_rate: float = 0.0
    urgent_orders: int = 0
    available_backup_vehicles: int = 0
    avg_delay_minutes: float = 0.0
    affected_routes: int = 0
    time_window_pressure: float = 0.0
    customer_priority_mix: float = 0.0
    cost_before: float = 0.0
    cost_after: float = 0.0
    candidate_costs: dict = field(default_factory=dict)
    solved: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "severity": self.severity,
            "scenario": self.scenario,
            "vehicles_count": self.vehicles_count,
            "orders_count": self.orders_count,
            "action": self.action,
            "outcome": self.outcome,
            "reasoning": self.reasoning,
            "before_distance": self.before_distance,
            "after_distance": self.after_distance,
            "unassigned_before": self.unassigned_before,
            "unassigned_after": self.unassigned_after,
            "timestamp": self.timestamp,
            "current_load_rate": self.current_load_rate,
            "urgent_orders": self.urgent_orders,
            "available_backup_vehicles": self.available_backup_vehicles,
            "avg_delay_minutes": self.avg_delay_minutes,
            "affected_routes": self.affected_routes,
            "time_window_pressure": self.time_window_pressure,
            "customer_priority_mix": self.customer_priority_mix,
            "cost_before": self.cost_before,
            "cost_after": self.cost_after,
            "candidate_costs": self.candidate_costs,
            "solved": self.solved,
        }


from .experience_model import ExperienceModel, ExperienceModelV2, LearnedRetrieverV2
__all__ = ["Case", "CaseRetriever", "RetrievedCase", "Rule", "RuleRetriever", "RetrievedRule", "ExperienceModel", "ExperienceModelV2", "LearnedRetrieverV2"]

from .case_generator import Case
from .case_retriever import CaseRetriever, RetrievedCase
from .rule_retriever import Rule, RuleRetriever, RetrievedRule
