"""RAG data utilities for the logistics scheduling environment."""

from __future__ import annotations

from dataclasses import dataclass


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
        }


from .experience_model import ExperienceModel, ExperienceModelV2
__all__ = ["Case", "CaseRetriever", "RetrievedCase", "Rule", "RuleRetriever", "RetrievedRule", "ExperienceModel", "ExperienceModelV2"]

from .case_generator import Case
from .case_retriever import CaseRetriever, RetrievedCase
from .rule_retriever import Rule, RuleRetriever, RetrievedRule