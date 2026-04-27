"""Rule-based knowledge retrieval for logistics anomaly response."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class Rule:
    """A business rule for logistics scheduling decisions."""
    id: str
    category: Literal["policy", "priority", "best_practice", "guideline"]
    trigger_event_types: list[str]
    trigger_severity_min: int
    trigger_severity_max: int
    trigger_scenarios: list[str]
    action: str
    priority: int
    explanation: str


@dataclass
class RetrievedRule:
    """A rule returned from the RuleRetriever."""
    rule: Rule
    match_reason: str


class RuleRetriever:
    """
    Retrieves business rules that match a given anomaly context.

    Rules are matched by checking whether the query satisfies all of a rule's
    trigger conditions: event type, severity range, and scenario (if specified).
    """

    def __init__(self, rules_path: str = "D:/Code/logistic-ai/data/rules/rules.jsonl"):
        self.rules: list[Rule] = []
        self._load(rules_path)

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self.rules.append(Rule(**d))

    def retrieve(
        self,
        event_type: str,
        severity: int,
        scenario: str,
        top_k: int = 3,
        categories: list[Literal["policy", "priority", "best_practice", "guideline"]] | None = None,
    ) -> list[RetrievedRule]:
        """
        Retrieve rules matching the given anomaly context.

        Args:
            event_type: The type of the detected anomaly.
            severity: Severity level (1-10).
            scenario: The scenario size (small, medium, large, stress).
            top_k: Maximum number of rules to return.
            categories: Optional filter to restrict rule categories.

        Returns:
            List of RetrievedRule, sorted by priority descending.
        """
        candidates = self.rules

        # Filter by category if specified
        if categories:
            candidates = [r for r in candidates if r.category in categories]

        matched = []
        for rule in candidates:
            reason_parts = []

            # Check event type match
            if rule.trigger_event_types and event_type not in rule.trigger_event_types:
                continue
            if rule.trigger_event_types:
                reason_parts.append(f"事件类型匹配({event_type})")

            # Check severity range
            if not (rule.trigger_severity_min <= severity <= rule.trigger_severity_max):
                continue
            reason_parts.append(f"severity={severity}在范围[{rule.trigger_severity_min},{rule.trigger_severity_max}]内")

            # Check scenario match (empty means universal)
            if rule.trigger_scenarios and scenario not in rule.trigger_scenarios:
                continue
            if rule.trigger_scenarios:
                reason_parts.append(f"场景匹配({scenario})")

            matched.append((rule, "; ".join(reason_parts)))

        # Sort by priority descending
        matched.sort(key=lambda x: x[0].priority, reverse=True)
        return [
            RetrievedRule(rule=r, match_reason=reason)
            for r, reason in matched[:top_k]
        ]
