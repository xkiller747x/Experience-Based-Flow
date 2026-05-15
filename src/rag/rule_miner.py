"""SE-RAG Phase 1: Decision Rule Miner.

Mines IF-THEN decision rules from historical cases by grouping on
discretised context features and computing dominant action + confidence.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Re-use Case from the package
from src.rag import Case


# ---------------------------------------------------------------------------
# Discretisation helpers
# ---------------------------------------------------------------------------

def _bucket_severity(value: int) -> str:
    if value <= 3:
        return "low"
    if value <= 6:
        return "mid"
    return "high"


def _bucket_load_rate(value: float) -> str:
    if value <= 0.4:
        return "low"
    if value <= 0.7:
        return "mid"
    return "high"


def _bucket_backup(value: int) -> str:
    if value == 0:
        return "none"
    if value <= 2:
        return "few"
    return "enough"


def _bucket_pressure(value: float) -> str:
    if value <= 0.4:
        return "low"
    return "high"


def _bucket_delay(value: float) -> str:
    if value <= 20:
        return "low"
    if value <= 60:
        return "mid"
    return "high"


def _bucket_priority_mix(value: float) -> str:
    if value <= 0.15:
        return "low"
    if value <= 0.35:
        return "mid"
    return "high"


def _bucket_urgent(value: int) -> str:
    if value == 0:
        return "none"
    if value <= 3:
        return "few"
    return "many"


# Feature → discretise function mapping
FEATURE_DISCRETISERS: dict[str, Any] = {
    "severity": _bucket_severity,
    "current_load_rate": _bucket_load_rate,
    "available_backup_vehicles": _bucket_backup,
    "time_window_pressure": _bucket_pressure,
    "avg_delay_minutes": _bucket_delay,
    "urgent_orders": _bucket_urgent,
    "customer_priority_mix": _bucket_priority_mix,
}

# Features used for rule conditions (ordered by expected discriminative power)
RULE_FEATURES = [
    "severity",
    "current_load_rate",
    "available_backup_vehicles",
    "time_window_pressure",
    "customer_priority_mix",
    "avg_delay_minutes",
    "urgent_orders",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DecisionRule:
    """A mined decision rule: IF conditions THEN action (confidence)."""

    event_type: str
    conditions: dict[str, str]          # feature → bucket
    action: str
    confidence: float                   # dominant_count / total_in_bucket
    support_count: int
    total_count: int
    outcome_dist: dict[str, int]        # action → count (within matching cases)
    action_weight: float = 1.0          # rarity weight for the dominant action
    example_cases: list[dict] = field(default_factory=list)  # up to 3 example case dicts

    @property
    def condition_key(self) -> tuple[str, tuple[tuple[str, str], ...]]:
        """Hashable key for deduplication."""
        return (self.event_type, tuple(sorted(self.conditions.items())))

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "conditions": dict(self.conditions),
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "support_count": self.support_count,
            "total_count": self.total_count,
            "outcome_dist": dict(self.outcome_dist),
            "action_weight": round(self.action_weight, 4),
            "example_cases": self.example_cases,
        }


@dataclass
class MinedRuleBase:
    """The complete output of decision rule mining."""

    rules: list[DecisionRule] = field(default_factory=list)
    event_type_index: dict[str, list[int]] = field(default_factory=dict)
    action_index: dict[str, list[int]] = field(default_factory=dict)
    action_frequencies: dict[str, float] = field(default_factory=dict)

    # ---- persistence ----

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "rules": [r.to_dict() for r in self.rules],
            "meta": {
                "total_rules": len(self.rules),
                "event_types": sorted(self.event_type_index.keys()),
                "actions": sorted(self.action_index.keys()),
                "action_frequencies": dict(self.action_frequencies),
            },
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> MinedRuleBase:
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        rb = cls()
        rb.action_frequencies = dict(data.get("meta", {}).get("action_frequencies", {}))
        for rd in data["rules"]:
            rule = DecisionRule(
                event_type=rd["event_type"],
                conditions=rd["conditions"],
                action=rd["action"],
                confidence=rd["confidence"],
                support_count=rd["support_count"],
                total_count=rd["total_count"],
                outcome_dist=rd["outcome_dist"],
                action_weight=rd.get("action_weight", 1.0),
                example_cases=rd.get("example_cases", []),
            )
            idx = len(rb.rules)
            rb.rules.append(rule)
            rb.event_type_index.setdefault(rule.event_type, []).append(idx)
            rb.action_index.setdefault(rule.action, []).append(idx)
        return rb

    # ---- lookup helpers ----

    def rules_for_event(self, event_type: str) -> list[DecisionRule]:
        return [self.rules[i] for i in self.event_type_index.get(event_type, [])]

    def rules_for_action(self, action: str) -> list[DecisionRule]:
        return [self.rules[i] for i in self.action_index.get(action, [])]


# ---------------------------------------------------------------------------
# Miner
# ---------------------------------------------------------------------------

class DecisionRuleMiner:
    """Mine decision rules from a case store.

    Strategy: for each ``event_type``, iterate over increasing subsets of
    discretised features.  For each feature combination, group cases and
    emit a rule if the dominant action exceeds the confidence threshold.
    """

    def __init__(
        self,
        cases_path: str | Path,
        *,
        min_confidence: float = 0.70,
        min_support: int = 10,
        max_example_cases: int = 3,
    ):
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.max_example_cases = max_example_cases
        self.cases: list[Case] = []
        self._action_freq: dict[str, float] = {}
        self._load(cases_path)

    def _load(self, path: str | Path) -> None:
        action_counts: Counter[str] = Counter()
        with Path(path).open(encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                data.pop('original_action', None)
                data.pop('solver_relabeled', None)
                case = Case(**data)
                self.cases.append(case)
                action_counts[case.action] += 1

        total = sum(action_counts.values())
        if total:
            self._action_freq = {
                action: count / total for action, count in action_counts.items()
            }

    # ---- public API ----

    def mine(self) -> MinedRuleBase:
        """Run the mining pipeline and return a ``MinedRuleBase``."""
        rb = MinedRuleBase(action_frequencies=dict(self._action_freq))

        # Group cases by event_type
        by_event: dict[str, list[Case]] = defaultdict(list)
        for case in self.cases:
            by_event[case.event_type].append(case)

        for event_type, event_cases in sorted(by_event.items()):
            event_rules = self._mine_event(event_type, event_cases)
            for rule in event_rules:
                idx = len(rb.rules)
                rb.rules.append(rule)
                rb.event_type_index.setdefault(rule.event_type, []).append(idx)
                rb.action_index.setdefault(rule.action, []).append(idx)

        return rb

    # ---- internal ----

    def _mine_event(self, event_type: str, cases: list[Case]) -> list[DecisionRule]:
        """Mine rules for one event type across feature subsets."""
        seen_keys: set[tuple[str, tuple]] = set()
        rules: list[DecisionRule] = []

        # Try subsets of size 1..len(RULE_FEATURES)
        from itertools import combinations

        for combo_size in range(1, len(RULE_FEATURES) + 1):
            for features in combinations(RULE_FEATURES, combo_size):
                groups: dict[tuple[tuple[str, str], ...], list[Case]] = defaultdict(list)
                for case in cases:
                    bucket = tuple(
                        (feat, FEATURE_DISCRETISERS[feat](getattr(case, feat, 0)))
                        for feat in features
                    )
                    groups[bucket].append(case)

                for bucket, group_cases in groups.items():
                    # Dedup: skip if we already have a more specific rule for same conditions
                    key = (event_type, bucket)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    # Check if subsumed by an existing (simpler) rule with same action and higher/equal confidence
                    action_counts = Counter(c.action for c in group_cases)
                    dominant_action, dominant_count = action_counts.most_common(1)[0]
                    min_support = self._min_support_for_action(dominant_action)
                    if dominant_count < min_support:
                        continue

                    confidence = dominant_count / len(group_cases)

                    if confidence < self.min_confidence:
                        continue

                    # Pick example cases (prefer diverse severities)
                    examples = self._pick_examples(group_cases, dominant_action)

                    rule = DecisionRule(
                        event_type=event_type,
                        conditions=dict(bucket),
                        action=dominant_action,
                        confidence=confidence,
                        support_count=dominant_count,
                        total_count=len(group_cases),
                        outcome_dist=dict(action_counts),
                        action_weight=self._action_weight(dominant_action),
                        example_cases=[c.to_dict() for c in examples],
                    )
                    rules.append(rule)

        # Deduplicate: keep only the most specific rule per (event_type, conditions)
        # Already handled by seen_keys above; sort by specificity descending
        rules.sort(key=lambda r: (-len(r.conditions), -r.confidence))
        return rules

    def _hard_action_threshold(self) -> float:
        """Frequency below which an action is treated as a hard/non-dominant action."""
        if not self._action_freq:
            return 0.0
        return max(self._action_freq.values())

    def _min_support_for_action(self, action: str) -> int:
        threshold = self._hard_action_threshold()
        action_freq = self._action_freq.get(action, 1.0)
        if threshold and action_freq < threshold:
            hard_support_cap = max(3, self.min_support // 2)
            scaled_support = round(hard_support_cap * action_freq / threshold)
            return max(3, min(hard_support_cap, scaled_support))
        return self.min_support

    def _action_weight(self, action: str) -> float:
        threshold = self._hard_action_threshold()
        action_freq = self._action_freq.get(action, 1.0)
        if not threshold or action_freq >= threshold:
            return 1.0
        return threshold / max(action_freq, 1e-9)

    def _pick_examples(self, cases: list[Case], dominant_action: str) -> list[Case]:
        """Pick up to ``max_example_cases`` diverse examples."""
        matching = [c for c in cases if c.action == dominant_action]
        if not matching:
            return cases[: self.max_example_cases]

        # Try to pick different severity levels
        by_sev: dict[int, list[Case]] = defaultdict(list)
        for c in matching:
            by_sev[c.severity].append(c)

        picked: list[Case] = []
        for sev_cases in by_sev.values():
            if len(picked) >= self.max_example_cases:
                break
            picked.append(sev_cases[0])

        # Fill remaining slots
        for c in matching:
            if len(picked) >= self.max_example_cases:
                break
            if c not in picked:
                picked.append(c)

        return picked


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cases_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\Code\logistic-ai\data\cases\cases_30k.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else r"D:\Code\logistic-ai\output\se_rag\rule_base.json"

    print(f"Loading cases from {cases_path}")
    miner = DecisionRuleMiner(cases_path)
    print(f"Loaded {len(miner.cases)} cases")

    print("Mining decision rules...")
    rb = miner.mine()
    print(f"Mined {len(rb.rules)} rules")

    # Stats
    event_types = set()
    actions = set()
    for rule in rb.rules:
        event_types.add(rule.event_type)
        actions.add(rule.action)
    print(f"Covering {len(event_types)} event types, {len(actions)} actions")
    print(f"Average confidence: {sum(r.confidence for r in rb.rules) / len(rb.rules):.4f}")

    rb.save(output_path)
    print(f"Saved rule base to {output_path}")
