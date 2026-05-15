"""SE-RAG Phase 3 & 4: Rule Matcher + Evidence Assembler.

Online retrieval: given a query, match rules from the rule base,
assemble structured evidence (triggered rules + causal chains +
contrast rules + support cases), and build the LLM prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.rag import Case
from src.rag.causal_extractor import CausalGraph
from src.rag.rule_miner import (
    FEATURE_DISCRETISERS,
    MinedRuleBase,
    DecisionRule,
)


# ---------------------------------------------------------------------------
# Evidence structures
# ---------------------------------------------------------------------------

@dataclass
class TriggeredRule:
    """A rule matched by the query with match details."""
    rule: DecisionRule
    match_type: str           # "exact" | "partial"
    match_diff: list[str]     # features that differ (for partial matches)
    rule_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "conditions": self.rule.conditions,
            "action": self.rule.action,
            "confidence": round(self.rule.confidence, 4),
            "support_count": self.rule.support_count,
            "match_type": self.match_type,
            "match_diff": self.match_diff,
            "rule_summary": self.rule_summary or self._auto_summary(),
        }

    def _auto_summary(self) -> str:
        conds = " + ".join(f"{k}={v}" for k, v in sorted(self.rule.conditions.items()))
        return f"{self.rule.event_type} + {conds} → {self.rule.action} (置信度{self.rule.confidence:.0%}, {self.rule.support_count}条案例支持)"


@dataclass
class ContrastRule:
    """A rule with similar conditions but different action (for 'why not X')."""
    rule: DecisionRule
    differing_feature: str
    differing_value: str

    def to_dict(self) -> dict:
        conds = " + ".join(f"{k}={v}" for k, v in sorted(self.rule.conditions.items()))
        return {
            "conditions": conds,
            "action": self.rule.action,
            "reason": f"当 {self.differing_feature}={self.differing_value} 时应选择 {self.rule.action}",
            "confidence": round(self.rule.confidence, 4),
        }


@dataclass
class CausalExplanation:
    """A causal chain explaining why a feature matters."""
    feature: str
    value: Any
    impact_on_action: str
    mi_strength: float

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "value": self.value,
            "impact": self.impact_on_action,
            "strength": round(self.mi_strength, 4),
        }


@dataclass
class StructuredEvidence:
    """Complete structured evidence for a single query."""
    query_event_type: str
    query_severity: int
    query_scenario: str
    query_context: dict

    triggered_rules: list[TriggeredRule] = field(default_factory=list)
    contrast_rules: list[ContrastRule] = field(default_factory=list)
    causal_explanations: list[CausalExplanation] = field(default_factory=list)
    support_cases: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": {
                "event_type": self.query_event_type,
                "severity": self.query_severity,
                "scenario": self.query_scenario,
                "context": self.query_context,
            },
            "triggered_rules": [r.to_dict() for r in self.triggered_rules],
            "contrast_rules": [c.to_dict() for c in self.contrast_rules],
            "causal_explanations": [e.to_dict() for e in self.causal_explanations],
            "support_cases": self.support_cases,
        }


# ---------------------------------------------------------------------------
# Rule Matcher
# ---------------------------------------------------------------------------

class RuleMatcher:
    """Match incoming queries against a mined rule base."""

    def __init__(
        self,
        rule_base: MinedRuleBase,
        causal_graph: CausalGraph | None = None,
        action_frequencies: dict[str, float] | None = None,
    ):
        self.rule_base = rule_base
        self.causal_graph = causal_graph
        self._action_freq_map = (
            dict(action_frequencies)
            if action_frequencies is not None
            else dict(getattr(rule_base, "action_frequencies", {}))
        )
        self._hard_action_threshold = (
            max(self._action_freq_map.values()) if self._action_freq_map else 0.0
        )

    def match(
        self,
        event_type: str,
        severity: int,
        scenario: str,
        context: dict[str, Any],
        *,
        max_triggered: int = 5,
        max_contrast: int = 3,
        max_support_cases: int = 3,
    ) -> StructuredEvidence:
        """Match a query and assemble structured evidence."""
        # Discretise query features
        query_buckets: dict[str, str] = {}
        for feat, func in FEATURE_DISCRETISERS.items():
            if feat == "severity":
                query_buckets[feat] = func(severity)
            else:
                query_buckets[feat] = func(context.get(feat, 0))

        # Get candidate rules for this event type
        candidates = self.rule_base.rules_for_event(event_type)

        # 1. Exact match
        triggered: list[TriggeredRule] = []
        for rule in candidates:
            if self._is_exact_match(rule, query_buckets):
                triggered.append(TriggeredRule(
                    rule=rule,
                    match_type="exact",
                    match_diff=[],
                ))

        # 2. Partial match (allow 2 feature diffs)
        if len(triggered) < max_triggered:
            for rule in candidates:
                diff = self._partial_match(rule, query_buckets, max_diff=2)
                if diff is not None:
                    # Skip if already triggered as exact
                    if any(t.rule is rule for t in triggered):
                        continue
                    triggered.append(TriggeredRule(
                        rule=rule,
                        match_type="partial",
                        match_diff=diff,
                    ))

        # Sort: exact first, then by confidence plus hard-action bonus,
        # then specificity descending.
        def _sort_key(t: TriggeredRule) -> tuple:
            hard_bonus = 0.0
            if self._hard_action_threshold:
                action_freq = self._action_freq_map.get(t.rule.action, 1.0)
                if action_freq < self._hard_action_threshold:
                    hard_bonus = 0.25 * (
                        1.0 - action_freq / self._hard_action_threshold
                    )
            return (
                0 if t.match_type == "exact" else 1,
                -(t.rule.confidence + hard_bonus),
                -len(t.rule.conditions),
            )

        triggered.sort(key=_sort_key)
        triggered = triggered[:max_triggered]

        # 3. Contrast rules: same event_type, similar conditions, different action
        triggered_actions = {t.rule.action for t in triggered}
        contrast: list[ContrastRule] = []
        if triggered:
            for rule in candidates:
                if rule.action in triggered_actions:
                    continue
                diff = self._partial_match(rule, query_buckets, max_diff=2)
                if diff is not None:
                    diff_feat = diff[0]
                    contrast.append(ContrastRule(
                        rule=rule,
                        differing_feature=diff_feat,
                        differing_value=rule.conditions.get(diff_feat, "?"),
                    ))
                    if len(contrast) >= max_contrast:
                        break

        # 4. Causal explanations
        causal_exps: list[CausalExplanation] = []
        if self.causal_graph:
            for edge in self.causal_graph.feature_to_action:
                if edge.strength > 0.01:
                    feat_val = query_buckets.get(edge.source, "?")
                    raw_val = severity if edge.source == "severity" else context.get(edge.source, "?")
                    causal_exps.append(CausalExplanation(
                        feature=edge.source,
                        value=raw_val,
                        impact_on_action=edge.direction,
                        mi_strength=edge.strength,
                    ))
            causal_exps.sort(key=lambda e: -e.mi_strength)
            causal_exps = causal_exps[:6]

        # 5. Collect support cases from top triggered rule
        support: list[dict] = []
        if triggered:
            for case_dict in triggered[0].rule.example_cases[:max_support_cases]:
                support.append({
                    "event_type": case_dict.get("event_type"),
                    "severity": case_dict.get("severity"),
                    "action": case_dict.get("action"),
                    "outcome": case_dict.get("outcome"),
                    "reasoning": case_dict.get("reasoning", ""),
                    "current_load_rate": case_dict.get("current_load_rate"),
                    "time_window_pressure": case_dict.get("time_window_pressure"),
                })

        return StructuredEvidence(
            query_event_type=event_type,
            query_severity=severity,
            query_scenario=scenario,
            query_context=context,
            triggered_rules=triggered,
            contrast_rules=contrast,
            causal_explanations=causal_exps,
            support_cases=support,
        )

    # ---- matching helpers ----

    @staticmethod
    def _is_exact_match(rule: DecisionRule, query_buckets: dict[str, str]) -> bool:
        """Check if all rule conditions match the query's discretised features."""
        for feat, bucket in rule.conditions.items():
            if query_buckets.get(feat) != bucket:
                return False
        return True

    @staticmethod
    def _partial_match(
        rule: DecisionRule, query_buckets: dict[str, str], max_diff: int = 1
    ) -> list[str] | None:
        """Return differing features if <= max_diff, else None."""
        diffs = []
        for feat, bucket in rule.conditions.items():
            if query_buckets.get(feat) != bucket:
                diffs.append(feat)
                if len(diffs) > max_diff:
                    return None
        return diffs


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

SE_RAG_SYSTEM_PROMPT = (
    "你是一个物流调度决策专家。基于结构化经验库的检索结果，为当前异常做决策。\n\n"
    "你会收到：\n"
    "1. 触发的决策规则（从历史案例中提炼）\n"
    "2. 因果推理链（解释关键因素如何影响决策）\n"
    "3. 对比规则（解释为什么不应选择其他行动）\n"
    "4. 支撑案例（具体的历史案例）\n\n"
    "请基于以上结构化证据做出决策，并提供因果推理。\n"
    "可选行动：reroute, ignore, adjust_capacity, reassign_order, delay_tolerant\n"
    '输出 JSON: {"action": "...", "reasoning": "...", "reroute_needed": true/false}'
)


def build_se_rag_prompt(evidence: StructuredEvidence) -> tuple[str, str]:
    """Build the system + user prompt from structured evidence."""
    ctx = evidence.query_context

    parts = []

    # Current anomaly
    parts.append("## 当前异常")
    parts.append(f"  event_type: {evidence.query_event_type}")
    parts.append(f"  severity: {evidence.query_severity}")
    parts.append(f"  scenario: {evidence.query_scenario}")
    parts.append(f"  current_load_rate: {ctx.get('current_load_rate', 0)}")
    parts.append(f"  available_backup_vehicles: {ctx.get('available_backup_vehicles', 0)}")
    parts.append(f"  time_window_pressure: {ctx.get('time_window_pressure', 0)}")
    parts.append(f"  avg_delay_minutes: {ctx.get('avg_delay_minutes', 0)}")
    parts.append(f"  urgent_orders: {ctx.get('urgent_orders', 0)}")
    parts.append(f"  customer_priority_mix: {ctx.get('customer_priority_mix', 0)}")

    # Triggered rules
    if evidence.triggered_rules:
        parts.append("\n## 触发的决策规则（从历史案例提炼）")
        for i, tr in enumerate(evidence.triggered_rules, 1):
            match_label = "精确匹配" if tr.match_type == "exact" else f"近似匹配(差异: {', '.join(tr.match_diff)})"
            parts.append(
                f"  Rule #{i} [{match_label}]: {tr.to_dict()['rule_summary']}"
            )

    # Causal explanations
    if evidence.causal_explanations:
        parts.append("\n## 因果推理链")
        for exp in evidence.causal_explanations:
            if exp.impact_on_action:
                parts.append(f"  - {exp.feature}={exp.value}: {exp.impact_on_action} (因果强度={exp.mi_strength:.3f})")
            else:
                parts.append(f"  - {exp.feature}={exp.value} (因果强度={exp.mi_strength:.3f})")

    # Contrast rules
    if evidence.contrast_rules:
        parts.append("\n## 对比（为什么不是其他行动）")
        for cr in evidence.contrast_rules:
            d = cr.to_dict()
            parts.append(f"  - 不选 {d['action']}: {d['reason']} (置信度{d['confidence']:.0%})")

    # Support cases
    if evidence.support_cases:
        parts.append("\n## 支撑案例")
        for i, sc in enumerate(evidence.support_cases, 1):
            parts.append(
                f"  Case {i}: event={sc.get('event_type')}, severity={sc.get('severity')}, "
                f"action={sc.get('action')}, outcome={sc.get('outcome')}, "
                f"load_rate={sc.get('current_load_rate')}, "
                f"pressure={sc.get('time_window_pressure')}"
            )
            if sc.get("reasoning"):
                parts.append(f"    原因: {sc['reasoning'][:120]}")

    parts.append("\n请基于以上结构化证据做出决策。")

    return SE_RAG_SYSTEM_PROMPT, "\n".join(parts)
