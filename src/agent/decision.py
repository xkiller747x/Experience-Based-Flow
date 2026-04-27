"""Decision-making module for logistics anomaly response recommendations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from src.agent.llm import LLMGateway
from src.agent.prompts import ActionType, build_decision_prompt
from src.rag import RetrievedCase, RetrievedRule


@dataclass
class DecisionResult:
    """Structured output from the decision maker."""

    action: Literal[
        "reroute", "ignore", "adjust_capacity", "reassign_order", "delay_tolerant"
    ]
    reasoning: str
    reroute_needed: bool

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reasoning": self.reasoning,
            "reroute_needed": self.reroute_needed,
        }


class DecisionMaker:
    """
    Recommends corrective actions for detected anomalies using an LLM.

    Args:
        llm: An initialized LLMGateway instance.
    """

    VALID_ACTIONS: set[str] = {
        "reroute",
        "ignore",
        "adjust_capacity",
        "reassign_order",
        "delay_tolerant",
    }

    def __init__(self, llm: LLMGateway) -> None:
        self._llm = llm

    def recommend(
        self,
        anomaly_result: dict,
        route_plan,
        vehicles: list,
        retrieved_cases: list[RetrievedCase] | None = None,
        retrieved_rules: list[RetrievedRule] | None = None,
    ) -> DecisionResult:
        """
        Recommend a response action for an anomaly given the current routing plan.

        Args:
            anomaly_result: A dict with keys is_anomaly, severity, reason,
                            and optionally affected_orders.
            route_plan: A RoutePlan (or duck-type) with total_distance,
                        total_travel_time, unassigned_order_ids.
            vehicles: List of Vehicle objects in the current state.

        Returns:
            A DecisionResult with action (str), reasoning (str), reroute_needed (bool).

        Raises:
            LLMError: If the LLM call fails.
            ValueError: If the LLM response is not valid JSON or missing fields.
        """
        orders = anomaly_result.get("affected_orders", [])
        # If orders is a list of str IDs, resolve to actual Order objects
        if orders and isinstance(orders[0], str):
            all_orders = getattr(route_plan, '_orders', [])
            id_to_order = {o.id: o for o in all_orders}
            orders = [id_to_order[oid] for oid in orders if oid in id_to_order]

        system_prompt, user_prompt = build_decision_prompt(
            anomaly_result=anomaly_result,
            route_plan=route_plan,
            vehicles=vehicles,
            orders=orders,
            retrieved_cases=retrieved_cases,
            retrieved_rules=retrieved_rules,
        )

        raw_response = self._llm.generate(user_prompt, system_prompt=system_prompt)

        result = self._parse_response(raw_response)

        return result

    def _parse_response(self, raw: str) -> DecisionResult:
        """Extract and validate JSON from the LLM response."""

        json_str = self._extract_json(raw)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"DecisionMaker received invalid JSON from LLM: {exc}\n"
                f"Raw response: {raw[:500]}"
            ) from exc

        for required_key in ("action", "reasoning", "reroute_needed"):
            if required_key not in data:
                raise ValueError(
                    f"DecisionMaker response missing required key '{required_key}': {data}"
                )

        action = str(data["action"]).lower()
        if action not in self.VALID_ACTIONS:
            raise ValueError(
                f"DecisionMaker returned invalid action '{action}'. "
                f"Must be one of {sorted(self.VALID_ACTIONS)}"
            )

        return DecisionResult(
            action=action,
            reasoning=str(data["reasoning"]),
            reroute_needed=bool(data["reroute_needed"]),
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Pull the first JSON object from LLM output."""

        code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_match:
            return code_match.group(1)

        brace_start = text.find("{")
        if brace_start == -1:
            raise ValueError(f"No JSON object found in LLM response: {text[:200]}")

        depth = 0
        start = brace_start
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        raise ValueError(f"Unclosed braces in LLM response: {text[start:300]}")