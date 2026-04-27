"""Anomaly detection powered by an LLM gateway."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.agent.llm import LLMGateway
from src.agent.prompts import build_anomaly_prompt


@dataclass
class AnomalyResult:
    """Structured output from anomaly detection."""

    is_anomaly: bool
    severity: int
    reason: str

    def to_dict(self) -> dict:
        return {
            "is_anomaly": self.is_anomaly,
            "severity": self.severity,
            "reason": self.reason,
        }


class AnomalyDetector:
    """
    Classifies simulation events as normal or anomalous using an LLM.

    Args:
        llm: An initialized LLMGateway instance.
    """

    def __init__(self, llm: LLMGateway) -> None:
        self._llm = llm

    def detect(
        self,
        event,
        current_state: dict,
    ) -> AnomalyResult:
        """
        Classify a simulation event as anomalous or normal.

        Args:
            event: A simulation Event object (must have: timestamp, type,
                   location, severity, affected_orders, description).
            current_state: Dict with keys current_time, vehicles, active_order_count,
                          total_order_count.

        Returns:
            An AnomalyResult with is_anomaly (bool), severity (int 1-5), and reason (str).

        Raises:
            LLMError: If the LLM call fails.
            ValueError: If the LLM response is not valid JSON or missing fields.
        """
        vehicles = current_state.get("vehicles", [])

        system_prompt, user_prompt = build_anomaly_prompt(
            event=event,
            current_time=current_state.get("current_time", 0.0),
            vehicles=vehicles,
            active_order_count=current_state.get("active_order_count", 0),
            total_order_count=current_state.get("total_order_count", 0),
        )

        raw_response = self._llm.generate(user_prompt, system_prompt=system_prompt)

        result = self._parse_response(raw_response)

        # Clamp severity to [1, 5]
        result.severity = max(1, min(5, result.severity))

        return result

    def _parse_response(self, raw: str) -> AnomalyResult:
        """Extract JSON from the LLM response and validate required fields."""

        # Try to find JSON block
        json_str = self._extract_json(raw)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"AnomalyDetector received invalid JSON from LLM: {exc}\n"
                f"Raw response: {raw[:500]}"
            ) from exc

        for required_key in ("is_anomaly", "severity", "reason"):
            if required_key not in data:
                raise ValueError(
                    f"AnomalyDetector response missing required key '{required_key}': {data}"
                )

        return AnomalyResult(
            is_anomaly=bool(data["is_anomaly"]),
            severity=int(data["severity"]),
            reason=str(data["reason"]),
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Pull the first JSON object or array from LLM output."""

        # Try to find a code block first
        code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # Fall back to first { ... } block
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

        raise ValueError(f"Unclosed braces in LLM response: {text[start:]}")