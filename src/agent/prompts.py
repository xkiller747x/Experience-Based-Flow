"""Prompt templates for the LLM-based logistics agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation.models import Event, Vehicle
    from src.rag.case_retriever import RetrievedCase
    from src.rag.negative_aware_retriever import DualRetrievalResult
    from src.rag.rule_retriever import RetrievedRule


class AnomalyType(Enum):
    """Classification categories for detected anomalies."""

    VEHICLE_BREAKDOWN = "vehicle_breakdown"
    TRAFFIC_ACCIDENT = "traffic_accident"
    ROAD_CLOSED = "road_closed"
    ORDER_CANCEL = "order_cancel"
    SEVERE_CONGESTION = "severe_congestion"
    CAPACITY_ISSUE = "capacity_issue"
    TIME_VIOLATION = "time_violation"
    UNKNOWN = "unknown"


class ActionType(Enum):
    """Recommended action types from the decision maker."""

    REROUTE = "reroute"
    IGNORE = "ignore"
    ADJUST_CAPACITY = "adjust_capacity"
    REASSIGN_ORDER = "reassign_order"
    DELAY_TOLERANT = "delay_tolerant"


@dataclass
class PromptTemplate:
    """Composable prompt with system/user slots and few-shot examples."""

    system_template: str
    user_template: str
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)

    def build(self, **kwargs: object) -> tuple[str, str]:
        """Render the prompt with provided variable values."""

        system = self.system_template
        user = self.user_template

        for key, val in kwargs.items():
            double = f"{{{{{key}}}}}"
            single = f"{{{key}}}"
            for pattern in (double, single):
                system = system.replace(pattern, str(val))
                user = user.replace(pattern, str(val))

        if self.few_shot_examples:
            shots = ""
            for ex in self.few_shot_examples:
                shots += f"\n## Example\n"
                shots += f"Input: {ex['input']}\n"
                shots += f"Output: {ex['output']}\n"
            user = user + shots

        return system, user


ANOMALY_DETECTION_TEMPLATE = PromptTemplate(
    system_template=(
        "You are a logistics operations AI. Your role is to classify events "
        "as normal operational variation or genuine anomalies requiring attention.\n"
        "Be precise, cite specific evidence, and use the severity scale consistently:\n"
        "  1 = Minor irregularity, no immediate action needed\n"
        "  2 = Moderate issue, monitor closely\n"
        "  3 = Significant disruption, action may be required\n"
        "  4 = Major incident, response needed\n"
        "  5 = Critical emergency, immediate intervention required\n"
        "\nAnomaly types you may encounter:\n"
        "  - VEHICLE_BREAKDOWN: A vehicle is no longer operational\n"
        "  - TRAFFIC_ACCIDENT: Traffic incident causing delays on a road segment\n"
        "  - ROAD_CLOSED: A road segment is fully blocked\n"
        "  - ORDER_CANCEL: A customer has cancelled an order\n"
        "  - SEVERE_CONGESTION: Unusually high congestion on a route\n"
        "  - CAPACITY_ISSUE: Vehicle capacity mismatch detected\n"
        "  - TIME_VIOLATION: A delivery is at risk of missing its time window\n"
        "\nOutput format: JSON with keys is_anomaly (bool), severity (int 1-5), "
        "reason (string explaining the classification)."
    ),
    user_template=(
        "Analyze the following event in the context of the current simulation state.\n"
        "\n"
        "## Event\n"
        "  timestamp: {timestamp}\n"
        "  type: {event_type}\n"
        "  location: {location}\n"
        "  severity (raw): {raw_severity}\n"
        "  affected_orders: {affected_orders}\n"
        "  description: {description}\n"
        "\n"
        "## Current Simulation State\n"
        "  current_time: {current_time}\n"
        "  total_vehicles: {total_vehicles}\n"
        "  active_vehicles: {active_vehicles}\n"
        "  total_orders: {total_orders}\n"
        "  active_orders: {active_orders}\n"
        "\n"
        "## Vehicles Status\n"
        "{vehicle_summary}\n"
        "\n"
        "Classify this event. Is it a genuine anomaly that warrants attention? "
        "Consider whether the raw severity accurately reflects operational impact "
        "given the current state (e.g., a severity-3 event affecting a single "
        "vehicle may be more or less impactful depending on available alternatives).\n"
        "\n"
        "Respond with a JSON object only, no extra text:\n"
        '{{"is_anomaly": true/false, "severity": 1-5, "reason": "..."}}'
    ),
    few_shot_examples=[
        {
            "input": (
                "Event: timestamp=14.0, type=TRAFFIC_ACCIDENT, location='A->B', "
                "raw_severity=2, affected_orders=['O1'], description='Minor fender-bender'\n"
                "State: current_time=10.0, 8 vehicles active, 40 orders active\n"
                "Vehicles: mostly on-time, no congestion reported\n"
            ),
            "output": (
                '{"is_anomaly": false, "severity": 1, '
                '"reason": "Minor incident with low raw severity; alternative routes '
                'available; no significant impact on active deliveries."}'
            ),
        },
        {
            "input": (
                "Event: timestamp=15.5, type=VEHICLE_BREAKDOWN, location='V3', "
                "raw_severity=4, affected_orders=['O5', 'O12'], description='Engine failure'\n"
                "State: current_time=15.5, only 2 vehicles available, 15 orders pending\n"
                "Vehicles: V1(dispatch), V2(on route), V3(broken), remaining capacity low\n"
            ),
            "output": (
                '{"is_anomaly": true, "severity": 5, '
                '"reason": "Only vehicle V3 available for additional orders; engine failure '
                'at peak hours leaves critical gap; re-routing alone cannot compensate; '
                'immediate reassignment or capacity adjustment required."}'
            ),
        },
    ],
)


DECISION_MAKING_TEMPLATE = PromptTemplate(
    system_template=(
        "You are a logistics dispatch advisor. Given an anomaly and the current "
        "routing plan, recommend the single best action to take.\n"
        "\n"
        "Actions you may recommend:\n"
        '  - "reroute": Re-optimize routes for affected vehicles (best when alternatives exist)\n'
        '  - "ignore": No action needed; system handles it automatically\n'
        '  - "adjust_capacity": Reassign orders to vehicles with spare capacity\n'
        '  - "reassign_order": Move specific orders to different vehicles\n'
        '  - "delay_tolerant": Accept the delay and let the schedule absorb it\n'
        "\n"
        "Consider: affected order count, vehicle availability, time urgency, "
        "downstream consequences. Prefer the least disruptive effective action.\n"
        "\n"
        "Output format: JSON with keys action (string from the list above), "
        "reasoning (string explaining why), reroute_needed (bool)."
    ),
    user_template=(
        "An anomaly has been detected. Evaluate the routing impact and recommend action.\n"
        "\n"
        "## Anomaly Detection Result\n"
        "  is_anomaly: {is_anomaly}\n"
        "  severity: {severity}\n"
        "  reason: {reason}\n"
        "\n"
        "## Current Route Plan\n"
        "  total_distance: {total_distance} km\n"
        "  total_travel_time: {total_travel_time} hours\n"
        "  unassigned_orders: {unassigned_orders}\n"
        "\n"
        "## Vehicle States\n"
        "{vehicle_details}\n"
        "\n"
        "## Affected Order Details\n"
        "{order_details}\n"
        "\n"
        "What is the best single action to take? Consider the severity, "
        "availability of alternative vehicles, time window constraints, "
        "and downstream impact on other orders.\n"
        "\n"
        "Respond with JSON only, no extra text:\n"
        '{{"action": "reroute"|"ignore"|"adjust_capacity"|"reassign_order"|"delay_tolerant", '
        '"reasoning": "...", "reroute_needed": true/false}}'
    ),
    few_shot_examples=[
        {
            "input": (
                "Anomaly: is_anomaly=True, severity=2, "
                "reason='Minor congestion delay on route to customer C3, 15min behind schedule'\n"
                "RoutePlan: 3 reroutes available, total_distance=120km, 2 unassigned orders\n"
                "Vehicles: V1(on-time), V2(delayed 10min), V3(idle)\n"
                "Orders: O1(urgent, window 16:00), O2(standard)\n"
            ),
            "output": (
                '{"action": "reroute", '
                '"reasoning": "Minor delay with 15min impact; V3 is idle and can take over O1 delivery to meet urgent window; re-routing V2 to avoid congested segment.", '
                '"reroute_needed": true}'
            ),
        },
        {
            "input": (
                "Anomaly: is_anomaly=True, severity=1, "
                "reason='Small package bumped from V1, still within capacity'\n"
                "RoutePlan: all deliveries on schedule, no unassigned orders\n"
                "Vehicles: all within capacity, no bottlenecks\n"
                "Orders: standard priorities, all time windows met\n"
            ),
            "output": (
                '{"action": "ignore", '
                '"reasoning": "Minor load shift within vehicle capacity; all deliveries on schedule; no time windows at risk; re-routing would add cost with no measurable benefit.", '
                '"reroute_needed": false}'
            ),
        },
    ],
)


NEGATIVE_AWARE_DECISION_TEMPLATE = PromptTemplate(
    system_template=(
        "你是一个物流调度顾问。基于历史案例的成功经验和失败教训，"
        "为当前异常推荐最佳行动。\n"
        "\n"
        "可选行动：\n"
        '  - "reroute": 重新优化受影响车辆路线\n'
        '  - "ignore": 无需处理\n'
        '  - "adjust_capacity": 调整车辆容量分配\n'
        '  - "reassign_order": 将订单重新分配给其他车辆\n'
        '  - "delay_tolerant": 接受延迟\n'
        "\n"
        "关键规则：\n"
        "1. 优先采取成功案例中高频出现的行动\n"
        "2. 严格避免采取失败案例中出现过的行动\n"
        "3. 如果成功案例的行动建议不一致，结合当前 severity 和 load_rate 判断\n"
        "4. 只输出 JSON，不要有其他文字\n"
    ),
    user_template=(
        "## 当前异常\n"
        "  event_type: {event_type}\n"
        "  severity: {severity}\n"
        "  scenario: {scenario}\n"
        "  current_load_rate: {current_load_rate}\n"
        "  available_backup_vehicles: {available_backup_vehicles}\n"
        "  time_window_pressure: {time_window_pressure}\n"
        "  avg_delay_minutes: {avg_delay_minutes}\n"
        "  urgent_orders: {urgent_orders}\n"
        "  customer_priority_mix: {customer_priority_mix}\n"
        "\n"
        "## 成功经验（类似情况下成功的案例）\n"
        "{success_cases_block}\n"
        "\n"
        "## 失败教训（类似情况下失败的案例，请避免这些行动）\n"
        "{failure_cases_block}\n"
        "\n"
        "请基于以上正负经验推荐最佳行动。"
        '输出 JSON: {{"action": "...", "reasoning": "...", "reroute_needed": true/false}}'
    ),
)


def build_anomaly_prompt(
    event: "Event",
    current_time: float,
    vehicles: list["Vehicle"],
    active_order_count: int,
    total_order_count: int,
) -> tuple[str, str]:
    """Render the anomaly detection prompt for a specific event and state."""

    active_vehicle_count = sum(1 for v in vehicles if v.available)
    vehicle_summary = "\n".join(
        f"  - {v.id}: location={v.currentlocation}, "
        f"capacity_w={v.capacityweight}, capacity_v={v.capacityvolume}, "
        f"available={v.available}"
        for v in vehicles
    )

    system, user = ANOMALY_DETECTION_TEMPLATE.build(
        timestamp=event.timestamp,
        event_type=event.type.value,
        location=event.location,
        raw_severity=event.severity,
        affected_orders=", ".join(event.affected_orders) or "(none)",
        description=event.description or "(none)",
        current_time=current_time,
        total_vehicles=len(vehicles),
        active_vehicles=active_vehicle_count,
        total_orders=total_order_count,
        active_orders=active_order_count,
        vehicle_summary=vehicle_summary or "(no vehicles)",
    )
    return system, user


def build_decision_prompt(
    anomaly_result: dict,
    route_plan,
    vehicles: list["Vehicle"],
    orders: list,
    retrieved_cases: list["RetrievedCase"] | None = None,
    retrieved_rules: list["RetrievedRule"] | None = None,
    dual_retrieval_result: "DualRetrievalResult" | None = None,
) -> tuple[str, str]:
    """Render the decision-making prompt for an anomaly and routing plan."""

    order_details = ""
    if anomaly_result.get("affected_orders"):
        for oid in anomaly_result["affected_orders"]:
            for order in orders:
                if order.id == oid:
                    order_details += (
                        f"  - Order {order.id}: pickup={order.pickup}, "
                        f"delivery={order.delivery}, "
                        f"pickup_window=[{order.pickupwindow.starthour:.1f}, "
                        f"{order.pickupwindow.endhour:.1f}], "
                        f"delivery_window=[{order.deliverywindow.starthour:.1f}, "
                        f"{order.deliverywindow.endhour:.1f}], "
                        f"priority={order.priority}, weight={order.weight}, volume={order.volume}\n"
                    )
                    break

    vehicle_details = "\n".join(
        f"  - {v.id}: location={v.currentlocation}, speed={v.speed}, "
        f"capacity_w={v.capacityweight:.0f}, capacity_v={v.capacityvolume:.0f}, "
        f"available={v.available}, cost_per_km={v.costper_km:.2f}"
        for v in vehicles
    )

    context = anomaly_result.get("context", {}) or {}
    if dual_retrieval_result is not None:
        system, user = NEGATIVE_AWARE_DECISION_TEMPLATE.build(
            event_type=anomaly_result.get("event_type", "unknown"),
            severity=anomaly_result.get("severity", 0),
            scenario=anomaly_result.get("scenario", "unknown"),
            current_load_rate=_get_context_value(
                anomaly_result, context, "current_load_rate"
            ),
            available_backup_vehicles=_get_context_value(
                anomaly_result, context, "available_backup_vehicles"
            ),
            time_window_pressure=_get_context_value(
                anomaly_result, context, "time_window_pressure"
            ),
            avg_delay_minutes=_get_context_value(
                anomaly_result, context, "avg_delay_minutes"
            ),
            urgent_orders=_get_context_value(anomaly_result, context, "urgent_orders"),
            customer_priority_mix=_get_context_value(
                anomaly_result, context, "customer_priority_mix"
            ),
            success_cases_block=_format_retrieved_case_block(
                dual_retrieval_result.success_cases
            ),
            failure_cases_block=_format_retrieved_case_block(
                dual_retrieval_result.failure_cases
            ),
        )
        return system, user

    unassigned = getattr(route_plan, "unassigned_order_ids", [])
    system, user = DECISION_MAKING_TEMPLATE.build(
        is_anomaly=anomaly_result.get("is_anomaly", False),
        severity=anomaly_result.get("severity", 0),
        reason=anomaly_result.get("reason", "N/A"),
        total_distance=getattr(route_plan, "total_distance", 0.0),
        total_travel_time=getattr(route_plan, "total_travel_time", 0.0),
        unassigned_orders=", ".join(unassigned) or "(none)",
        vehicle_details=vehicle_details or "(no vehicles)",
        order_details=order_details or "(no affected orders)",
    )

    if retrieved_cases:
        cases_lines = []
        for i, rc in enumerate(retrieved_cases):
            cases_lines.append(
                f"  Case {i+1}: event={rc.case.event_type}, "
                f"severity={rc.case.severity}, outcome={rc.case.outcome}, "
                f"action={rc.case.action}, reasoning={rc.case.reasoning}"
            )
        cases_block = "\n".join(cases_lines)
        user = user + f"\n\n## Retrieved Similar Cases\n{cases_block}"

    # Inject relevant business rules
    if retrieved_rules:
        rules_section = "\n\n## Relevant Business Rules:\n"
        for rr in retrieved_rules:
            r = rr.rule
            rules_section += f"- [{r.category.upper()}] {r.explanation} (recommended action: {r.action}, priority={r.priority})\n"
        user += rules_section

    return system, user


def _get_context_value(anomaly_result: dict, context: dict, key: str) -> object:
    return anomaly_result.get(key, context.get(key, 0))


def _format_retrieved_case_block(cases: list["RetrievedCase"]) -> str:
    if not cases:
        return "  (none)"

    lines = []
    for index, item in enumerate(cases, start=1):
        case = item.case
        lines.append(
            f"  Case {index}: event={case.event_type}, severity={case.severity}, "
            f"scenario={case.scenario}, outcome={case.outcome}, action={case.action}, "
            f"score={item.score:.4f}, load_rate={case.current_load_rate}, "
            f"backup={case.available_backup_vehicles}, pressure={case.time_window_pressure}, "
            f"delay={case.avg_delay_minutes}, urgent={case.urgent_orders}, "
            f"priority_mix={case.customer_priority_mix}, reasoning={case.reasoning}"
        )
    return "\n".join(lines)
