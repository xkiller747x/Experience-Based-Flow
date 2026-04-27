"""Streamlit visualization app for the Logistics AI system."""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simulation import SimulationConfig, LogisticsSimulator, Event, EventType
from src.optimizer.vrp_solver import ORToolsSolver
from src.agent import AnomalyDetector, DecisionMaker, LLMGateway, LLMError
from src.rag import CaseRetriever, RuleRetriever

# ── Page config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Logistics AI — 调度异常决策系统",
    layout="wide",
)

# ── Helper ────────────────────────────────────────────────────────────

@st.cache_resource
def get_llm():
    try:
        return LLMGateway()
    except ValueError:
        return None


def run_simulation(num_vehicles: int, num_orders: int, area_size_km: float, seed: int):
    config = SimulationConfig(
        num_vehicles=num_vehicles,
        num_orders=num_orders,
        area_size_km=area_size_km,
        seed=seed,
    )
    simulator = LogisticsSimulator(config)
    simulator.generate()

    solver = ORToolsSolver(time_limit_seconds=5)
    route_plan = solver.solve(
        vehicles=simulator.vehicles,
        orders=simulator.orders,
        road_network=simulator.road_network,
    )
    return simulator, route_plan


# ── Sidebar: 参数配置 ─────────────────────────────────────────────────

st.sidebar.title("仿真参数配置")

num_vehicles = st.sidebar.slider("车辆数", 3, 20, 5)
num_orders = st.sidebar.slider("订单数", 5, 50, 10)
area_size_km = st.sidebar.slider("区域大小 (km)", 10.0, 100.0, 30.0)
seed = st.sidebar.number_input("随机种子", value=42)
scenario = st.sidebar.selectbox("场景", ["small", "medium", "large", "stress"])

generate_clicked = st.sidebar.button("🚀 生成仿真场景")

# ── Main title ───────────────────────────────────────────────────────

st.title("🚛 Logistics AI — 调度异常决策系统")

# ── Section 1: 仿真结果 ───────────────────────────────────────────────

st.header("1. 仿真场景")

if generate_clicked:
    with st.spinner("生成仿真数据中..."):
        simulator, route_plan = run_simulation(num_vehicles, num_orders, area_size_km, seed)

    col1, col2, col3 = st.columns(3)
    col1.metric("车辆数", len(simulator.vehicles))
    col2.metric("订单数", len(simulator.orders))
    col3.metric("总里程", f"{route_plan.total_distance:.1f} km")

    st.subheader("车辆列表")
    vehicle_data = [
        {
            "ID": v.id,
            "当前位置": v.currentlocation,
            "载重容量": v.capacityweight,
            "体积容量": v.capacityvolume,
            "可用": "✅" if v.available else "❌",
        }
        for v in simulator.vehicles
    ]
    st.dataframe(vehicle_data, use_container_width=True)

    st.subheader("订单列表")
    order_data = [
        {
            "ID": o.id,
            "提货点": o.pickup,
            "配送点": o.delivery,
            "重量": o.weight,
            "体积": o.volume,
            "优先级": o.priority,
        }
        for o in simulator.orders
    ]
    st.dataframe(order_data, use_container_width=True)

    st.session_state["simulator"] = simulator
    st.session_state["route_plan"] = route_plan
else:
    st.info("请在侧边栏配置参数并点击「生成仿真场景」")

# ── Section 2: 注入异常 ───────────────────────────────────────────────

st.header("2. 异常注入")

if "simulator" in st.session_state:
    simulator = st.session_state["simulator"]

    col_ev_type, col_sev, col_loc = st.columns(3)
    ev_type = col_ev_type.selectbox(
        "异常类型",
        ["vehicle_breakdown", "order_cancel", "traffic_accident", "road_closed"],
    )
    severity = col_sev.slider("严重程度", 1, 10, 5)
    loc = col_loc.selectbox("位置", [v.currentlocation for v in simulator.vehicles])

    if st.button("⚠️ 注入异常"):
        event = Event(
            timestamp=simulator.current_time + 0.5,
            type=EventType[ev_type.upper().replace("_", "")],
            location=loc,
            severity=severity,
            affected_orders=[o.id for o in simulator.orders[:3]],
            description=f"{ev_type} at {loc}",
        )
        simulator.inject_event(event)
        st.session_state["event"] = event
        st.success(f"已注入: {ev_type} (severity={severity}) @ {loc}")
else:
    st.warning("先生成仿真场景")

# ── Section 3: 异常检测 & 决策 ───────────────────────────────────────

st.header("3. 异常检测 & 决策")

if "event" in st.session_state:
    llm = get_llm()
    if llm is None:
        st.error("请先复制 config/llm.local.example.json 为 config/llm.local.json，并填写模型与 API Key")
    else:
        event = st.session_state["event"]
        simulator = st.session_state["simulator"]

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.subheader("🔍 异常检测")
            detector = AnomalyDetector(llm)
            current_state = {
                "current_time": simulator.current_time,
                "vehicles": simulator.vehicles,
                "active_order_count": len(simulator.get_active_orders()),
                "total_order_count": len(simulator.orders),
            }
            with st.spinner("检测中..."):
                detect_result = detector.detect(event, current_state)

            st.write(f"**是否异常:** {'🔴 是' if detect_result.is_anomaly else '🟢 否'}")
            st.write(f"**严重程度:** {detect_result.severity}")
            st.write(f"**原因:** {detect_result.reason}")

        with col_d2:
            st.subheader("🤖 决策建议")
            if detect_result.is_anomaly:
                maker = DecisionMaker(llm)

                # Case retrieval
                retriever = CaseRetriever()
                retrieved_cases = retriever.retrieve(
                    event_type=event.type.value,
                    severity=event.severity,
                    scenario="small",
                    top_k=5,
                )

                # Rule retrieval
                rule_retriever = RuleRetriever()
                retrieved_rules = rule_retriever.retrieve(
                    event_type=event.type.value,
                    severity=event.severity,
                    scenario="small",
                    top_k=3,
                )

                route_plan = st.session_state["route_plan"]
                anomaly_dict = detect_result.to_dict()
                anomaly_dict["affected_orders"] = event.affected_orders

                with st.spinner("决策中..."):
                    decision_result = maker.recommend(
                        anomaly_result=anomaly_dict,
                        route_plan=route_plan,
                        vehicles=simulator.vehicles,
                        retrieved_cases=retrieved_cases,
                        retrieved_rules=retrieved_rules,
                    )

                st.write(f"**建议动作:** `{decision_result.action}`")
                st.write(f"**需要重规划:** {'是' if decision_result.reroute_needed else '否'}")
                st.write(f"**推理过程:** {decision_result.reasoning}")

                # Show retrieved cases
                with st.expander("📋 检索到的相似 Case"):
                    for rc in retrieved_cases:
                        st.markdown(
                            f"- **{rc.case.id}** | score={rc.score:.1f} | "
                            f"action={rc.case.action} | outcome={rc.case.outcome}"
                        )

                # Show retrieved rules
                with st.expander("📜 命中的业务规则"):
                    for rr in retrieved_rules:
                        st.markdown(
                            f"- **[{rr.rule.category.upper()}]** {rr.rule.explanation} "
                            f"(action={rr.rule.action}, priority={rr.rule.priority})"
                        )
else:
    st.warning("请先注入异常")

# ── Section 4: RAG 独立查询 ──────────────────────────────────────────

st.header("4. RAG 检索工具")

col_q1, col_q2, col_q3, col_q4 = st.columns(4)
q_event_type = col_q1.text_input("事件类型", value="vehicle_breakdown")
q_severity = col_q2.slider("严重程度", 1, 10, 4)
q_scenario = col_q3.selectbox("场景", ["small", "medium", "large", "stress"])
q_topk = col_q4.number_input("返回数量 (top_k)", value=3, step=1)

tab_cases, tab_rules = st.tabs(["📂 Case 检索", "📜 规则检索"])

with tab_cases:
    if st.button("查询 Cases", key="query_cases"):
        retriever = CaseRetriever()
        results = retriever.retrieve(
            event_type=q_event_type,
            severity=q_severity,
            scenario=q_scenario,
            top_k=int(q_topk),
        )
        if results:
            for rc in results:
                with st.container():
                    st.markdown(f"### {rc.case.id}")
                    col_a, col_b = st.columns(2)
                    col_a.write(f"**事件类型:** {rc.case.event_type}")
                    col_a.write(f"**动作:** {rc.case.action}")
                    col_b.write(f"**结果:** {rc.case.outcome}")
                    col_b.write(f"**匹配分:** {rc.score:.2f}")
                    st.markdown(f"**推理:** {rc.case.reasoning}")
                    st.markdown(f"*匹配原因: {rc.match_reason}*")
                    st.divider()
        else:
            st.warning("未找到匹配的 case")

with tab_rules:
    if st.button("查询 Rules", key="query_rules"):
        retriever = RuleRetriever()
        results = retriever.retrieve(
            event_type=q_event_type,
            severity=q_severity,
            scenario=q_scenario,
            top_k=int(q_topk),
        )
        if results:
            for rr in results:
                with st.container():
                    st.markdown(f"### {rr.rule.id} **[{rr.rule.category.upper()}]**")
                    st.markdown(f"**动作:** {rr.rule.action} | **优先级:** {rr.rule.priority}")
                    st.markdown(f"**说明:** {rr.rule.explanation}")
                    st.markdown(f"*匹配原因: {rr.match_reason}*")
                    st.divider()
        else:
            st.warning("未找到匹配的规则")
