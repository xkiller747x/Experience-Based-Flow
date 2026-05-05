"""
Experience Model — 基于决策树的经验补全模型。

给定部分条件（event_type, severity, scenario），预测：
- action: 建议动作
- outcome: 预期结果
- distance_ratio: 里程开销比例 (after/before)
- unassigned_delta: 未分配订单变化量
"""

from __future__ import annotations

import sys
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import tree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.case_retriever import CaseRetriever


# ── Feature encoding ───────────────────────────────────────────────────

# 所有可能的事件类型（必须和仿真层一致）
ALL_EVENT_TYPES = [
    "vehicle_breakdown",
    "traffic_accident",
    "road_closed",
    "order_cancel",
    "vehicle_maintenance",
    "fuel_shortage",
    "driver_unavailable",
    "traffic_congestion",
    "road_narrowing",
    "bridge_weight_limit",
    "order_modify",
    "order_update",
    "priority_order_urgent",
    "delivery_failure",
    "demand_surge",
    "demand_drop",
    "weather_delay",
    "natural_disaster",
    "public_event",
    "warehouse_delay",
    "inventory_stockout",
]

ALL_SCENARIOS = ["small", "medium", "large", "stress"]
ALL_ACTIONS = ["reroute", "ignore", "adjust_capacity", "reassign_order", "delay_tolerant"]
ALL_OUTCOMES = ["success", "failure"]


class ExperienceModel:
    """
    决策树经验模型。
    学习从 (event_type, severity, scenario, vehicles_count, orders_count)
    预测 (action, outcome, distance_ratio, unassigned_delta)。
    """

    def __init__(self):
        self._event_encoder = LabelEncoder()
        self._scenario_encoder = LabelEncoder()
        self._action_encoder = LabelEncoder()
        self._outcome_encoder = LabelEncoder()

        self._action_tree = None  # 预测 action
        self._outcome_tree = None  # 预测 outcome
        self._distance_tree = None  # 预测 distance_ratio (回归)
        self._unassigned_tree = None  # 预测 unassigned_delta (回归)

        self._distance_mean = 0.0
        self._distance_std = 1.0
        self._unassigned_mean = 0.0
        self._unassigned_std = 1.0

        self._event_encoder.fit(ALL_EVENT_TYPES)
        self._scenario_encoder.fit(ALL_SCENARIOS)
        self._action_encoder.fit(ALL_ACTIONS)
        self._outcome_encoder.fit(ALL_OUTCOMES)

    def _features_from_case(self, case) -> list:
        return [
            self._event_encoder.transform([case.event_type])[0],
            case.severity,
            self._scenario_encoder.transform([case.scenario])[0],
            case.vehicles_count,
            case.orders_count,
        ]

    def train(self, cases: list):
        """用 case 列表训练决策树模型。"""
        if not cases:
            raise ValueError("No cases provided for training")

        X = [self._features_from_case(c) for c in cases]
        y_action = [self._action_encoder.transform([c.action])[0] for c in cases]
        y_outcome = [self._outcome_encoder.transform([c.outcome])[0] for c in cases]

        distance_ratios = []
        for c in cases:
            ratio = c.after_distance / c.before_distance if c.before_distance > 0 else 1.0
            distance_ratios.append(ratio)
        y_distance = np.array(distance_ratios)

        unassigned_deltas = [c.unassigned_after - c.unassigned_before for c in cases]
        y_unassigned = np.array(unassigned_deltas)

        # 标准化回归目标
        self._distance_mean = float(np.mean(y_distance))
        self._distance_std = float(np.std(y_distance)) + 1e-8
        y_distance_norm = (y_distance - self._distance_mean) / self._distance_std

        self._unassigned_mean = float(np.mean(y_unassigned))
        self._unassigned_std = float(np.std(y_unassigned)) + 1e-8
        y_unassigned_norm = (y_unassigned - self._unassigned_mean) / self._unassigned_std

        # 训练决策树
        self._action_tree = tree.DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=3, random_state=42
        )
        self._action_tree.fit(X, y_action)

        self._outcome_tree = tree.DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=3, random_state=42
        )
        self._outcome_tree.fit(X, y_outcome)

        self._distance_tree = tree.DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=3, random_state=42
        )
        self._distance_tree.fit(X, y_distance_norm)

        self._unassigned_tree = tree.DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=3, random_state=42
        )
        self._unassigned_tree.fit(X, y_unassigned_norm)

    def predict(self, event_type: str, severity: int, scenario: str,
                vehicles_count: int | None = None, orders_count: int | None = None) -> dict:
        """
        给定部分条件，预测典型经验。

        Args:
            event_type: 事件类型
            severity: 严重程度 (1-10)
            scenario: 场景大小
            vehicles_count: 车辆数量（可选，默认用该场景均值）
            orders_count: 订单数量（可选，默认用该场景均值）

        Returns:
            dict: {
                "action": str,  # 建议动作
                "action_confidence": float,
                "outcome": str,  # 预期结果
                "outcome_confidence": float,
                "distance_ratio": float,  # 里程比例（after/before）
                "unassigned_delta": int,  # 未分配订单变化
                "action_distribution": dict,  # 动作概率分布
                "outcome_distribution": dict,  # 结果概率分布
            }
        """
        if self._action_tree is None:
            raise RuntimeError("Model not trained yet")

        # 编码输入
        try:
            ev_code = self._event_encoder.transform([event_type])[0]
        except ValueError:
            ev_code = 0  # fallback

        try:
            sc_code = self._scenario_encoder.transform([scenario])[0]
        except ValueError:
            sc_code = 0

        # 用场景均值填充缺失值
        _default_vehicles = {"small": 5, "medium": 10, "large": 20, "stress": 40}
        _default_orders = {"small": 10, "medium": 30, "large": 80, "stress": 150}
        vc = vehicles_count if vehicles_count is not None else _default_vehicles.get(scenario, 5)
        oc = orders_count if orders_count is not None else _default_orders.get(scenario, 10)

        X = [[ev_code, severity, sc_code, vc, oc]]

        # 预测
        action_code = int(self._action_tree.predict(X)[0])
        action_probs = self._action_tree.predict_proba(X)[0]
        action = self._action_encoder.inverse_transform([action_code])[0]
        action_confidence = float(action_probs[action_code])

        outcome_code = int(self._outcome_tree.predict(X)[0])
        outcome_probs = self._outcome_tree.predict_proba(X)[0]
        outcome = self._outcome_encoder.inverse_transform([outcome_code])[0]
        outcome_confidence = float(outcome_probs[outcome_code])

        dist_norm = float(self._distance_tree.predict(X)[0])
        distance_ratio = round(dist_norm * self._distance_std + self._distance_mean, 3)

        unassigned_norm = float(self._unassigned_tree.predict(X)[0])
        unassigned_delta = int(round(unassigned_norm * self._unassigned_std + self._unassigned_mean))

        # 概率分布
        action_dist = {
            self._action_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(action_probs)
        }
        outcome_dist = {
            self._outcome_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(outcome_probs)
        }

        return {
            "action": action,
            "action_confidence": round(action_confidence, 3),
            "outcome": outcome,
            "outcome_confidence": round(outcome_confidence, 3),
            "distance_ratio": distance_ratio,
            "unassigned_delta": unassigned_delta,
            "action_distribution": action_dist,
            "outcome_distribution": outcome_dist,
        }

    def get_rules_text(self, event_type: str | None = None, max_depth: int = 3) -> str:
        """
        以文字形式导出决策树学到的经验规则。
        如果 event_type 指定，只导出与该类型相关的规则。
        """
        if self._action_tree is None:
            raise RuntimeError("Model not trained yet")

        # 导出 action 决策树的文字规则
        from sklearn.tree import export_text
        rules = export_text(self._action_tree, max_depth=max_depth, feature_names=[
            "event_type", "severity", "scenario", "vehicles", "orders"
        ])
        return rules

    def save(self, path: str | Path):
        """保存模型到文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceModel":
        """从文件加载模型。"""
        with open(path, "rb") as f:
            return pickle.load(f)


class ExperienceModelV2:
    """
    经验模型 v2：学习 query-case 的相关性权重。

    不再预测 action/outcome，而是学习：
    "给定一个 query (event_type, severity, scenario)，
    这个候选 case 对该 query 有参考价值吗？"

    训练数据：1000 条 case，两两构成 (query, case) 样本
    正样本：case.action == ground_truth_action(query) 且 outcome == success
    负样本：case.action != ground_truth_action(query) 或 outcome == failure
    """

    def __init__(self):
        self._clf = None
        self._event_encoder = LabelEncoder()
        self._scenario_encoder = LabelEncoder()
        self._action_encoder = LabelEncoder()
        self._pos_count = 0
        self._neg_count = 0

        ALL_EVENT_TYPES = [
            "vehicle_breakdown", "traffic_accident", "road_closed", "order_cancel",
            "vehicle_maintenance", "fuel_shortage", "driver_unavailable", "traffic_congestion",
            "road_narrowing", "bridge_weight_limit", "order_modify", "order_update",
            "priority_order_urgent", "delivery_failure", "demand_surge", "demand_drop",
            "weather_delay", "natural_disaster", "public_event", "warehouse_delay",
            "inventory_stockout",
        ]
        ALL_SCENARIOS = ["small", "medium", "large", "stress"]
        ALL_ACTIONS = ["reroute", "ignore", "adjust_capacity", "reassign_order", "delay_tolerant"]

        self._event_encoder.fit(ALL_EVENT_TYPES)
        self._scenario_encoder.fit(ALL_SCENARIOS)
        self._action_encoder.fit(ALL_ACTIONS)

    def _context_from_case(self, case) -> dict:
        return {
            "current_load_rate": case.current_load_rate,
            "urgent_orders": case.urgent_orders,
            "available_backup_vehicles": case.available_backup_vehicles,
            "avg_delay_minutes": case.avg_delay_minutes,
            "affected_routes": case.affected_routes,
            "time_window_pressure": case.time_window_pressure,
            "customer_priority_mix": case.customer_priority_mix,
        }

    def _context_from_cases(self, cases: list) -> dict:
        if not cases:
            return {
                "current_load_rate": 0.0,
                "urgent_orders": 0,
                "available_backup_vehicles": 0,
                "avg_delay_minutes": 0.0,
                "affected_routes": 0,
                "time_window_pressure": 0.0,
                "customer_priority_mix": 0.0,
            }

        contexts = [self._context_from_case(case) for case in cases]
        count = len(contexts)
        return {
            "current_load_rate": sum(ctx["current_load_rate"] for ctx in contexts) / count,
            "urgent_orders": int(round(sum(ctx["urgent_orders"] for ctx in contexts) / count)),
            "available_backup_vehicles": int(round(sum(ctx["available_backup_vehicles"] for ctx in contexts) / count)),
            "avg_delay_minutes": sum(ctx["avg_delay_minutes"] for ctx in contexts) / count,
            "affected_routes": int(round(sum(ctx["affected_routes"] for ctx in contexts) / count)),
            "time_window_pressure": sum(ctx["time_window_pressure"] for ctx in contexts) / count,
            "customer_priority_mix": sum(ctx["customer_priority_mix"] for ctx in contexts) / count,
        }

    def _query_case_features(self, q_ev: str, q_sev: int, q_sc: str, case, gt_action: str) -> list:
        try:
            q_ev_code = self._event_encoder.transform([q_ev])[0]
        except ValueError:
            q_ev_code = 0
        try:
            case_ev_code = self._event_encoder.transform([case.event_type])[0]
        except ValueError:
            case_ev_code = 0
        try:
            q_sc_code = self._scenario_encoder.transform([q_sc])[0]
        except ValueError:
            q_sc_code = 0
        try:
            case_sc_code = self._scenario_encoder.transform([case.scenario])[0]
        except ValueError:
            case_sc_code = 0

        ev_match = 1 if q_ev == case.event_type else 0
        sc_match = 1 if q_sc == case.scenario else 0
        sev_diff = abs(q_sev - case.severity)
        action_match = 1 if case.action == gt_action else 0
        outcome_bonus = 1.0 if case.outcome == "success" else 0.0
        sev_x_action = sev_diff * action_match
        ev_x_outcome = (1 if case.event_type == q_ev else 0) * outcome_bonus
        ev_code_diff = abs(q_ev_code - case_ev_code)
        sc_code_diff = abs(q_sc_code - case_sc_code)

        return [
            ev_match,
            sc_match,
            1 / (1 + sev_diff),
            action_match,
            outcome_bonus,
            sev_x_action,
            ev_x_outcome,
            ev_code_diff,
            sc_code_diff,
            1 / (1 + abs(q_sev - 5)),
        ]

    def train(self, cases: list):
        from src.rag.solver_ground_truth import SolverDrivenGroundTruth

        X, y = [], []
        self._pos_count = 0
        self._neg_count = 0
        max_neg_per_query = 10
        solver_gt = SolverDrivenGroundTruth(seed=20260428)

        for query_case in cases:
            q_ev, q_sev, q_sc = query_case.event_type, query_case.severity, query_case.scenario
            context = self._context_from_case(query_case)
            gt_action, _ = solver_gt.best_action(
                query_case.scenario,
                query_case.event_type,
                query_case.severity,
                context,
            )
            pos_feats, neg_feats = [], []

            for cand_case in cases:
                if cand_case is query_case:
                    continue
                is_pos = (cand_case.action == gt_action and cand_case.outcome == "success")
                is_neg = not is_pos
                if is_pos:
                    feat = self._query_case_features(q_ev, q_sev, q_sc, cand_case, gt_action)
                    pos_feats.append(feat)
                    self._pos_count += 1
                elif is_neg and len(neg_feats) < max_neg_per_query:
                    feat = self._query_case_features(q_ev, q_sev, q_sc, cand_case, gt_action)
                    neg_feats.append(feat)
                    self._neg_count += 1

            X.extend(pos_feats)
            y.extend([1] * len(pos_feats))
            X.extend(neg_feats)
            y.extend([0] * len(neg_feats))

        print(f"  V2 training: {len(X)} samples (pos={self._pos_count}, neg={self._neg_count})")
        self._clf = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=8,
            learning_rate=0.1,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=42,
            class_weight="balanced",
        )
        self._clf.fit(X, y)
        print(f"  V2 model trained.")

    def predict_relevance(self, q_ev: str, q_sev: int, q_sc: str, cases: list, context: dict | None = None) -> list[float]:
        from src.rag.solver_ground_truth import SolverDrivenGroundTruth

        if self._clf is None:
            raise RuntimeError("V2 model not trained")
        solver_gt = SolverDrivenGroundTruth(seed=20260428)
        context = context or self._context_from_cases(cases)
        gt_action, _ = solver_gt.best_action(q_sc, q_ev, q_sev, context)
        probs = []
        for case in cases:
            feat = self._query_case_features(q_ev, q_sev, q_sc, case, gt_action)
            prob = float(self._clf.predict_proba([feat])[0][1])
            probs.append(prob)
        return probs

    def save(self, path: str | Path):
        path = Path(path)
        with open(path.with_suffix(".v2.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceModelV2":
        with open(str(Path(path).with_suffix(".v2.pkl")), "rb") as f:
            return pickle.load(f)


class ExperienceModelV2Legacy:
    """
    Legacy v2 experience model using a single decision tree relevance classifier.
    """

    def __init__(self):
        self._clf = None
        self._event_encoder = LabelEncoder()
        self._scenario_encoder = LabelEncoder()
        self._action_encoder = LabelEncoder()
        self._pos_count = 0
        self._neg_count = 0

        ALL_EVENT_TYPES = [
            "vehicle_breakdown", "traffic_accident", "road_closed", "order_cancel",
            "vehicle_maintenance", "fuel_shortage", "driver_unavailable", "traffic_congestion",
            "road_narrowing", "bridge_weight_limit", "order_modify", "order_update",
            "priority_order_urgent", "delivery_failure", "demand_surge", "demand_drop",
            "weather_delay", "natural_disaster", "public_event", "warehouse_delay",
            "inventory_stockout",
        ]
        ALL_SCENARIOS = ["small", "medium", "large", "stress"]
        ALL_ACTIONS = ["reroute", "ignore", "adjust_capacity", "reassign_order", "delay_tolerant"]

        self._event_encoder.fit(ALL_EVENT_TYPES)
        self._scenario_encoder.fit(ALL_SCENARIOS)
        self._action_encoder.fit(ALL_ACTIONS)

    def _context_from_case(self, case) -> dict:
        return {
            "current_load_rate": case.current_load_rate,
            "urgent_orders": case.urgent_orders,
            "available_backup_vehicles": case.available_backup_vehicles,
            "avg_delay_minutes": case.avg_delay_minutes,
            "affected_routes": case.affected_routes,
            "time_window_pressure": case.time_window_pressure,
            "customer_priority_mix": case.customer_priority_mix,
        }

    def _context_from_cases(self, cases: list) -> dict:
        if not cases:
            return {
                "current_load_rate": 0.0,
                "urgent_orders": 0,
                "available_backup_vehicles": 0,
                "avg_delay_minutes": 0.0,
                "affected_routes": 0,
                "time_window_pressure": 0.0,
                "customer_priority_mix": 0.0,
            }

        contexts = [self._context_from_case(case) for case in cases]
        count = len(contexts)
        return {
            "current_load_rate": sum(ctx["current_load_rate"] for ctx in contexts) / count,
            "urgent_orders": int(round(sum(ctx["urgent_orders"] for ctx in contexts) / count)),
            "available_backup_vehicles": int(round(sum(ctx["available_backup_vehicles"] for ctx in contexts) / count)),
            "avg_delay_minutes": sum(ctx["avg_delay_minutes"] for ctx in contexts) / count,
            "affected_routes": int(round(sum(ctx["affected_routes"] for ctx in contexts) / count)),
            "time_window_pressure": sum(ctx["time_window_pressure"] for ctx in contexts) / count,
            "customer_priority_mix": sum(ctx["customer_priority_mix"] for ctx in contexts) / count,
        }

    def _query_case_features(self, q_ev: str, q_sev: int, q_sc: str, case, gt_action: str) -> list:
        try:
            q_ev_code = self._event_encoder.transform([q_ev])[0]
        except ValueError:
            q_ev_code = 0
        try:
            case_ev_code = self._event_encoder.transform([case.event_type])[0]
        except ValueError:
            case_ev_code = 0
        try:
            q_sc_code = self._scenario_encoder.transform([q_sc])[0]
        except ValueError:
            q_sc_code = 0
        try:
            case_sc_code = self._scenario_encoder.transform([case.scenario])[0]
        except ValueError:
            case_sc_code = 0

        ev_match = 1 if q_ev == case.event_type else 0
        sc_match = 1 if q_sc == case.scenario else 0
        sev_diff = abs(q_sev - case.severity)
        action_match = 1 if case.action == gt_action else 0
        outcome_bonus = 1.0 if case.outcome == "success" else 0.0
        sev_x_action = sev_diff * action_match
        ev_x_outcome = (1 if case.event_type == q_ev else 0) * outcome_bonus
        ev_code_diff = abs(q_ev_code - case_ev_code)
        sc_code_diff = abs(q_sc_code - case_sc_code)

        return [
            ev_match,
            sc_match,
            1 / (1 + sev_diff),
            action_match,
            outcome_bonus,
            sev_x_action,
            ev_x_outcome,
            ev_code_diff,
            sc_code_diff,
            1 / (1 + abs(q_sev - 5)),
        ]

    def train(self, cases: list):
        from src.rag.solver_ground_truth import SolverDrivenGroundTruth

        X, y = [], []
        self._pos_count = 0
        self._neg_count = 0
        max_neg_per_query = 10
        solver_gt = SolverDrivenGroundTruth(seed=20260428)

        for query_case in cases:
            q_ev, q_sev, q_sc = query_case.event_type, query_case.severity, query_case.scenario
            context = self._context_from_case(query_case)
            gt_action, _ = solver_gt.best_action(
                query_case.scenario,
                query_case.event_type,
                query_case.severity,
                context,
            )
            pos_feats, neg_feats = [], []

            for cand_case in cases:
                if cand_case is query_case:
                    continue
                is_pos = (cand_case.action == gt_action and cand_case.outcome == "success")
                is_neg = not is_pos
                if is_pos:
                    feat = self._query_case_features(q_ev, q_sev, q_sc, cand_case, gt_action)
                    pos_feats.append(feat)
                    self._pos_count += 1
                elif is_neg and len(neg_feats) < max_neg_per_query:
                    feat = self._query_case_features(q_ev, q_sev, q_sc, cand_case, gt_action)
                    neg_feats.append(feat)
                    self._neg_count += 1

            X.extend(pos_feats)
            y.extend([1] * len(pos_feats))
            X.extend(neg_feats)
            y.extend([0] * len(neg_feats))

        print(f"  V2 training: {len(X)} samples (pos={self._pos_count}, neg={self._neg_count})")
        self._clf = tree.DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, random_state=42, class_weight="balanced"
        )
        self._clf.fit(X, y)
        print(f"  V2 model trained.")

    def predict_relevance(self, q_ev: str, q_sev: int, q_sc: str, cases: list, context: dict | None = None) -> list[float]:
        from src.rag.solver_ground_truth import SolverDrivenGroundTruth

        if self._clf is None:
            raise RuntimeError("V2 model not trained")
        solver_gt = SolverDrivenGroundTruth(seed=20260428)
        context = context or self._context_from_cases(cases)
        gt_action, _ = solver_gt.best_action(q_sc, q_ev, q_sev, context)
        probs = []
        for case in cases:
            feat = self._query_case_features(q_ev, q_sev, q_sc, case, gt_action)
            prob = float(self._clf.predict_proba([feat])[0][1])
            probs.append(prob)
        return probs

    def save(self, path: str | Path):
        path = Path(path)
        with open(path.with_suffix(".v2.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceModelV2Legacy":
        with open(str(Path(path).with_suffix(".v2.pkl")), "rb") as f:
            return pickle.load(f)


class LearnedRetrieverV2:
    """LightGBM LambdaRank retriever for query-case relevance."""

    FEATURE_NAMES = [
        "event_type_match",
        "event_type_hierarchy",
        "scenario_match",
        "severity_diff",
        "vehicles_diff",
        "orders_diff",
        "load_rate_diff",
        "load_rate_ratio",
        "urgent_orders_diff",
        "urgent_orders_ratio",
        "backup_vehicles_diff",
        "avg_delay_diff",
        "avg_delay_ratio",
        "affected_routes_diff",
        "time_window_pressure_diff",
        "time_window_pressure_ratio",
        "customer_priority_diff",
        "customer_priority_ratio",
        "load_x_urgent",
        "pressure_x_priority",
        "query_load_rate",
        "query_urgent_ratio",
        "query_time_pressure",
        "query_capacity_reserve",
    ]

    EVENT_TYPE_GROUPS = (
        {"traffic_congestion", "traffic_accident", "road_closed"},
        {"weather_delay", "traffic_accident"},
        {"order_modify", "order_cancel", "order_update"},
    )

    CONTEXT_DEFAULTS = {
        "current_load_rate": 0.0,
        "urgent_orders": 0,
        "available_backup_vehicles": 0,
        "avg_delay_minutes": 0.0,
        "affected_routes": 0,
        "time_window_pressure": 0.0,
        "customer_priority_mix": 0.0,
    }

    def __init__(
        self,
        random_state: int = 42,
        context_sample_size: int = 1000,
        max_train_queries: int | None = None,
    ):
        self._model = None
        self.random_state = random_state
        self.context_sample_size = context_sample_size
        self.max_train_queries = max_train_queries
        self._pos_count = 0
        self._neg_count = 0
        self._group_count = 0

    @staticmethod
    def _get(obj: Any, key: str, default: Any = 0) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def _context_value(cls, obj: Any, key: str) -> float:
        return float(cls._get(obj, key, cls.CONTEXT_DEFAULTS.get(key, 0.0)) or 0.0)

    @staticmethod
    def _inverse_abs_diff(left: float, right: float) -> float:
        return 1.0 / (1.0 + abs(float(left) - float(right)))

    @staticmethod
    def _ratio(left: float, right: float) -> float:
        left = float(left)
        right = float(right)
        high = max(abs(left), abs(right))
        if high <= 1e-8:
            return 1.0
        return min(abs(left), abs(right)) / (high + 1e-8)

    @classmethod
    def _event_type_hierarchy_match(cls, query_type: str, case_type: str) -> int:
        if query_type == case_type:
            return 0
        for group in cls.EVENT_TYPE_GROUPS:
            if query_type in group and case_type in group:
                return 1
        return 0

    @classmethod
    def _surface_match(cls, query_case: Any, candidate_case: Any) -> bool:
        return (
            cls._get(query_case, "event_type") == cls._get(candidate_case, "event_type")
            and cls._get(query_case, "scenario") == cls._get(candidate_case, "scenario")
        )

    @classmethod
    def _context_pressure_similar(cls, query_case: Any, candidate_case: Any) -> bool:
        load_diff = abs(
            cls._context_value(query_case, "current_load_rate")
            - cls._context_value(candidate_case, "current_load_rate")
        )
        pressure_diff = abs(
            cls._context_value(query_case, "time_window_pressure")
            - cls._context_value(candidate_case, "time_window_pressure")
        )
        severity_diff = abs(
            int(cls._get(query_case, "severity", 0) or 0)
            - int(cls._get(candidate_case, "severity", 0) or 0)
        )
        return load_diff < 0.2 and pressure_diff < 0.2 and severity_diff < 3

    @classmethod
    def _is_positive_pair(cls, query_case: Any, candidate_case: Any) -> bool:
        if cls._get(candidate_case, "outcome") != "success":
            return False

        load_diff = abs(
            cls._context_value(query_case, "current_load_rate")
            - cls._context_value(candidate_case, "current_load_rate")
        )
        severity_diff = abs(
            int(cls._get(query_case, "severity", 0) or 0)
            - int(cls._get(candidate_case, "severity", 0) or 0)
        )

        condition_1 = (
            cls._surface_match(query_case, candidate_case)
            and load_diff < 0.15
            and severity_diff < 2
        )
        condition_2 = (
            not cls._surface_match(query_case, candidate_case)
            and cls._context_pressure_similar(query_case, candidate_case)
        )
        return condition_1 or condition_2

    @classmethod
    def _is_negative_pair(cls, query_case: Any, candidate_case: Any) -> bool:
        if cls._get(candidate_case, "outcome") == "failure":
            return True

        if not cls._surface_match(query_case, candidate_case):
            return False

        load_diff = abs(
            cls._context_value(query_case, "current_load_rate")
            - cls._context_value(candidate_case, "current_load_rate")
        )
        severity_diff = abs(
            int(cls._get(query_case, "severity", 0) or 0)
            - int(cls._get(candidate_case, "severity", 0) or 0)
        )
        return load_diff > 0.4 or severity_diff > 5

    @classmethod
    def relevance_label(cls, query_case: Any, candidate_case: Any) -> int:
        """Return the binary relevance label used by training and evaluation."""
        return 1 if cls._is_positive_pair(query_case, candidate_case) else 0

    def _extract_features(self, query_case: Any, candidate_case: Any) -> list[float]:
        q_event = str(self._get(query_case, "event_type", ""))
        c_event = str(self._get(candidate_case, "event_type", ""))
        q_scenario = str(self._get(query_case, "scenario", ""))
        c_scenario = str(self._get(candidate_case, "scenario", ""))

        q_severity = int(self._get(query_case, "severity", 0) or 0)
        c_severity = int(self._get(candidate_case, "severity", 0) or 0)
        q_vehicles = int(self._get(query_case, "vehicles_count", 0) or 0)
        c_vehicles = int(self._get(candidate_case, "vehicles_count", 0) or 0)
        q_orders = int(self._get(query_case, "orders_count", 0) or 0)
        c_orders = int(self._get(candidate_case, "orders_count", 0) or 0)

        q_load = self._context_value(query_case, "current_load_rate")
        c_load = self._context_value(candidate_case, "current_load_rate")
        q_urgent = self._context_value(query_case, "urgent_orders")
        c_urgent = self._context_value(candidate_case, "urgent_orders")
        q_backup = self._context_value(query_case, "available_backup_vehicles")
        c_backup = self._context_value(candidate_case, "available_backup_vehicles")
        q_delay = self._context_value(query_case, "avg_delay_minutes")
        c_delay = self._context_value(candidate_case, "avg_delay_minutes")
        q_routes = self._context_value(query_case, "affected_routes")
        c_routes = self._context_value(candidate_case, "affected_routes")
        q_pressure = self._context_value(query_case, "time_window_pressure")
        c_pressure = self._context_value(candidate_case, "time_window_pressure")
        q_priority = self._context_value(query_case, "customer_priority_mix")
        c_priority = self._context_value(candidate_case, "customer_priority_mix")

        load_rate_diff = abs(q_load - c_load)
        urgent_orders_diff = abs(q_urgent - c_urgent)
        time_window_pressure_diff = abs(q_pressure - c_pressure)
        customer_priority_diff = abs(q_priority - c_priority)

        return [
            1.0 if q_event == c_event else 0.0,
            float(self._event_type_hierarchy_match(q_event, c_event)),
            1.0 if q_scenario == c_scenario else 0.0,
            self._inverse_abs_diff(q_severity, c_severity),
            self._inverse_abs_diff(q_vehicles, c_vehicles),
            self._inverse_abs_diff(q_orders, c_orders),
            load_rate_diff,
            self._ratio(q_load, c_load),
            urgent_orders_diff,
            self._ratio(q_urgent, c_urgent),
            abs(q_backup - c_backup),
            abs(q_delay - c_delay),
            self._ratio(q_delay, c_delay),
            abs(q_routes - c_routes),
            time_window_pressure_diff,
            self._ratio(q_pressure, c_pressure),
            customer_priority_diff,
            self._ratio(q_priority, c_priority),
            load_rate_diff * urgent_orders_diff,
            time_window_pressure_diff * customer_priority_diff,
            q_load,
            q_urgent / max(q_orders, 1),
            q_pressure,
            q_backup / max(q_vehicles, 1),
        ]

    def _make_query_context(self, case: Any) -> dict:
        return {
            "event_type": self._get(case, "event_type"),
            "severity": self._get(case, "severity"),
            "scenario": self._get(case, "scenario"),
            "vehicles_count": self._get(case, "vehicles_count"),
            "orders_count": self._get(case, "orders_count"),
            "current_load_rate": self._context_value(case, "current_load_rate"),
            "urgent_orders": self._context_value(case, "urgent_orders"),
            "available_backup_vehicles": self._context_value(case, "available_backup_vehicles"),
            "avg_delay_minutes": self._context_value(case, "avg_delay_minutes"),
            "affected_routes": self._context_value(case, "affected_routes"),
            "time_window_pressure": self._context_value(case, "time_window_pressure"),
            "customer_priority_mix": self._context_value(case, "customer_priority_mix"),
        }

    def _sample_context_successes(
        self,
        query_case: Any,
        success_cases: list,
        rng: random.Random,
    ) -> list:
        if len(success_cases) <= self.context_sample_size:
            sampled = success_cases
        else:
            sampled = rng.sample(success_cases, self.context_sample_size)
        return [
            case
            for case in sampled
            if case is not query_case and self._is_positive_pair(query_case, case)
        ]

    def train(self, cases: list):
        """Train a LambdaRank retriever from historical Case objects."""
        if not cases:
            raise ValueError("No cases provided for training")

        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "LearnedRetrieverV2 requires lightgbm. Install it in the logistic conda env."
            ) from exc

        rng = random.Random(self.random_state)
        train_queries = list(cases)
        if self.max_train_queries is not None and len(train_queries) > self.max_train_queries:
            train_queries = rng.sample(train_queries, self.max_train_queries)

        surface_buckets: dict[tuple[str, str], list] = {}
        success_cases = []
        for case in cases:
            key = (self._get(case, "event_type"), self._get(case, "scenario"))
            surface_buckets.setdefault(key, []).append(case)
            if self._get(case, "outcome") == "success":
                success_cases.append(case)

        X: list[list[float]] = []
        y: list[int] = []
        group_sizes: list[int] = []
        self._pos_count = 0
        self._neg_count = 0
        self._group_count = 0

        for query_case in train_queries:
            key = (self._get(query_case, "event_type"), self._get(query_case, "scenario"))
            surface_candidates = [
                case for case in surface_buckets.get(key, []) if case is not query_case
            ]
            rng.shuffle(surface_candidates)

            positives = [
                case for case in surface_candidates if self._is_positive_pair(query_case, case)
            ]
            if len(positives) < 20:
                positives.extend(
                    case
                    for case in self._sample_context_successes(query_case, success_cases, rng)
                    if case not in positives
                )
            positives = positives[:20]

            hard_negatives = [
                case
                for case in surface_candidates
                if self._is_negative_pair(query_case, case)
            ][:20]

            selected_ids = {id(case) for case in positives + hard_negatives}
            random_negatives = []
            attempts = 0
            while len(random_negatives) < 10 and attempts < 200:
                candidate = rng.choice(cases)
                attempts += 1
                if candidate is query_case or id(candidate) in selected_ids:
                    continue
                if self._is_negative_pair(query_case, candidate):
                    random_negatives.append(candidate)
                    selected_ids.add(id(candidate))

            query_pairs = [(case, 1) for case in positives]
            query_pairs.extend((case, 0) for case in hard_negatives)
            query_pairs.extend((case, 0) for case in random_negatives)

            if not positives or len(query_pairs) == len(positives):
                continue

            for candidate, label in query_pairs:
                X.append(self._extract_features(query_case, candidate))
                y.append(label)
            group_sizes.append(len(query_pairs))
            self._pos_count += len(positives)
            self._neg_count += len(query_pairs) - len(positives)

        if not X or not group_sizes:
            raise ValueError("No valid ranking pairs generated for training")

        train_data = lgb.Dataset(np.asarray(X, dtype=np.float32), label=y, group=group_sizes)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_at": [5, 10],
            "learning_rate": 0.1,
            "num_leaves": 63,
            "min_data_in_leaf": 20,
            "seed": self.random_state,
            "verbosity": -1,
        }
        self._model = lgb.train(params, train_data, num_boost_round=300)
        self._group_count = len(group_sizes)
        print(
            "  LearnedRetrieverV2 training: "
            f"{len(X)} samples, groups={self._group_count}, "
            f"pos={self._pos_count}, neg={self._neg_count}"
        )

    def _build_match_reason(self, query_context: dict, candidate_case: Any) -> str:
        reasons = []
        if self._get(query_context, "event_type") == self._get(candidate_case, "event_type"):
            reasons.append("event_type match")
        elif self._event_type_hierarchy_match(
            str(self._get(query_context, "event_type", "")),
            str(self._get(candidate_case, "event_type", "")),
        ):
            reasons.append("event_type hierarchy match")
        if self._get(query_context, "scenario") == self._get(candidate_case, "scenario"):
            reasons.append("scenario match")
        if self._context_pressure_similar(query_context, candidate_case):
            reasons.append("context pressure close")
        if self._get(candidate_case, "outcome") == "success":
            reasons.append("successful outcome")
        return "; ".join(reasons) if reasons else "learned relevance score"

    def _build_context_diff(self, query_context: dict, candidate_case: Any) -> dict:
        q_load = self._context_value(query_context, "current_load_rate")
        c_load = self._context_value(candidate_case, "current_load_rate")
        q_urgent = self._context_value(query_context, "urgent_orders")
        c_urgent = self._context_value(candidate_case, "urgent_orders")
        q_pressure = self._context_value(query_context, "time_window_pressure")
        c_pressure = self._context_value(candidate_case, "time_window_pressure")
        severity_diff = abs(
            int(self._get(query_context, "severity", 0) or 0)
            - int(self._get(candidate_case, "severity", 0) or 0)
        )

        load_diff = abs(q_load - c_load)
        urgent_diff = abs(q_urgent - c_urgent)
        pressure_diff = abs(q_pressure - c_pressure)
        overall_similarity = float(
            np.mean(
                [
                    max(0.0, 1.0 - min(load_diff, 1.0)),
                    1.0 / (1.0 + urgent_diff),
                    max(0.0, 1.0 - min(pressure_diff, 1.0)),
                    1.0 / (1.0 + severity_diff),
                ]
            )
        )
        if overall_similarity >= 0.8:
            insight = (
                "Load rate and time-window pressure are close; "
                "historical case is highly reusable"
            )
        elif overall_similarity >= 0.6:
            insight = "Context is partly aligned; use as a moderate reference"
        else:
            insight = "Context differs materially; use as a weak reference"

        return {
            "load_rate": {
                "query": round(q_load, 4),
                "case": round(c_load, 4),
                "diff": round(load_diff, 4),
            },
            "urgent_orders": {
                "query": q_urgent,
                "case": c_urgent,
                "diff": urgent_diff,
            },
            "time_window_pressure": {
                "query": round(q_pressure, 4),
                "case": round(c_pressure, 4),
                "diff": round(pressure_diff, 4),
            },
            "overall_similarity": round(overall_similarity, 4),
            "key_insight": insight,
        }

    def retrieve(
        self,
        query_context: dict,
        candidate_cases: list,
        top_k: int = 5,
    ) -> list[dict]:
        if self._model is None:
            raise RuntimeError("LearnedRetrieverV2 model not trained")
        if not candidate_cases:
            return []

        X = np.asarray(
            [self._extract_features(query_context, case) for case in candidate_cases],
            dtype=np.float32,
        )
        scores = self._model.predict(X)
        ranked = sorted(
            zip(candidate_cases, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results = []
        for case, score in ranked[:top_k]:
            results.append(
                {
                    "case": case,
                    "score": float(score),
                    "match_reason": self._build_match_reason(query_context, case),
                    "context_diff": self._build_context_diff(query_context, case),
                }
            )
        return results

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedRetrieverV2":
        with open(path, "rb") as f:
            return pickle.load(f)


# CLI for training
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))


    print("Training Experience Model...")

    # 加载 cases
    retriever = CaseRetriever()
    print(f"Loaded {len(retriever.cases)} cases")

    # 训练
    model = ExperienceModel()
    model.train(retriever.cases)

    # 保存
    model_path = Path(__file__).parent / "experience_model.pkl"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # 测试预测
    test_cases = [
        ("vehicle_breakdown", 5, "small"),
        ("order_cancel", 1, "medium"),
        ("traffic_congestion", 4, "large"),
        ("demand_surge", 3, "small"),
    ]
    print("\nPrediction tests:")
    for ev, sev, sc in test_cases:
        pred = model.predict(ev, sev, sc)
        print(f"\n  [{ev}] severity={sev} scenario={sc}")
        print(f"    action={pred['action']} (conf={pred['action_confidence']:.2f})")
        print(f"    outcome={pred['outcome']} (conf={pred['outcome_confidence']:.2f})")
        print(f"    distance_ratio={pred['distance_ratio']:.3f}")
        print(f"    unassigned_delta={pred['unassigned_delta']}")
