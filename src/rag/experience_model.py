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
from pathlib import Path

import numpy as np
from sklearn import tree
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

    def _ground_truth_action(self, event_type: str, severity: int) -> str:
        if event_type in ("vehicle_breakdown", "vehicle_maintenance", "driver_unavailable"):
            if severity >= 5: return "reassign_order"
            elif severity >= 3: return "adjust_capacity"
            else: return "reroute"
        if event_type == "fuel_shortage":
            return "adjust_capacity" if severity >= 4 else "reroute"
        if event_type in ("traffic_accident", "traffic_congestion", "road_narrowing"):
            return "reroute" if severity >= 4 else "ignore"
        if event_type in ("road_closed", "bridge_weight_limit"):
            return "reroute" if severity >= 3 else "delay_tolerant"
        if event_type == "order_cancel":
            if severity >= 4: return "reroute"
            elif severity >= 2: return "delay_tolerant"
            else: return "ignore"
        if event_type == "order_modify":
            return "reroute" if severity >= 4 else "delay_tolerant"
        if event_type == "order_update": return "ignore"
        if event_type in ("priority_order_urgent", "delivery_failure"): return "reassign_order"
        if event_type == "demand_surge": return "adjust_capacity" if severity >= 4 else "delay_tolerant"
        if event_type == "demand_drop": return "delay_tolerant"
        if event_type in ("weather_delay", "natural_disaster", "public_event"):
            return "delay_tolerant" if severity >= 4 else "reroute"
        if event_type in ("warehouse_delay", "inventory_stockout"): return "delay_tolerant"
        return "reroute"

    def _query_case_features(self, q_ev: str, q_sev: int, q_sc: str, case) -> list:
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
        gt_action = self._ground_truth_action(q_ev, q_sev)
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
        import numpy as np
        X, y = [], []
        self._pos_count = 0
        self._neg_count = 0
        max_neg_per_query = 10

        for query_case in cases:
            gt_action = self._ground_truth_action(query_case.event_type, query_case.severity)
            q_ev, q_sev, q_sc = query_case.event_type, query_case.severity, query_case.scenario
            pos_feats, neg_feats = [], []

            for cand_case in cases:
                if cand_case is query_case:
                    continue
                is_pos = (cand_case.action == gt_action and cand_case.outcome == "success")
                is_neg = not is_pos
                if is_pos:
                    feat = self._query_case_features(q_ev, q_sev, q_sc, cand_case)
                    pos_feats.append(feat)
                    self._pos_count += 1
                elif is_neg and len(neg_feats) < max_neg_per_query:
                    feat = self._query_case_features(q_ev, q_sev, q_sc, cand_case)
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

        print(f"  V2 training: {len(X)} samples (pos={self._pos_count}, neg={self._neg_count})")
        self._clf = tree.DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, random_state=42, class_weight="balanced"
        )
        self._clf.fit(X, y)
        print(f"  V2 model trained.")

    def predict_relevance(self, q_ev: str, q_sev: int, q_sc: str, cases: list) -> list[float]:
        if self._clf is None:
            raise RuntimeError("V2 model not trained")
        probs = []
        for case in cases:
            feat = self._query_case_features(q_ev, q_sev, q_sc, case)
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


# ── CLI for training ─────────────────────────────────────────────────

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