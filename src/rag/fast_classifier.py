"""Fast XGBoost classifier for cascade logistics decisions.

The classifier is intended to handle high-confidence cases before a slower
LLM+RAG fallback is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_PATH = ROOT / "data" / "cases" / "cases_30k.jsonl"
DEFAULT_MODEL_PATH = ROOT / "output" / "models" / "fast_classifier.json"

FEATURE_COLUMNS = [
    "severity",
    "event_type",
    "scenario",
    "current_load_rate",
    "urgent_orders",
    "available_backup_vehicles",
    "avg_delay_minutes",
    "time_window_pressure",
    "customer_priority_mix",
    "affected_routes",
    "cost_before",
]

NUMERIC_FEATURES = [
    "severity",
    "current_load_rate",
    "urgent_orders",
    "available_backup_vehicles",
    "avg_delay_minutes",
    "time_window_pressure",
    "customer_priority_mix",
    "affected_routes",
    "cost_before",
]

CATEGORICAL_FEATURES = ["event_type", "scenario"]
ACTION_CLASSES = ["adjust_capacity", "delay_tolerant", "ignore", "reassign_order", "reroute"]


class CascadeClassifier:
    """A lightweight multiclass classifier for logistics action selection."""

    def __init__(self) -> None:
        self.model: XGBClassifier | None = None
        self.event_encoder = LabelEncoder()
        self.scenario_encoder = LabelEncoder()
        self.action_encoder = LabelEncoder()
        self._event_map: dict[str, int] = {}
        self._scenario_map: dict[str, int] = {}
        self._cost_before_by_scenario: dict[str, float] = {}
        self._global_cost_before: float = 0.0
        self._is_loaded = False

    def train(
        self,
        cases_path: str | Path = DEFAULT_CASES_PATH,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        max_cases: int = 30_000,
        random_state: int = 20260428,
    ) -> dict[str, Any]:
        """Train the XGBoost model and save it to disk."""

        rows = self._load_cases(Path(cases_path), max_cases=max_cases)
        if not rows:
            raise ValueError(f"No cases found in {cases_path}")

        event_values = [str(row.get("event_type", "")) for row in rows]
        scenario_values = [str(row.get("scenario", "")) for row in rows]
        action_values = [str(row.get("action", "")) for row in rows]

        self.event_encoder.fit(event_values)
        self.scenario_encoder.fit(scenario_values)
        self.action_encoder.fit(action_values)
        self._refresh_maps()
        self._fit_cost_before_imputer(rows)

        x = self._rows_to_matrix(rows)
        y = self.action_encoder.transform(action_values)

        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=y,
        )

        self.model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=len(self.action_encoder.classes_),
            n_estimators=1000,
            early_stopping_rounds=10,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )
        self.model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

        y_pred = self.model.predict(x_val)
        val_accuracy = float(accuracy_score(y_val, y_pred))
        classwise = self._classwise_accuracy(y_val, y_pred)

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_metadata()
        self.model.save_model(str(model_path))
        self._is_loaded = True

        return {
            "val_accuracy": round(val_accuracy, 6),
            "classwise_accuracies": classwise,
            "train_size": int(len(y_train)),
            "val_size": int(len(y_val)),
            "model_path": str(model_path),
        }

    def load(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> "CascadeClassifier":
        """Load a trained classifier from disk."""

        self.model = XGBClassifier()
        self.model.load_model(str(model_path))
        metadata_raw = self.model.get_booster().attr("cascade_metadata")
        if not metadata_raw:
            raise ValueError(f"Model at {model_path} does not contain cascade metadata")

        metadata = json.loads(metadata_raw)
        self.event_encoder.classes_ = np.array(metadata["event_type_classes"], dtype=object)
        self.scenario_encoder.classes_ = np.array(metadata["scenario_classes"], dtype=object)
        self.action_encoder.classes_ = np.array(metadata["action_classes"], dtype=object)
        self._cost_before_by_scenario = {
            str(key): float(value)
            for key, value in metadata.get("cost_before_by_scenario", {}).items()
        }
        self._global_cost_before = float(metadata.get("global_cost_before", 0.0))
        self._refresh_maps()
        self._is_loaded = True
        return self

    def predict(self, query_features: dict[str, Any]) -> tuple[str, float]:
        """Predict a single action and confidence score."""

        if self.model is None:
            raise RuntimeError("Classifier is not trained or loaded")

        x = self._query_to_matrix(query_features)
        probs = self.model.predict_proba(x)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        action = str(self.action_encoder.inverse_transform([pred_idx])[0])
        return action, confidence

    def predict_batch(self, queries: list[dict[str, Any]]) -> list[tuple[str, float]]:
        """Predict actions and confidence scores for a batch of queries."""

        if self.model is None:
            raise RuntimeError("Classifier is not trained or loaded")
        if not queries:
            return []

        x = np.vstack([self._query_to_matrix(query) for query in queries])
        probs = self.model.predict_proba(x)
        pred_indices = np.argmax(probs, axis=1)
        actions = self.action_encoder.inverse_transform(pred_indices.astype(int))
        confidences = np.max(probs, axis=1)
        return [(str(action), float(conf)) for action, conf in zip(actions, confidences)]

    @staticmethod
    def _load_cases(path: Path, *, max_cases: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if len(rows) >= max_cases:
                    break
        return rows

    def _rows_to_matrix(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.array([self._feature_vector(row) for row in rows], dtype=float)

    def _query_to_matrix(self, query: dict[str, Any]) -> np.ndarray:
        return np.array([self._feature_vector(query)], dtype=float)

    def _feature_vector(self, item: dict[str, Any]) -> list[float]:
        context = item.get("context") if isinstance(item.get("context"), dict) else {}
        event_type = str(item.get("event_type", context.get("event_type", "")))
        scenario = str(item.get("scenario", context.get("scenario", "")))

        return [
            self._to_float(self._lookup(item, context, "severity")),
            float(self._event_map.get(event_type, -1)),
            float(self._scenario_map.get(scenario, -1)),
            self._to_float(self._lookup(item, context, "current_load_rate")),
            self._to_float(self._lookup(item, context, "urgent_orders")),
            self._to_float(self._lookup(item, context, "available_backup_vehicles")),
            self._to_float(self._lookup(item, context, "avg_delay_minutes")),
            self._to_float(self._lookup(item, context, "time_window_pressure")),
            self._to_float(self._lookup(item, context, "customer_priority_mix")),
            self._to_float(self._lookup(item, context, "affected_routes")),
            self._cost_before_value(item, context, scenario),
        ]

    @staticmethod
    def _lookup(item: dict[str, Any], context: dict[str, Any], key: str) -> Any:
        return item.get(key, context.get(key, 0))

    @staticmethod
    def _to_float(value: Any) -> float:
        if value is None or value == "":
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _refresh_maps(self) -> None:
        self._event_map = {str(value): int(i) for i, value in enumerate(self.event_encoder.classes_)}
        self._scenario_map = {str(value): int(i) for i, value in enumerate(self.scenario_encoder.classes_)}

    def _fit_cost_before_imputer(self, rows: list[dict[str, Any]]) -> None:
        by_scenario: dict[str, list[float]] = {}
        all_values: list[float] = []
        for row in rows:
            value = self._to_float(row.get("cost_before"))
            scenario = str(row.get("scenario", ""))
            by_scenario.setdefault(scenario, []).append(value)
            all_values.append(value)

        self._cost_before_by_scenario = {
            scenario: float(np.median(values))
            for scenario, values in by_scenario.items()
            if values
        }
        self._global_cost_before = float(np.median(all_values)) if all_values else 0.0

    def _cost_before_value(self, item: dict[str, Any], context: dict[str, Any], scenario: str) -> float:
        if "cost_before" in item or "cost_before" in context:
            return self._to_float(self._lookup(item, context, "cost_before"))
        return self._cost_before_by_scenario.get(scenario, self._global_cost_before)

    def _store_metadata(self) -> None:
        if self.model is None:
            raise RuntimeError("Cannot store metadata before model exists")

        metadata = {
            "feature_columns": FEATURE_COLUMNS,
            "event_type_classes": [str(x) for x in self.event_encoder.classes_],
            "scenario_classes": [str(x) for x in self.scenario_encoder.classes_],
            "action_classes": [str(x) for x in self.action_encoder.classes_],
            "cost_before_by_scenario": self._cost_before_by_scenario,
            "global_cost_before": self._global_cost_before,
        }
        self.model.get_booster().set_attr(cascade_metadata=json.dumps(metadata, ensure_ascii=False))

    def _classwise_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        result: dict[str, float] = {}
        for class_idx, action in enumerate(self.action_encoder.classes_):
            mask = y_true == class_idx
            if not np.any(mask):
                result[str(action)] = 0.0
                continue
            result[str(action)] = round(float(accuracy_score(y_true[mask], y_pred[mask])), 6)
        return result


def main() -> None:
    classifier = CascadeClassifier()
    metrics = classifier.train()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
