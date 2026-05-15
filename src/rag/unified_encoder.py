"""Unified dual-purpose encoder for action classification and dense RAG.

The model consumes structured logistics case features, predicts the best action,
and emits a normalized 64-dimensional embedding suitable for nearest-neighbor
retrieval.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CASES_PATH = ROOT / "data" / "cases" / "cases_30k.jsonl"
DEFAULT_MODEL_PATH = ROOT / "output" / "models" / "unified_encoder.pt"
DEFAULT_INDEX_PATH = ROOT / "output" / "models" / "case_index.faiss"

NUMERIC_FEATURES = [
    "severity",
    "vehicles_count",
    "orders_count",
    "current_load_rate",
    "urgent_orders",
    "available_backup_vehicles",
    "avg_delay_minutes",
    "affected_routes",
    "time_window_pressure",
    "customer_priority_mix",
    "cost_before",
    "before_distance",
    "unassigned_before",
]

ACTION_CLASSES = [
    "delay_tolerant",
    "reassign_order",
    "adjust_capacity",
    "reroute",
    "ignore",
]

SCENARIO_CLASSES = ["small", "medium", "large", "stress"]
RANDOM_STATE = 20260428


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _lookup(item: dict[str, Any], key: str, default: Any = 0) -> Any:
    context = item.get("context") if isinstance(item.get("context"), dict) else {}
    return item.get(key, context.get(key, default))


def _load_cases(path: str | Path, max_cases: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if "action" not in row:
                    raise ValueError("missing action")
                rows.append(row)
                if max_cases is not None and len(rows) >= max_cases:
                    break
            except Exception as exc:
                skipped += 1
                if skipped <= 5:
                    print(f"Skipping malformed case at line {line_no}: {exc}")
    if skipped:
        print(f"Skipped {skipped} malformed rows while loading {path}")
    return rows


def _fit_event_map(rows: Iterable[dict[str, Any]], event_type_size: int) -> dict[str, int]:
    values = sorted({str(row.get("event_type", _lookup(row, "event_type", ""))) for row in rows})
    if len(values) > event_type_size:
        raise ValueError(
            f"Found {len(values)} event types, but event_type_size={event_type_size}"
        )
    return {value: idx for idx, value in enumerate(values)}


def _fit_scenario_map(rows: Iterable[dict[str, Any]], scenario_size: int) -> dict[str, int]:
    observed = {str(row.get("scenario", _lookup(row, "scenario", ""))) for row in rows}
    values = [value for value in SCENARIO_CLASSES if value in observed]
    values.extend(sorted(observed - set(values)))
    if len(values) > scenario_size:
        raise ValueError(f"Found {len(values)} scenarios, but scenario_size={scenario_size}")
    return {value: idx for idx, value in enumerate(values)}


def _numeric_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [[_to_float(_lookup(row, feature)) for feature in NUMERIC_FEATURES] for row in rows],
        dtype=np.float32,
    )


@dataclass
class PreparedArrays:
    numeric: np.ndarray
    event_type: np.ndarray
    scenario: np.ndarray
    labels: np.ndarray
    case_ids: np.ndarray


class CaseTensorDataset(Dataset):
    """Torch dataset for preprocessed logistics cases."""

    def __init__(self, arrays: PreparedArrays):
        self.numeric = torch.tensor(arrays.numeric, dtype=torch.float32)
        self.event_type = torch.tensor(arrays.event_type, dtype=torch.long)
        self.scenario = torch.tensor(arrays.scenario, dtype=torch.long)
        self.labels = torch.tensor(arrays.labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.numeric[idx], self.event_type[idx], self.scenario[idx], self.labels[idx]


class ActionBalancedSampler(Sampler[int]):
    """Yield indices ordered so each batch includes examples from every action class."""

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        min_per_class: int = 2,
        generator: torch.Generator | None = None,
    ):
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError("labels must be a 1D numpy array")
        if labels.shape[0] == 0:
            raise ValueError("labels must not be empty")

        self.labels = labels.astype(np.int64, copy=False)
        self.batch_size = int(batch_size)
        self.min_per_class = int(min_per_class)
        self.generator = generator

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.min_per_class <= 0:
            raise ValueError("min_per_class must be positive")
        self.num_batches = int(math.ceil(len(self.labels) / self.batch_size))

        self.class_indices: list[torch.Tensor] = [
            torch.as_tensor(np.flatnonzero(self.labels == label), dtype=torch.long)
            for label in sorted(np.unique(self.labels).tolist())
        ]
        required = len(self.class_indices) * self.min_per_class
        if required > self.batch_size:
            raise ValueError(
                f"batch_size={self.batch_size} is too small for "
                f"{self.min_per_class} samples across {len(self.class_indices)} classes"
            )

    def __iter__(self):
        all_indices = torch.arange(len(self.labels), dtype=torch.long)
        for _ in range(self.num_batches):
            batch_parts: list[torch.Tensor] = []
            for indices in self.class_indices:
                if len(indices) >= self.min_per_class:
                    order = torch.randperm(len(indices), generator=self.generator)[: self.min_per_class]
                    batch_parts.append(indices[order])
                else:
                    sampled = torch.randint(
                        len(indices),
                        (self.min_per_class,),
                        generator=self.generator,
                        dtype=torch.long,
                    )
                    batch_parts.append(indices[sampled])

            batch = torch.cat(batch_parts)
            fill_count = self.batch_size - int(batch.shape[0])
            if fill_count > 0:
                selected = set(int(idx) for idx in batch.tolist())
                remaining = [idx for idx in range(len(self.labels)) if idx not in selected]
                pool = (
                    torch.as_tensor(remaining, dtype=torch.long)
                    if len(remaining) >= fill_count
                    else all_indices
                )
                if len(pool) >= fill_count:
                    order = torch.randperm(len(pool), generator=self.generator)[:fill_count]
                    fill = pool[order]
                else:
                    sampled = torch.randint(
                        len(pool),
                        (fill_count,),
                        generator=self.generator,
                        dtype=torch.long,
                    )
                    fill = pool[sampled]
                batch = torch.cat([batch, fill])

            order = torch.randperm(len(batch), generator=self.generator)
            for idx in batch[order].tolist():
                yield int(idx)

    def __len__(self) -> int:
        return self.num_batches * self.batch_size


class UnifiedEncoder(nn.Module):
    """MLP encoder with categorical embeddings and a classification head."""

    def __init__(
        self,
        num_numeric: int = 13,
        event_type_size: int = 21,
        event_type_emb_dim: int = 8,
        scenario_size: int = 4,
        scenario_emb_dim: int = 4,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        num_classes: int = 5,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims or [128, 64])
        if len(hidden_dims) != 2 or hidden_dims[-1] != 64:
            raise ValueError("hidden_dims must end with the 64-dimensional embedding size")

        self.num_numeric = int(num_numeric)
        self.event_type_size = int(event_type_size)
        self.event_type_emb_dim = int(event_type_emb_dim)
        self.scenario_size = int(scenario_size)
        self.scenario_emb_dim = int(scenario_emb_dim)
        self.hidden_dims = hidden_dims
        self.dropout = float(dropout)
        self.num_classes = int(num_classes)

        self.event_embedding = nn.Embedding(event_type_size, event_type_emb_dim)
        self.scenario_embedding = nn.Embedding(scenario_size, scenario_emb_dim)

        input_dim = num_numeric + event_type_emb_dim + scenario_emb_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

        self.scaler_mean: list[float] | None = None
        self.scaler_std: list[float] | None = None
        self.event_type_to_idx: dict[str, int] = {}
        self.scenario_to_idx: dict[str, int] = {}
        self.action_to_idx: dict[str, int] = {
            action: idx for idx, action in enumerate(ACTION_CLASSES)
        }
        self.idx_to_action: dict[int, str] = {
            idx: action for action, idx in self.action_to_idx.items()
        }
        self.case_ids: np.ndarray | None = None

    def forward(
        self,
        numeric: torch.Tensor,
        event_type: torch.Tensor,
        scenario: torch.Tensor,
    ) -> torch.Tensor:
        event_emb = self.event_embedding(event_type.long())
        scenario_emb = self.scenario_embedding(scenario.long())
        x = torch.cat([numeric.float(), event_emb, scenario_emb], dim=1)
        return self.encoder(x)

    def logits(
        self,
        numeric: torch.Tensor,
        event_type: torch.Tensor,
        scenario: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(self.forward(numeric, event_type, scenario))

    def predict(
        self,
        numeric: torch.Tensor | np.ndarray | list[float],
        event_type: torch.Tensor | np.ndarray | int,
        scenario: torch.Tensor | np.ndarray | int,
    ) -> tuple[int, float]:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            numeric_t = torch.as_tensor(numeric, dtype=torch.float32)
            if numeric_t.ndim == 1:
                numeric_t = numeric_t.unsqueeze(0)
            event_t = torch.as_tensor(event_type, dtype=torch.long).reshape(-1)
            scenario_t = torch.as_tensor(scenario, dtype=torch.long).reshape(-1)
            probs = torch.softmax(self.logits(numeric_t, event_t, scenario_t), dim=1)
            confidence, pred = torch.max(probs, dim=1)
        if was_training:
            self.train()
        return int(pred[0].item()), float(confidence[0].item())

    def config(self) -> dict[str, Any]:
        return {
            "num_numeric": self.num_numeric,
            "event_type_size": self.event_type_size,
            "event_type_emb_dim": self.event_type_emb_dim,
            "scenario_size": self.scenario_size,
            "scenario_emb_dim": self.scenario_emb_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "num_classes": self.num_classes,
        }

    def save_model(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config(),
                "numeric_features": NUMERIC_FEATURES,
                "action_classes": [self.idx_to_action[i] for i in range(self.num_classes)],
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
                "event_type_to_idx": self.event_type_to_idx,
                "scenario_to_idx": self.scenario_to_idx,
                "case_ids": self.case_ids.tolist() if self.case_ids is not None else None,
            },
            path,
        )

    @classmethod
    def load_model(cls, path: str | Path = DEFAULT_MODEL_PATH, device: str = "cpu") -> "UnifiedEncoder":
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.scaler_mean = checkpoint.get("scaler_mean")
        model.scaler_std = checkpoint.get("scaler_std")
        model.event_type_to_idx = {
            str(key): int(value) for key, value in checkpoint.get("event_type_to_idx", {}).items()
        }
        model.scenario_to_idx = {
            str(key): int(value) for key, value in checkpoint.get("scenario_to_idx", {}).items()
        }
        actions = checkpoint.get("action_classes", ACTION_CLASSES)
        model.action_to_idx = {str(action): idx for idx, action in enumerate(actions)}
        model.idx_to_action = {idx: str(action) for idx, action in enumerate(actions)}
        case_ids = checkpoint.get("case_ids")
        model.case_ids = np.array(case_ids, dtype=object) if case_ids is not None else None
        model.to(device)
        model.eval()
        return model


class ContrastiveLoss(nn.Module):
    """Supervised NT-Xent loss using same-action positives within a batch."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.shape[0] < 2:
            return embeddings.sum() * 0.0

        z = nn.functional.normalize(embeddings, p=2, dim=1)
        logits = torch.matmul(z, z.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = labels.reshape(-1, 1)
        positives = torch.eq(labels, labels.T).float().to(embeddings.device)
        self_mask = torch.eye(embeddings.shape[0], device=embeddings.device)
        positives = positives * (1.0 - self_mask)

        exp_logits = torch.exp(logits) * (1.0 - self_mask)
        denominator = exp_logits.sum(dim=1).clamp_min(1e-12)
        numerator = (exp_logits * positives).sum(dim=1).clamp_min(1e-12)

        valid = positives.sum(dim=1) > 0
        if not torch.any(valid):
            return embeddings.sum() * 0.0
        return (-torch.log(numerator[valid] / denominator[valid])).mean()


class EncoderTrainer:
    """Training helper for joint CE and contrastive optimization."""

    def __init__(
        self,
        model: UnifiedEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        alpha: float = 0.7,
        beta: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | None = None,
        contrastive_epochs: int = 0,
        contrastive_lr: float = 1e-3,
        use_balanced_sampler: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.original_train_loader = train_loader
        self.val_loader = val_loader
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.contrastive_epochs = int(contrastive_epochs)
        self.contrastive_lr = float(contrastive_lr)
        self.use_balanced_sampler = bool(use_balanced_sampler)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss(temperature=0.1)
        self.history: list[dict[str, Any]] = []

    def _snapshot_state(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def _make_balanced_train_loader(self) -> DataLoader:
        labels = getattr(self.original_train_loader.dataset, "labels", None)
        if labels is None:
            raise ValueError("Balanced sampling requires a dataset with a labels attribute")
        if isinstance(labels, torch.Tensor):
            labels_array = labels.detach().cpu().numpy()
        else:
            labels_array = np.asarray(labels)

        batch_size = self.original_train_loader.batch_size
        if batch_size is None:
            raise ValueError("Balanced sampling requires train_loader.batch_size to be set")

        generator = torch.Generator()
        generator.manual_seed(RANDOM_STATE)
        sampler = ActionBalancedSampler(
            labels_array,
            batch_size=int(batch_size),
            min_per_class=2,
            generator=generator,
        )
        loader_kwargs: dict[str, Any] = {
            "batch_size": int(batch_size),
            "sampler": sampler,
            "drop_last": False,
            "num_workers": self.original_train_loader.num_workers,
            "pin_memory": self.original_train_loader.pin_memory,
            "collate_fn": self.original_train_loader.collate_fn,
        }
        if self.original_train_loader.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.original_train_loader.persistent_workers
        return DataLoader(self.original_train_loader.dataset, **loader_kwargs)

    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        self.model.train()
        train_total = 0.0
        train_ce = 0.0
        train_ctr = 0.0
        seen = 0

        for numeric, event_type, scenario, labels in loader:
            numeric = numeric.to(self.device)
            event_type = event_type.to(self.device)
            scenario = scenario.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            embeddings = self.model(numeric, event_type, scenario)
            logits = self.model.classifier(embeddings)
            ce = self.ce_loss(logits, labels)
            ctr = self.contrastive_loss(embeddings, labels)
            loss = self.alpha * ce + self.beta * ctr
            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            train_total += float(loss.item()) * batch_size
            train_ce += float(ce.item()) * batch_size
            train_ctr += float(ctr.item()) * batch_size
            seen += batch_size

        return {
            "train_loss": train_total / max(seen, 1),
            "train_ce_loss": train_ce / max(seen, 1),
            "train_contrastive_loss": train_ctr / max(seen, 1),
        }

    def train(self, epochs: int = 200, patience: int = 15) -> list[dict[str, Any]]:
        original_alpha = self.alpha
        original_beta = self.beta

        if self.contrastive_epochs > 0:
            self.alpha = 0.0
            self.beta = 1.0
            self.train_loader = (
                self._make_balanced_train_loader()
                if self.use_balanced_sampler
                else self.original_train_loader
            )
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.contrastive_lr,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.contrastive_epochs),
            )
            best_state: dict[str, torch.Tensor] | None = None
            best_val_loss = math.inf
            best_val_accuracy = 0.0

            for epoch in range(1, self.contrastive_epochs + 1):
                train_metrics = self._train_one_epoch(self.train_loader, optimizer)
                scheduler.step()
                val_metrics = self.evaluate(self.val_loader)
                val_loss = float(val_metrics["loss"])
                val_accuracy = float(val_metrics["accuracy"])

                metrics = {
                    "stage": 1,
                    "epoch": epoch,
                    "train_loss": train_metrics["train_contrastive_loss"],
                    "train_ce_loss": train_metrics["train_ce_loss"],
                    "train_contrastive_loss": train_metrics["train_contrastive_loss"],
                    "val_loss": val_loss,
                    "val_ce_loss": float(val_metrics["ce_loss"]),
                    "val_contrastive_loss": float(val_metrics["contrastive_loss"]),
                    "val_accuracy": val_accuracy,
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                self.history.append(metrics)
                print(
                    f"[Stage 1] Epoch {epoch:03d}/{self.contrastive_epochs:03d} | "
                    f"train_ctr_loss={metrics['train_contrastive_loss']:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_accuracy * 100:.2f}%"
                )

                epoch_state = self._snapshot_state()
                if val_loss + 1e-6 < best_val_loss:
                    best_val_loss = val_loss
                    best_val_accuracy = val_accuracy
                    best_state = epoch_state

            if best_state is not None:
                self.model.load_state_dict(best_state)
                self.model.to(self.device)
            print(f"Stage 1 complete: best val accuracy {best_val_accuracy * 100:.2f}%")

        self.alpha = original_alpha
        self.beta = original_beta
        self.train_loader = self.original_train_loader
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
        )

        best_state: dict[str, torch.Tensor] | None = None
        best_val_ce = math.inf
        stale_epochs = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_one_epoch(self.train_loader, optimizer)
            scheduler.step()
            val_metrics = self.evaluate(self.val_loader)
            train_loss = train_metrics["train_loss"]
            train_ce_loss = train_metrics["train_ce_loss"]
            train_ctr_loss = train_metrics["train_contrastive_loss"]
            val_ce = float(val_metrics["ce_loss"])

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ce_loss": train_ce_loss,
                "train_contrastive_loss": train_ctr_loss,
                "val_loss": float(val_metrics["loss"]),
                "val_ce_loss": val_ce,
                "val_contrastive_loss": float(val_metrics["contrastive_loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "lr": float(scheduler.get_last_lr()[0]),
            }
            if self.contrastive_epochs > 0:
                metrics["stage"] = 2
            self.history.append(metrics)
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} ce={train_ce_loss:.4f} ctr={train_ctr_loss:.4f} | "
                f"val_loss={metrics['val_loss']:.4f} val_ce={val_ce:.4f} "
                f"val_acc={metrics['val_accuracy'] * 100:.2f}%"
            )

            if val_ce + 1e-6 < best_val_ce:
                best_val_ce = val_ce
                stale_epochs = 0
                best_state = self._snapshot_state()
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"Early stopping at epoch {epoch} (best val CE {best_val_ce:.4f})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return self.history

    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_ce = 0.0
        total_ctr = 0.0
        total = 0
        correct = 0
        class_total = {idx: 0 for idx in range(self.model.num_classes)}
        class_correct = {idx: 0 for idx in range(self.model.num_classes)}

        with torch.no_grad():
            for numeric, event_type, scenario, labels in loader:
                numeric = numeric.to(self.device)
                event_type = event_type.to(self.device)
                scenario = scenario.to(self.device)
                labels = labels.to(self.device)

                embeddings = self.model(numeric, event_type, scenario)
                logits = self.model.classifier(embeddings)
                ce = self.ce_loss(logits, labels)
                ctr = self.contrastive_loss(embeddings, labels)
                loss = self.alpha * ce + self.beta * ctr

                preds = torch.argmax(logits, dim=1)
                batch_size = labels.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_ce += float(ce.item()) * batch_size
                total_ctr += float(ctr.item()) * batch_size
                total += batch_size
                correct += int((preds == labels).sum().item())

                for label_idx in range(self.model.num_classes):
                    mask = labels == label_idx
                    count = int(mask.sum().item())
                    if count:
                        class_total[label_idx] += count
                        class_correct[label_idx] += int((preds[mask] == label_idx).sum().item())

        per_class = {}
        for idx in range(self.model.num_classes):
            action = self.model.idx_to_action.get(idx, str(idx))
            count = class_total[idx]
            per_class[action] = {
                "accuracy": float(class_correct[idx] / count) if count else 0.0,
                "correct": int(class_correct[idx]),
                "total": int(count),
            }

        return {
            "loss": total_loss / max(total, 1),
            "ce_loss": total_ce / max(total, 1),
            "contrastive_loss": total_ctr / max(total, 1),
            "accuracy": correct / max(total, 1),
            "correct": correct,
            "total": total,
            "per_class_accuracy": per_class,
        }


def _prepare_arrays(
    rows: list[dict[str, Any]],
    scaler: StandardScaler,
    event_map: dict[str, int],
    scenario_map: dict[str, int],
    action_map: dict[str, int],
    *,
    fit_scaler: bool,
) -> PreparedArrays:
    numeric = _numeric_matrix(rows)
    numeric = scaler.fit_transform(numeric) if fit_scaler else scaler.transform(numeric)
    numeric = numeric.astype(np.float32)

    return PreparedArrays(
        numeric=numeric,
        event_type=np.array(
            [event_map.get(str(row.get("event_type", _lookup(row, "event_type", ""))), 0) for row in rows],
            dtype=np.int64,
        ),
        scenario=np.array(
            [scenario_map.get(str(row.get("scenario", _lookup(row, "scenario", ""))), 0) for row in rows],
            dtype=np.int64,
        ),
        labels=np.array([action_map[str(row["action"])] for row in rows], dtype=np.int64),
        case_ids=np.array([str(row.get("id", i)) for i, row in enumerate(rows)], dtype=object),
    )


def _query_to_tensors(
    model: UnifiedEncoder,
    query_features: dict[str, Any],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = np.array(
        [[_to_float(_lookup(query_features, feature)) for feature in NUMERIC_FEATURES]],
        dtype=np.float32,
    )
    if model.scaler_mean is not None and model.scaler_std is not None:
        mean = np.array(model.scaler_mean, dtype=np.float32)
        std = np.array(model.scaler_std, dtype=np.float32)
        std = np.where(std == 0.0, 1.0, std)
        values = (values - mean) / std

    event_type = str(query_features.get("event_type", _lookup(query_features, "event_type", "")))
    scenario = str(query_features.get("scenario", _lookup(query_features, "scenario", "")))
    event_idx = model.event_type_to_idx.get(event_type, 0)
    scenario_idx = model.scenario_to_idx.get(scenario, 0)

    return (
        torch.tensor(values, dtype=torch.float32, device=device),
        torch.tensor([event_idx], dtype=torch.long, device=device),
        torch.tensor([scenario_idx], dtype=torch.long, device=device),
    )


class NumpyIPIndex:
    """Small FAISS-like inner-product index used when faiss-cpu is unavailable."""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.ntotal = int(self.embeddings.shape[0])
        self.case_ids: np.ndarray | None = None

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sims = np.asarray(queries, dtype=np.float32) @ self.embeddings.T
        k = min(int(k), self.ntotal)
        if k <= 0:
            return np.empty((queries.shape[0], 0), dtype=np.float32), np.empty((queries.shape[0], 0), dtype=np.int64)
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-scores, axis=1)
        idx = np.take_along_axis(idx, order, axis=1).astype(np.int64)
        scores = np.take_along_axis(scores, order, axis=1).astype(np.float32)
        return scores, idx

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as f:
            np.save(f, self.embeddings)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyIPIndex":
        with Path(path).open("rb") as f:
            embeddings = np.load(f)
        return cls(embeddings)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (embeddings / norms).astype(np.float32)


def _try_import_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except Exception:
        return None


def build_index(
    model: UnifiedEncoder,
    cases_file: str | Path = DEFAULT_CASES_PATH,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    device: str = "cpu",
    batch_size: int = 512,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Build a cosine-similarity case index from model embeddings."""

    rows = _load_cases(cases_file)
    if not rows:
        raise ValueError(f"No cases found in {cases_file}")

    scaler = StandardScaler()
    if model.scaler_mean is None or model.scaler_std is None:
        raise ValueError("Model is missing scaler metadata; train or load a saved model first")
    scaler.mean_ = np.array(model.scaler_mean, dtype=np.float64)
    scaler.scale_ = np.array(model.scaler_std, dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(NUMERIC_FEATURES)

    arrays = _prepare_arrays(
        rows,
        scaler,
        model.event_type_to_idx,
        model.scenario_to_idx,
        model.action_to_idx,
        fit_scaler=False,
    )

    loader = DataLoader(CaseTensorDataset(arrays), batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for numeric, event_type, scenario, _labels in loader:
            numeric = numeric.to(device)
            event_type = event_type.to(device)
            scenario = scenario.to(device)
            emb = model(numeric, event_type, scenario).cpu().numpy()
            chunks.append(emb)
    embeddings = _normalize_embeddings(np.vstack(chunks))

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    case_ids_path = index_path.with_name(index_path.stem + "_ids.npy")
    np.save(case_ids_path, arrays.case_ids)

    faiss = _try_import_faiss()
    if faiss is not None:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(index_path))
        print(f"Saved FAISS index to {index_path}")
    else:
        index = NumpyIPIndex(embeddings)
        index.save(index_path)
        print(f"faiss-cpu is unavailable; saved numpy fallback index to {index_path}")

    try:
        index.case_ids = arrays.case_ids
        index.embeddings = embeddings
    except Exception:
        pass
    model.case_ids = arrays.case_ids
    return index, arrays.case_ids, embeddings


def load_index(index_path: str | Path = DEFAULT_INDEX_PATH) -> tuple[Any, np.ndarray]:
    """Load a FAISS index, or the numpy fallback index if FAISS is unavailable."""

    index_path = Path(index_path)
    case_ids = np.load(index_path.with_name(index_path.stem + "_ids.npy"), allow_pickle=True)
    faiss = _try_import_faiss()
    if faiss is not None:
        try:
            index = faiss.read_index(str(index_path))
        except Exception:
            index = NumpyIPIndex.load(index_path)
    else:
        index = NumpyIPIndex.load(index_path)
    try:
        index.case_ids = case_ids
    except Exception:
        pass
    return index, case_ids


def retrieve(
    model: UnifiedEncoder,
    index: Any,
    query_features: dict[str, Any],
    k: int = 5,
    device: str = "cpu",
) -> list[tuple[str, float, np.ndarray | None]]:
    """Retrieve top-k similar case ids, scores, and embeddings for a query."""

    model.to(device)
    model.eval()
    with torch.no_grad():
        numeric, event_type, scenario = _query_to_tensors(model, query_features, device)
        query_emb = model(numeric, event_type, scenario).cpu().numpy()
    query_emb = _normalize_embeddings(query_emb)

    scores, indices = index.search(query_emb.astype(np.float32), int(k))
    case_ids = getattr(index, "case_ids", model.case_ids)
    indexed_embeddings = getattr(index, "embeddings", None)
    results: list[tuple[str, float, np.ndarray | None]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        case_id = str(case_ids[idx]) if case_ids is not None and idx < len(case_ids) else str(int(idx))
        embedding = indexed_embeddings[idx] if indexed_embeddings is not None else None
        results.append((case_id, float(score), embedding))
    return results


def _print_xgboost_comparison(val_rows: list[dict[str, Any]], encoder_accuracy: float) -> None:
    try:
        from src.rag.fast_classifier import CascadeClassifier, DEFAULT_MODEL_PATH as XGB_PATH

        xgb_path = Path(XGB_PATH)
        if not xgb_path.exists():
            print("XGBoost comparison skipped: output/models/fast_classifier.json not found")
            return
        classifier = CascadeClassifier().load(xgb_path)
        correct = 0
        total = 0
        for row in val_rows:
            pred, _conf = classifier.predict(row)
            correct += int(pred == row.get("action"))
            total += 1
        xgb_acc = correct / max(total, 1)
        print(
            f"Validation comparison: unified_encoder={encoder_accuracy * 100:.2f}% "
            f"vs XGBoost={xgb_acc * 100:.2f}%"
        )
    except Exception as exc:
        print(f"XGBoost comparison skipped: {exc}")


def main() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading cases from {DEFAULT_CASES_PATH}")
    rows = _load_cases(DEFAULT_CASES_PATH, max_cases=30_000)
    if not rows:
        raise RuntimeError(f"No training rows loaded from {DEFAULT_CASES_PATH}")

    action_map = {action: idx for idx, action in enumerate(ACTION_CLASSES)}
    rows = [row for row in rows if str(row.get("action")) in action_map]
    labels = [action_map[str(row["action"])] for row in rows]

    train_rows, val_rows = train_test_split(
        rows,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    model = UnifiedEncoder()
    model.action_to_idx = action_map
    model.idx_to_action = {idx: action for action, idx in action_map.items()}
    model.event_type_to_idx = _fit_event_map(rows, model.event_type_size)
    model.scenario_to_idx = _fit_scenario_map(rows, model.scenario_size)

    scaler = StandardScaler()
    train_arrays = _prepare_arrays(
        train_rows,
        scaler,
        model.event_type_to_idx,
        model.scenario_to_idx,
        action_map,
        fit_scaler=True,
    )
    val_arrays = _prepare_arrays(
        val_rows,
        scaler,
        model.event_type_to_idx,
        model.scenario_to_idx,
        action_map,
        fit_scaler=False,
    )
    model.scaler_mean = scaler.mean_.astype(float).tolist()
    model.scaler_std = scaler.scale_.astype(float).tolist()

    contrastive_epochs = 50
    train_loader = DataLoader(
        CaseTensorDataset(train_arrays),
        batch_size=256,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        CaseTensorDataset(val_arrays),
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )

    trainer = EncoderTrainer(
        model,
        train_loader,
        val_loader,
        device=device,
        contrastive_epochs=contrastive_epochs,
        use_balanced_sampler=True,
    )
    trainer.train(epochs=200, patience=15)
    val_metrics = trainer.evaluate(val_loader)

    model.save_model(DEFAULT_MODEL_PATH)
    print(f"Saved model to {DEFAULT_MODEL_PATH}")
    build_index(model, DEFAULT_CASES_PATH, DEFAULT_INDEX_PATH, device=device)

    print(f"Encoder trained. Val accuracy: {val_metrics['accuracy'] * 100:.2f}%")
    print("Per-action accuracy:")
    for action, stats in val_metrics["per_class_accuracy"].items():
        print(
            f"  {action}: {stats['accuracy'] * 100:.2f}% "
            f"({stats['correct']}/{stats['total']})"
        )
    _print_xgboost_comparison(val_rows, float(val_metrics["accuracy"]))


if __name__ == "__main__":
    main()
