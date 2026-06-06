"""
Linear probing utilities for PretrainEvaluator.

This module implements a small grid search over linear classifiers with different
learning rates, based on DINOv3's eval/linear.py, but operates directly on
pre-extracted feature tensors provided by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pimm.models.losses import build_criteria
from pimm.utils import comm

_DEFAULT_LR_LIST: Tuple[float, ...] = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1)

class LinearClassifier(nn.Module):
    """Simple linear classifier used for probing."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class AllClassifiers(nn.Module):
    """
    Container for multiple linear classifiers.

    Forward returns a dict[name -> logits].
    """

    def __init__(self, classifiers: Dict[str, LinearClassifier]) -> None:
        super().__init__()
        self.classifiers = nn.ModuleDict(classifiers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {k: m(x) for k, m in self.classifiers.items()}

    def __len__(self) -> int:
        return len(self.classifiers)


@dataclass
class LinearProbingConfig:
    learning_rates: Sequence[float] = field(default_factory=lambda: _DEFAULT_LR_LIST)
    epochs: int = 10
    batch_size: int = 32768
    weight_decay: float = 0.01
    device: Optional[torch.device] = None
    criteria: dict = field(default_factory=lambda: dict(type="CrossEntropyLoss"))

class LinearProbingTrainer:
    """
    Train a grid of linear classifiers on top of frozen features.

    This class assumes that feature extraction has already been performed and
    receives (X_train, y_train, X_test, y_test) tensors directly.
    """

    def __init__(
        self,
        *,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        num_classes: int,
        logger,
        config: Optional[dict] = None,
    ) -> None:
        assert X_train.ndim == 2, "X_train must be 2D [N, D]"
        assert X_test.ndim == 2, "X_test must be 2D [M, D]"
        assert X_train.size(1) == X_test.size(1), "Train/test feature dims must match"

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_classes = int(num_classes)
        self.logger = logger
        self.cfg = LinearProbingConfig(**config) if config is not None else LinearProbingConfig()  # type: ignore

        self.device = (
            self.cfg.device
            if self.cfg.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.world_size = comm.get_world_size()

        self._build_model()

    def _build_model(self) -> None:
        in_dim = int(self.X_train.size(1))

        classifiers: Dict[str, LinearClassifier] = {}
        param_groups: List[Dict[str, object]] = []
        for base_lr in self.cfg.learning_rates:
            name = f"lr_{base_lr:.6f}".replace(".", "_")
            clf = LinearClassifier(in_dim, self.num_classes)
            classifiers[name] = clf
            param_groups.append({"params": clf.parameters(), "lr": base_lr})

        self.model = AllClassifiers(classifiers).to(self.device)

        # We keep training single-process here
        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.cfg.weight_decay
        )
        self.criterion = build_criteria(self.cfg.criteria)

        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

    def _train(self) -> None:
        self.model.train()
        for epoch in range(self.cfg.epochs):
            running_loss = 0.0
            n_samples = 0
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device).contiguous()
                batch_y = batch_y.to(self.device).contiguous()

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                # Sum loss over all classifiers
                loss = 0.0
                for logits in outputs.values():
                    loss = loss + self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()

                running_loss += float(loss.item()) * batch_x.size(0)
                n_samples += batch_x.size(0)

            if self.logger is not None and n_samples > 0:
                avg_loss = running_loss / n_samples
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.cfg.epochs} "
                    f"Train Loss: {avg_loss:.4f}"
                )

    @torch.no_grad()
    def _evaluate_grid(self) -> Tuple[str, Dict[str, float]]:
        """
        Evaluate all classifiers on X_test and return best name and per-head mean F1.
        """
        self.model.eval()
        X = self.X_test.to(self.device).contiguous()
        y = self.y_test.to(self.device).contiguous()

        outputs = self.model(X)  # dict[name -> logits]
        mf1s: Dict[str, float] = {}
        best_name = ""
        best_mf1 = -1.0

        for name, logits in outputs.items():
            preds = logits.argmax(dim=-1)
            preds_np = preds.cpu().numpy()
            y_np = y.cpu().numpy()
            num_classes = self.num_classes

            f1_class = []
            for c in range(num_classes):
                pred_c = preds_np == c
                gt_c = y_np == c
                tp = np.logical_and(pred_c, gt_c).sum()
                fp = np.logical_and(pred_c, np.logical_not(gt_c)).sum()
                fn = np.logical_and(np.logical_not(pred_c), gt_c).sum()
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                f1_class.append(f1)
            mf1 = float(np.mean(f1_class))

            mf1s[name] = mf1
            if mf1 > best_mf1:
                best_mf1 = mf1
                best_name = name

        if self.logger is not None:
            for name, mf1 in mf1s.items():
                self.logger.info(
                    f"Head {name}: mf1={mf1 * 100.0:.2f}%"
                )
            self.logger.info(
                f"Best head: {best_name} "
                f"mf1={best_mf1 * 100.0:.2f}%"
            )

        return best_name, mf1s

    @torch.no_grad()
    def _compute_metrics_for_head(self, head_name: str) -> Dict[str, object]:
        """
        Compute IoU / precision / recall / F1 and confusion matrix for a given head.
        """
        self.model.eval()
        X = self.X_test.to(self.device)
        y = self.y_test.to(self.device)
        logits = self.model(X)[head_name]
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels = y.cpu().numpy()

        num_classes = self.num_classes

        # Per-class precision/recall/F1
        precision_class = np.zeros(num_classes, dtype=np.float64)
        recall_class = np.zeros(num_classes, dtype=np.float64)
        f1_class = np.zeros(num_classes, dtype=np.float64)
        intersection = np.zeros(num_classes, dtype=np.float64)
        union = np.zeros(num_classes, dtype=np.float64)

        for c in range(num_classes):
            pred_c = preds == c
            gt_c = labels == c
            if gt_c.sum() > 0 or pred_c.sum() > 0:
                tp = np.logical_and(pred_c, gt_c).sum()
                fp = np.logical_and(pred_c, np.logical_not(gt_c)).sum()
                fn = np.logical_and(np.logical_not(pred_c), gt_c).sum()

                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)

                precision_class[c] = precision
                recall_class[c] = recall
                f1_class[c] = f1

            inter = np.logical_and(pred_c, gt_c).sum()
            un = np.logical_or(pred_c, gt_c).sum()
            intersection[c] = inter
            union[c] = un

        iou_class = intersection / (union + 1e-10)

        m_precision = float(np.mean(precision_class))
        m_recall = float(np.mean(recall_class))
        m_f1 = float(np.mean(f1_class))
        m_iou = float(np.mean(iou_class))

        # Confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(labels, preds):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[int(t), int(p)] += 1

        class_support = cm.sum(axis=1)

        return {
            "m_iou": m_iou,
            "m_precision": m_precision,
            "m_recall": m_recall,
            "m_f1": m_f1,
            "iou_class": iou_class,
            "precision_class": precision_class,
            "recall_class": recall_class,
            "f1_class": f1_class,
            "confusion_matrix": cm,
            "class_support": class_support,
        }

    def train_and_evaluate(self) -> Dict[str, object]:
        """
        Run training and evaluation, returning a metrics dict.

        Returns keys:
          - best_classifier (str)
          - best_accuracy (float, 0-1)
          - m_iou, m_precision, m_recall, m_f1 (floats)
          - iou_class, precision_class, recall_class, f1_class (np.ndarray)
          - confusion_matrix (np.ndarray)
          - class_support (np.ndarray)
        """
        self._train()
        best_name, accuracies = self._evaluate_grid()
        metrics = self._compute_metrics_for_head(best_name)

        best_acc = float(accuracies.get(best_name, 0.0))
        metrics.update(
            {
                "best_classifier": best_name,
                "best_accuracy": best_acc,
            }
        )
        return metrics


