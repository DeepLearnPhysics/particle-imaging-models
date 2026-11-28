"""Frozen point-feature probing and linear classifier training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pimm.models.losses import build_criteria
import pimm.utils.comm as comm
from pimm.distributed import unwrap_model
from pimm.engines.hooks.default import HookBase, get_writer_step, unsharded_val_loader
from pimm.engines.hooks.builder import HOOKS
from pimm.observability import structured_logger as sl

# Linear probe grid adapted from facebookresearch/dinov3, dinov3/eval/linear.py.
_DEFAULT_LR_LIST: Tuple[float, ...] = (
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    0.1,
)


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


def hold_out(X, y, fraction=0.2, seed=0):
    """Split training rows into rows to fit probe heads on and rows to choose among them, so test rows are only scored."""
    if len(X) < 2:
        raise ValueError("Held-out probe selection requires at least two training rows")
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    held = max(1, int(len(X) * fraction))
    return X[order[held:]], y[order[held:]], X[order[:held]], y[order[:held]]


@dataclass
class LinearProbingConfig:
    learning_rates: Sequence[float] = field(default_factory=lambda: _DEFAULT_LR_LIST)
    held_out_selection: bool = False
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

        self.X_train, self.y_train = X_train, y_train
        self.X_select, self.y_select = X_test, y_test
        self.X_test = X_test
        self.y_test = y_test
        self.num_classes = int(num_classes)
        self.logger = logger
        self.cfg = LinearProbingConfig(**config) if config is not None else LinearProbingConfig()  # type: ignore

        if self.cfg.held_out_selection:
            self.X_train, self.y_train, self.X_select, self.y_select = hold_out(
                X_train, y_train
            )

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
        Evaluate all classifiers on the configured selection rows and return the best name and per-head mean F1.
        """
        self.model.eval()
        X = self.X_select.to(self.device).contiguous()
        y = self.y_select.to(self.device).contiguous()

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
                self.logger.info(f"Head {name}: mf1={mf1 * 100.0:.2f}%")
            self.logger.info(f"Best head: {best_name} " f"mf1={best_mf1 * 100.0:.2f}%")

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
          - best_accuracy (float, 0-1): the chosen head's mean F1 on the selection rows
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
        # expose the best head's per-class softmax scores + labels on the test
        # set, so callers can compute ROC AUC / efficiency-at-rejection (the
        # fiTQun-comparable PID metrics) without re-running the probe.
        with torch.no_grad():
            self.model.eval()
            logits = self.model(self.X_test.to(self.device))[best_name]
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        metrics["test_scores"] = probs  # (N, num_classes)
        metrics["test_labels"] = self.y_test.cpu().numpy()
        return metrics


@HOOKS.register_module()
class PretrainEvaluator(HookBase):
    """Evaluate frozen pretraining features with downstream linear probes.

    On rank 0, extracts per-point backbone features from a budget of validation
    events (the first ``max_train_events`` for probe training, the next
    ``max_test_events`` for probe testing), then fits a grid of linear
    classifiers (via ``LinearProbingTrainer``) for each configured label and
    reports per-class and macro IoU / precision / recall / F1 plus a confusion
    matrix. Handles Sonata (teacher backbone), standard backbones (PTv3),
    fusion-export models, and models exposing ``encode()`` (Point-M2AE),
    unrolling any ``pooling_parent`` chain to recover point-resolution features.
    It publishes ``mF1`` of the (last-evaluated) label as the
    checkpoint-selection metric (``current_metric_value`` = ``m_f1``,
    ``current_metric_name`` = ``mF1``). Runs after every step when
    ``every_n_steps > 0`` (when ``(global_iter + 1) % every_n_steps == 0``),
    otherwise after each epoch; only when ``cfg.evaluate`` is true. Registered as
    ``PretrainEvaluator`` (use as ``type`` in a ``hooks=[...]`` entry).

    Args:
        label (str | Sequence): Label key(s) to probe; a string probes one
            label, a sequence probes several. An entry that is itself a
            tuple/list of keys, e.g. ``("segment_motif", "segment_pid")``,
            probes the JOINT label: each unique combination of the component
            values observed in the collected events becomes one probe class
            (indices are compacted over train+test together, so only
            combinations that actually occur become classes). The joint label
            is reported under the ``"+"``-joined name, e.g.
            ``segment_motif+segment_pid``. Defaults to ``"segment"``.
        write_cls_iou (bool): Also log per-class IoU/F1/precision/recall to the
            writer. Defaults to ``True``.
        every_n_steps (int): Step cadence; ``0`` evaluates once per epoch.
            Defaults to ``1``.
        max_train_events (int): Number of events used to fit the linear probe.
            Defaults to ``250``.
        max_test_events (int): Number of events used to evaluate the probe.
            Defaults to ``250``.
        class_names (list | dict | None): Class names; a list (shared across
            labels) or a per-label dict. Falls back to ``cfg.data.names``.
            For joint labels, provide a dict keyed by the COMPONENT keys
            (e.g. ``{"segment_motif": [...], "segment_pid": [...]}``); each
            joint class is named by joining its component names with ``"+"``
            (components without names fall back to their integer value).
            Defaults to ``None``.
        exclude_combinations (Sequence | dict | None): Joint-label
            combinations to drop from probing entirely (points with these
            combinations are removed from both the probe's train and test
            sets, so they are neither trained on nor scored). Each combination
            is a tuple with one entry per component, given as a class name
            (resolved via ``class_names``) or a raw integer value, e.g.
            ``[("track", "electron"), ("shower", "muon")]``. Pass a plain
            list to apply to every joint label, or a dict keyed by the
            joint label's ``"+"``-joined name. Ignored for non-joint labels.
            Defaults to ``None``.
        prefix (str): Writer/metric prefix; the ``segment`` label uses the root
            namespace. Defaults to ``""``.
        train_config (dict | None): Config forwarded to ``LinearProbingTrainer``
            (probe LR grid, epochs, etc.). Defaults to ``None``.
        feature_sampling (None | str | dict): How to turn the backbone's pooling
            hierarchy into point-resolution features. ``None`` (default) uses the
            legacy upcast concatenation. A string (``"trilinear"``, ``"nearest"``
            or ``"msm"``) or a dict ``{"method": ..., "levels": [...]}`` enables
            the MSM multi-resolution feature sampling of
            https://arxiv.org/abs/2504.06719 (``levels`` selects hierarchy
            levels, ``0`` = finest; ``None`` = all). ``"nearest"`` with all
            levels reproduces the default. Defaults to ``None``.
        force_fp32_features (bool): Disable AMP during feature extraction.
            Defaults to ``False``.

    Note:
        The selection metric is the linear-probe ``mF1`` (higher is better);
        ``current_metric_name`` is set when no selection value is present,
        while ``current_metric_value`` is overwritten each eval. Evaluation
        runs on rank 0 only; other ranks synchronize.

    Example:
        Add to ``cfg.hooks`` for SSL pretraining; it fits linear probes on frozen
        backbone features and sets the selection metric:

        .. code-block:: python

            hooks = [
                dict(type="PretrainEvaluator", label="segment",
                     every_n_steps=1000, max_train_events=250,
                     max_test_events=250),
            ]
            # → every 1000 steps (rank 0) trains a linear probe on 250 events / tests
            #   on the next 250 and logs  val/mIoU, val/mPrecision, val/mRecall,
            #   val/mF1  (+ per-class IoU/F1 with write_cls_iou), setting the
            #   checkpoint-selection metric to mF1
    """

    def __init__(
        self,
        label="segment",
        write_cls_iou=True,
        every_n_steps=1,
        max_train_events=250,
        max_test_events=250,
        class_names=None,
        exclude_combinations=None,
        prefix="",
        train_config=None,
        feature_sampling=None,
        force_fp32_features=False,
    ):
        """Configure label heads, feature budget, cadence, and probe trainer."""
        self.write_cls_iou = write_cls_iou
        self.every_n_steps = every_n_steps
        self.max_train_events = max_train_events
        self.max_test_events = max_test_events
        self.prefix = prefix
        self.feature_sampling = feature_sampling
        self.force_fp32_features = bool(force_fp32_features)
        raw_labels = [label] if isinstance(label, str) else list(label)
        self.labels = []
        self.joint_components = {}
        self._fetch_labels = []
        for entry in raw_labels:
            if isinstance(entry, str):
                name, components = entry, (entry,)
            else:
                components = tuple(entry)
                name = "+".join(components)
                self.joint_components[name] = components
            self.labels.append(name)
            for key in components:
                if key not in self._fetch_labels:
                    self._fetch_labels.append(key)
        self.train_config = train_config

        # support per-label class_names
        # class_names can be:
        # - None: use default names from cfg.data.names
        # - list: same names for all labels
        # - dict: {label_name: names_list} for per-label names
        if class_names is None or not isinstance(class_names, dict):
            # single set of names or None - apply to all labels
            self.class_names_dict = {
                label_name: class_names for label_name in self.labels
            }
        else:
            # dict of per-label names
            self.class_names_dict = class_names

        # Resolve named exclusions before feature extraction starts.
        self.exclude_combinations = {}
        if exclude_combinations:
            for name, components in self.joint_components.items():
                spec = (
                    exclude_combinations.get(name, [])
                    if isinstance(exclude_combinations, dict)
                    else exclude_combinations
                )
                resolved = []
                for combo in spec:
                    assert len(combo) == len(
                        components
                    ), f"exclusion {combo} does not match components {components}"
                    values = []
                    for key, value in zip(components, combo):
                        if isinstance(value, str):
                            names = self.class_names_dict.get(key)
                            assert names is not None and value in list(names), (
                                f"cannot resolve class name '{value}' for "
                                f"component '{key}'; provide class_names[{key!r}]"
                            )
                            value = list(names).index(value)
                        values.append(int(value))
                    resolved.append(tuple(values))
                self.exclude_combinations[name] = resolved

    def after_step(self):
        """Run probing evaluation on the configured step cadence."""
        if self.trainer.cfg.evaluate and self.every_n_steps > 0:
            global_iter = (
                self.trainer.comm_info["iter"]
                + self.trainer.comm_info["iter_per_epoch"]
                * self.trainer.comm_info["epoch"]
            )
            if (global_iter + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        """Run probing evaluation after epochs when step cadence is disabled."""
        if self.trainer.cfg.evaluate and self.every_n_steps == 0:
            self.eval()

    def get_backbone(self):
        """Return the backbone module used to produce probe features."""
        model = unwrap_model(self.trainer.model)
        if hasattr(model, "teacher"):  # sonata
            return model.teacher["backbone"]
        elif hasattr(model, "backbone"):  # else
            return model.backbone
        else:
            raise ValueError(f"Model {model} has no backbone")

    def _has_encode(self):
        """Check if the model has a direct encode() method (e.g. Point-M2AE)."""
        model = unwrap_model(self.trainer.model)
        return hasattr(model, "encode") and callable(model.encode)

    def _uses_exported_point(self):
        """Fusion-enabled Sonata exports features outside the backbone."""
        model = unwrap_model(self.trainer.model)
        return bool(getattr(model, "representation_fusion_enabled", False))

    def _process_batch_with_offsets(self, input_dict):
        """Process a batch and extract features properly using offsets to handle multiple events"""
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        mode = torch.no_grad if self.force_fp32_features else torch.inference_mode
        with mode():
            if self._has_encode():
                # Models with encode() (e.g. Point-M2AE): returns packed (N, C) tensor
                model = unwrap_model(self.trainer.model)
                if getattr(self.trainer.cfg, "enable_amp", False):
                    amp_dtype = getattr(self.trainer.cfg, "amp_dtype", "bfloat16")
                    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        features = model.encode(input_dict).float()
                else:
                    features = model.encode(input_dict).float()

                offsets = [0] + input_dict["offset"].cpu().tolist()

                all_labels = {}
                for label_name in self._fetch_labels:
                    all_labels[label_name] = input_dict[label_name].squeeze(-1)

                batch_features = []
                batch_labels_dict = {
                    label_name: [] for label_name in self._fetch_labels
                }
                for i in range(len(offsets) - 1):
                    start_idx = offsets[i]
                    end_idx = offsets[i + 1]
                    batch_features.append(features[start_idx:end_idx].cpu())
                    for label_name in self._fetch_labels:
                        batch_labels_dict[label_name].append(
                            all_labels[label_name][start_idx:end_idx].cpu()
                        )
                return batch_features, batch_labels_dict

            enable_amp = (
                getattr(self.trainer.cfg, "enable_amp", False)
                and not self.force_fp32_features
            )
            amp_dtype = getattr(self.trainer.cfg, "amp_dtype", "bfloat16")
            dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
            with torch.amp.autocast(
                device_type="cuda", enabled=enable_amp, dtype=dtype
            ):
                if self._uses_exported_point():
                    point = unwrap_model(self.trainer.model)(
                        input_dict, return_point=True
                    )["point"]
                else:
                    point = self.get_backbone()(input_dict)
            point = point.upcast(feature_sampling=self.feature_sampling)

        # Get features and offset information
        features = point.feat.float()  # [N, C]
        point_offset = (
            point["offset"] if "offset" in point.keys() else input_dict["offset"]
        )
        offsets = [0] + point_offset.cpu().tolist()  # Batch offsets

        # Extract all label types
        all_labels = {}
        for label_name in self._fetch_labels:
            if label_name in point.keys():
                all_labels[label_name] = point[label_name].squeeze(-1)  # [N]
            else:
                all_labels[label_name] = input_dict[label_name].squeeze(-1)  # [N]

        # Process features by batch using offsets
        batch_features = []
        batch_labels_dict = {label_name: [] for label_name in self._fetch_labels}

        # Use offsets to separate points from different events in the batch
        for i in range(len(offsets) - 1):
            start_idx = offsets[i]
            end_idx = offsets[i + 1]

            # Extract features for this event
            event_features = features[start_idx:end_idx]
            batch_features.append(event_features.cpu())
            # Extract labels for each label type
            for label_name in self._fetch_labels:
                event_labels = all_labels[label_name][start_idx:end_idx]
                batch_labels_dict[label_name].append(event_labels.cpu())

        return batch_features, batch_labels_dict

    @sl.log_trace_span("evaluation.pretrain_semantic_segmentation")
    def eval(self):
        """Collect event features, train probes, and log downstream metrics."""
        sl.add_step_tag("evaluation")
        # only run on rank 0
        rank = comm.get_rank()
        if rank != 0:
            # wait for rank 0 to finish evaluation
            if comm.get_world_size() > 1:
                comm.synchronize()
            return

        self.trainer.model.eval()
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        # collect features and labels from events
        train_features = []
        train_labels_dict = {label_name: [] for label_name in self._fetch_labels}
        test_features = []
        test_labels_dict = {label_name: [] for label_name in self._fetch_labels}

        event_count = 0
        total_events_needed = self.max_train_events + self.max_test_events

        for i, input_dict in enumerate(unsharded_val_loader(self.trainer)):
            self.trainer.logger.info(f"Processing batch {i}")
            batch_features, batch_labels_dict = self._process_batch_with_offsets(
                input_dict
            )

            for event_idx, event_features in enumerate(batch_features):
                if event_count < self.max_train_events:
                    train_features.append(event_features)
                    for label_name in self._fetch_labels:
                        train_labels_dict[label_name].append(
                            batch_labels_dict[label_name][event_idx]
                        )
                elif event_count < total_events_needed:
                    test_features.append(event_features)
                    for label_name in self._fetch_labels:
                        test_labels_dict[label_name].append(
                            batch_labels_dict[label_name][event_idx]
                        )
                else:
                    break

                event_count += 1

            if event_count >= total_events_needed:
                break

        # truncate to exact counts
        train_features = train_features[: self.max_train_events]
        test_features = test_features[: self.max_test_events]
        for label_name in self._fetch_labels:
            train_labels_dict[label_name] = train_labels_dict[label_name][
                : self.max_train_events
            ]
            test_labels_dict[label_name] = test_labels_dict[label_name][
                : self.max_test_events
            ]

        X_train = torch.cat(train_features, dim=0)
        X_test = torch.cat(test_features, dim=0)

        self.trainer.logger.info(
            f"Train events: {len(train_features)}, Test events: {len(test_features)}"
        )
        self.trainer.logger.info(
            f"Train features: {X_train.shape}, Test features: {X_test.shape}"
        )

        for label_name in self.labels:
            self.trainer.logger.info(
                f"\n{'=' * 60}\nEvaluating label: {label_name}\n{'=' * 60}"
            )

            X_train_label, X_test_label = X_train, X_test
            if label_name in self.joint_components:
                (
                    y_train,
                    y_test,
                    label_class_names,
                    train_keep,
                    test_keep,
                ) = self._build_joint_label(
                    label_name, train_labels_dict, test_labels_dict
                )
                if train_keep is not None:
                    n_dropped = int((~train_keep).sum() + (~test_keep).sum())
                    self.trainer.logger.info(
                        f"Joint label {label_name}: excluded {n_dropped} points "
                        f"matching {self.exclude_combinations[label_name]}"
                    )
                    X_train_label = X_train[train_keep]
                    X_test_label = X_test[test_keep]
                self.trainer.logger.info(
                    f"Joint label {label_name}: {len(label_class_names)} "
                    f"observed combinations: {label_class_names}"
                )
            else:
                y_train = torch.cat(train_labels_dict[label_name], dim=0)
                y_test = torch.cat(test_labels_dict[label_name], dim=0)
                label_class_names = self.class_names_dict.get(label_name, None)

            if len(self.labels) > 1:
                eval_prefix = (
                    label_name if not self.prefix else f"{self.prefix}_{label_name}"
                )
            else:
                eval_prefix = self.prefix if self.prefix else label_name

            self._evaluate_single_label(
                X_train_label,
                y_train,
                X_test_label,
                y_test,
                eval_prefix,
                label_class_names,
            )

        self.trainer.model.train()

        # signal other ranks that evaluation is complete
        if comm.get_world_size() > 1:
            comm.synchronize()

    def _build_joint_label(self, label_name, train_labels_dict, test_labels_dict):
        """Compact-encode each unique combination of component labels as one class.

        The mapping is built over train+test together, so class indices are
        consistent between the two splits and only combinations that actually
        occur become classes. Combination order (and thus index order) follows
        ``torch.unique``'s lexicographic sort of the component values, which is
        deterministic for a fixed validation set.

        Points whose combination is listed in ``exclude_combinations`` are
        dropped: the returned keep masks (aligned with the unfiltered
        train/test point order, or ``None`` when nothing is excluded) must be
        applied to the feature matrices, and excluded combinations never
        become classes.
        """
        components = self.joint_components[label_name]
        columns = [
            torch.cat(train_labels_dict[key] + test_labels_dict[key], dim=0)
            .reshape(-1)
            .long()
            for key in components
        ]
        stacked = torch.stack(columns, dim=1)
        n_train = sum(t.numel() for t in train_labels_dict[components[0]])

        excluded = self.exclude_combinations.get(label_name, [])
        train_keep = test_keep = None
        if excluded:
            keep = torch.ones(stacked.shape[0], dtype=torch.bool)
            for combo in excluded:
                keep &= ~(stacked == torch.tensor(combo)).all(dim=1)
            train_keep, test_keep = keep[:n_train], keep[n_train:]
            stacked = stacked[keep]
            n_train = int(train_keep.sum())

        combos, joint = torch.unique(stacked, dim=0, return_inverse=True)
        component_names = [self.class_names_dict.get(key) for key in components]
        class_names = [
            "+".join(
                (
                    str(names[value])
                    if names is not None and 0 <= value < len(names)
                    else str(value)
                )
                for names, value in zip(component_names, row)
            )
            for row in combos.tolist()
        ]
        return joint[:n_train], joint[n_train:], class_names, train_keep, test_keep

    def _evaluate_single_label(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        eval_prefix,
        label_class_names,
    ):
        """Train and evaluate a grid of linear classifiers for a single label type."""

        # Use provided class names or fall back to default
        if label_class_names is None:
            label_class_names = self.trainer.cfg.data.names

        num_classes = int(y_train.max().item()) + 1
        if self.joint_components:
            num_classes = int(max(y_train.max().item(), y_test.max().item())) + 1

        trainer = LinearProbingTrainer(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_classes=num_classes,
            logger=self.trainer.logger,
            config=self.train_config,
        )
        metrics = trainer.train_and_evaluate()

        m_iou = metrics["m_iou"]
        m_precision = metrics["m_precision"]
        m_recall = metrics["m_recall"]
        m_f1 = metrics["m_f1"]
        iou_class = metrics["iou_class"]
        precision_class = metrics["precision_class"]
        recall_class = metrics["recall_class"]
        f1_class = metrics["f1_class"]
        cm = metrics["confusion_matrix"]
        class_support = metrics["class_support"]

        self.trainer.storage.put_scalar(
            f"{eval_prefix}_val_intersection", iou_class * (class_support + 1e-10)
        )
        self.trainer.storage.put_scalar(
            f"{eval_prefix}_val_union", (class_support + 1e-10)
        )
        self.trainer.storage.put_scalar(f"{eval_prefix}_val_target", class_support)

        self.trainer.logger.info(
            "Val result: mIoU/mPrec/mRec/mF1 {:.4f}/{:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_precision, m_recall, m_f1
            )
        )

        try:
            from rich.table import Table
            from rich.console import Console
        except ImportError:
            Table = None

        if Table is not None:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("ClassIdx", justify="right")
            table.add_column("Name")
            table.add_column("IoU", justify="right")
            table.add_column("Precision", justify="right")
            table.add_column("Recall", justify="right")
            table.add_column("F1", justify="right")
            table.add_column("Support", justify="right")

            for i in range(num_classes):
                table.add_row(
                    str(i),
                    str(label_class_names[i]),
                    f"{iou_class[i]:.4f}",
                    f"{precision_class[i]:.4f}",
                    f"{recall_class[i]:.4f}",
                    f"{f1_class[i]:.4f}",
                    str(class_support[i]),
                )

            console = Console(file=None, width=100)
            console.print(table)

        import pandas as pd

        label_names = [str(n) for n in label_class_names[:num_classes]]
        cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
        self.trainer.logger.info(
            "Confusion Matrix (rows=true, cols=pred):\n" + cm_df.to_string()
        )

        _prefix = eval_prefix
        eval_prefix = eval_prefix + "/"
        if eval_prefix == "segment/":
            eval_prefix = ""

        if self.trainer.writer is not None:
            step = get_writer_step(self.trainer)
            self.trainer.writer.add_scalar(f"{eval_prefix}val/mIoU", m_iou, step)
            self.trainer.writer.add_scalar(
                f"{eval_prefix}val/mPrecision", m_precision, step
            )
            self.trainer.writer.add_scalar(f"{eval_prefix}val/mRecall", m_recall, step)
            self.trainer.writer.add_scalar(f"{eval_prefix}val/mF1", m_f1, step)

            if self.write_cls_iou:
                for i in range(num_classes):
                    self.trainer.writer.add_scalar(
                        f"{eval_prefix}val/cls_{i}-{label_class_names[i]} IoU",
                        iou_class[i],
                        step,
                    )
                    self.trainer.writer.add_scalar(
                        f"{eval_prefix}val/cls_{i}-{label_class_names[i]} F1",
                        f1_class[i],
                        step,
                    )
                    self.trainer.writer.add_scalar(
                        f"{eval_prefix}val/cls_{i}-{label_class_names[i]} Precision",
                        precision_class[i],
                        step,
                    )
                    self.trainer.writer.add_scalar(
                        f"{eval_prefix}val/cls_{i}-{label_class_names[i]} Recall",
                        recall_class[i],
                        step,
                    )

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        if "current_metric_value" not in self.trainer.comm_info:
            self.trainer.comm_info["current_metric_name"] = "mF1"
        self.trainer.comm_info["current_metric_value"] = m_f1
        self.trainer.comm_info[f"{_prefix}_current_metric_value"] = m_f1

        # Stash for cross-hook readout (e.g. AutoresearchSummary).
        if not hasattr(self, "last_metrics_by_label"):
            self.last_metrics_by_label = {}
        self.last_metrics_by_label[_prefix or "segment"] = {
            "mF1": float(m_f1),
            "mIoU": float(m_iou),
            "mPrecision": float(m_precision),
            "mRecall": float(m_recall),
        }
