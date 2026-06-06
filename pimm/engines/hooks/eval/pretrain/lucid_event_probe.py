"""
Event-level linear probing hook for LUCiD SSL pretraining.
"""

from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Subset

import pimm.utils.comm as comm
from pimm.distributed import unwrap_model

from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase


def _get_writer_step(trainer):
    comm_info = getattr(trainer, "comm_info", {})
    if "epoch" in comm_info and "iter" in comm_info:
        return (
            comm_info.get("epoch", 0) * comm_info.get("iter_per_epoch", 0)
            + comm_info.get("iter", 0)
            + 1
        )
    return getattr(trainer, "epoch", 0)


@HOOKS.register_module()
class EventLinearProbeEvaluator(HookBase):
    """Evaluate event identity with frozen, mean-pooled pretraining features."""

    def __init__(
        self,
        label_key="event_label",
        every_n_steps=0,
        train_fraction=0.5,
        max_events_per_class=None,
        seed=0,
        prefix="event_probe",
        class_names=None,
        train_config=None,
        write_cls_metrics=True,
        require_heldout_data=True,
    ):
        self.label_key = label_key
        self.every_n_steps = int(every_n_steps)
        self.train_fraction = float(train_fraction)
        self.max_events_per_class = max_events_per_class
        self.seed = int(seed)
        self.prefix = str(prefix).strip("/")
        self.class_names = class_names
        self.train_config = train_config
        self.write_cls_metrics = bool(write_cls_metrics)
        self.require_heldout_data = bool(require_heldout_data)
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")

    def after_step(self):
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        if self.every_n_steps > 0:
            global_iter = (
                self.trainer.comm_info["iter"]
                + self.trainer.comm_info["iter_per_epoch"]
                * self.trainer.comm_info["epoch"]
            )
            if (global_iter + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        if self.every_n_steps == 0:
            self.eval()

    def _move_to_device(self, input_dict):
        """Move tensor batch entries to the evaluator device in place."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(device, non_blocking=True)
        return input_dict

    def _forward_point(self, input_dict):
        """Run the model and return the Point object required by the probe."""
        model = unwrap_model(self.trainer.model)
        use_amp = bool(getattr(self.trainer.cfg, "enable_amp", False))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amp_dtype = getattr(self.trainer.cfg, "amp_dtype", "bfloat16")
        dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16

        with torch.inference_mode():
            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    output = model(input_dict, return_point=True)
            else:
                output = model(input_dict, return_point=True)

        if isinstance(output, dict) and "point" in output:
            return output["point"]
        raise KeyError(
            "EventLinearProbeEvaluator expected model(..., return_point=True) "
            "to return a dict containing 'point'."
        )

    @staticmethod
    def _labels_by_event(labels, offsets, n_points):
        labels = labels.detach().long().view(-1)
        n_events = len(offsets) - 1
        if labels.numel() == n_events:
            return labels.cpu()
        if labels.numel() == n_points:
            event_labels = []
            for start, end in zip(offsets[:-1], offsets[1:]):
                event_labels.append(labels[start:end][0])
            return torch.stack(event_labels).cpu()
        raise ValueError(
            f"Cannot map {labels.numel()} labels to {n_events} events "
            f"and {n_points} points."
        )

    @staticmethod
    def _loader_dataset(loader):
        return getattr(loader, "dataset", None)

    @classmethod
    def _base_dataset(cls, dataset):
        while isinstance(dataset, Subset):
            dataset = dataset.dataset
        return dataset

    @classmethod
    def _dataset_split(cls, dataset):
        dataset = cls._base_dataset(dataset)
        split = getattr(dataset, "split", None)
        if split is None:
            return None
        if isinstance(split, (list, tuple, set)):
            return tuple(str(item) for item in split)
        return str(split)

    @staticmethod
    def _source_key(source, source_idx):
        if isinstance(source, dict):
            source_root = source.get("source_root") or source.get("data_root")
            if source_root is not None:
                return os.path.realpath(str(source_root))
            if source.get("name") is not None:
                return str(source["name"])
        return str(source_idx)

    @classmethod
    def _event_keys(cls, dataset):
        if dataset is None:
            return None

        subset_indices = None
        while isinstance(dataset, Subset):
            indices = [int(idx) for idx in dataset.indices]
            if subset_indices is None:
                subset_indices = indices
            else:
                subset_indices = [indices[idx] for idx in subset_indices]
            dataset = dataset.dataset

        data_list = getattr(dataset, "data_list", None)
        sources = getattr(dataset, "datasets", None)
        if data_list is None or sources is None or len(data_list) == 0:
            return None

        if subset_indices is None:
            indices = range(len(data_list))
        else:
            indices = subset_indices

        keys = set()
        base_len = len(data_list)
        for dataset_idx in indices:
            source_idx, event_idx = data_list[int(dataset_idx) % base_len]
            source_key = cls._source_key(sources[int(source_idx)], int(source_idx))
            keys.add((source_key, int(event_idx)))
        return keys

    @staticmethod
    def _format_event_key(key):
        source_key, event_idx = key
        return f"{source_key}:{event_idx}"

    def _validate_heldout_source(self):
        val_dataset = self._loader_dataset(self.trainer.val_loader)
        val_split = self._dataset_split(val_dataset)
        heldout_splits = {"holdout", "val", "test"}
        forbidden_splits = {"train", "all"}

        if val_split is None:
            raise RuntimeError(
                "EventLinearProbeEvaluator cannot verify heldout evaluation: "
                "val_loader.dataset has no split metadata."
            )

        val_split_set = {val_split} if isinstance(val_split, str) else set(val_split)
        if val_split_set & forbidden_splits or not val_split_set <= heldout_splits:
            raise RuntimeError(
                "EventLinearProbeEvaluator refuses to run on non-heldout data: "
                f"val_loader split={val_split!r}. Expected one of "
                f"{sorted(heldout_splits)}."
            )

        train_loader = getattr(self.trainer, "train_loader", None)
        train_dataset = self._loader_dataset(train_loader)
        train_keys = self._event_keys(train_dataset)
        val_keys = self._event_keys(val_dataset)
        if train_keys is None or val_keys is None:
            raise RuntimeError(
                "EventLinearProbeEvaluator cannot verify train/heldout "
                "disjointness for these datasets. Use a dataset exposing "
                "`data_list` and `datasets`, or disable require_heldout_data "
                "only for a deliberate diagnostic run."
            )

        overlap = train_keys & val_keys
        if overlap:
            examples = ", ".join(
                self._format_event_key(key) for key in sorted(overlap)[:5]
            )
            raise RuntimeError(
                "EventLinearProbeEvaluator detected train/heldout leakage: "
                f"{len(overlap)} events appear in both loaders. Examples: "
                f"{examples}"
            )

        self.trainer.logger.info(
            "Event probe data guard: val split=%s, train events=%d, "
            "heldout events=%d, overlap=0",
            val_split,
            len(train_keys),
            len(val_keys),
        )

    def _process_batch(self, input_dict):
        input_dict = self._move_to_device(input_dict)
        point = self._forward_point(input_dict)

        features = point.feat.float()
        point_offset = point["offset"] if "offset" in point.keys() else input_dict["offset"]
        offsets = [0] + point_offset.detach().cpu().tolist()
        labels = self._labels_by_event(
            input_dict[self.label_key], offsets, features.shape[0]
        )

        embeddings = []
        kept_labels = []
        for event_idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            if end <= start:
                continue
            embeddings.append(features[start:end].mean(dim=0).cpu())
            kept_labels.append(labels[event_idx])
        return embeddings, kept_labels

    def _collect_features(self):
        embeddings = []
        labels = []
        for batch_idx, input_dict in enumerate(self.trainer.val_loader):
            batch_embeddings, batch_labels = self._process_batch(input_dict)
            embeddings.extend(batch_embeddings)
            labels.extend(batch_labels)
            if (batch_idx + 1) % 10 == 0:
                self.trainer.logger.info(
                    "Event probe feature extraction: "
                    f"{batch_idx + 1}/{len(self.trainer.val_loader)} batches"
                )

        if not embeddings:
            raise RuntimeError("No event embeddings were collected from val_loader")
        X = torch.stack(embeddings, dim=0)
        y = torch.stack(labels, dim=0).long()
        return X, y

    def _split_train_test(self, X, y):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        train_indices = []
        test_indices = []
        classes = sorted(int(c) for c in y.unique().tolist())
        if len(classes) < 2:
            raise RuntimeError(
                "Event probe requires at least two classes in the holdout set; "
                f"got {classes}"
            )

        for class_id in classes:
            idx = torch.nonzero(y == class_id, as_tuple=False).flatten()
            idx = idx[torch.randperm(idx.numel(), generator=generator)]
            if self.max_events_per_class is not None:
                idx = idx[: int(self.max_events_per_class)]
            if idx.numel() < 2:
                raise RuntimeError(
                    f"Need at least two holdout events for class {class_id}, "
                    f"got {idx.numel()}"
                )
            n_train = int(round(idx.numel() * self.train_fraction))
            n_train = max(1, min(n_train, idx.numel() - 1))
            train_indices.append(idx[:n_train])
            test_indices.append(idx[n_train:])

        train_idx = torch.cat(train_indices, dim=0)
        test_idx = torch.cat(test_indices, dim=0)
        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def eval(self):
        if comm.get_rank() != 0:
            if comm.get_world_size() > 1:
                comm.synchronize()
            return

        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Event Linear Probe Evaluation >>>>>>>>>>>>>>>>"
        )
        self.trainer.model.eval()

        if self.require_heldout_data:
            self._validate_heldout_source()

        X, y = self._collect_features()
        X_train, y_train, X_test, y_test = self._split_train_test(X, y)
        self.trainer.logger.info(
            "Event probe heldout split: "
            f"probe_train={tuple(X_train.shape)}, "
            f"heldout_val={tuple(X_test.shape)}"
        )

        self._evaluate_probe(X_train, y_train, X_test, y_test)

        self.trainer.model.train()
        if comm.get_world_size() > 1:
            comm.synchronize()

    def _evaluate_probe(self, X_train, y_train, X_test, y_test):
        from .linear import LinearProbingTrainer

        class_names = self.class_names
        if class_names is None:
            class_names = getattr(self.trainer.cfg.data, "names", None)
        if class_names is None:
            num_classes = int(max(y_train.max(), y_test.max()).item()) + 1
            class_names = [str(i) for i in range(num_classes)]
        else:
            num_classes = max(
                len(class_names),
                int(max(y_train.max(), y_test.max()).item()) + 1,
            )

        probe = LinearProbingTrainer(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_classes=num_classes,
            logger=self.trainer.logger,
            config=self.train_config,
        )
        metrics = probe.train_and_evaluate()

        m_iou = metrics["m_iou"]
        m_precision = metrics["m_precision"]
        m_recall = metrics["m_recall"]
        m_f1 = metrics["m_f1"]
        iou_class = metrics["iou_class"]
        precision_class = metrics["precision_class"]
        recall_class = metrics["recall_class"]
        f1_class = metrics["f1_class"]
        cm = metrics["confusion_matrix"]
        support = metrics["class_support"]

        self.trainer.logger.info(
            "Event probe result: mIoU/mPrec/mRec/mF1 "
            f"{m_iou:.4f}/{m_precision:.4f}/{m_recall:.4f}/{m_f1:.4f}."
        )
        for class_id in range(num_classes):
            name = class_names[class_id] if class_id < len(class_names) else str(class_id)
            self.trainer.logger.info(
                "Event probe class "
                f"{class_id} ({name}): IoU={iou_class[class_id]:.4f}, "
                f"Precision={precision_class[class_id]:.4f}, "
                f"Recall={recall_class[class_id]:.4f}, "
                f"F1={f1_class[class_id]:.4f}, Support={support[class_id]}"
            )
        self.trainer.logger.info(
            "Event probe confusion matrix rows=true cols=pred:\n"
            + np.array2string(cm)
        )

        if self.trainer.writer is not None:
            step = _get_writer_step(self.trainer)
            prefix = f"{self.prefix}/" if self.prefix else ""
            self.trainer.writer.add_scalar(f"{prefix}val/mIoU", m_iou, step)
            self.trainer.writer.add_scalar(
                f"{prefix}val/mPrecision", m_precision, step
            )
            self.trainer.writer.add_scalar(f"{prefix}val/mRecall", m_recall, step)
            self.trainer.writer.add_scalar(f"{prefix}val/mF1", m_f1, step)
            if self.write_cls_metrics:
                for class_id in range(num_classes):
                    name = (
                        class_names[class_id]
                        if class_id < len(class_names)
                        else str(class_id)
                    )
                    self.trainer.writer.add_scalar(
                        f"{prefix}val/cls_{class_id}-{name} IoU",
                        iou_class[class_id],
                        step,
                    )
                    self.trainer.writer.add_scalar(
                        f"{prefix}val/cls_{class_id}-{name} F1",
                        f1_class[class_id],
                        step,
                    )

        metric_name = f"{self.prefix}/mF1" if self.prefix else "mF1"
        self.trainer.comm_info["current_metric_name"] = metric_name
        self.trainer.comm_info["current_metric_value"] = m_f1
        self.trainer.comm_info[f"{self.prefix}_current_metric_value"] = m_f1
        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<<< End Event Linear Probe Evaluation <<<<<<<<<<<<<<<<<"
        )
