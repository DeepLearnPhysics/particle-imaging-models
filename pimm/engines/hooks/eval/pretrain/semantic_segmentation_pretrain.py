import torch
import pimm.utils.comm as comm
from pimm.distributed import unwrap_model
from pimm.engines.hooks.default import HookBase
from pimm.engines.hooks.builder import HOOKS

def _get_writer_step(trainer):
    """Local train step for logging; writer applies any configured offset."""
    ci = getattr(trainer, "comm_info", {})
    if "epoch" in ci and "iter" in ci:
        return ci.get("epoch", 0) * ci.get("iter_per_epoch", 0) + ci.get("iter", 0) + 1
    return getattr(trainer, "epoch", 0)

@HOOKS.register_module()
class PretrainEvaluator(HookBase):
    """Evaluate frozen pretraining features with downstream linear probes."""

    def __init__(
        self,
        label="segment",
        write_cls_iou=True,
        every_n_steps=1,
        max_train_events=250,
        max_test_events=250,
        class_names=None,
        prefix="",
        train_config=None,
    ):
        """Configure label heads, feature budget, cadence, and probe trainer."""
        self.write_cls_iou = write_cls_iou
        self.every_n_steps = every_n_steps
        self.max_train_events = max_train_events
        self.max_test_events = max_test_events
        self.prefix = prefix
        # support both single label and multiple labels
        self.labels = [label] if isinstance(label, str) else list(label)
        self.train_config = train_config
        
        # support per-label class_names
        # class_names can be:
        # - None: use default names from cfg.data.names
        # - list: same names for all labels
        # - dict: {label_name: names_list} for per-label names
        if class_names is None or not isinstance(class_names, dict):
            # single set of names or None - apply to all labels
            self.class_names_dict = {label_name: class_names for label_name in self.labels}
        else:
            # dict of per-label names
            self.class_names_dict = class_names
        
    def after_step(self):
        """Run probing evaluation on the configured step cadence."""
        if self.trainer.cfg.evaluate and self.every_n_steps > 0:
            global_iter = self.trainer.comm_info['iter'] + self.trainer.comm_info['iter_per_epoch'] * self.trainer.comm_info['epoch']
            if (global_iter + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        """Run probing evaluation after epochs when step cadence is disabled."""
        if self.trainer.cfg.evaluate and self.every_n_steps == 0:
            self.eval()

    def get_backbone(self):
        """Return the backbone module used to produce probe features."""
        model = unwrap_model(self.trainer.model)
        if hasattr(model, "teacher"): # sonata
            return model.teacher["backbone"]
        elif hasattr(model, "backbone"): # else
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

        with torch.inference_mode():
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
                for label_name in self.labels:
                    all_labels[label_name] = input_dict[label_name].squeeze(-1)

                batch_features = []
                batch_labels_dict = {label_name: [] for label_name in self.labels}
                for i in range(len(offsets) - 1):
                    start_idx = offsets[i]
                    end_idx = offsets[i + 1]
                    batch_features.append(features[start_idx:end_idx].cpu())
                    for label_name in self.labels:
                        batch_labels_dict[label_name].append(
                            all_labels[label_name][start_idx:end_idx].cpu()
                        )
                return batch_features, batch_labels_dict

            # Standard backbone path (Sonata, PTv3, etc.)
            if getattr(self.trainer.cfg, "enable_amp", False):
                amp_dtype = getattr(self.trainer.cfg, "amp_dtype", "bfloat16")
                dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    if self._uses_exported_point():
                        point = unwrap_model(self.trainer.model)(
                            input_dict, return_point=True
                        )["point"]
                    else:
                        point = self.get_backbone()(input_dict)
            else:
                if self._uses_exported_point():
                    point = unwrap_model(self.trainer.model)(
                        input_dict, return_point=True
                    )["point"]
                else:
                    point = self.get_backbone()(input_dict)
            while "pooling_parent" in point.keys():
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent

        # Get features and offset information
        features = point.feat.float()  # [N, C]
        point_offset = point["offset"] if "offset" in point.keys() else input_dict["offset"]
        offsets = [0] + point_offset.cpu().tolist()  # Batch offsets

        # Extract all label types
        all_labels = {}
        for label_name in self.labels:
            if label_name in point.keys():
                all_labels[label_name] = point[label_name].squeeze(-1)  # [N]
            else:
                all_labels[label_name] = input_dict[label_name].squeeze(-1)  # [N]

        # Process features by batch using offsets
        batch_features = []
        batch_labels_dict = {label_name: [] for label_name in self.labels}

        # Use offsets to separate points from different events in the batch
        for i in range(len(offsets) - 1):
            start_idx = offsets[i]
            end_idx = offsets[i + 1]

            # Extract features for this event
            event_features = features[start_idx:end_idx]
            batch_features.append(event_features.cpu())
            # Extract labels for each label type
            for label_name in self.labels:
                event_labels = all_labels[label_name][start_idx:end_idx]
                batch_labels_dict[label_name].append(event_labels.cpu())

        return batch_features, batch_labels_dict

    def eval(self):
        """Collect event features, train probes, and log downstream metrics."""
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
        train_labels_dict = {label_name: [] for label_name in self.labels}
        test_features = []
        test_labels_dict = {label_name: [] for label_name in self.labels}

        event_count = 0
        total_events_needed = self.max_train_events + self.max_test_events

        for i, input_dict in enumerate(self.trainer.val_loader):
            self.trainer.logger.info(f"Processing batch {i}")
            batch_features, batch_labels_dict = self._process_batch_with_offsets(
                input_dict
            )

            for event_idx, event_features in enumerate(batch_features):
                if event_count < self.max_train_events:
                    train_features.append(event_features)
                    for label_name in self.labels:
                        train_labels_dict[label_name].append(
                            batch_labels_dict[label_name][event_idx]
                        )
                elif event_count < total_events_needed:
                    test_features.append(event_features)
                    for label_name in self.labels:
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
        for label_name in self.labels:
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

            y_train = torch.cat(train_labels_dict[label_name], dim=0)
            y_test = torch.cat(test_labels_dict[label_name], dim=0)

            if len(self.labels) > 1:
                eval_prefix = (
                    label_name if not self.prefix else f"{self.prefix}_{label_name}"
                )
            else:
                eval_prefix = self.prefix if self.prefix else label_name

            label_class_names = self.class_names_dict.get(label_name, None)

            self._evaluate_single_label(
                X_train,
                y_train,
                X_test,
                y_test,
                eval_prefix,
                label_class_names,
            )

        self.trainer.model.train()

        # signal other ranks that evaluation is complete
        if comm.get_world_size() > 1:
            comm.synchronize()

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
        from pimm.engines.hooks.eval.pretrain.linear import (
            LinearProbingTrainer,
        )

        # Use provided class names or fall back to default
        if label_class_names is None:
            label_class_names = self.trainer.cfg.data.names

        num_classes = int(y_train.max().item()) + 1

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

        self.trainer.storage.put_scalar(f"{eval_prefix}_val_intersection", iou_class * (class_support + 1e-10))
        self.trainer.storage.put_scalar(f"{eval_prefix}_val_union", (class_support + 1e-10))
        self.trainer.storage.put_scalar(f"{eval_prefix}_val_target", class_support)

        self.trainer.logger.info(
            "Val result: mIoU/mPrec/mRec/mF1 {:.4f}/{:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_precision, m_recall, m_f1
            )
        )

        from rich.table import Table
        from rich.console import Console

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

        console = Console(file=None, width=100, record=True)
        console.print(table)
        table_str = console.export_text()  # noqa: F841

        import pandas as pd
        label_names = [str(n) for n in label_class_names[:num_classes]]
        cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
        self.trainer.logger.info("Confusion Matrix (rows=true, cols=pred):\n" + cm_df.to_string())

        _prefix = eval_prefix
        eval_prefix = eval_prefix + "/"
        if eval_prefix == "segment/":
            eval_prefix = ""

        if self.trainer.writer is not None:
            step = _get_writer_step(self.trainer)
            self.trainer.writer.add_scalar(f"{eval_prefix}val/mIoU", m_iou, step)
            self.trainer.writer.add_scalar(f"{eval_prefix}val/mPrecision", m_precision, step)
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
        if "current_metric_value" not in self.trainer.comm_info.keys():
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