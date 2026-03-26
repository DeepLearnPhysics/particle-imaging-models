"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import wandb
import torch
import torch.distributed as dist
import pointops
from uuid import uuid4

try:
    from torch_cluster import knn_graph
except ImportError:
    knn_graph = None

import pimm.utils.comm as comm
from pimm.utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS
from pimm.models.utils.structure import Point
from pimm.models.utils.misc import offset2bincount
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(self, write_cls_iou=False, every_n_steps=0, ignore_index=-1, macro_ignore_class_ids=None, per_instance_metrics=False):
        self.write_cls_iou = write_cls_iou
        self.every_n_steps = every_n_steps
        self.ignore_index = ignore_index
        self.macro_ignore_class_ids = tuple(sorted(set(macro_ignore_class_ids or [])))
        self.per_instance_metrics = per_instance_metrics

    def after_step(self):
        if self.trainer.cfg.evaluate and self.every_n_steps > 0:
            global_iter = self.trainer.comm_info['iter'] + self.trainer.comm_info['iter_per_epoch'] * self.trainer.comm_info['epoch']
            if (global_iter + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        if self.trainer.cfg.evaluate and self.every_n_steps == 0:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        
        all_preds = []
        all_segments = []
        all_instances = []
        event_sizes = []  # track number of points per event for per-instance metrics
        has_instance = False
        
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            if "seg_logits" in output_dict:
                output = output_dict["seg_logits"]
            elif "sem_logits" in output_dict:
                output = output_dict["sem_logits"]
            else:
                raise KeyError("No semantic logits found in model output (expected 'seg_logits' or 'sem_logits').")
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]
                offsets = input_dict["origin_offset"].cpu().tolist()
            else:
                offsets = input_dict["offset"].cpu().tolist()

            segment = segment.squeeze(-1)
            
            # track event sizes from offsets for per-instance metrics
            prev_offset = 0
            for offset in offsets:
                event_size = offset - prev_offset
                event_sizes.append(event_size)
                prev_offset = offset
            
            all_preds.append(pred.cpu())
            all_segments.append(segment.cpu())
            # collect instance ids if available and requested
            if self.per_instance_metrics and "instance" in input_dict:
                instance = input_dict["instance"]
                if "origin_coord" in input_dict.keys() and "origin_instance" in input_dict:
                    instance = input_dict["origin_instance"]
                instance = instance.squeeze(-1).cpu()
                all_instances.append(instance)
                has_instance = True

            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(info + f"Loss {loss.item():.4f} ")
        
        if comm.get_world_size() > 1:
            all_preds_gathered = comm.gather(all_preds, dst=0)
            all_segments_gathered = comm.gather(all_segments, dst=0)
            event_sizes_gathered = comm.gather(event_sizes, dst=0)
            if has_instance:
                all_instances_gathered = comm.gather(all_instances, dst=0)
            if comm.get_rank() == 0:
                all_preds = [p for preds in all_preds_gathered for p in preds]
                all_segments = [s for segments in all_segments_gathered for s in segments]
                event_sizes = [s for sizes in event_sizes_gathered for s in sizes]
                if has_instance:
                    all_instances = [ins for insts in all_instances_gathered for ins in insts]
        
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_segments = torch.cat(all_segments, dim=0).numpy()
        if has_instance:
            all_instances = torch.cat(all_instances, dim=0).numpy()
        
        # store event boundaries for per-instance metrics
        self._event_boundaries = event_sizes
        
        num_classes = self.trainer.cfg.data.num_classes
        precision_class = np.zeros(num_classes)
        recall_class = np.zeros(num_classes)
        f1_class = np.zeros(num_classes)
        
        for i in range(num_classes):
            pred_i = (all_preds == i)
            gt_i = (all_segments == i)
            if gt_i.sum() > 0 or pred_i.sum() > 0:
                tp = np.logical_and(pred_i, gt_i).sum()
                fp = np.logical_and(pred_i, np.logical_not(gt_i)).sum()
                fn = np.logical_and(np.logical_not(pred_i), gt_i).sum()
                
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                
                precision_class[i] = precision
                recall_class[i] = recall
                f1_class[i] = f1
        
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)

        macro_mask = np.ones(num_classes, dtype=bool)
        for idx in self.macro_ignore_class_ids:
            if 0 <= idx < num_classes:
                macro_mask[idx] = False

        precision_valid = precision_class[macro_mask]
        recall_valid = recall_class[macro_mask]
        f1_valid = f1_class[macro_mask]
        iou_valid = iou_class[macro_mask]
        acc_valid = acc_class[macro_mask]

        if precision_valid.size == 0:
            precision_valid = precision_class
        if recall_valid.size == 0:
            recall_valid = recall_class
        if f1_valid.size == 0:
            f1_valid = f1_class
        if iou_valid.size == 0:
            iou_valid = iou_class
        if acc_valid.size == 0:
            acc_valid = acc_class

        m_precision = np.mean(precision_valid)
        m_recall = np.mean(recall_valid)
        m_f1 = np.mean(f1_valid)
        m_iou = np.mean(iou_valid)
        m_acc = np.mean(acc_valid)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc/mPrec/mRec/mF1 {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc, m_precision, m_recall, m_f1
            )
        )
        table_header = "| Class ID | Class Name | IoU | Accuracy | Precision | Recall | F1 |"
        table_separator = "|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 8 + "|" + "-" * 10 + "|" + "-" * 11 + "|" + "-" * 8 + "|" + "-" * 6 + "|"
        
        self.trainer.logger.info("Per-class metrics:")
        self.trainer.logger.info(table_header)
        self.trainer.logger.info(table_separator)
        
        if not macro_mask.all():
            self.trainer.logger.info("* indicates class ignored in macro metrics")

        for i in range(self.trainer.cfg.data.num_classes):
            ignored_marker = "*" if not macro_mask[i] else ""
            self.trainer.logger.info(
                "| {idx:8d} | {name:10s} | {iou:.4f} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                    idx=i,
                    name=(self.trainer.cfg.data.names[i] + ignored_marker),
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                    precision=precision_class[i],
                    recall=recall_class[i],
                    f1=f1_class[i]
                )
            )
        current_iter = self.trainer.comm_info['iter']+1  # noqa: F841
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, self.trainer.writer.run.step)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, self.trainer.writer.run.step)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, self.trainer.writer.run.step)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, self.trainer.writer.run.step)
            self.trainer.writer.add_scalar("val/mPrecision", m_precision, self.trainer.writer.run.step)
            self.trainer.writer.add_scalar("val/mRecall", m_recall, self.trainer.writer.run.step)
            self.trainer.writer.add_scalar("val/mF1", m_f1, self.trainer.writer.run.step)
            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        self.trainer.writer.run.step
                    )
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} F1",
                        f1_class[i],
                        self.trainer.writer.run.step
                    )

                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} Precision",
                        precision_class[i],
                        self.trainer.writer.run.step
                    )

                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} Recall",
                        recall_class[i],
                        self.trainer.writer.run.step
                    )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver
        self.trainer.model.train()

        # per-instance metrics if enabled and instance info available
        if self.per_instance_metrics and has_instance:
            self.eval_per_instance(all_preds, all_segments, all_instances)

    def eval_per_instance(self, all_preds, all_segments, all_instances):
        # all_preds, all_segments, all_instances are numpy arrays, shape [N]
        # group by (event_id, instance_id) to respect per-event instance ID reuse
        from collections import defaultdict, Counter
        import numpy as np

        # check if we have event boundaries
        if not hasattr(self, '_event_boundaries'):
            self.trainer.logger.warning(
                "Per-instance metrics disabled: requires event boundary tracking. "
                "Instance IDs are reused per event and cannot be grouped globally."
            )
            return
        
        # group by (event_id, instance_id) tuple
        event_instance_to_idx = defaultdict(list)
        point_idx = 0
        for event_id, event_size in enumerate(self._event_boundaries):
            for local_idx in range(event_size):
                inst_id = all_instances[point_idx]
                event_instance_to_idx[(event_id, inst_id)].append(point_idx)
                point_idx += 1
        
        pred_labels = []
        gt_labels = []
        for (event_id, inst_id), idxs in event_instance_to_idx.items():
            pred_votes = all_preds[idxs]
            gt_votes = all_segments[idxs]
            # ignore instances with ignore_index in gt
            valid_gt = gt_votes[gt_votes != self.ignore_index]
            if len(valid_gt) == 0:
                continue
            # majority vote
            pred_label = Counter(pred_votes).most_common(1)[0][0]
            gt_label = Counter(valid_gt).most_common(1)[0][0]
            pred_labels.append(pred_label)
            gt_labels.append(gt_label)
        pred_labels = np.array(pred_labels)
        gt_labels = np.array(gt_labels)
        num_classes = self.trainer.cfg.data.num_classes

        # support: number of instances per class in gt
        support = np.zeros(num_classes, dtype=int)
        for i in range(num_classes):
            support[i] = np.sum(gt_labels == i)
        self.trainer.logger.info("[Per-instance] Num instances / class:")
        for i in range(num_classes):
            self.trainer.logger.info(f"  {self.trainer.cfg.data.names[i]}: {support[i]}")

        # confusion matrix
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pred in zip(gt_labels, pred_labels):
            if 0 <= gt < num_classes and 0 <= pred < num_classes:
                confusion[gt, pred] += 1
        self.trainer.logger.info("[Per-instance] Confusion matrix (rows=gt, cols=pred):")
        header = "      " + " ".join([f"{self.trainer.cfg.data.names[j]:>8s}" for j in range(num_classes)])
        self.trainer.logger.info(header)
        for i in range(num_classes):
            row = f"{self.trainer.cfg.data.names[i]:>6s} " + " ".join([f"{confusion[i, j]:8d}" for j in range(num_classes)])
            self.trainer.logger.info(row)

        precision_class = np.zeros(num_classes)
        recall_class = np.zeros(num_classes)
        f1_class = np.zeros(num_classes)
        for i in range(num_classes):
            pred_i = (pred_labels == i)
            gt_i = (gt_labels == i)
            if gt_i.sum() > 0 or pred_i.sum() > 0:
                tp = np.logical_and(pred_i, gt_i).sum()
                fp = np.logical_and(pred_i, np.logical_not(gt_i)).sum()
                fn = np.logical_and(np.logical_not(pred_i), gt_i).sum()
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                precision_class[i] = precision
                recall_class[i] = recall
                f1_class[i] = f1
        macro_mask = np.ones(num_classes, dtype=bool)
        for idx in self.macro_ignore_class_ids:
            if 0 <= idx < num_classes:
                macro_mask[idx] = False
        precision_valid = precision_class[macro_mask]
        recall_valid = recall_class[macro_mask]
        f1_valid = f1_class[macro_mask]
        if precision_valid.size == 0:
            precision_valid = precision_class
        if recall_valid.size == 0:
            recall_valid = recall_class
        if f1_valid.size == 0:
            f1_valid = f1_class
        m_precision = np.mean(precision_valid)
        m_recall = np.mean(recall_valid)
        m_f1 = np.mean(f1_valid)
        self.trainer.logger.info(
            "[Per-instance] mPrec/mRec/mF1 {:.4f}/{:.4f}/{:.4f}".format(
                m_precision, m_recall, m_f1
            )
        )
        table_header = "| Class ID | Class Name | Precision | Recall | F1 |"
        table_separator = "|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 11 + "|" + "-" * 8 + "|" + "-" * 6 + "|"
        self.trainer.logger.info("[Per-instance] Per-class metrics:")
        self.trainer.logger.info(table_header)
        self.trainer.logger.info(table_separator)
        if not macro_mask.all():
            self.trainer.logger.info("* indicates class ignored in macro metrics")
        for i in range(num_classes):
            ignored_marker = "*" if not macro_mask[i] else ""
            self.trainer.logger.info(
                "| {idx:8d} | {name:10s} | {precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                    idx=i,
                    name=(self.trainer.cfg.data.names[i] + ignored_marker),
                    precision=precision_class[i],
                    recall=recall_class[i],
                    f1=f1_class[i]
                )
            )

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )

@HOOKS.register_module()
class PretrainEvaluator(HookBase):
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
        if self.trainer.cfg.evaluate and self.every_n_steps > 0:
            global_iter = self.trainer.comm_info['iter'] + self.trainer.comm_info['iter_per_epoch'] * self.trainer.comm_info['epoch']
            if (global_iter + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        if self.trainer.cfg.evaluate and self.every_n_steps == 0:
            self.eval()

    def _unwrap_model(self):
        if isinstance(self.trainer.model, torch.nn.parallel.DistributedDataParallel):
            return self.trainer.model.module
        return self.trainer.model

    def get_backbone(self):
        model = self._unwrap_model()
        if hasattr(model, "teacher"): # sonata
            return model.teacher["backbone"]
        elif hasattr(model, "backbone"): # else
            return model.backbone
        else:
            raise ValueError(f"Model {model} has no backbone")

    def _has_encode(self):
        """Check if the model has a direct encode() method (e.g. Point-M2AE)."""
        model = self._unwrap_model()
        return hasattr(model, "encode") and callable(model.encode)

    def _process_batch_with_offsets(self, input_dict):
        """Process a batch and extract features properly using offsets to handle multiple events"""
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        with torch.inference_mode():
            if self._has_encode():
                # Models with encode() (e.g. Point-M2AE): returns packed (N, C) tensor
                model = self._unwrap_model()
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
                    point = self.get_backbone()(input_dict)
            else:
                point = self.get_backbone()(input_dict)
            while "pooling_parent" in point.keys():
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent

        # Get features and offset information
        features = point.feat.float()  # [N, C]
        offsets = [0] + input_dict["offset"].cpu().tolist()  # Batch offsets

        # Extract all label types
        all_labels = {}
        for label_name in self.labels:
            all_labels[label_name] = getattr(point, label_name).squeeze(-1)  # [N]

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
        from pimm.engines.hooks.eval.linear import (
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
            self.trainer.writer.add_scalar(f"{eval_prefix}val/mIoU", m_iou, self.trainer.writer.run.step)
            self.trainer.writer.add_scalar(f"{eval_prefix}val/mPrecision", m_precision, self.trainer.writer.run.step)
            self.trainer.writer.add_scalar(f"{eval_prefix}val/mRecall", m_recall, self.trainer.writer.run.step)
            self.trainer.writer.add_scalar(f"{eval_prefix}val/mF1", m_f1, self.trainer.writer.run.step)

            if self.write_cls_iou:
                for i in range(num_classes):
                    self.trainer.writer.add_scalar(
                        f"{eval_prefix}val/cls_{i}-{label_class_names[i]} IoU",
                        iou_class[i],
                        self.trainer.writer.run.step
                    )
                    self.trainer.writer.add_scalar(
                        f"{eval_prefix}val/cls_{i}-{label_class_names[i]} F1",
                        f1_class[i],
                        self.trainer.writer.run.step
                    )
                    self.trainer.writer.add_scalar(
                        f"{eval_prefix}val/cls_{i}-{label_class_names[i]} Precision",
                        precision_class[i],
                        self.trainer.writer.run.step
                    )
                    self.trainer.writer.add_scalar(
                        f"{eval_prefix}val/cls_{i}-{label_class_names[i]} Recall",
                        recall_class[i],
                        self.trainer.writer.run.step
                    )

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        if "current_metric_value" not in self.trainer.comm_info.keys():
            self.trainer.comm_info["current_metric_name"] = "mF1"
        self.trainer.comm_info["current_metric_value"] = m_f1
        self.trainer.comm_info[f"{_prefix}_current_metric_value"] = m_f1


@HOOKS.register_module()
class InstanceSegmentationEvaluator(HookBase):
    """Instance-level segmentation metrics including ARI and detection/class stats."""

    def __init__(
        self,
        every_n_steps=0,
        stuff_threshold=0.5,
        mask_threshold=0.5,
        class_names=None,
        stuff_classes=None,
        iou_thresh=0.5,
        require_class_for_match=False,
    ):
        self.every_n_steps = int(every_n_steps)
        self.stuff_threshold = float(stuff_threshold)
        self.mask_threshold = float(mask_threshold)
        self.iou_thresh = float(iou_thresh)
        self.require_class_for_match = bool(require_class_for_match)
        self.class_names = tuple(class_names or [])
        self.stuff_classes = tuple(sorted(stuff_classes or ()))

    def after_step(self):
        if self.trainer.cfg.evaluate and self.every_n_steps > 0:
            global_iter = (
                self.trainer.comm_info["iter"]
                + self.trainer.comm_info["iter_per_epoch"] * self.trainer.comm_info["epoch"]
            )
            if (global_iter + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        if self.trainer.cfg.evaluate and self.every_n_steps == 0:
            self.eval()

    def eval(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>> Start Instance Segmentation Evaluation >>>>>>>>>>>>>>>>"
        )
        self.trainer.model.eval()

        class_names = (
            self.class_names
            if len(self.class_names)
            else tuple(getattr(self.trainer.cfg.data, "names", []))
        )
        if not class_names:
            class_names = tuple(range(self.trainer.cfg.data.num_classes))

        all_stats = []
        ari_scores = []
        momentum_stats = []

        for input_dict in self.trainer.val_loader:
            assert (
                len(input_dict["offset"]) == 1
            ), "InstanceSegmentationEvaluator requires bs=1"

            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.trainer.model(input_dict, return_point=True)

            point = output_dict.get("point")
            if point is None:
                self.trainer.logger.warning(
                    "InstanceSegmentationEvaluator: missing point data"
                )
                continue

            gt_instance = point.instance
            if gt_instance is None:
                self.trainer.logger.warning(
                    "InstanceSegmentationEvaluator: missing instance labels"
                )
                continue

            gt_segment = getattr(point, "segment", None)
            if gt_segment is None:
                self.trainer.logger.warning(
                    "InstanceSegmentationEvaluator: missing PID labels"
                )
                continue

            pred_masks_list = output_dict.get("pred_masks")
            pred_logits_list = output_dict.get("pred_logits")
            pred_momentum_list = output_dict.get("pred_momentum")
            if not pred_masks_list or pred_logits_list is None:
                self.trainer.logger.warning(
                    "InstanceSegmentationEvaluator: missing predictions"
                )
                continue

            # Use detector's postprocess
            model = self._unwrap(self.trainer.model)
            
            # construct input for postprocess
            stuff_probs = (
                point.outputs.get("stuff_probs")
                if hasattr(point, "outputs")
                else None
            )
            point_counts = offset2bincount(point.offset)
            
            post_input = {
                "pred_masks": pred_masks_list,
                "pred_logits": pred_logits_list,
                "stuff_probs": stuff_probs,
                "point_counts": point_counts,
                "pred_momentum": pred_momentum_list,
            }
            
            results = model.postprocess(
                post_input,
                stuff_threshold=self.stuff_threshold,
                mask_threshold=self.mask_threshold,
            )
            
            pred_instance_labels = results["instance_labels"].cpu()
            pred_pid_labels = results["class_labels"].cpu()
            
            # Use instance_momentum from postprocess results if available
            pred_instance_momentum = None
            if "pred_momentum" in results:
                pred_instance_momentum = results["pred_momentum"].cpu()

            gt_inst = gt_instance.squeeze(-1).cpu().numpy().astype(np.int64)
            gt_pid = gt_segment.squeeze(-1).cpu().numpy().astype(np.int64)
            pr_inst = pred_instance_labels.numpy().astype(np.int64)
            pr_pid = pred_pid_labels.numpy().astype(np.int64)

            stats = self._eval_instances(
                gt_inst,
                pr_inst,
                gt_pid,
                pr_pid,
                class_names=class_names,
                iou_thresh=self.iou_thresh,
                require_class_for_match=self.require_class_for_match,
            )
            all_stats.append(stats)

            # evaluate momentum regression if available
            momentum_gt = input_dict.get("momentum")
            if pred_instance_momentum is not None and momentum_gt is not None:
                # counts is [num_points] for the batch item
                counts = offset2bincount(point.offset)
                counts_list = counts.cpu().tolist()
                
                # criterion is needed. Retrieve from model
                criterion = model.criteria.criteria[0].criterion

                mom_stats = self._eval_momentum(
                    stats["matches"],
                    pred_instance_momentum,
                    momentum_gt,
                    gt_instance,
                    counts_list,
                    criterion,
                    num_classes=self.trainer.cfg.data.num_classes,
                    pred_instance_labels=pr_inst,
                )
                if mom_stats is not None:
                    momentum_stats.append(mom_stats)

            led_mask = pr_inst != -1
            if led_mask.any():
                ari_scores.append(
                    adjusted_rand_score(gt_inst[led_mask], pr_inst[led_mask])
                )
            else:
                ari_scores.append(float("nan"))

        if comm.get_world_size() > 1:
            gathered = comm.gather((all_stats, ari_scores, momentum_stats), dst=0)
            if comm.get_rank() == 0:
                merged_stats = []
                merged_ari = []
                merged_momentum = []
                for stats_chunk, ari_chunk, momentum_chunk in gathered:
                    merged_stats.extend(stats_chunk)
                    merged_ari.extend(ari_chunk)
                    merged_momentum.extend(momentum_chunk)
                all_stats = merged_stats
                ari_scores = merged_ari
                momentum_stats = merged_momentum
            else:
                all_stats = []
                ari_scores = []
                momentum_stats = []

        if comm.get_world_size() > 1 and comm.get_rank() != 0:
            self.trainer.model.train()
            return

        if not all_stats:
            self.trainer.logger.warning(
                "InstanceSegmentationEvaluator: no stats computed"
            )
            self.trainer.model.train()
            return

        aggregated = self._aggregate_instance_results(
            all_stats, require_class_for_match=self.require_class_for_match
        )

        det = aggregated["detection"]
        cls = aggregated["classification_on_matched"]
        det_prec = det["precision"]
        det_rec = det["recall"]
        det_f1 = det["f1"]
        det_iou = det["mean_matched_iou"]
        total_gt = int(det.get("num_gt", 0))
        total_pred = int(det.get("num_pred", 0))
        total_matched = int(det.get("num_matched", 0))
        fp_det = max(total_pred - total_matched, 0)
        fn_det = max(total_gt - total_matched, 0)
        fp_per_gt = (fp_det / total_gt) if total_gt > 0 else 0.0
        fn_per_gt = (fn_det / total_gt) if total_gt > 0 else 0.0

        support = cls["support"]
        if np.any(support > 0):
            precision_macro = float(np.mean(cls["precision"][support > 0]))
            recall_macro = float(np.mean(cls["recall"][support > 0]))
            f1_macro = float(np.mean(cls["f1"][support > 0]))
        else:
            precision_macro = recall_macro = f1_macro = 0.0

        ari_clean = np.asarray(ari_scores, dtype=float)
        ari_clean = ari_clean[~np.isnan(ari_clean)]
        ari_mean = float(np.mean(ari_clean)) if ari_clean.size else float("nan")

        self.trainer.logger.info(
            "Detection P={:.3f} R={:.3f} F1={:.3f} IoU={:.3f}".format(
                det_prec, det_rec, det_f1, det_iou
            )
        )
        self.trainer.logger.info(
            "Counts GT={} Pred={} TP={} FP={} FN={} (FP/GT={:.3f} FN/GT={:.3f})".format(
                total_gt, total_pred, total_matched, fp_det, fn_det, fp_per_gt, fn_per_gt
            )
        )
        self.trainer.logger.info(
            "Classification macro P={:.3f} R={:.3f} F1={:.3f}".format(
                precision_macro, recall_macro, f1_macro
            )
        )
        if not np.isnan(ari_mean):
            self.trainer.logger.info("ARI mean={:.3f}".format(ari_mean))
        else:
            self.trainer.logger.info("ARI mean=n/a")

        # evaluate momentum regression if available
        if momentum_stats:
            momentum_aggregated = self._aggregate_momentum_results(momentum_stats, class_names)
            self._log_momentum_metrics(momentum_aggregated, class_names)

        if self.trainer.writer is not None:
            step = self.trainer.writer.run.step
            self.trainer.writer.add_scalar("val/ins_det_precision", det_prec, step)
            self.trainer.writer.add_scalar("val/ins_det_recall", det_rec, step)
            self.trainer.writer.add_scalar("val/ins_det_f1", det_f1, step)
            self.trainer.writer.add_scalar("val/ins_det_mean_iou", det_iou, step)
            self.trainer.writer.add_scalar("val/ins_det_fp", fp_det, step)
            self.trainer.writer.add_scalar("val/ins_det_fn", fn_det, step)
            self.trainer.writer.add_scalar("val/ins_det_fp_per_gt", fp_per_gt, step)
            self.trainer.writer.add_scalar("val/ins_det_fn_per_gt", fn_per_gt, step)
            self.trainer.writer.add_scalar(
                "val/ins_cls_macro_precision", precision_macro, step
            )
            self.trainer.writer.add_scalar(
                "val/ins_cls_macro_recall", recall_macro, step
            )
            self.trainer.writer.add_scalar("val/ins_cls_macro_f1", f1_macro, step)
            if not np.isnan(ari_mean):
                self.trainer.writer.add_scalar("val/ins_ari", ari_mean, step)

        self.trainer.comm_info["current_metric_value"] = det_f1
        self.trainer.comm_info["current_metric_name"] = "ins_det_f1"
        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<<< End Instance Segmentation Evaluation <<<<<<<<<<<<<<<<<"
        )
        self.trainer.model.train()

    def _eval_instances(
        self,
        gt_inst,  # (N,) int, ground-truth instance id per point
        pr_inst,  # (N,) int, predicted instance id per point
        gt_pid,  # either (N,) int per-point PID (constant within GT instance) OR (M_gt,) int table indexed by inst id
        pr_pid,  # either (N,) int per-point PID (constant within Pred instance) OR (M_pr,) int table indexed by inst id
        class_names=("photon", "electron", "muon", "pion", "proton"),
        iou_thresh=0.5,
        require_class_for_match=False,  # True → panoptic-style class-aware matching
    ):
        """
        Returns:
        {
            "detection": {precision, recall, f1, num_gt, num_pred, num_matched, mean_matched_iou, iou_thresh, matching},
            "classification_on_matched": {precision[K], recall[K], f1[K], confusion[KxK], support[K], class_names},
            "pq": {PQ, RQ, SQ}  # only when require_class_for_match=True
            "matches": [ {pred_id, gt_id, iou, pred_cls, gt_cls, intersection, pred_size, gt_size}, ... ]
        }
        """
        K = len(class_names)
        N = gt_inst.shape[0]
        assert pr_inst.shape[0] == N

        # Filter out background and invalid PIDs (bg_id=-1, pid=5)
        valid_mask = (gt_inst >= 0) & (pr_inst >= 0)
        
        # sizes per instance (counts of points)
        pr_ids, pr_sizes = np.unique(pr_inst[pr_inst >= 0], return_counts=True)
        gt_ids, gt_sizes = np.unique(gt_inst[gt_inst >= 0], return_counts=True)
        pr_size = {int(i): int(c) for i, c in zip(pr_ids, pr_sizes)}
        gt_size = {int(i): int(c) for i, c in zip(gt_ids, gt_sizes)}

        # intersections via pair counting in O(N)
        if valid_mask.any():
            pairs = np.stack([pr_inst[valid_mask], gt_inst[valid_mask]], axis=1)
            uniq_pairs, inter_counts = np.unique(pairs, axis=0, return_counts=True)
            # mapping (p,g) -> inter
            inter_map = {
                (int(p), int(g)): int(c) for (p, g), c in zip(uniq_pairs, inter_counts)
            }
        else:
            inter_map = {}

        # build per-instance PID from arrays
        def build_inst_pid_map(inst_ids_present, inst_ids_per_point, pid_array):
            inst_ids_present = list(map(int, inst_ids_present))
            m = int(np.max(inst_ids_present)) + 1 if inst_ids_present else 0

            # case A: pid_array is per-instance table (len > max inst id)
            if pid_array.ndim == 1 and pid_array.shape[0] >= m and m > 0:
                return {i: int(pid_array[i]) for i in inst_ids_present if 0 <= int(pid_array[i]) < K}

            # case B: pid_array is per-point; take the mode for each instance
            pid_map = {}
            if len(inst_ids_present) == 0:
                return pid_map
            # restrict to points with valid instances and PIDs
            use = (inst_ids_per_point >= 0) & (pid_array >= 0) & (pid_array < K)
            inst_vals = inst_ids_per_point[use].astype(np.int64)
            pid_vals = pid_array[use].astype(np.int64)
            # count (inst, pid)
            ip = np.stack([inst_vals, pid_vals], axis=1)
            uniq_ip, cnts = np.unique(ip, axis=0, return_counts=True)
            # for each inst, choose pid with max count
            # sort by inst then count desc
            order = np.lexsort(
                (-cnts, uniq_ip[:, 0])
            )  # primary: inst asc, secondary: count desc
            uniq_ip_sorted = uniq_ip[order]
            cnts_sorted = cnts[order]
            # first occurrence per inst in this order is the mode
            _, first_idx = np.unique(uniq_ip_sorted[:, 0], return_index=True)
            for idx in first_idx:
                inst_i = int(uniq_ip_sorted[idx, 0])
                pid_i = int(uniq_ip_sorted[idx, 1])
                if 0 <= pid_i < K:  # Only include valid PIDs
                    pid_map[inst_i] = pid_i
            return pid_map

        gt_pid_map = build_inst_pid_map(gt_ids, gt_inst, gt_pid)
        pr_pid_map = build_inst_pid_map(pr_ids, pr_inst, pr_pid)

        # optional class-aware gating for matching
        def class_ok(p_id, g_id):
            if not require_class_for_match:
                return True
            return pr_pid_map.get(p_id, -999) == gt_pid_map.get(g_id, -998)

        # candidate pairs with IoU
        cand = []
        for (p, g), inter in inter_map.items():
            if p not in pr_size or g not in gt_size:
                continue
            if not class_ok(p, g):
                continue
            union = pr_size[p] + gt_size[g] - inter
            if union <= 0:
                continue
            iou = inter / union
            cand.append((iou, p, g, inter))

        # greedy one-to-one matching by IoU
        cand.sort(reverse=True, key=lambda t: t[0])
        used_p, used_g = set(), set()
        matches = []
        for iou, p, g, inter in cand:
            if iou < iou_thresh:
                break
            if p in used_p or g in used_g:
                continue
            matches.append((p, g, iou, inter))
            used_p.add(p)
            used_g.add(g)

        num_gt = len(gt_ids)
        num_pred = len(pr_ids)
        num_matched = len(matches)
        fp = num_pred - num_matched
        fn = num_gt - num_matched

        det_prec = num_matched / (num_matched + fp) if (num_matched + fp) else 0.0
        det_rec = num_matched / (num_matched + fn) if (num_matched + fn) else 0.0
        det_f1 = (
            (2 * det_prec * det_rec / (det_prec + det_rec)) if (det_prec + det_rec) else 0.0
        )
        mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0

        # classification on matched pairs
        K = len(class_names)
        confusion = np.zeros((K, K), dtype=int)
        out_matches = []
        iou_sum_for_pq = 0.0
        tp_for_pq = 0
        for p, g, iou, inter in matches:
            pred_cls = pr_pid_map.get(p, -1)
            gt_cls = gt_pid_map.get(g, -1)
            if 0 <= gt_cls < K and 0 <= pred_cls < K:
                confusion[gt_cls, pred_cls] += 1
            out_matches.append(
                {
                    "pred_id": p,
                    "gt_id": g,
                    "iou": float(iou),
                    "pred_cls": int(pred_cls),
                    "gt_cls": int(gt_cls),
                    "intersection": int(inter),
                    "pred_size": pr_size[p],
                    "gt_size": gt_size[g],
                }
            )
            if require_class_for_match and pred_cls == gt_cls and 0 <= pred_cls < K:
                tp_for_pq += 1
                iou_sum_for_pq += iou

        support = np.array([confusion[i, :].sum() for i in range(K)], dtype=int)
        precision = np.zeros(K)
        recall = np.zeros(K)
        f1 = np.zeros(K)
        for i in range(K):
            tp = confusion[i, i]
            fp_c = confusion[:, i].sum() - tp
            fn_c = confusion[i, :].sum() - tp
            pr = tp / (tp + fp_c) if (tp + fp_c) else 0.0
            rc = tp / (tp + fn_c) if (tp + fn_c) else 0.0
            precision[i], recall[i] = pr, rc
            f1[i] = (2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0

        out = {
            "detection": {
                "precision": det_prec,
                "recall": det_rec,
                "f1": det_f1,
                "num_gt": num_gt,
                "num_pred": num_pred,
                "num_matched": num_matched,
                "mean_matched_iou": mean_iou,
                "iou_thresh": iou_thresh,
                "matching": "class-aware" if require_class_for_match else "unlabeled",
            },
            "classification_on_matched": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion": confusion,
                "support": support,
                "class_names": list(class_names),
            },
            "matches": out_matches,
        }

        if require_class_for_match:
            # PQ = SQ * RQ, with SQ = mean IoU over TPs (class-correct)
            # RQ = TP / (TP + 0.5 FP + 0.5 FN)
            sq = (iou_sum_for_pq / tp_for_pq) if tp_for_pq else 0.0
            rq_den = tp_for_pq + 0.5 * fp + 0.5 * fn
            rq = (tp_for_pq / rq_den) if rq_den > 0 else 0.0
            out["pq"] = {"PQ": rq * sq, "RQ": rq, "SQ": sq}

        return out

    def _aggregate_instance_results(self, stats_list, require_class_for_match=False):
        """Pool counts across events to compute global micro metrics and confusion."""
        if len(stats_list) == 0:
            raise ValueError("stats_list is empty")

        # Assume consistent class set across events
        cls0 = stats_list[0]["classification_on_matched"]
        class_names = list(cls0["class_names"])
        K = len(class_names)

        total_gt = total_pred = total_matched = 0
        # IoU sums
        iou_sum_all_matches = 0.0
        iou_sum_pq = 0.0  # class-correct only (for PQ)
        tp_pq = 0  # class-correct matched count

        # pooled confusion/support
        pooled_conf = np.zeros((K, K), dtype=int)
        pooled_support = np.zeros(K, dtype=int)

        for s in stats_list:
            det = s["detection"]
            total_gt += det["num_gt"]
            total_pred += det["num_pred"]
            total_matched += det["num_matched"]

            # accumulate IoUs from matches
            for m in s["matches"]:
                iou_sum_all_matches += float(m["iou"])
                if 0 <= m["gt_cls"] < K and m["gt_cls"] == m["pred_cls"]:
                    tp_pq += 1
                    iou_sum_pq += float(m["iou"])

            cls = s["classification_on_matched"]
            pooled_conf += np.asarray(cls["confusion"], dtype=int)
            pooled_support += np.asarray(cls["support"], dtype=int)

        # Detection micro metrics
        tp_det = total_matched
        fp_det = total_pred - total_matched
        fn_det = total_gt - total_matched

        det_prec = tp_det / (tp_det + fp_det) if (tp_det + fp_det) else 0.0
        det_rec = tp_det / (tp_det + fn_det) if (tp_det + fn_det) else 0.0
        det_f1 = (
            (2 * det_prec * det_rec / (det_prec + det_rec)) if (det_prec + det_rec) else 0.0
        )
        mean_iou = (iou_sum_all_matches / tp_det) if tp_det else 0.0

        # Classification micro-from-pooled-confusion
        tp = np.diag(pooled_conf)
        fp = pooled_conf.sum(axis=0) - tp
        fn = pooled_conf.sum(axis=1) - tp

        precision = np.zeros(K)
        recall = np.zeros(K)
        f1 = np.zeros(K)
        for i in range(K):
            pr = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
            rc = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
            precision[i], recall[i] = pr, rc
            f1[i] = (2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0

        # Build a "global" result dict in the same schema
        res_global = {
            "detection": {
                "precision": det_prec,
                "recall": det_rec,
                "f1": det_f1,
                "num_gt": int(total_gt),
                "num_pred": int(total_pred),
                "num_matched": int(total_matched),
                "mean_matched_iou": float(mean_iou),
                "iou_thresh": stats_list[0]["detection"]["iou_thresh"],
                "matching": stats_list[0]["detection"]["matching"],
            },
            "classification_on_matched": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion": pooled_conf,
                "support": pooled_support,
                "class_names": class_names,
            },
            "matches": [],  # not retained at global level
        }

        # Global PQ if class-aware matching was used
        if require_class_for_match:
            rq_den = tp_pq + 0.5 * fp_det + 0.5 * fn_det
            rq = (tp_pq / rq_den) if rq_den > 0 else 0.0
            sq = (iou_sum_pq / tp_pq) if tp_pq > 0 else 0.0
            res_global["pq"] = {"PQ": rq * sq, "RQ": rq, "SQ": sq}

        return res_global

    def _eval_momentum(
        self,
        matches,
        pred_instance_momentum,
        momentum_gt,
        gt_instance,
        counts,
        criterion,
        num_classes,
        pred_instance_labels,
    ):
        """Evaluate momentum regression per class for matched instances."""
        try:
            b = 0
            # get batch momentum ground truth
            mom_gt_b = criterion._get_batch_tensor(
                momentum_gt, b, counts, torch.device("cpu"), None
            )
            
            # get inverse mapping from points to instances
            inst_b = gt_instance.squeeze(-1)
            if inst_b.dim() == 0:
                inst_b = inst_b.unsqueeze(0)
            if inst_b.dim() == 2 and inst_b.shape[1] == 1:
                inst_b = inst_b.squeeze(1)
            inst_b = inst_b.cpu()
            
            # Map GT ID -> Momentum
            unique_gt_ids = torch.unique(inst_b)
            unique_gt_ids = unique_gt_ids[unique_gt_ids >= 0]
            gt_mom_map = {}
            for gt_id in unique_gt_ids:
                mask = (inst_b == gt_id)
                if mask.any():
                    gt_mom_map[gt_id.item()] = mom_gt_b[mask].float().mean().item()
            
            # Map Pred ID -> Momentum
            if isinstance(pred_instance_labels, np.ndarray):
                pred_inst_t = torch.from_numpy(pred_instance_labels)
            else:
                pred_inst_t = pred_instance_labels
                
            # Sort points by instance ID
            sorted_ids, sort_idx = torch.sort(pred_inst_t)
            # Get unique IDs and their first occurrence index in the sorted array
            unique_ids_sorted, counts_sorted = torch.unique_consecutive(sorted_ids, return_counts=True)
            
            # Compute start indices
            cumsum_counts = torch.cat([torch.tensor([0]), counts_sorted.cumsum(0)[:-1]])
            
            # Extract values at first indices
            first_indices = sort_idx[cumsum_counts]
            mom_values = pred_instance_momentum[first_indices]
            
            # Build map
            pr_mom_map = {uid.item(): val.item() for uid, val in zip(unique_ids_sorted, mom_values) if uid.item() >= 0}

            matched_preds = []
            
            # Iterate matches
            for m in matches:
                pid = m["pred_id"]
                gid = m["gt_id"]
                pred_cls = m["pred_cls"] # used for per-class stats
                
                if pid in pr_mom_map and gid in gt_mom_map:
                    matched_preds.append({
                        "pred": pr_mom_map[pid],
                        "gt": gt_mom_map[gid],
                        "cls": pred_cls
                    })
            
            if not matched_preds:
                return None
                
            # Compute stats
            all_p = np.array([x["pred"] for x in matched_preds])
            all_g = np.array([x["gt"] for x in matched_preds])
            all_c = np.array([x["cls"] for x in matched_preds])
            all_e = all_p - all_g
            
            # Collect stats per class
            class_stats = {}
            for cls_idx in range(num_classes):
                cls_mask = (all_c == cls_idx)
                if not np.any(cls_mask):
                    continue
                
                cls_errors = all_e[cls_mask]
                cls_mae = float(np.mean(np.abs(cls_errors)))
                cls_rmse = float(np.sqrt(np.mean(cls_errors ** 2)))
                cls_count = int(np.sum(cls_mask))
                
                class_stats[cls_idx] = {
                    "mae": cls_mae,
                    "rmse": cls_rmse,
                    "count": cls_count,
                }
            
            # Overall
            overall_mae = float(np.mean(np.abs(all_e)))
            overall_rmse = float(np.sqrt(np.mean(all_e ** 2)))
            
            return {
                "per_class": class_stats,
                "overall": {
                    "mae": overall_mae,
                    "rmse": overall_rmse,
                    "count": len(all_e),
                },
            }

        except Exception as e:
            self.trainer.logger.warning(f"Error evaluating momentum: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _aggregate_momentum_results(self, momentum_stats_list, class_names):
        """Aggregate momentum evaluation results across batches."""
        num_classes = len(class_names)
        
        # initialize per-class accumulators
        class_mae = {i: [] for i in range(num_classes)}
        class_rmse = {i: [] for i in range(num_classes)}
        class_count = {i: 0 for i in range(num_classes)}
        
        # overall accumulators
        overall_mae = []
        overall_rmse = []
        overall_count = 0
        
        for stats in momentum_stats_list:
            if stats is None:
                continue
            
            # accumulate per-class stats
            for cls_idx, cls_stats in stats["per_class"].items():
                if cls_idx < num_classes:
                    class_mae[cls_idx].append(cls_stats["mae"])
                    class_rmse[cls_idx].append(cls_stats["rmse"])
                    class_count[cls_idx] += cls_stats["count"]
            
            # accumulate overall stats
            overall_mae.append(stats["overall"]["mae"])
            overall_rmse.append(stats["overall"]["rmse"])
            overall_count += stats["overall"]["count"]
        
        # compute aggregated per-class metrics
        aggregated_per_class = {}
        for cls_idx in range(num_classes):
            if len(class_mae[cls_idx]) > 0:
                aggregated_per_class[cls_idx] = {
                    "mae": float(np.mean(class_mae[cls_idx])),
                    "rmse": float(np.mean(class_rmse[cls_idx])),
                    "count": class_count[cls_idx],
                }
        
        # compute aggregated overall metrics
        aggregated_overall = {
            "mae": float(np.mean(overall_mae)) if overall_mae else 0.0,
            "rmse": float(np.mean(overall_rmse)) if overall_rmse else 0.0,
            "count": overall_count,
        }
        
        return {
            "per_class": aggregated_per_class,
            "overall": aggregated_overall,
        }

    def _log_momentum_metrics(self, momentum_aggregated, class_names):
        """Log momentum regression metrics."""
        self.trainer.logger.info("Momentum Regression Metrics:")
        
        # overall metrics
        overall = momentum_aggregated["overall"]
        self.trainer.logger.info(
            "Overall: MAE={:.4f} RMSE={:.4f} Count={}".format(
                overall["mae"], overall["rmse"], overall["count"]
            )
        )
        
        # per-class metrics
        per_class = momentum_aggregated["per_class"]
        if per_class:
            self.trainer.logger.info("Per-class metrics:")
            for cls_idx in sorted(per_class.keys()):
                cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
                cls_stats = per_class[cls_idx]
                self.trainer.logger.info(
                    "  {}: MAE={:.4f} RMSE={:.4f} Count={}".format(
                        cls_name, cls_stats["mae"], cls_stats["rmse"], cls_stats["count"]
                    )
                )
        
        # log to tensorboard if available
        if self.trainer.writer is not None:
            step = self.trainer.writer.run.step
            overall = momentum_aggregated["overall"]
            self.trainer.writer.add_scalar("val/momentum_overall_mae", overall["mae"], step)
            self.trainer.writer.add_scalar("val/momentum_overall_rmse", overall["rmse"], step)
            
            per_class = momentum_aggregated["per_class"]
            for cls_idx in sorted(per_class.keys()):
                cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
                cls_stats = per_class[cls_idx]
                self.trainer.writer.add_scalar(
                    f"val/momentum_{cls_name}_mae", cls_stats["mae"], step
                )
                self.trainer.writer.add_scalar(
                    f"val/momentum_{cls_name}_rmse", cls_stats["rmse"], step
                )

    def _unwrap(self, obj):
        if isinstance(obj, torch.nn.parallel.DistributedDataParallel):
            return obj.module
        return obj

@HOOKS.register_module()
class MAEEvaluator(HookBase):
    """
    Validation hook for MAE that logs reconstruction losses on the validation set.
    """

    def __init__(self, every_n_steps: int = 0, max_batches: int = None, log_pointclouds: bool = True):
        """
        Args:
            every_n_steps: run validation every N steps. If 0, run every epoch instead.
            max_batches: limit number of batches for faster validation (None = all)
            log_pointclouds: if True, log reconstruction visualizations to wandb
        """
        self.every_n_steps = every_n_steps
        self.max_batches = max_batches
        self.log_pointclouds = log_pointclouds

    def after_step(self):
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        if self.every_n_steps > 0:
            global_iter = (
                self.trainer.comm_info["iter"]
                + self.trainer.comm_info["iter_per_epoch"] * self.trainer.comm_info["epoch"]
            )
            if (global_iter + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        if self.every_n_steps == 0:
            self.eval()

    @torch.no_grad()
    def eval(self):
        # only run on rank 0
        rank = comm.get_rank()
        if rank != 0:
            if comm.get_world_size() > 1:
                comm.synchronize()
            return

        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start MAE Validation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        total_loss = 0.0
        total_coord_loss = 0.0
        total_feat_loss = 0.0
        total_mask_ratio = 0.0
        num_batches = 0
        num_valid = 0

        loader = self.trainer.val_loader
        for i, input_dict in enumerate(loader):
            if self.max_batches is not None and i >= self.max_batches:
                break

            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            # use AMP if enabled in config to match training conditions
            # on first batch, request visualization data
            return_viz = (i == 0 and self.log_pointclouds)
            if getattr(self.trainer.cfg, "enable_amp", False):
                amp_dtype = getattr(self.trainer.cfg, "amp_dtype", "bfloat16")
                dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    output_dict = self.trainer.model(input_dict, return_pred=return_viz)
            else:
                output_dict = self.trainer.model(input_dict, return_pred=return_viz)

            # log point cloud visualizations on first batch
            if return_viz and "viz_visible_coord" in output_dict:
                self._log_pointcloud_viz(output_dict)

            loss_val = output_dict.get("loss", 0.0)
            coord_loss_val = output_dict.get("coord_loss", 0.0)
            feat_loss_val = output_dict.get("feat_loss", 0.0)
            mask_ratio_val = output_dict.get("mask_ratio_actual", 0.0)

            # handle tensor vs scalar
            if hasattr(loss_val, "item"):
                loss_val = loss_val.item()
            if hasattr(coord_loss_val, "item"):
                coord_loss_val = coord_loss_val.item()
            if hasattr(feat_loss_val, "item"):
                feat_loss_val = feat_loss_val.item()
            if hasattr(mask_ratio_val, "item"):
                mask_ratio_val = mask_ratio_val.item()

            # skip invalid batches (loss=0 from empty masks)
            if loss_val == 0.0:
                num_batches += 1
                continue

            total_loss += loss_val
            total_coord_loss += coord_loss_val
            total_feat_loss += feat_loss_val
            total_mask_ratio += mask_ratio_val
            num_valid += 1
            num_batches += 1

            if (i + 1) % 10 == 0:
                self.trainer.logger.info(
                    f"Val: [{i + 1}/{len(loader)}] "
                    f"Loss: {loss_val:.6f} Coord: {coord_loss_val:.6f} "
                    f"Feat: {feat_loss_val:.6f} MaskRatio: {mask_ratio_val:.3f}"
                )

        if num_valid > 0:
            avg_loss = total_loss / num_valid
            avg_coord = total_coord_loss / num_valid
            avg_feat = total_feat_loss / num_valid
            avg_mask_ratio = total_mask_ratio / num_valid
        else:
            avg_loss = avg_coord = avg_feat = avg_mask_ratio = 0.0

        self.trainer.logger.info(
            f"Val Result: Loss: {avg_loss:.6f} Coord: {avg_coord:.6f} "
            f"Feat: {avg_feat:.6f} MaskRatio: {avg_mask_ratio:.3f} "
            f"({num_valid}/{num_batches} valid batches)"
        )

        # log to wandb/tensorboard
        if self.trainer.writer is not None:
            step = getattr(self.trainer.writer, "step", self.trainer.epoch)
            if hasattr(self.trainer.writer, "run") and self.trainer.writer.run is not None:
                step = self.trainer.writer.run.step
            self.trainer.writer.add_scalar("val/loss", avg_loss, step)
            self.trainer.writer.add_scalar("val/coord_loss", avg_coord, step)
            self.trainer.writer.add_scalar("val/feat_loss", avg_feat, step)
            self.trainer.writer.add_scalar("val/mask_ratio", avg_mask_ratio, step)

        self.trainer.comm_info["current_metric_value"] = -avg_loss  # negative since lower is better
        self.trainer.comm_info["current_metric_name"] = "neg_val_loss"

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End MAE Validation <<<<<<<<<<<<<<<<<")
        self.trainer.model.train()

        # synchronize other ranks
        if comm.get_world_size() > 1:
            comm.synchronize()

    def _log_pointcloud_viz(self, output_dict):
        """Log point cloud visualizations to wandb."""
        try:
            # check if wandb is available and active
            if self.trainer.writer is None:
                return
            if not hasattr(self.trainer.writer, "run") or self.trainer.writer.run is None:
                return

            visible_coord = output_dict["viz_visible_coord"].numpy()  # (N_vis, 3)
            pred_coord = output_dict["viz_pred_coord"].numpy()  # (N_masked, K, 3)
            target_coord = output_dict["viz_target_coord"].numpy()  # (N_target, 3)
            target_counts = output_dict["viz_target_counts"].numpy()  # (N_masked,)

            # flatten predicted coordinates
            pred_coord_flat = pred_coord.reshape(-1, 3)  # (N_masked * K, 3)

            # create colored point clouds
            # visible points: blue (0, 0, 255)
            # predicted points: red (255, 0, 0)
            # target points: green (0, 255, 0)

            n_vis = visible_coord.shape[0]
            n_pred = pred_coord_flat.shape[0]
            n_target = target_coord.shape[0]

            # subsample if too many points (wandb has limits)
            max_points = 50000
            if n_vis > max_points:
                idx = np.random.choice(n_vis, max_points, replace=False)
                visible_coord = visible_coord[idx]
                n_vis = max_points
            if n_pred > max_points:
                idx = np.random.choice(n_pred, max_points, replace=False)
                pred_coord_flat = pred_coord_flat[idx]
                n_pred = max_points
            if n_target > max_points:
                idx = np.random.choice(n_target, max_points, replace=False)
                target_coord = target_coord[idx]
                n_target = max_points

            # combine visible + target (ground truth)
            gt_combined = np.vstack([visible_coord, target_coord])
            gt_colors = np.vstack([
                np.full((n_vis, 3), [0, 0, 255]),  # blue for visible
                np.full((n_target, 3), [0, 255, 0]),  # green for masked (GT)
            ])
            gt_pointcloud = np.hstack([gt_combined, gt_colors])

            # combine visible + predicted (reconstruction)
            recon_combined = np.vstack([visible_coord, pred_coord_flat])
            recon_colors = np.vstack([
                np.full((n_vis, 3), [0, 0, 255]),  # blue for visible
                np.full((n_pred, 3), [255, 0, 0]),  # red for predicted
            ])
            recon_pointcloud = np.hstack([recon_combined, recon_colors])

            # log to wandb
            self.trainer.writer.run.log({
                "val/pointcloud_gt": wandb.Object3D(gt_pointcloud),
                "val/pointcloud_recon": wandb.Object3D(recon_pointcloud),
            })

            self.trainer.logger.info("Logged point cloud visualizations to wandb")

        except Exception as e:
            self.trainer.logger.warning(f"Failed to log point cloud viz: {e}")


@HOOKS.register_module()
class HMAEEvaluator(HookBase):
    """
    Validation hook for HMAE that logs chamfer losses on the validation set.
    """

    def __init__(self, every_n_steps: int = 0, max_batches: int = None):
        """
        Args:
            every_n_steps: run validation every N steps. If 0, run every epoch instead.
            max_batches: limit number of batches for faster validation (None = all)
        """
        self.every_n_steps = every_n_steps
        self.max_batches = max_batches

    def after_step(self):
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        if self.every_n_steps > 0:
            global_iter = (
                self.trainer.comm_info["iter"]
                + self.trainer.comm_info["iter_per_epoch"] * self.trainer.comm_info["epoch"]
            )
            if (global_iter + 1) % self.every_n_steps == 0:
                if comm.get_world_size() > 1:
                    if comm.get_rank() == 0:
                        self.eval()
                else:
                    self.eval()

    def after_epoch(self):
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        if self.every_n_steps == 0:
            if comm.get_world_size() > 1:
                if comm.get_rank() == 0:
                    self.eval()
            else:
                self.eval()

    @torch.no_grad()
    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start HMAE Validation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        total_loss = 0.0
        total_coord_loss = 0.0
        total_feat_loss = 0.0
        num_batches = 0
        num_valid = 0

        for i, input_dict in enumerate(self.trainer.test_loader):
            if self.max_batches is not None and i >= self.max_batches:
                break

            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            output_dict = self.trainer.model(input_dict)

            loss_val = output_dict.get("loss", 0.0)
            coord_loss_val = output_dict.get("coord_loss", 0.0)
            feat_loss_val = output_dict.get("feat_loss", 0.0)

            # handle tensor vs scalar
            if hasattr(loss_val, "item"):
                loss_val = loss_val.item()
            if hasattr(coord_loss_val, "item"):
                coord_loss_val = coord_loss_val.item()
            if hasattr(feat_loss_val, "item"):
                feat_loss_val = feat_loss_val.item()

            # skip invalid batches (loss=0 from hmae_valid=False)
            if loss_val == 0.0:
                continue

            total_loss += loss_val
            total_coord_loss += coord_loss_val
            total_feat_loss += feat_loss_val
            num_valid += 1
            num_batches += 1

            if (i + 1) % 10 == 0:
                self.trainer.logger.info(
                    f"Val: [{i + 1}/{len(self.trainer.val_loader)}] "
                    f"Loss: {loss_val:.6f} Coord: {coord_loss_val:.6f} Feat: {feat_loss_val:.6f}"
                )

        if num_valid > 0:
            avg_loss = total_loss / num_valid
            avg_coord = total_coord_loss / num_valid
            avg_feat = total_feat_loss / num_valid
        else:
            avg_loss = avg_coord = avg_feat = 0.0

        self.trainer.logger.info(
            f"Val Result: Loss: {avg_loss:.6f} Coord: {avg_coord:.6f} Feat: {avg_feat:.6f} "
            f"({num_valid}/{num_batches} valid batches)"
        )

        # log to wandb/tensorboard
        if self.trainer.writer is not None:
            step = getattr(self.trainer.writer, "step", self.trainer.epoch)
            if hasattr(self.trainer.writer, "run") and self.trainer.writer.run is not None:
                step = self.trainer.writer.run.step
            self.trainer.writer.add_scalar("val/loss", avg_loss, step)
            self.trainer.writer.add_scalar("val/coord_loss", avg_coord, step)
            self.trainer.writer.add_scalar("val/feat_loss", avg_feat, step)

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End HMAE Validation <<<<<<<<<<<<<<<<<")
        self.trainer.model.train()
