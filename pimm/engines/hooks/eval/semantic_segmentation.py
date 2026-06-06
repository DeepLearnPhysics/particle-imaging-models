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
from pimm.distributed import unwrap_model
from pimm.utils.misc import intersection_and_union_gpu
from pimm.engines.metrics import (
    aggregate_instance_results,
    compute_semseg_metrics,
    eval_instances,
)

from pimm.engines.hooks.default import HookBase
from pimm.engines.hooks.builder import HOOKS
from pimm.models.utils.structure import Point
from pimm.models.utils.misc import offset2bincount
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

def _get_writer_step(trainer):
    """Local train step for logging; writer applies any configured offset."""
    ci = getattr(trainer, "comm_info", {})
    if "epoch" in ci and "iter" in ci:
        return ci.get("epoch", 0) * ci.get("iter_per_epoch", 0) + ci.get("iter", 0) + 1
    return getattr(trainer, "epoch", 0)


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    """Evaluate point-wise semantic segmentation on the validation loader."""

    def __init__(self, write_cls_iou=False, every_n_steps=0, ignore_index=-1, macro_ignore_class_ids=None, per_instance_metrics=False):
        """Configure evaluation cadence, macro masking, and instance summaries."""
        self.write_cls_iou = write_cls_iou
        self.every_n_steps = every_n_steps
        self.ignore_index = ignore_index
        self.macro_ignore_class_ids = tuple(sorted(set(macro_ignore_class_ids or [])))
        self.per_instance_metrics = per_instance_metrics

    def after_step(self):
        """Run semantic evaluation on the configured step cadence."""
        if self.trainer.cfg.evaluate and self.every_n_steps > 0:
            global_iter = self.trainer.comm_info['iter'] + self.trainer.comm_info['iter_per_epoch'] * self.trainer.comm_info['epoch']
            if (global_iter + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        """Run semantic evaluation after epochs when step cadence is disabled."""
        if self.trainer.cfg.evaluate and self.every_n_steps == 0:
            self.eval()

    def eval(self):
        """Compute point-wise validation metrics and publish mIoU to comm_info."""
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
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        metrics = compute_semseg_metrics(
            all_preds,
            all_segments,
            num_classes,
            macro_ignore_class_ids=self.macro_ignore_class_ids,
            intersection=intersection,
            union=union,
            target_count=target,
        )
        precision_class = metrics.precision_class
        recall_class = metrics.recall_class
        f1_class = metrics.f1_class
        iou_class = metrics.iou_class
        acc_class = metrics.acc_class
        macro_mask = metrics.macro_mask
        m_precision = metrics.m_precision
        m_recall = metrics.m_recall
        m_f1 = metrics.m_f1
        m_iou = metrics.m_iou
        m_acc = metrics.m_acc
        all_acc = metrics.all_acc
        
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
            step = _get_writer_step(self.trainer)
            self.trainer.writer.add_scalar("val/loss", loss_avg, step)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, step)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, step)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, step)
            self.trainer.writer.add_scalar("val/mPrecision", m_precision, step)
            self.trainer.writer.add_scalar("val/mRecall", m_recall, step)
            self.trainer.writer.add_scalar("val/mF1", m_f1, step)
            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        step
                    )
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} F1",
                        f1_class[i],
                        step
                    )

                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} Precision",
                        precision_class[i],
                        step
                    )

                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} Recall",
                        recall_class[i],
                        step
                    )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver
        self.trainer.model.train()

        # per-instance metrics if enabled and instance info available
        if self.per_instance_metrics and has_instance:
            self.eval_per_instance(all_preds, all_segments, all_instances)

    def eval_per_instance(self, all_preds, all_segments, all_instances):
        """Compute majority-vote class metrics over per-event instances."""
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
        """Log the best semantic metric tracked by checkpoint hooks."""
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )