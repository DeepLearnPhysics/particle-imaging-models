"""Semantic segmentation and pooled point classification evaluators."""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
import pimm.utils.comm as comm
from pimm.utils.misc import intersection_and_union_gpu
from pimm.engines.metrics import compute_semseg_metrics
from pimm.engines.hooks.default import HookBase, get_writer_step
from pimm.engines.hooks.builder import HOOKS
from pimm.observability import structured_logger as sl
import os


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    """Evaluate point-wise semantic segmentation on the validation loader.

    Runs the model over ``trainer.val_loader``, argmaxes the semantic logits
    (``seg_logits`` or ``sem_logits``), and accumulates per-class
    intersection/union/target counts (all-reduced across ranks). It reports
    mIoU, mAcc, allAcc and macro precision/recall/F1 plus a per-class table,
    and publishes ``mIoU`` as the checkpoint-selection metric via
    ``trainer.comm_info["current_metric_value"]`` /
    ``["current_metric_name"]``. Only the model's output is scored: the logits,
    the labels ``segment`` and the event boundaries ``offset``, all at the same
    points. Runs after
    every step when ``every_n_steps > 0`` (on iterations where
    ``(global_iter + 1) % every_n_steps == 0``), otherwise after each epoch;
    only when ``cfg.evaluate`` is true. Registered as ``SemSegEvaluator`` (use
    as ``type`` in a ``hooks=[...]`` entry).

    Args:
        write_cls_iou (bool): Also log per-class IoU/F1/precision/recall to the
            writer. Defaults to ``False``.
        every_n_steps (int): Step cadence; ``0`` evaluates once per epoch.
            Defaults to ``0``.
        macro_ignore_class_ids (Sequence[int] | None): Class ids excluded from
            macro (mean) metrics; deduplicated and sorted. Defaults to ``None``.
        selection_metric (str): ``"mIoU"`` or ``"mF1"``, the checkpoint-selection
            metric. Defaults to ``"mIoU"``.

    Note:
        The selection metric is ``mIoU`` by default (higher is better).

    Example:
        Add to ``cfg.hooks`` before the checkpoint saver; every ``every_n_steps``
        it validates and sets the selection metric:

        .. code-block:: python

            hooks = [
                dict(type="SemSegEvaluator", every_n_steps=1000,
                     write_cls_iou=True),
            ]
            # → every 1000 steps logs  val/loss, val/mIoU, val/mAcc, val/allAcc,
            #   val/mPrecision, val/mRecall, val/mF1  (+ per-class IoU/F1/Precision/
            #   Recall with write_cls_iou) and sets the checkpoint-selection metric
            #   to mIoU
    """

    def __init__(
        self,
        write_cls_iou=False,
        every_n_steps=0,
        macro_ignore_class_ids=None,
        selection_metric="mIoU",
        reset_metrics_each_eval=False,
    ):
        """Configure evaluation cadence and macro masking."""
        self.write_cls_iou = write_cls_iou
        self.every_n_steps = every_n_steps
        self.macro_ignore_class_ids = tuple(sorted(set(macro_ignore_class_ids or [])))
        assert selection_metric in ("mIoU", "mF1"), selection_metric
        self.selection_metric = selection_metric
        self.reset_metrics_each_eval = bool(reset_metrics_each_eval)

    def after_step(self):
        """Run semantic evaluation on the configured step cadence."""
        if self.trainer.cfg.evaluate and self.every_n_steps > 0:
            global_iter = (
                self.trainer.comm_info["iter"]
                + self.trainer.comm_info["iter_per_epoch"]
                * self.trainer.comm_info["epoch"]
            )
            if (global_iter + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        """Run semantic evaluation after epochs when step cadence is disabled."""
        if self.trainer.cfg.evaluate and self.every_n_steps == 0:
            self.eval()

    @sl.log_trace_span("evaluation.semantic_segmentation")
    def eval(self):
        """Compute point-wise validation metrics and publish mIoU to comm_info."""
        sl.add_step_tag("evaluation")
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        all_preds = []
        all_segments = []
        intersection_total, union_total, target_total, losses = (
            0,
            0,
            0,
            [],
        )  # this evaluation only

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
                raise KeyError(
                    "No semantic logits found in model output (expected 'seg_logits' or 'sem_logits')."
                )
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = output_dict["segment"].squeeze(-1)

            all_preds.append(pred.cpu())
            all_segments.append(segment.cpu())

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
            intersection_total = intersection_total + intersection
            union_total = union_total + union
            target_total = target_total + target
            losses.append(loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            self.trainer.logger.info(info + f"Loss {loss.item():.4f} ")

        if comm.get_world_size() > 1:
            all_preds_gathered = comm.gather(all_preds, dst=0)
            all_segments_gathered = comm.gather(all_segments, dst=0)
            if comm.get_rank() == 0:
                all_preds = [p for preds in all_preds_gathered for p in preds]
                all_segments = [
                    s for segments in all_segments_gathered for s in segments
                ]

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_segments = torch.cat(all_segments, dim=0).numpy()

        num_classes = self.trainer.cfg.data.num_classes
        loss_avg = float(np.mean(losses))
        intersection, union, target = intersection_total, union_total, target_total
        if not self.reset_metrics_each_eval:
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
        table_header = (
            "| Class ID | Class Name | IoU | Accuracy | Precision | Recall | F1 |"
        )
        table_separator = (
            "|"
            + "-" * 10
            + "|"
            + "-" * 12
            + "|"
            + "-" * 8
            + "|"
            + "-" * 10
            + "|"
            + "-" * 11
            + "|"
            + "-" * 8
            + "|"
            + "-" * 6
            + "|"
        )

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
                    f1=f1_class[i],
                )
            )
        current_iter = self.trainer.comm_info["iter"] + 1  # noqa: F841
        if self.trainer.writer is not None:
            step = get_writer_step(self.trainer)
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
                        step,
                    )
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} F1",
                        f1_class[i],
                        step,
                    )

                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} Precision",
                        precision_class[i],
                        step,
                    )

                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} Recall",
                        recall_class[i],
                        step,
                    )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = (
            m_f1 if self.selection_metric == "mF1" else m_iou
        )  # for the saver
        self.trainer.comm_info["current_metric_name"] = self.selection_metric
        self.trainer.model.train()

    def after_train(self):
        """Log the best semantic metric tracked by checkpoint hooks."""
        self.trainer.logger.info(
            "Best {}: {:.4f}".format(
                self.selection_metric, self.trainer.best_metric_value
            )
        )


# FM4NPP/PP_collision, train/downstream/loss.py: compute_multiclass_metrics.
def pooled_metrics(confusion):
    """FM4NPP's metrics from a (true, pred) confusion matrix: per-class precision / recall / F1 over the classes seen
    in truth or predictions (sklearn ``average=None``, ``zero_division=0``), their means, and accuracy.
    """
    confusion = np.asarray(confusion, dtype=np.float64)
    tp, true, pred = np.diag(confusion), confusion.sum(1), confusion.sum(0)
    seen = (true > 0) | (pred > 0)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=precision + recall > 0,
    )
    return dict(
        accuracy=tp.sum() / max(confusion.sum(), 1),
        macro_precision=precision[seen].mean(),
        macro_recall=recall[seen].mean(),
        macro_f1=f1[seen].mean(),
        precision=precision,
        recall=recall,
    )


@HOOKS.register_module()
class PointClassificationEvaluator(HookBase):
    """FM4NPP's pooled per-hit metrics for each head of a multi-head ``DefaultSegmentorV2`` (module docstring),
    from the model's output: ``seg_logits[head]`` and each label under its own name, at the same points.

    Logs ``val/<head>/{accuracy,macro_precision,macro_recall,macro_f1}`` and per-class precision / recall, and gives
    the checkpoint saver ``val/score``, the mean macro F1 over heads. With ``groups`` (several probes trained in one
    run on a frozen backbone), each group is also selected on its own: ``val/<group>/score`` is its heads' mean macro
    F1, and at each new best its heads' weights go to ``model/best_<group>.pth`` (with ``step`` and ``score``). Runs
    every ``every_n_steps`` steps when ``cfg.evaluate`` is true.

    Args:
        heads (dict): head name -> the label key it predicts, e.g. ``dict(linear="segment", mlp="segment")``.
        names (dict): label key -> class names.
        every_n_steps (int): evaluation cadence in steps.
        groups (dict, optional): group name -> head names, e.g. ``dict(linear=["linear"], mlp=["mlp"])``.
    """

    def __init__(self, heads, names, every_n_steps, groups=None):
        self.heads, self.names, self.every_n_steps = (
            dict(heads),
            dict(names),
            int(every_n_steps),
        )
        self.groups, self.best = dict(groups or {}), {}

    def after_step(self):
        info = self.trainer.comm_info
        step = info["epoch"] * info["iter_per_epoch"] + info["iter"] + 1
        if self.trainer.cfg.evaluate and step % self.every_n_steps == 0:
            self.eval(step)

    def eval(self, step):
        self.trainer.model.eval()
        confusion = {
            head: torch.zeros(
                len(self.names[label]),
                len(self.names[label]),
                dtype=torch.long,
                device="cuda",
            )
            for head, label in self.heads.items()
        }
        for input_dict in self.trainer.val_loader:
            input_dict = {
                k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
                for k, v in input_dict.items()
            }
            with torch.no_grad():
                output = self.trainer.model(input_dict)
            for head, label in self.heads.items():
                n = len(self.names[label])
                pred = output["seg_logits"][head].argmax(1)
                true = output[label].view(-1).long()
                keep = true >= 0
                confusion[head] += torch.bincount(
                    true[keep] * n + pred[keep], minlength=n * n
                ).view(n, n)
        f1 = {}
        for head, label in self.heads.items():
            if comm.get_world_size() > 1:
                dist.all_reduce(confusion[head])
            m = pooled_metrics(confusion[head].cpu().numpy())
            f1[head] = m["macro_f1"]
            self.trainer.logger.info(
                f"val {head}: accuracy {m['accuracy']:.4f} macro precision "
                f"{m['macro_precision']:.4f} macro recall {m['macro_recall']:.4f} "
                f"macro F1 {m['macro_f1']:.4f}"
            )
            if self.trainer.writer is not None:
                for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
                    self.trainer.writer.add_scalar(f"val/{head}/{name}", m[name], step)
                for i, cls in enumerate(self.names[label]):
                    self.trainer.writer.add_scalar(
                        f"val/{head}/{cls}_precision", m["precision"][i], step
                    )
                    self.trainer.writer.add_scalar(
                        f"val/{head}/{cls}_recall", m["recall"][i], step
                    )
        for group, heads in self.groups.items():
            score = float(np.mean([f1[head] for head in heads]))
            path = os.path.join(
                self.trainer.cfg.save_path, "model", f"best_{group}.pth"
            )
            if group not in self.best:  # a resumed run continues from the saved best
                self.best[group] = (
                    torch.load(path, map_location="cpu")["score"]
                    if os.path.exists(path)
                    else -1.0
                )
            if score > self.best[group]:
                self.best[group] = score
                if comm.is_main_process():
                    keep = tuple(f"seg_head.{head}." for head in heads)
                    state = {
                        k.removeprefix("module."): v.cpu()
                        for k, v in self.trainer.model.state_dict().items()
                        if any(p in k for p in keep)
                    }
                    torch.save(
                        dict(state_dict=state, step=step, score=score), path + ".tmp"
                    )
                    os.replace(path + ".tmp", path)
            self.trainer.logger.info(
                f"val {group}: score {score:.4f} (best {self.best[group]:.4f})"
            )
            if self.trainer.writer is not None:
                self.trainer.writer.add_scalar(f"val/{group}/score", score, step)
        score = float(np.mean(list(f1.values())))
        self.trainer.comm_info["current_metric_value"] = score
        self.trainer.comm_info["current_metric_name"] = "val/score"
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/score", score, step)
        self.trainer.model.train()
