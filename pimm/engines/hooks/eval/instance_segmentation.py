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
class InstanceSegmentationEvaluator(HookBase):
    """Instance-level segmentation metrics including ARI and detection/class stats.

    Runs the model with ``return_point=True`` on ``trainer.val_loader`` (batch
    size MUST be 1), postprocesses predicted instance masks via
    ``model.postprocess``, and greedily matches them to ground-truth instances
    by mask IoU. For each configured label it reports detection
    precision/recall/F1 and mean matched IoU, FP/FN counts, macro
    classification precision/recall/F1 on matched instances, mean Adjusted Rand
    Index (ARI), and, when momentum predictions and targets are present,
    per-class momentum regression MAE/RMSE. Supports a single default label
    (``labels=(None,)``) or multiple auxiliary labels via
    ``outputs_by_label``. Results are gathered to rank 0 on multi-GPU. The
    primary label's detection F1 is published as the checkpoint-selection metric
    (``current_metric_value`` = ``det_f1``, ``current_metric_name`` =
    ``<prefix>/ins_det_f1``). Runs after every step when ``every_n_steps > 0``
    (when ``(global_iter + 1) % every_n_steps == 0``), otherwise after each
    epoch; only when ``cfg.evaluate`` is true. Registered as
    ``InstanceSegmentationEvaluator`` (use as ``type`` in a ``hooks=[...]``
    entry).

    Args:
        every_n_steps (int): Step cadence; ``0`` evaluates once per epoch.
            Defaults to ``0``.
        stuff_threshold (float): Threshold passed to ``model.postprocess`` for
            stuff/semantic regions. Defaults to ``0.5``.
        mask_threshold (float): Per-point probability threshold for binarizing
            predicted masks in postprocessing. Defaults to ``0.5``.
        class_names (list | dict | None): Class names for logging; a list
            (shared) or a per-label dict. Falls back to data-config/model
            metadata. Defaults to ``None``.
        stuff_classes (list | None): Class ids treated as stuff. Defaults to
            ``None``.
        iou_thresh (float): Minimum mask IoU for a prediction to count as a
            true-positive match. Defaults to ``0.5``.
        require_class_for_match (bool): Require predicted and GT classes to
            agree for a match to count. Defaults to ``False``.
        labels (str | Sequence | None): Label(s) to evaluate; ``None`` evaluates
            the default instance/segment label. Defaults to ``None`` (treated as
            ``(None,)``).
        prefix (str | None): Writer metric prefix; defaults to ``val`` (or
            ``val/<label>`` for named labels) when ``None``. Defaults to
            ``None``.
        primary_label (Any): Label whose ``det_f1`` becomes the selection
            metric; defaults to the first entry of ``labels``. Defaults to
            ``None``.
        set_current_metric (bool): Whether to publish a checkpoint-selection
            metric at all. Defaults to ``True``.
        instance_key (str | dict | None): Override input key for GT instance ids
            (per label). Defaults to ``None``.
        segment_key (str | dict | None): Override input key for GT semantic
            labels (per label). Defaults to ``None``.
        segment_fallback_key (str | dict | None): Fallback semantic key used
            when ``segment_key`` is absent. Defaults to ``None``.
        log_ari (bool): Compute and log mean ARI over events. Defaults to
            ``True``.

    Note:
        Requires validation batch size 1 (asserted). The selection metric is
        instance detection F1 of the primary label (higher is better).
        Momentum regression metrics are only computed when the model exposes a
        regression criterion and the batch carries a ``momentum`` target.

    Example:
        Add to ``cfg.hooks`` (validation batch size MUST be 1); it scores
        instance masks and sets the selection metric:

        .. code-block:: python

            hooks = [
                dict(type="InstanceSegmentationEvaluator", every_n_steps=2000,
                     iou_thresh=0.5, labels="particle"),
            ]
            # → every 2000 steps logs  val/particle/ins_det_{precision,recall,f1,
            #   mean_iou,fp,fn,fp_per_gt,fn_per_gt}, val/particle/ins_cls_macro_
            #   {precision,recall,f1}, val/particle/ins_ari  and sets the
            #   checkpoint-selection metric to val/particle/ins_det_f1
    """

    def __init__(
        self,
        every_n_steps=0,
        stuff_threshold=0.5,
        mask_threshold=0.5,
        class_names=None,
        stuff_classes=None,
        iou_thresh=0.5,
        require_class_for_match=False,
        labels=None,
        prefix=None,
        primary_label=None,
        set_current_metric=True,
        instance_key=None,
        segment_key=None,
        segment_fallback_key=None,
        log_ari=True,
    ):
        """Configure label-specific instance metrics and logging behavior."""
        self.every_n_steps = int(every_n_steps)
        self.stuff_threshold = float(stuff_threshold)
        self.mask_threshold = float(mask_threshold)
        self.iou_thresh = float(iou_thresh)
        self.require_class_for_match = bool(require_class_for_match)
        if labels is None:
            labels = (None,)
        elif isinstance(labels, str):
            labels = (labels,)
        self.labels = tuple(labels)
        self.prefix = prefix
        self.primary_label = primary_label
        self.set_current_metric = bool(set_current_metric)
        self.instance_key = instance_key
        self.segment_key = segment_key
        self.segment_fallback_key = segment_fallback_key
        self.log_ari = bool(log_ari)
        self.class_names = class_names
        self.stuff_classes = stuff_classes

    @staticmethod
    def _select_for_label(value, label, default=None):
        """Resolve scalar or label-keyed config values for one label."""
        if value is None:
            return default
        if isinstance(value, dict):
            if label in value:
                return value[label]
            label_key = str(label)
            if label_key in value:
                return value[label_key]
            return value.get("default", default)
        return value

    def _metric_prefix(self, label):
        """Return the writer prefix used for a label's metrics."""
        prefix = self.prefix
        if prefix is None:
            prefix = "val" if label is None else f"val/{label}"
        return str(prefix).strip("/")

    def _metric_key(self, label, name):
        """Return the full writer metric key for a label and metric name."""
        return f"{self._metric_prefix(label)}/{name}"

    @staticmethod
    def _label_name(label):
        """Return a readable name for default or explicit labels."""
        return "default" if label is None else str(label)

    def _label_specs(self, model):
        """Return optional model-provided label metadata."""
        return getattr(model, "label_specs", {}) or {}

    @staticmethod
    def _spec_value(spec, name, default=None):
        """Read a field from a dictionary or dataclass label config."""
        return (
            spec.get(name, default)
            if isinstance(spec, dict)
            else getattr(spec, name, default)
        )

    def _resolve_class_names(self, label, model):
        """Resolve class names from hook config, data config, or model specs."""
        class_names = self._select_for_label(self.class_names, label, None)
        if class_names:
            return tuple(class_names)

        data_cfg = self.trainer.cfg.data
        if label is not None:
            label_names_key = f"{label}_names"
            if hasattr(data_cfg, label_names_key):
                return tuple(getattr(data_cfg, label_names_key))
            if label == "interaction" and hasattr(data_cfg, "interaction_names"):
                return tuple(data_cfg.interaction_names)
            if label == "particle" and hasattr(data_cfg, "names"):
                return tuple(data_cfg.names)

            spec = self._label_specs(model).get(label, {})
            num_classes = self._spec_value(spec, "num_classes")
            if num_classes is not None:
                return tuple(range(int(num_classes)))

        if hasattr(data_cfg, "names"):
            return tuple(data_cfg.names)
        return tuple(range(data_cfg.num_classes))

    def _resolve_gt_labels(self, input_dict, point, model, label):
        """Return instance and segment ground truth tensors for one label."""
        if label is None:
            return (
                getattr(point, "instance", None),
                getattr(point, "segment", None),
            )

        spec = self._label_specs(model).get(label, {})
        instance_key = (
            self._select_for_label(self.instance_key, label, None)
            or self._spec_value(spec, "instance_key")
            or f"instance_{label}"
        )
        segment_key = (
            self._select_for_label(self.segment_key, label, None)
            or self._spec_value(spec, "segment_key")
            or f"segment_{label}"
        )
        fallback_key = self._select_for_label(
            self.segment_fallback_key, label, None
        ) or self._spec_value(spec, "segment_fallback_key")
        if segment_key not in input_dict and fallback_key in input_dict:
            segment_key = fallback_key
        return input_dict.get(instance_key), input_dict.get(segment_key)

    def _get_label_predictions(self, output_dict, label):
        """Return model predictions for an auxiliary label when available."""
        if label is not None and "outputs_by_label" in output_dict:
            outputs_by_label = output_dict["outputs_by_label"]
            if label in outputs_by_label:
                return outputs_by_label[label]
        return None

    def _should_set_current_metric(self, label):
        """Return whether this label should update checkpoint selection metrics."""
        if not self.set_current_metric:
            return False
        primary = self.primary_label
        if primary is None:
            primary = self.labels[0] if self.labels else None
        return label == primary

    def after_step(self):
        """Run instance evaluation on the configured step cadence."""
        if self.trainer.cfg.evaluate and self.every_n_steps > 0:
            global_iter = (
                self.trainer.comm_info["iter"]
                + self.trainer.comm_info["iter_per_epoch"] * self.trainer.comm_info["epoch"]
            )
            if (global_iter + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        """Run instance evaluation after epochs when step cadence is disabled."""
        if self.trainer.cfg.evaluate and self.every_n_steps == 0:
            self.eval()

    def eval(self):
        """Evaluate instance masks, class labels, ARI, and momentum metrics."""
        label_msg = ", ".join(self._label_name(label) for label in self.labels)
        self.trainer.logger.info(
            f">>>>>>>>>>>>>> Start Instance Segmentation Evaluation [{label_msg}] >>>>>>>>>>>>>>>>"
        )
        self.trainer.model.eval()
        model = unwrap_model(self.trainer.model)

        all_stats_by_label = {label: [] for label in self.labels}
        ari_scores_by_label = {label: [] for label in self.labels}
        momentum_stats_by_label = {label: [] for label in self.labels}

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

            point_counts = output_dict.get("point_counts")
            if point_counts is None:
                point_counts = offset2bincount(point.offset)

            for label in self.labels:
                gt_instance, gt_segment = self._resolve_gt_labels(
                    input_dict, point, model, label
                )
                label_name = self._label_name(label)
                if gt_instance is None:
                    self.trainer.logger.warning(
                        f"InstanceSegmentationEvaluator[{label_name}]: missing instance labels"
                    )
                    continue
                if gt_segment is None:
                    self.trainer.logger.warning(
                        f"InstanceSegmentationEvaluator[{label_name}]: missing semantic labels"
                    )
                    continue

                task_output = self._get_label_predictions(output_dict, label)
                if task_output is not None:
                    pred_masks_list = task_output.get("pred_masks")
                    pred_logits_list = task_output.get("pred_logits")
                    pred_momentum_list = task_output.get("pred_momentum")
                    post_input = {
                        "outputs_by_label": output_dict["outputs_by_label"],
                        "point_counts": point_counts,
                    }
                else:
                    pred_masks_list = output_dict.get("pred_masks")
                    pred_logits_list = output_dict.get("pred_logits")
                    pred_momentum_list = output_dict.get("pred_momentum")
                    stuff_probs = (
                        point.outputs.get("stuff_probs")
                        if hasattr(point, "outputs")
                        else None
                    )
                    post_input = {
                        "pred_masks": pred_masks_list,
                        "pred_logits": pred_logits_list,
                        "stuff_probs": stuff_probs,
                        "point_counts": point_counts,
                        "pred_momentum": pred_momentum_list,
                    }

                if not pred_masks_list or pred_logits_list is None:
                    self.trainer.logger.warning(
                        f"InstanceSegmentationEvaluator[{label_name}]: missing predictions"
                    )
                    continue

                post_kwargs = dict(
                    stuff_threshold=self.stuff_threshold,
                    mask_threshold=self.mask_threshold,
                )
                if label is not None:
                    post_kwargs["label"] = label
                results = model.postprocess(post_input, **post_kwargs)

                pred_instance_labels = results["instance_labels"].cpu()
                pred_pid_labels = results["class_labels"].cpu()

                pred_instance_momentum = None
                if "pred_momentum" in results:
                    pred_instance_momentum = results["pred_momentum"].cpu()

                gt_inst = gt_instance.detach().reshape(-1).cpu().numpy().astype(np.int64)
                gt_pid = gt_segment.detach().reshape(-1).cpu().numpy().astype(np.int64)
                pr_inst = pred_instance_labels.numpy().astype(np.int64)
                pr_pid = pred_pid_labels.numpy().astype(np.int64)
                class_names = self._resolve_class_names(label, model)

                stats = eval_instances(
                    gt_inst,
                    pr_inst,
                    gt_pid,
                    pr_pid,
                    class_names=class_names,
                    iou_thresh=self.iou_thresh,
                    require_class_for_match=self.require_class_for_match,
                )
                all_stats_by_label[label].append(stats)

                # evaluate momentum regression if available
                momentum_gt = input_dict.get("momentum")
                if pred_instance_momentum is not None and momentum_gt is not None:
                    counts = offset2bincount(point.offset)
                    counts_list = counts.cpu().tolist()
                    criterion = None
                    criteria = getattr(model, "criteria", None)
                    if criteria is not None and getattr(criteria, "criteria", None):
                        criterion = getattr(criteria.criteria[0], "criterion", None)

                    if criterion is not None:
                        mom_stats = self._eval_momentum(
                            stats["matches"],
                            pred_instance_momentum,
                            momentum_gt,
                            gt_instance,
                            counts_list,
                            criterion,
                            num_classes=len(class_names),
                            pred_instance_labels=pr_inst,
                        )
                        if mom_stats is not None:
                            momentum_stats_by_label[label].append(mom_stats)

                if self.log_ari:
                    valid_ari_mask = pr_inst != -1
                    if valid_ari_mask.any():
                        ari_scores_by_label[label].append(
                            adjusted_rand_score(
                                gt_inst[valid_ari_mask], pr_inst[valid_ari_mask]
                            )
                        )
                    else:
                        ari_scores_by_label[label].append(float("nan"))

        if comm.get_world_size() > 1:
            gathered = comm.gather(
                (all_stats_by_label, ari_scores_by_label, momentum_stats_by_label),
                dst=0,
            )
            if comm.get_rank() == 0:
                merged_stats = {label: [] for label in self.labels}
                merged_ari = {label: [] for label in self.labels}
                merged_momentum = {label: [] for label in self.labels}
                for stats_chunk, ari_chunk, momentum_chunk in gathered:
                    for label in self.labels:
                        merged_stats[label].extend(stats_chunk.get(label, []))
                        merged_ari[label].extend(ari_chunk.get(label, []))
                        merged_momentum[label].extend(momentum_chunk.get(label, []))
                all_stats_by_label = merged_stats
                ari_scores_by_label = merged_ari
                momentum_stats_by_label = merged_momentum
            else:
                all_stats_by_label = {label: [] for label in self.labels}
                ari_scores_by_label = {label: [] for label in self.labels}
                momentum_stats_by_label = {label: [] for label in self.labels}

        if comm.get_world_size() > 1 and comm.get_rank() != 0:
            self.trainer.model.train()
            return

        logged_any = False
        current_metric_set = False
        for label in self.labels:
            all_stats = all_stats_by_label[label]
            label_name = self._label_name(label)
            if not all_stats:
                self.trainer.logger.warning(
                    f"InstanceSegmentationEvaluator[{label_name}]: no stats computed"
                )
                continue

            aggregated = aggregate_instance_results(
                all_stats, require_class_for_match=self.require_class_for_match
            )
            class_names = self._resolve_class_names(label, model)

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

            ari_clean = np.asarray(ari_scores_by_label[label], dtype=float)
            ari_clean = ari_clean[~np.isnan(ari_clean)]
            ari_mean = float(np.mean(ari_clean)) if ari_clean.size else float("nan")

            self.trainer.logger.info(
                "[{}] Detection P={:.3f} R={:.3f} F1={:.3f} IoU={:.3f}".format(
                    label_name, det_prec, det_rec, det_f1, det_iou
                )
            )
            self.trainer.logger.info(
                "[{}] Counts GT={} Pred={} TP={} FP={} FN={} (FP/GT={:.3f} FN/GT={:.3f})".format(
                    label_name,
                    total_gt,
                    total_pred,
                    total_matched,
                    fp_det,
                    fn_det,
                    fp_per_gt,
                    fn_per_gt,
                )
            )
            self.trainer.logger.info(
                "[{}] Classification macro P={:.3f} R={:.3f} F1={:.3f}".format(
                    label_name, precision_macro, recall_macro, f1_macro
                )
            )
            if self.log_ari:
                if not np.isnan(ari_mean):
                    self.trainer.logger.info(
                        "[{}] ARI mean={:.3f}".format(label_name, ari_mean)
                    )
                else:
                    self.trainer.logger.info("[{}] ARI mean=n/a".format(label_name))

            momentum_stats = momentum_stats_by_label[label]
            if momentum_stats:
                momentum_aggregated = self._aggregate_momentum_results(
                    momentum_stats, class_names
                )
                self._log_momentum_metrics(momentum_aggregated, class_names)

            if self.trainer.writer is not None:
                step = _get_writer_step(self.trainer)
                self.trainer.writer.add_scalar(
                    self._metric_key(label, "ins_det_precision"), det_prec, step
                )
                self.trainer.writer.add_scalar(
                    self._metric_key(label, "ins_det_recall"), det_rec, step
                )
                self.trainer.writer.add_scalar(
                    self._metric_key(label, "ins_det_f1"), det_f1, step
                )
                self.trainer.writer.add_scalar(
                    self._metric_key(label, "ins_det_mean_iou"), det_iou, step
                )
                self.trainer.writer.add_scalar(
                    self._metric_key(label, "ins_det_fp"), fp_det, step
                )
                self.trainer.writer.add_scalar(
                    self._metric_key(label, "ins_det_fn"), fn_det, step
                )
                self.trainer.writer.add_scalar(
                    self._metric_key(label, "ins_det_fp_per_gt"), fp_per_gt, step
                )
                self.trainer.writer.add_scalar(
                    self._metric_key(label, "ins_det_fn_per_gt"), fn_per_gt, step
                )
                self.trainer.writer.add_scalar(
                    self._metric_key(label, "ins_cls_macro_precision"),
                    precision_macro,
                    step,
                )
                self.trainer.writer.add_scalar(
                    self._metric_key(label, "ins_cls_macro_recall"),
                    recall_macro,
                    step,
                )
                self.trainer.writer.add_scalar(
                    self._metric_key(label, "ins_cls_macro_f1"), f1_macro, step
                )
                if self.log_ari and not np.isnan(ari_mean):
                    self.trainer.writer.add_scalar(
                        self._metric_key(label, "ins_ari"), ari_mean, step
                    )

            if self._should_set_current_metric(label):
                self.trainer.comm_info["current_metric_value"] = det_f1
                self.trainer.comm_info["current_metric_name"] = self._metric_key(
                    label, "ins_det_f1"
                )
                current_metric_set = True
            elif self.set_current_metric and not current_metric_set:
                self.trainer.comm_info.setdefault("current_metric_value", det_f1)
                self.trainer.comm_info.setdefault(
                    "current_metric_name", self._metric_key(label, "ins_det_f1")
                )
            logged_any = True

        if not logged_any:
            self.trainer.model.train()
            return

        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<<< End Instance Segmentation Evaluation <<<<<<<<<<<<<<<<<"
        )
        self.trainer.model.train()

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
            step = _get_writer_step(self.trainer)
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
