from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pimm.models.losses.builder import LOSSES
from pimm.models.losses.instance import SingleLayerInstanceLoss
from pimm.models.panda_detector.matcher_fast import (
    FastHungarianMatcher,
    build_target_cache,
    get_target_masks,
    split_by_counts,
)


@LOSSES.register_module()
class FastInstanceSegmentationLoss(nn.Module):
    """Mask2Former-style instance segmentation loss (cached, vectorized).

    Computes a set-prediction loss between per-query predicted masks and
    ground-truth instances. Targets are cached once per call (built from the
    per-point ``truth_label`` ids), a fast Hungarian matcher assigns queries to
    instances, and matched pairs incur a focal mask loss, a Dice loss, and a
    classification loss (matched -> instance class, unmatched -> a no-object
    class). Optional momentum and IoU terms are added when their weights are
    positive and the predictions/targets are present. Deep supervision over
    ``aux_outputs`` is summed and scaled by ``aux_loss_weight``.

    ``forward(pred, input_dict)`` expects ``pred`` to be a dict with
    ``pred_masks`` (list of ``(Q, P)`` tensors per batch element) and optionally
    ``pred_logits``/``pred_momentum``/``pred_iou``/``aux_outputs``, and
    ``input_dict`` to provide ``truth_label`` (and ``segment`` for the class
    term). Returns ``(loss, components)`` where ``components`` is a dict of scalar
    diagnostics (``focal``, ``dice``, ``cls_matched``, ``num_pairs``, ...).
    Registered as ``FastInstanceSegmentationLoss``.

    Args:
        cost_mask (float): Matcher weight on the focal mask cost. Defaults to
            ``1.0``.
        cost_dice (float): Matcher weight on the Dice cost. Defaults to ``1.0``.
        cost_class (float): Matcher weight on the classification cost. Defaults to
            ``0.0``.
        num_points (int): Number of sampled points for the mask cost (``0`` uses
            all points). Defaults to ``0``.
        ignore_index (int): Instance/segment id treated as void. Defaults to
            ``-1``.
        loss_weight_focal (float): Weight on the focal mask loss. Defaults to
            ``1.0``.
        loss_weight_dice (float): Weight on the Dice loss. Defaults to ``1.0``.
        cls_weight_matched (float): Weight on the matched-query CE term. Defaults
            to ``2.0``.
        cls_weight_noobj (float): Weight on the no-object CE term. Defaults to
            ``0.1``.
        focal_alpha (float): Focal-loss positive-class weight. Defaults to
            ``0.25``.
        focal_gamma (float): Focal-loss focusing exponent. Defaults to ``2.0``.
        truth_label (str): ``input_dict`` key holding per-point instance ids.
            Defaults to ``"segment"``.
        aux_loss_weight (float): Scale on the summed auxiliary (deep-supervision)
            loss. Defaults to ``1.0``.
        momentum_loss_weight (float): Weight on the per-instance momentum
            regression term (``0`` disables). Defaults to ``0.0``.
        iou_loss_weight (float): Weight on the mask-IoU prediction term (``0``
            disables). Defaults to ``0.0``.

    Example:
        .. code-block:: python

            >>> import torch
            >>> from pimm.models.losses.instance_fast import FastInstanceSegmentationLoss
            >>> crit = FastInstanceSegmentationLoss(truth_label="instance")
            >>> # pred: dict with per-batch mask logits; here 1 event, Q=8 queries, P=50 points
            >>> pred = {"pred_masks": [torch.randn(8, 50)]}
            >>> inst = torch.zeros(50, dtype=torch.long)  # per-point instance ids
            >>> inst[10:30] = 1; inst[30:50] = 2
            >>> input_dict = {"instance": inst.view(-1, 1)}
            >>> loss, comp = crit(pred, input_dict)       # scalar loss + diagnostics dict
            >>> float(loss)                               # e.g. 0.71 (focal + dice)
            0.7063...
            >>> sorted(comp)[:4]
            ['cls_matched', 'cls_noobj', 'dice', 'focal']
            >>> # add pred["pred_logits"]/["pred_momentum"] + input_dict["segment"]/["momentum"]
            >>> # to activate the classification / momentum terms.
    """

    def __init__(
        self,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
        cost_class: float = 0.0,
        num_points: int = 0,
        ignore_index: int = -1,
        loss_weight_focal: float = 1.0,
        loss_weight_dice: float = 1.0,
        cls_weight_matched: float = 2.0,
        cls_weight_noobj: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        truth_label: str = "segment",
        aux_loss_weight: float = 1.0,
        momentum_loss_weight: float = 0.0,
        iou_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.aux_loss_weight = aux_loss_weight
        self.criterion = FastSingleLayerInstanceLoss(
            cost_mask=cost_mask,
            cost_dice=cost_dice,
            cost_class=cost_class,
            num_points=num_points,
            ignore_index=ignore_index,
            loss_weight_focal=loss_weight_focal,
            loss_weight_dice=loss_weight_dice,
            cls_weight_matched=cls_weight_matched,
            cls_weight_noobj=cls_weight_noobj,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            truth_label=truth_label,
            momentum_loss_weight=momentum_loss_weight,
            iou_loss_weight=iou_loss_weight,
        )

    def _target_cache(self, pred: Dict, input_dict: Dict) -> List[Dict]:
        pred_masks_list = pred["pred_masks"]
        counts = [pm.shape[1] for pm in pred_masks_list]
        return build_target_cache(
            input_dict[self.criterion.truth_label],
            counts,
            segment=input_dict.get("segment", None),
            ignore_index=self.criterion.ignore_index,
            device=pred_masks_list[0].device if pred_masks_list else None,
        )

    def forward(self, pred: Dict, input_dict: Dict) -> torch.Tensor:
        if hasattr(pred, "outputs"):
            pred = pred.outputs

        target_cache = self._target_cache(pred, input_dict)
        final_loss, components = self.criterion(pred, input_dict, target_cache)

        if "aux_outputs" in pred and pred["aux_outputs"]:
            aux_loss = pred["pred_masks"][0].new_tensor(0.0)
            for layer_idx, aux_pred in enumerate(pred["aux_outputs"]):
                aux_loss_val, aux_comp = self.criterion(
                    aux_pred, input_dict, target_cache
                )
                aux_loss = aux_loss + aux_loss_val
                for key in ("focal", "dice", "cls_matched", "cls_noobj", "momentum", "iou"):
                    if key in aux_comp:
                        components[f"aux_{key}_L{layer_idx}"] = aux_comp[key]

            final_loss = final_loss + self.aux_loss_weight * aux_loss

        return final_loss, components


class FastSingleLayerInstanceLoss(SingleLayerInstanceLoss):
    def __init__(
        self,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
        cost_class: float = 0.0,
        num_points: int = 0,
        ignore_index: int = -1,
        loss_weight_focal: float = 1.0,
        loss_weight_dice: float = 1.0,
        cls_weight_matched: float = 2.0,
        cls_weight_noobj: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        truth_label: str = "segment",
        segment_key: str = "segment",
        momentum_loss_weight: float = 0.0,
        iou_loss_weight: float = 0.0,
    ):
        super().__init__(
            cost_mask=cost_mask,
            cost_dice=cost_dice,
            cost_class=cost_class,
            num_points=num_points,
            ignore_index=ignore_index,
            loss_weight_focal=loss_weight_focal,
            loss_weight_dice=loss_weight_dice,
            cls_weight_matched=cls_weight_matched,
            cls_weight_noobj=cls_weight_noobj,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            truth_label=truth_label,
            momentum_loss_weight=momentum_loss_weight,
            iou_loss_weight=iou_loss_weight,
        )
        self.segment_key = segment_key
        self.matcher = FastHungarianMatcher(
            cost_class=cost_class,
            cost_mask=cost_mask,
            cost_dice=cost_dice,
            num_points=num_points,
            ignore_index=ignore_index,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

    def _build_target_cache(self, pred_masks_list, input_dict):
        counts = [pm.shape[1] for pm in pred_masks_list]
        return build_target_cache(
            input_dict[self.truth_label],
            counts,
            segment=input_dict.get(self.segment_key),
            ignore_index=self.ignore_index,
            device=pred_masks_list[0].device if pred_masks_list else None,
        )

    def _batch_tensor(self, tensor_or_list, batch_idx, counts, device, valid_mask):
        tensor_b = split_by_counts(tensor_or_list, counts)[batch_idx].to(device)
        if valid_mask is not None:
            tensor_b = tensor_b[valid_mask]
        return tensor_b

    def _mean_per_instance(self, values, inverse, num_instances):
        out = values.new_zeros((num_instances,) + values.shape[1:])
        counts = values.new_zeros((num_instances,) + (1,) * (values.dim() - 1))
        out.index_add_(0, inverse, values)
        ones = torch.ones_like(inverse, dtype=values.dtype)
        counts.index_add_(0, inverse, ones.view(-1, *([1] * (values.dim() - 1))))
        return out / counts.clamp_min(1)

    def _classification_terms(self, logits_b, inst_class, idx_q, idx_gt):
        C = logits_b.shape[-1] - 1
        cls_loss_b = logits_b.new_tensor(0.0)
        cls_count_b = 0
        ce_matched = None
        ce_noobj = None

        if idx_q.numel() > 0 and inst_class is not None:
            logits_matched = logits_b[idx_q.long()]
            target_matched = inst_class[idx_gt.long()]
            ce_matched = F.cross_entropy(
                logits_matched, target_matched, reduction="mean"
            )
            cls_loss_b = cls_loss_b + self.cls_weight_matched * ce_matched
            cls_count_b += 1

        mask_unmatched = torch.ones(logits_b.shape[0], dtype=torch.bool, device=logits_b.device)
        if idx_q.numel() > 0:
            mask_unmatched[idx_q.long()] = False
        if mask_unmatched.any():
            logits_unmatched = logits_b[mask_unmatched]
            target_noobj = logits_unmatched.new_full(
                (logits_unmatched.shape[0],), C, dtype=torch.long
            )
            ce_noobj = F.cross_entropy(
                logits_unmatched, target_noobj, reduction="mean"
            )
            cls_loss_b = cls_loss_b + self.cls_weight_noobj * ce_noobj
            cls_count_b += 1

        return cls_loss_b, cls_count_b, ce_matched, ce_noobj

    def _matched_mask_losses(self, pred_sel, target_masks, target_sizes, idx_gt):
        gt_sel = target_masks[idx_gt]
        prob = pred_sel.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(
            pred_sel, gt_sel, reduction="none"
        )
        p_t = prob * gt_sel + (1 - prob) * (1 - gt_sel)
        focal_weight = (1 - p_t) ** self.focal_gamma
        alpha_t = self.focal_alpha * gt_sel + (1 - self.focal_alpha) * (1 - gt_sel)
        focal_per_pair = (alpha_t * focal_weight * ce_loss).mean(dim=1)

        numerator = 2 * (prob * gt_sel).sum(dim=1)
        denominator = prob.sum(dim=1) + target_sizes[idx_gt].to(prob.dtype)
        dice = 1 - (numerator + 1) / (denominator + 1)
        return focal_per_pair, dice

    def forward(self, pred: Dict[str, List[torch.Tensor]], input_dict: Dict, target_cache=None):
        assert isinstance(pred, dict) and "pred_masks" in pred, (
            "pred must be a dict with key 'pred_masks'"
        )
        assert self.truth_label in input_dict, (
            f"input_dict must contain key '{self.truth_label}'"
        )

        pred_masks_list = pred["pred_masks"]
        counts = [pm.shape[1] for pm in pred_masks_list]
        if target_cache is None:
            target_cache = self._build_target_cache(pred_masks_list, input_dict)

        indices = self.matcher(
            {
                "pred_masks": pred_masks_list,
                "pred_logits": pred.get("pred_logits", None),
            },
            target_cache,
        )

        total_loss_focal = pred_masks_list[0].new_tensor(0.0)
        total_loss_dice = pred_masks_list[0].new_tensor(0.0)
        total_loss_cls = pred_masks_list[0].new_tensor(0.0)
        total_loss_momentum = pred_masks_list[0].new_tensor(0.0)
        total_loss_iou = pred_masks_list[0].new_tensor(0.0)
        num_batches_with_loss = 0
        num_batches_with_momentum = 0
        num_batches_with_iou = 0

        total_focal = pred_masks_list[0].new_tensor(0.0)
        total_dice = pred_masks_list[0].new_tensor(0.0)
        total_pairs = 0
        total_ce_matched = pred_masks_list[0].new_tensor(0.0)
        count_ce_matched = pred_masks_list[0].new_tensor(0.0)
        total_ce_noobj = pred_masks_list[0].new_tensor(0.0)
        count_ce_noobj = pred_masks_list[0].new_tensor(0.0)
        total_momentum = pred_masks_list[0].new_tensor(0.0)
        count_momentum_batches = 0
        total_iou = pred_masks_list[0].new_tensor(0.0)
        count_iou_batches = 0
        queries_total = pred_masks_list[0].new_tensor(0.0)
        gt_instances_total = pred_masks_list[0].new_tensor(0.0)

        for batch_idx, (pm_b_full, meta, (idx_q, idx_gt)) in enumerate(
            zip(pred_masks_list, target_cache, indices)
        ):
            queries_total = queries_total + pm_b_full.shape[0]

            if idx_q.numel() == 0 or meta["num_instances"] == 0:
                continue

            pm_b = pm_b_full[:, meta["valid_mask"]]
            gt_instances_total = gt_instances_total + meta["num_instances"]
            target_masks = get_target_masks(meta, pm_b.dtype)

            idx_q = idx_q.to(pm_b.device).long()
            idx_gt = idx_gt.to(pm_b.device).long()
            pred_sel = pm_b[idx_q]
            num_pairs_b = pred_sel.shape[0]
            if num_pairs_b == 0:
                continue

            focal_per_pair, dice = self._matched_mask_losses(
                pred_sel,
                target_masks,
                meta["target_sizes"].to(pm_b.device),
                idx_gt,
            )
            focal_loss_b = focal_per_pair.sum() / num_pairs_b
            dice_loss_b = dice.sum() / num_pairs_b
            total_loss_focal = total_loss_focal + focal_loss_b
            total_loss_dice = total_loss_dice + dice_loss_b
            total_focal = total_focal + focal_per_pair.sum()
            total_dice = total_dice + dice.sum()

            total_pairs += num_pairs_b
            num_batches_with_loss += 1

            if "pred_logits" in pred and self.segment_key in input_dict:
                logits_b = pred["pred_logits"][batch_idx]
                C = logits_b.shape[-1] - 1
                inst_class = meta["inst_class"].to(pm_b.device)
                cls_loss_b, cls_count_b, ce_matched, ce_noobj = self._classification_terms(
                    logits_b, inst_class, idx_q, idx_gt
                )
                if cls_count_b > 0:
                    total_loss_cls = total_loss_cls + (cls_loss_b / cls_count_b)
                if ce_matched is not None:
                    total_ce_matched = total_ce_matched + ce_matched
                    count_ce_matched = count_ce_matched + 1.0
                if ce_noobj is not None:
                    total_ce_noobj = total_ce_noobj + ce_noobj
                    count_ce_noobj = count_ce_noobj + 1.0

            if (
                self.momentum_loss_weight > 0
                and "pred_momentum" in pred
                and "momentum" in input_dict
            ):
                momentum_gt = self._batch_tensor(
                    input_dict["momentum"],
                    batch_idx,
                    counts,
                    pm_b.device,
                    meta["valid_mask"],
                )
                mom_pred_b = pred["pred_momentum"][batch_idx].to(pm_b.device)
                mom_gt_per_inst = self._mean_per_instance(
                    momentum_gt, meta["inverse"], meta["num_instances"]
                )
                loss_momentum_b = self._compute_momentum_loss(
                    mom_pred_b, mom_gt_per_inst, idx_q, idx_gt
                )
                total_loss_momentum = total_loss_momentum + loss_momentum_b
                total_momentum = total_momentum + loss_momentum_b
                num_batches_with_momentum += 1
                count_momentum_batches += 1

            if self.iou_loss_weight > 0 and "pred_iou" in pred:
                gt_sel = target_masks[idx_gt]
                iou_pred_b = pred["pred_iou"][batch_idx].to(pm_b.device)
                loss_iou_b = self._compute_iou_loss(
                    iou_pred_b, pred_sel, gt_sel, idx_q
                )
                total_loss_iou = total_loss_iou + loss_iou_b
                total_iou = total_iou + loss_iou_b
                num_batches_with_iou += 1
                count_iou_batches += 1

        if num_batches_with_momentum > 0:
            total_loss_momentum = total_loss_momentum / num_batches_with_momentum
        if num_batches_with_iou > 0:
            total_loss_iou = total_loss_iou / num_batches_with_iou

        denom = max(num_batches_with_loss, 1)
        loss_masks = self.loss_weight_focal * (
            total_loss_focal / denom
        ) + self.loss_weight_dice * (total_loss_dice / denom)
        loss_cls = total_loss_cls / denom
        loss = (
            loss_masks
            + loss_cls
            + self.momentum_loss_weight * total_loss_momentum
            + self.iou_loss_weight * total_loss_iou
        )

        unmatched_queries = queries_total - total_pairs
        unmatched_gt = gt_instances_total - total_pairs
        components = {
            "focal": total_focal / max(total_pairs, 1),
            "dice": total_dice / max(total_pairs, 1),
            "cls_matched": total_ce_matched / count_ce_matched.clamp_min(1),
            "cls_noobj": total_ce_noobj / count_ce_noobj.clamp_min(1),
            "momentum": total_momentum / max(count_momentum_batches, 1),
            "iou": total_iou / max(count_iou_batches, 1),
            "num_pairs": total_pairs,
            "queries_total": queries_total,
            "gt_instances_total": gt_instances_total,
            "unmatched_queries": unmatched_queries,
            "unmatched_gt": unmatched_gt,
            "num_cls_matched": count_ce_matched,
            "num_cls_noobj": count_ce_noobj,
        }
        return loss, components


@LOSSES.register_module()
class InstanceSegmentationLoss(FastInstanceSegmentationLoss):
    """DEPRECATED alias for :class:`FastInstanceSegmentationLoss`.

    The original O(QxJxP) non-fast instance loss + ``HungarianMatcher`` were removed;
    they were verified bitwise-equal to the fast path. This name is kept registered
    only so historical dumped configs (`type="InstanceSegmentationLoss"`) still build.
    New configs should use ``FastInstanceSegmentationLoss``.
    """
