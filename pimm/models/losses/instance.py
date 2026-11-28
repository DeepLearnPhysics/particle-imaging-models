"""Instance mask, class, and query-head losses for Panda detectors."""

# Focal/Dice matching follows OneFormer3D (github.com/oneformer3d/oneformer3d).

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from pimm.models.losses.builder import LOSSES
from pimm.models.panda_v1.detector_utils.matcher_fast import (
    FastHungarianMatcher as PublicHungarianMatcher,
    build_target_cache,
    get_target_masks,
    split_by_counts,
)

_HEAD_CRITERIA = {
    "continuous": {"type": "SmoothL1RegressionLoss", "beta": 1.0, "reduction": "mean"},
    "categorical": {"type": "CrossEntropyHeadLoss", "reduction": "mean"},
}


@LOSSES.register_module()
class FastUnifiedInstanceLoss(nn.Module):
    """Combine instance mask/class losses with configured query-head losses.

    Non-overlap mode builds masks from per-point instance ids and aggregates
    each matched instance's point targets for its query heads. Overlap mode uses
    the configured multi-hot membership matrix, validity mask, and per-instance
    head targets. ``aux_outputs`` provide deep supervision and are scaled by
    ``aux_loss_weight``.

    ``panda_v2=True`` selects all-query classification and the V2 matcher;
    the default retains the public matched/no-object classification objective.
    ``consistent_matching=True`` reuses the final assignment for auxiliary layers.

    ``query_heads`` is a list of named mappings. Continuous heads use a
    regression criterion; categorical heads use cross-entropy. Each mapping
    selects its prediction key, target key, target aggregation, and loss weight.
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
        truth_label: str = "instance",
        segment_key: str = "segment",
        aux_loss_weight: float = 1.0,
        *,
        query_heads,
        overlap: bool = False,
        membership_key: str = "inst_pe",
        valid_key: str = "ring_valid",
        mask_pe_thresh: float = 0.0,
        noobj_mask_loss_weight: float = 0.0,
        mask_loss_normalization: str = "event",
        charge_weighted_masks: bool = False,
        fused: bool = False,
        panda_v2: bool = False,
        consistent_matching: bool = False,
    ):
        super().__init__()
        self.aux_loss_weight = aux_loss_weight
        self.overlap = bool(overlap)
        self.consistent_matching = bool(consistent_matching)
        if self.consistent_matching and fused:
            raise ValueError("consistent_matching requires fused=False")
        self.criterion = FastUnifiedSingleLayerLoss(
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
            segment_key=segment_key,
            query_heads=query_heads,
            overlap=overlap,
            membership_key=membership_key,
            valid_key=valid_key,
            mask_pe_thresh=mask_pe_thresh,
            noobj_mask_loss_weight=noobj_mask_loss_weight,
            mask_loss_normalization=mask_loss_normalization,
            charge_weighted_masks=charge_weighted_masks,
            fused=fused,
            panda_v2=panda_v2,
        )

    def forward(self, pred: Dict, input_dict: Dict):
        target_cache = self.criterion.build_targets(pred, input_dict)
        indices = (
            self.criterion.match_targets(pred, target_cache)
            if self.consistent_matching
            else None
        )
        final_loss, components = self.criterion(pred, input_dict, target_cache, indices)

        if "aux_outputs" in pred and pred["aux_outputs"]:
            aux_loss = pred["pred_masks"][0].new_tensor(0.0)
            for layer_idx, aux_pred in enumerate(pred["aux_outputs"]):
                aux_loss_val, aux_comp = self.criterion(
                    aux_pred, input_dict, target_cache, indices
                )
                aux_loss = aux_loss + aux_loss_val
                for key, value in aux_comp.items():
                    if key in self.criterion.aux_component_keys:
                        components[f"aux_{key}_L{layer_idx}"] = value
            final_loss = final_loss + self.aux_loss_weight * aux_loss

        return final_loss, components


class FastUnifiedSingleLayerLoss(nn.Module):
    def __init__(
        self,
        cost_mask=1.0,
        cost_dice=1.0,
        cost_class=0.0,
        num_points=0,
        ignore_index=-1,
        loss_weight_focal=1.0,
        loss_weight_dice=1.0,
        cls_weight_matched=2.0,
        cls_weight_noobj=0.1,
        focal_alpha=0.25,
        focal_gamma=2.0,
        truth_label="instance",
        segment_key="segment",
        *,
        query_heads,
        overlap=False,
        membership_key="inst_pe",
        valid_key="ring_valid",
        mask_pe_thresh=0.0,
        noobj_mask_loss_weight=0.0,
        mask_loss_normalization="event",
        charge_weighted_masks=False,
        fused=False,
        panda_v2=False,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight_focal = loss_weight_focal
        self.loss_weight_dice = loss_weight_dice
        self.cls_weight_matched = cls_weight_matched
        self.cls_weight_noobj = cls_weight_noobj
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_points = num_points
        self.truth_label = truth_label
        self.segment_key = segment_key
        self.panda_v2 = bool(panda_v2)
        matcher = PublicHungarianMatcher
        if self.panda_v2:
            from pimm.models.panda_v2.detector_utils.matcher import FastHungarianMatcher

            matcher = FastHungarianMatcher
        self.matcher = matcher(
            cost_class=cost_class,
            cost_mask=cost_mask,
            cost_dice=cost_dice,
            num_points=num_points,
            ignore_index=-1 if self.panda_v2 else ignore_index,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        # The overlap path uses its own Hungarian cost.
        self.cost_mask = float(self.matcher.cost_mask)
        self.cost_dice = float(self.matcher.cost_dice)
        self.cost_class = float(self.matcher.cost_class)
        self.overlap = bool(overlap)
        self.membership_key = membership_key
        self.valid_key = valid_key
        self.mask_pe_thresh = float(mask_pe_thresh)
        self.noobj_mask_loss_weight = float(noobj_mask_loss_weight)
        # Charge weighting reduces the contribution from diffuse single-photon sites.
        self.charge_weighted_masks = bool(charge_weighted_masks)
        # Instance normalization gives each matched instance equal weight across events.
        if mask_loss_normalization not in ("event", "instance"):
            raise ValueError(
                f"mask_loss_normalization must be 'event' or 'instance', got {mask_loss_normalization!r}"
            )
        self.mask_loss_normalization = mask_loss_normalization
        self.query_heads = query_heads
        self.head_losses = nn.ModuleDict()
        for head in self.query_heads:
            kind = head.get("kind", "continuous")
            self.head_losses[head["name"]] = LOSSES.build(
                head.get("criterion", _HEAD_CRITERIA[kind])
            )
        self.aux_component_keys = {
            "focal",
            "dice",
            "cls_matched",
            "cls_noobj",
            *[head["name"] for head in self.query_heads],
        }
        # all scenes in one pass (batched targets, costs and terms); scipy still solves each scene
        self.fused = bool(fused)
        if self.charge_weighted_masks and not self.overlap:
            raise ValueError("charge_weighted_masks needs the overlap path")
        if self.fused and not self.panda_v2:
            raise ValueError("fused=True requires panda_v2=True")
        if self.fused and (
            not self.overlap
            or self.query_heads
            or self.noobj_mask_loss_weight > 0
            or self.charge_weighted_masks
        ):
            raise ValueError(
                "fused=True takes the overlap path without query heads or a no-object mask loss"
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

        mask_unmatched = torch.ones(
            logits_b.shape[0], dtype=torch.bool, device=logits_b.device
        )
        if idx_q.numel() > 0:
            mask_unmatched[idx_q.long()] = False
        if mask_unmatched.any():
            logits_unmatched = logits_b[mask_unmatched]
            target_noobj = logits_unmatched.new_full(
                (logits_unmatched.shape[0],), C, dtype=torch.long
            )
            ce_noobj = F.cross_entropy(logits_unmatched, target_noobj, reduction="mean")
            cls_loss_b = cls_loss_b + self.cls_weight_noobj * ce_noobj
            cls_count_b += 1

        return cls_loss_b, cls_count_b, ce_matched, ce_noobj

    def _all_query_classification_terms(self, logits_b, inst_class, idx_q, idx_gt):
        no_object = logits_b.shape[-1] - 1
        ce_matched = None
        ce_noobj = None

        if logits_b.shape[0] == 0:
            return logits_b.new_tensor(0.0), ce_matched, ce_noobj

        target = torch.full(
            (logits_b.shape[0],),
            no_object,
            dtype=torch.long,
            device=logits_b.device,
        )
        if idx_q.numel() > 0:
            if inst_class is None:
                raise ValueError("Matched queries require per-instance class targets")
            target[idx_q.long()] = inst_class[idx_gt.long()].to(logits_b.device)

        class_weight = logits_b.new_ones(logits_b.shape[-1])
        class_weight[no_object] = self.cls_weight_noobj
        cls_loss_b = self.cls_weight_matched * F.cross_entropy(
            logits_b,
            target,
            weight=class_weight,
            reduction="mean",
        )

        if idx_q.numel() > 0:
            logits_matched = logits_b[idx_q.long()]
            target_matched = target[idx_q.long()]
            ce_matched = F.cross_entropy(
                logits_matched, target_matched, reduction="mean"
            )

        mask_unmatched = torch.ones(
            logits_b.shape[0], dtype=torch.bool, device=logits_b.device
        )
        if idx_q.numel() > 0:
            mask_unmatched[idx_q.long()] = False
        if mask_unmatched.any():
            logits_unmatched = logits_b[mask_unmatched]
            ce_noobj = F.cross_entropy(
                logits_unmatched, target[mask_unmatched], reduction="mean"
            )

        return cls_loss_b, ce_matched, ce_noobj

    def _matched_mask_losses(self, pred_sel, target_masks, target_sizes, idx_gt):
        gt_sel = target_masks[idx_gt]
        prob = pred_sel.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(pred_sel, gt_sel, reduction="none")
        p_t = prob * gt_sel + (1 - prob) * (1 - gt_sel)
        focal_weight = (1 - p_t) ** self.focal_gamma
        alpha_t = self.focal_alpha * gt_sel + (1 - self.focal_alpha) * (1 - gt_sel)
        focal_per_pair = (alpha_t * focal_weight * ce_loss).mean(dim=1)

        numerator = 2 * (prob * gt_sel).sum(dim=1)
        denominator = prob.sum(dim=1) + target_sizes[idx_gt].to(prob.dtype)
        dice = 1 - (numerator + 1) / (denominator + 1)
        return focal_per_pair, dice

    def build_targets(self, pred, input_dict):
        if "pred_logits" in pred and self.segment_key not in input_dict:
            raise KeyError(f"the model predicts classes but the batch has no {self.segment_key!r} to train them on")
        pred_masks_list = pred["pred_masks"]
        if self.fused:
            return self._batched_targets(pred_masks_list, input_dict)
        if self.overlap:
            return self._build_overlap_target_cache(pred_masks_list, input_dict)
        return self._build_target_cache(pred_masks_list, input_dict)

    def _build_overlap_target_cache(self, pred_masks_list, input_dict):
        """Build multi-hot masks and per-instance targets from a membership matrix."""
        assert (
            self.membership_key in input_dict
        ), f"overlap loss needs membership matrix input_dict[{self.membership_key!r}]"
        device = pred_masks_list[0].device
        counts = [pm.shape[1] for pm in pred_masks_list]
        membership = input_dict[self.membership_key]  # (N_total, R_max)
        R_max = membership.shape[1]
        valid_all = input_dict.get(self.valid_key)
        point_offsets = torch.tensor(
            [0] + list(torch.tensor(counts).cumsum(0).tolist())
        )

        cache = []
        for b, P_b in enumerate(counts):
            p0, p1 = point_offsets[b].item(), point_offsets[b + 1].item()
            r0, r1 = b * R_max, (b + 1) * R_max
            mem_b = membership[p0:p1].to(device)  # (P_b, R_max)
            if valid_all is not None:
                valid_b = valid_all[r0:r1].to(device).bool()
            else:
                valid_b = (mem_b > self.mask_pe_thresh).any(0)
            masks = (mem_b[:, valid_b] > self.mask_pe_thresh).T.float()  # (R, P_b)
            inst_class = None
            if self.panda_v2 and self.segment_key in input_dict:
                inst_class = input_dict[self.segment_key][r0:r1].to(device)
                if inst_class.dim() > 1:
                    assert (
                        inst_class.shape[-1] == 1
                    ), "overlap class targets must be scalar per instance"
                    inst_class = inst_class.squeeze(-1)
                inst_class = inst_class[valid_b].long()
            meta = {
                "num_instances": int(masks.shape[0]),
                "valid_mask": torch.ones(P_b, dtype=torch.bool, device=device),
                "target_masks": masks,
                "target_sizes": masks.sum(1),
                "inst_class": inst_class,
                "ring_slice": (r0, r1),
                "valid_rings": valid_b,
            }
            if self.charge_weighted_masks:
                charge = mem_b[:, valid_b].float().sum(1)
                meta["site_weight"] = charge / charge.mean().clamp(min=1e-6)
                meta["target_sizes"] = masks @ meta["site_weight"]
            cache.append(meta)
        return cache

    @staticmethod
    def _align_regression_shapes(pred, target):
        if pred.dim() == 1 and target.dim() == 2 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        if pred.dim() == 2 and pred.shape[-1] == 1 and target.dim() == 1:
            pred = pred.squeeze(-1)
        return pred, target

    def _mode_per_instance(self, values, inverse, num_instances):
        values = values.long().view(-1)
        out = values.new_zeros(num_instances)
        for inst in range(num_instances):
            v = values[inverse == inst]
            if v.numel() == 0:
                continue
            out[inst] = torch.bincount(v).argmax()
        return out

    def _aggregate_target(self, values, inverse, num_instances, aggregation):
        if aggregation == "mean":
            return self._mean_per_instance(values.float(), inverse, num_instances)
        if aggregation == "first":
            out = values.new_zeros((num_instances,) + values.shape[1:])
            for inst in range(num_instances):
                sel = values[inverse == inst]
                if sel.numel() > 0:
                    out[inst] = sel[0]
            return out
        if aggregation == "mode":
            return self._mode_per_instance(values, inverse, num_instances)
        raise ValueError(f"Unknown head aggregation: {aggregation!r}")

    def _compute_head_losses(
        self, pred, input_dict, batch_idx, counts, meta, idx_q, idx_gt, device
    ):
        losses = {}
        for head in self.query_heads:
            name = head["name"]
            kind = head.get("kind", "continuous")
            pred_key = head.get("pred_key", f"pred_{name}")
            target_key = head.get("target_key", name)
            weight = float(head.get("loss_weight", 1.0))
            if weight == 0.0:
                continue
            missing = pred_key not in pred or target_key not in input_dict
            if missing:
                if head.get("required", True):
                    raise KeyError(
                        f"Missing query head {name!r}: expected prediction "
                        f"{pred_key!r} and input {target_key!r}"
                    )
                continue

            if self.overlap:
                # per-instance targets: input is a per-instance (ring) tensor.
                r0, r1 = meta["ring_slice"]
                target_full = input_dict[target_key][r0:r1].to(device)
                target_per_inst = target_full[meta["valid_rings"]]
            else:
                target = self._batch_tensor(
                    input_dict[target_key],
                    batch_idx,
                    counts,
                    device,
                    meta["valid_mask"],
                )
                target_per_inst = self._aggregate_target(
                    target,
                    meta["inverse"],
                    meta["num_instances"],
                    head.get("aggregation", "mean" if kind == "continuous" else "mode"),
                )

            pred_b = pred[pred_key][batch_idx].to(device)
            pred_matched = pred_b[idx_q.long()]
            target_matched = target_per_inst[idx_gt.long()]
            if head.get(
                "valid_key"
            ):  # supervise only the instances whose target is known
                if not self.overlap:
                    raise ValueError(
                        f"query head {name!r}: valid_key needs the overlap path"
                    )
                known = (
                    input_dict[head["valid_key"]][r0:r1]
                    .to(device)[meta["valid_rings"]]
                    .bool()[idx_gt.long()]
                )
                if not known.any():
                    losses[name] = pred_matched.sum() * 0.0
                    continue
                pred_matched, target_matched = (
                    pred_matched[known],
                    target_matched[known],
                )
            if kind == "continuous":
                target_matched = target_matched.to(pred_matched.dtype)
                pred_matched, target_matched = self._align_regression_shapes(
                    pred_matched, target_matched
                )
            losses[name] = weight * self.head_losses[name](pred_matched, target_matched)
        return losses

    def _weighted_mask_losses(self, pred_sel, target_masks, weight, idx_gt):
        """_matched_mask_losses with each site counted by ``weight`` (N,)."""
        gt_sel = target_masks[idx_gt]
        prob = pred_sel.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(pred_sel, gt_sel, reduction="none")
        p_t = prob * gt_sel + (1 - prob) * (1 - gt_sel)
        alpha_t = self.focal_alpha * gt_sel + (1 - self.focal_alpha) * (1 - gt_sel)
        w = weight.to(prob.dtype)
        focal_per_pair = (alpha_t * (1 - p_t) ** self.focal_gamma * ce_loss * w).mean(
            dim=1
        )
        numerator = 2 * (prob * gt_sel * w).sum(dim=1)
        denominator = (prob * w).sum(dim=1) + (gt_sel * w).sum(dim=1)
        dice = 1 - (numerator + 1) / (denominator + 1)
        return focal_per_pair, dice

    def _overlap_match(
        self,
        pred_masks_b,
        pred_logits_b,
        gt_masks,
        inst_class=None,
        site_weight=None,
    ):
        with torch.autocast(device_type=pred_masks_b.device.type, enabled=False):
            ml = pred_masks_b.float()  # (Q, P)
            R = gt_masks.shape[0]
            if R == 0 or ml.shape[0] == 0:
                empty = torch.zeros(0, dtype=torch.long, device=ml.device)
                return empty, empty
            N = ml.shape[1]
            prob = ml.sigmoid()
            a, g = self.focal_alpha, self.focal_gamma
            pos = a * (1 - prob) ** g * (-F.logsigmoid(ml))
            neg = (1 - a) * prob**g * (-F.logsigmoid(-ml))
            if site_weight is not None:  # charge-weighted sites
                w = site_weight.to(ml.dtype)
                pos, neg, prob_w = pos * w, neg * w, prob * w
            else:
                w, prob_w = None, prob
            cost_mask = (pos @ gt_masks.T + neg @ (1 - gt_masks).T) / max(N, 1)
            num = 2 * (prob_w @ gt_masks.T)
            den = (
                prob_w.sum(1, keepdim=True)
                + (gt_masks.sum(1) if w is None else gt_masks @ w)[None, :]
            )
            cost_dice = 1 - (num + 1) / (den + 1)
            C = self.cost_mask * cost_mask + self.cost_dice * cost_dice
            if (
                pred_logits_b is not None
                and self.cost_class > 0
                and inst_class is not None
            ):
                num_classes = pred_logits_b.shape[-1] - 1
                out_prob = F.softmax(
                    pred_logits_b[:, :num_classes].float(),
                    dim=-1,
                )
                class_cost = -torch.log(
                    out_prob[
                        :,
                        inst_class.to(ml.device).clamp(0, num_classes - 1),
                    ]
                    + 1e-8
                )
                C = C + self.cost_class * class_cost
            C = torch.nan_to_num(
                C, nan=1e4, posinf=1e4, neginf=-1e4
            )  # an fp16 overflow step: GradScaler skips it, scipy must not raise
            qi, ri = linear_sum_assignment(C.detach().cpu().numpy())
        dev = ml.device
        return (
            torch.as_tensor(qi, dtype=torch.long, device=dev),
            torch.as_tensor(ri, dtype=torch.long, device=dev),
        )

    def _batched_targets(self, pred_masks_list, input_dict):
        """``_build_overlap_target_cache`` for all scenes, padded to (scenes, slots, points), without host reads."""
        device = pred_masks_list[0].device
        queries, points = [pm.shape[0] for pm in pred_masks_list], [
            pm.shape[1] for pm in pred_masks_list
        ]
        counts = torch.tensor([queries, points]).to(device, non_blocking=True)
        query_valid = torch.arange(max(queries), device=device) < counts[0, :, None]
        point_valid = torch.arange(max(points), device=device) < counts[1, :, None]
        membership = input_dict[self.membership_key].to(device)  # (N_total, slots)
        scenes, slots = len(queries), membership.shape[1]
        member = membership.new_zeros(point_valid.shape + (slots,))
        member = (
            member.masked_scatter_(point_valid[..., None], membership)
            > self.mask_pe_thresh
        )
        valid = input_dict.get(self.valid_key)
        valid = (
            member.any(1)
            if valid is None
            else valid.to(device).view(scenes, slots).bool()
        )
        masks = (
            (member & valid[:, None]).transpose(1, 2).float()
        )  # (scenes, slots, points)
        inst_class = input_dict.get(self.segment_key)
        return dict(
            masks=masks,
            sizes=masks.sum(-1),
            valid=valid,
            counts=counts,
            query_counts=queries,
            inst_class=(
                None
                if inst_class is None
                else inst_class.to(device).view(scenes, slots).long()
            ),
            query_valid=query_valid,
            entries=query_valid[:, :, None] & point_valid[:, None, :],
        )

    @torch.no_grad()
    def _batched_match(self, masks, logits, targets):
        """``_overlap_match`` for all scenes: the costs in one batch, one copy to the host, scipy per scene.

        Returns a (5, pairs) device tensor of (scene, query, slot, pairs in its scene, query row in the batch)
        and each scene's pair count.
        """
        with torch.autocast(device_type=masks.device.type, enabled=False):
            ml = (
                masks.float()
            )  # (scenes, queries, points); padding at -1e4 adds nothing
            prob = ml.sigmoid()
            a, g = self.focal_alpha, self.focal_gamma
            pos = a * (1 - prob) ** g * (-F.logsigmoid(ml))
            neg = (1 - a) * prob**g * (-F.logsigmoid(-ml))
            gt = targets["masks"].transpose(1, 2)
            cost_mask = (pos @ gt + neg @ (1 - gt)) / targets["counts"][
                1, :, None, None
            ].clamp_min(1)
            num = 2 * (prob @ gt)
            den = prob.sum(-1, keepdim=True) + targets["sizes"][:, None, :]
            cost_dice = 1 - (num + 1) / (den + 1)
            C = self.cost_mask * cost_mask + self.cost_dice * cost_dice
            if (
                logits is not None
                and self.cost_class > 0
                and targets["inst_class"] is not None
            ):
                num_classes = logits.shape[-1] - 1
                padded = logits.new_zeros(
                    targets["query_valid"].shape + logits.shape[-1:]
                )
                padded = padded.masked_scatter(
                    targets["query_valid"][..., None], logits
                )
                out_prob = F.softmax(padded[..., :num_classes].float(), dim=-1)
                index = (
                    targets["inst_class"]
                    .clamp(0, num_classes - 1)[:, None, :]
                    .expand(-1, C.shape[1], -1)
                )
                C = C + self.cost_class * -torch.log(out_prob.gather(2, index) + 1e-8)
            C = torch.nan_to_num(
                C, nan=1e4, posinf=1e4, neginf=-1e4
            )  # an fp16 overflow step: GradScaler skips it, scipy must not raise
            host = torch.cat((C.flatten(), targets["valid"].flatten().float())).cpu()
        cost = host[: C.numel()].view(C.shape).numpy()
        valid = host[C.numel() :].view(targets["valid"].shape).numpy() > 0
        rows, pair_counts, start = [], [], 0
        for scene, queries in enumerate(targets["query_counts"]):
            slots = np.flatnonzero(valid[scene])
            empty = queries == 0 or len(slots) == 0
            qi, ri = (
                (slots[:0], slots[:0])
                if empty
                else linear_sum_assignment(cost[scene, :queries][:, slots])
            )
            rows.append(
                np.stack(
                    [
                        np.full(len(qi), scene),
                        qi,
                        slots[ri],
                        np.full(len(qi), len(qi)),
                        start + qi,
                    ]
                )
            )
            pair_counts.append(len(qi))
            start += queries
        pairs = torch.from_numpy(np.concatenate(rows, 1).astype(np.int64))
        return pairs.to(masks.device, non_blocking=True), pair_counts

    def _fused_forward(self, pred, input_dict, targets):
        """``forward`` with every scene in one batch: the same matching, terms and normalization."""
        flat = torch.cat([pm.reshape(-1) for pm in pred["pred_masks"]])
        # zero that keeps the mask head in DDP's static graph when a rank's scenes have no instances
        zero = flat.sum() * 0.0
        masks = flat.new_full(targets["entries"].shape, -1e4).masked_scatter(
            targets["entries"], flat
        )
        logits = torch.cat(pred["pred_logits"]) if "pred_logits" in pred else None
        pairs, pair_counts = self._batched_match(masks, logits, targets)
        scene, query, slot, scene_pairs, row = pairs
        num_pairs = sum(pair_counts)
        query_counts = targets["query_counts"]

        total_loss_cls = total_ce_matched = total_ce_noobj = zero
        num_batches_with_cls = count_ce_matched = count_ce_noobj = 0
        if logits is not None and self.segment_key in input_dict:
            no_object = logits.shape[-1] - 1
            target = torch.full(
                logits.shape[:1], no_object, dtype=torch.long, device=logits.device
            )
            target[row] = targets["inst_class"][scene, slot]
            ce = F.cross_entropy(logits, target, reduction="none")
            matched = ce.new_zeros(ce.shape)
            matched[row] = 1
            weight = torch.where(target == no_object, self.cls_weight_noobj, 1.0)
            terms = torch.stack(
                (ce * weight, weight, ce * matched, ce * (1 - matched), matched)
            )
            per_scene = terms.new_zeros((len(terms),) + targets["query_valid"].shape)
            weighted, weights, ce_matched, ce_noobj, n_matched = (
                per_scene.masked_scatter(targets["query_valid"], terms).sum(-1)
            )
            total_loss_cls = (
                zero
                + (self.cls_weight_matched * weighted / weights.clamp_min(1e-12)).sum()
            )
            total_ce_matched = zero + (ce_matched / n_matched.clamp_min(1)).sum()
            total_ce_noobj = (
                zero
                + (ce_noobj / (targets["counts"][0] - n_matched).clamp_min(1)).sum()
            )
            num_batches_with_cls = len(query_counts)
            count_ce_matched = sum(n > 0 for n in pair_counts)
            count_ce_noobj = sum(q > n for q, n in zip(query_counts, pair_counts))

        total_focal = total_dice = total_loss_focal = total_loss_dice = zero
        if num_pairs:
            selected = masks[scene, query]
            focal, dice = self._matched_mask_losses(
                selected,
                targets["masks"].flatten(0, 1).to(selected.dtype),
                targets["sizes"].flatten(),
                scene * targets["masks"].shape[1] + slot,
            )
            focal = focal * (
                selected.shape[1] / targets["counts"][1][scene]
            )  # its mean ran over the padding
            total_focal, total_dice = zero + focal.sum(), zero + dice.sum()
            total_loss_focal = zero + (focal / scene_pairs).sum()
            total_loss_dice = zero + (dice / scene_pairs).sum()

        if self.mask_loss_normalization == "instance":
            mask_focal, mask_dice = total_focal / max(num_pairs, 1), total_dice / max(
                num_pairs, 1
            )
        else:
            denom = max(sum(n > 0 for n in pair_counts), 1)
            mask_focal, mask_dice = total_loss_focal / denom, total_loss_dice / denom
        loss = (
            self.loss_weight_focal * mask_focal
            + self.loss_weight_dice * mask_dice
            + total_loss_cls
            / (max(num_batches_with_cls, 1) if self.panda_v2 else denom)
        )
        queries_total = zero + sum(query_counts)
        gt_instances_total = zero + targets["valid"].sum()
        components = {
            "focal": total_focal / max(num_pairs, 1),
            "dice": total_dice / max(num_pairs, 1),
            "cls_matched": total_ce_matched / max(count_ce_matched, 1),
            "cls_noobj": total_ce_noobj / max(count_ce_noobj, 1),
            "num_pairs": num_pairs,
            "queries_total": queries_total,
            "gt_instances_total": gt_instances_total,
            "unmatched_queries": queries_total - num_pairs,
            "unmatched_gt": gt_instances_total - num_pairs,
        }
        return loss, components

    def match_targets(self, pred, target_cache):
        pred_masks_list = pred["pred_masks"]
        if self.overlap:
            indices = []
            for b, (pm_b, meta) in enumerate(zip(pred_masks_list, target_cache)):
                logits_b = pred["pred_logits"][b] if "pred_logits" in pred else None
                indices.append(
                    self._overlap_match(
                        pm_b,
                        logits_b,
                        meta["target_masks"],
                        meta.get("inst_class"),
                        meta.get("site_weight"),
                    )
                )
        else:
            indices = self.matcher(
                {
                    "pred_masks": pred_masks_list,
                    "pred_logits": pred.get("pred_logits", None),
                },
                target_cache,
            )

        return indices

    def forward(self, pred, input_dict, target_cache=None, indices=None):
        assert isinstance(pred, dict) and "pred_masks" in pred
        pred_masks_list = pred["pred_masks"]
        counts = [pm.shape[1] for pm in pred_masks_list]
        if target_cache is None:
            target_cache = self.build_targets(pred, input_dict)
        if self.fused:
            return self._fused_forward(pred, input_dict, target_cache)

        if indices is None:
            indices = self.match_targets(pred, target_cache)

        # Empty V2 scenes still connect the mask head for static DDP graphs.
        z = (
            sum(pm.sum() for pm in pred_masks_list) * 0.0
            if self.panda_v2
            else pred_masks_list[0].new_tensor(0.0)
        )
        total_loss_focal = z.clone()
        total_loss_dice = z.clone()
        total_loss_cls = z.clone()
        total_loss_noobj_mask = z.clone()
        total_head = {h["name"]: z.clone() for h in self.query_heads}
        num_batches_with_head = {h["name"]: 0 for h in self.query_heads}
        num_batches_with_loss = 0
        num_batches_with_cls = 0

        total_focal = z.clone()
        total_dice = z.clone()
        total_pairs = 0
        total_ce_matched = z.clone()
        count_ce_matched = z.clone()
        total_ce_noobj = z.clone()
        count_ce_noobj = z.clone()
        queries_total = z.clone()
        gt_instances_total = z.clone()

        for batch_idx, (pm_b_full, meta, (idx_q, idx_gt)) in enumerate(
            zip(pred_masks_list, target_cache, indices)
        ):
            queries_total = queries_total + pm_b_full.shape[0]
            if self.panda_v2:
                gt_instances_total = gt_instances_total + meta["num_instances"]

            idx_q = idx_q.to(pm_b_full.device).long()
            idx_gt = idx_gt.to(pm_b_full.device).long()

            # Empty-truth V2 events teach every query to predict no-object.
            if (
                self.panda_v2
                and "pred_logits" in pred
                and self.segment_key in input_dict
            ):
                logits_b = pred["pred_logits"][batch_idx]
                inst_class = meta.get("inst_class")
                if inst_class is not None:
                    inst_class = inst_class.to(logits_b.device)
                cls_loss_b, ce_m, ce_n = self._all_query_classification_terms(
                    logits_b, inst_class, idx_q, idx_gt
                )
                total_loss_cls = total_loss_cls + cls_loss_b
                num_batches_with_cls += 1
                if ce_m is not None:
                    total_ce_matched = total_ce_matched + ce_m
                    count_ce_matched = count_ce_matched + 1.0
                if ce_n is not None:
                    total_ce_noobj = total_ce_noobj + ce_n
                    count_ce_noobj = count_ce_noobj + 1.0

            if idx_q.numel() == 0 or meta["num_instances"] == 0:
                continue

            pm_b = pm_b_full[:, meta["valid_mask"]]
            if not self.panda_v2:
                gt_instances_total = gt_instances_total + meta["num_instances"]
            if self.overlap:
                target_masks = meta["target_masks"].to(pm_b.dtype)
            else:
                target_masks = get_target_masks(meta, pm_b.dtype)

            pred_sel = pm_b[idx_q]
            num_pairs_b = pred_sel.shape[0]
            if num_pairs_b == 0:
                continue

            if "site_weight" in meta:
                focal_per_pair, dice = self._weighted_mask_losses(
                    pred_sel, target_masks, meta["site_weight"].to(pm_b.device), idx_gt
                )
            else:
                focal_per_pair, dice = self._matched_mask_losses(
                    pred_sel, target_masks, meta["target_sizes"].to(pm_b.device), idx_gt
                )
            total_loss_focal = total_loss_focal + focal_per_pair.sum() / num_pairs_b
            total_loss_dice = total_loss_dice + dice.sum() / num_pairs_b
            total_focal = total_focal + focal_per_pair.sum()
            total_dice = total_dice + dice.sum()
            total_pairs += num_pairs_b
            num_batches_with_loss += 1

            # optional: push unmatched query masks to empty
            if self.noobj_mask_loss_weight > 0:
                unmatched = torch.ones(
                    pm_b.shape[0], dtype=torch.bool, device=pm_b.device
                )
                unmatched[idx_q] = False
                if unmatched.any():
                    nm = pm_b[unmatched]
                    total_loss_noobj_mask = (
                        total_loss_noobj_mask
                        + F.binary_cross_entropy_with_logits(nm, torch.zeros_like(nm))
                    )

            # primary classification (matched -> pid, unmatched -> no-obj)
            if (
                not self.panda_v2
                and not self.overlap
                and "pred_logits" in pred
                and self.segment_key in input_dict
            ):
                logits_b = pred["pred_logits"][batch_idx]
                inst_class = meta["inst_class"].to(pm_b.device)
                cls_loss_b, cls_count_b, ce_m, ce_n = self._classification_terms(
                    logits_b, inst_class, idx_q, idx_gt
                )
                if cls_count_b > 0:
                    total_loss_cls = total_loss_cls + cls_loss_b / cls_count_b
                if ce_m is not None:
                    total_ce_matched = total_ce_matched + ce_m
                    count_ce_matched = count_ce_matched + 1.0
                if ce_n is not None:
                    total_ce_noobj = total_ce_noobj + ce_n
                    count_ce_noobj = count_ce_noobj + 1.0

            head_losses = self._compute_head_losses(
                pred, input_dict, batch_idx, counts, meta, idx_q, idx_gt, pm_b.device
            )
            for name, value in head_losses.items():
                total_head[name] = total_head[name] + value
                num_batches_with_head[name] += 1

        denom = max(num_batches_with_loss, 1)
        if self.mask_loss_normalization == "instance":
            mask_focal = total_focal / max(total_pairs, 1)
            mask_dice = total_dice / max(total_pairs, 1)
        else:
            mask_focal = total_loss_focal / denom
            mask_dice = total_loss_dice / denom
        loss = (
            self.loss_weight_focal * mask_focal
            + self.loss_weight_dice * mask_dice
            + total_loss_cls
            / (max(num_batches_with_cls, 1) if self.panda_v2 else denom)
        )
        if self.noobj_mask_loss_weight > 0:
            loss = loss + self.noobj_mask_loss_weight * (total_loss_noobj_mask / denom)
        for name, value in total_head.items():
            loss = loss + value / max(num_batches_with_head[name], 1)

        components = {
            "focal": total_focal / max(total_pairs, 1),
            "dice": total_dice / max(total_pairs, 1),
            "cls_matched": total_ce_matched / count_ce_matched.clamp_min(1),
            "cls_noobj": total_ce_noobj / count_ce_noobj.clamp_min(1),
            "num_pairs": total_pairs,
            "queries_total": queries_total,
            "gt_instances_total": gt_instances_total,
            "unmatched_queries": queries_total - total_pairs,
            "unmatched_gt": gt_instances_total - total_pairs,
        }
        for name, value in total_head.items():
            components[name] = value / max(num_batches_with_head[name], 1)
        return loss, components

