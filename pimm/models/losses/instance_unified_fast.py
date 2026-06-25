"""Unified instance loss for detector-v4.

Superset of the fast instance-segmentation loss that additionally supervises an
arbitrary set of per-query heads (categorical *and* continuous), and supports an
overlap-aware (multi-hot) mode for overlapping instances.

Two modes, selected by the ``overlap`` flag (injected per-label by
``UnifiedDetector``):

* ``overlap=False`` (default) -- mutually-exclusive masks. Ground truth comes
  from per-point ``instance`` ids (one-hot), matched with the cached fast
  Hungarian matcher, mask/dice/cls identical to ``FastInstanceSegmentationLoss``.
  Extra heads are supervised on matched pairs: continuous heads aggregate the
  per-point target per-instance (mean/first) and apply a regression criterion;
  categorical heads aggregate per-instance by mode/first and apply CE.

* ``overlap=True`` -- overlapping masks. Ground truth comes from a multi-hot
  membership matrix (``membership_key``, e.g. ``inst_pe``) plus a per-event
  ``valid_key`` (e.g. ``ring_valid``); per-instance head targets are read from
  per-instance tensors (``target_key``). Matching is a per-event scipy
  Hungarian over float masks (focal mask + dice + class cost), exactly as in the
  ring panoptic detector. This path requires a dataset that emits the membership
  matrix; it mirrors ``ring_panoptic_detector`` math.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from pimm.models.losses.builder import LOSSES
from pimm.models.losses.instance_fast import FastSingleLayerInstanceLoss
from pimm.models.panda_detector.matcher_fast import get_target_masks, split_by_counts


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, dict):
        if "name" in value:
            return [value]
        return [dict(cfg, name=name) for name, cfg in value.items()]
    return list(value)


_DEFAULT_CONTINUOUS_CRITERION = dict(
    type="SmoothL1RegressionLoss", beta=1.0, reduction="mean"
)
_DEFAULT_CATEGORICAL_CRITERION = dict(type="CrossEntropyHeadLoss", reduction="mean")


def _normalize_query_heads(query_heads):
    """Loss-side normalization (mirrors the model-side normalizer; kept here so
    the loss is usable standalone)."""
    heads = []
    seen = set()
    for cfg in _as_list(query_heads):
        cfg = deepcopy(cfg)
        name = cfg.get("name")
        if not name:
            raise ValueError("Each query head needs a non-empty 'name'")
        if name in seen:
            raise ValueError(f"Duplicate query head name: {name!r}")
        seen.add(name)
        kind = cfg.setdefault("kind", "continuous")
        cfg.setdefault("pred_key", f"pred_{name}")
        cfg.setdefault("target_key", name)
        cfg.setdefault("loss_weight", 1.0)
        cfg.setdefault("required", True)
        if kind == "continuous":
            cfg.setdefault("aggregation", "mean")
            cfg.setdefault("criterion", deepcopy(_DEFAULT_CONTINUOUS_CRITERION))
        else:
            cfg.setdefault("aggregation", "mode")
            cfg.setdefault("criterion", deepcopy(_DEFAULT_CATEGORICAL_CRITERION))
        heads.append(cfg)
    return heads


@LOSSES.register_module()
class FastUnifiedInstanceLoss(nn.Module):
    """Outer wrapper: final-layer loss + auxiliary deep supervision."""

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
        aux_loss_weight: float = 1.0,
        query_heads=None,
        overlap: bool = False,
        membership_key: str = "inst_pe",
        valid_key: str = "ring_valid",
        mask_pe_thresh: float = 0.0,
        noobj_mask_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.aux_loss_weight = aux_loss_weight
        self.overlap = bool(overlap)
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
            query_heads=query_heads,
            overlap=overlap,
            membership_key=membership_key,
            valid_key=valid_key,
            mask_pe_thresh=mask_pe_thresh,
            noobj_mask_loss_weight=noobj_mask_loss_weight,
        )

    def forward(self, pred: Dict, input_dict: Dict):
        if hasattr(pred, "outputs"):
            pred = pred.outputs

        target_cache = self.criterion.build_targets(pred, input_dict)
        final_loss, components = self.criterion(pred, input_dict, target_cache)

        if "aux_outputs" in pred and pred["aux_outputs"]:
            aux_loss = pred["pred_masks"][0].new_tensor(0.0)
            for layer_idx, aux_pred in enumerate(pred["aux_outputs"]):
                aux_loss_val, aux_comp = self.criterion(
                    aux_pred, input_dict, target_cache
                )
                aux_loss = aux_loss + aux_loss_val
                for key, value in aux_comp.items():
                    if key in self.criterion.aux_component_keys:
                        components[f"aux_{key}_L{layer_idx}"] = value
            final_loss = final_loss + self.aux_loss_weight * aux_loss

        return final_loss, components


class FastUnifiedSingleLayerLoss(FastSingleLayerInstanceLoss):
    def __init__(
        self,
        *args,
        query_heads=None,
        overlap=False,
        membership_key="inst_pe",
        valid_key="ring_valid",
        mask_pe_thresh=0.0,
        noobj_mask_loss_weight=0.0,
        **kwargs,
    ):
        kwargs.pop("momentum_loss_weight", None)
        kwargs.pop("iou_loss_weight", None)
        super().__init__(*args, momentum_loss_weight=0.0, iou_loss_weight=0.0, **kwargs)
        self.overlap = bool(overlap)
        self.membership_key = membership_key
        self.valid_key = valid_key
        self.mask_pe_thresh = float(mask_pe_thresh)
        self.noobj_mask_loss_weight = float(noobj_mask_loss_weight)
        self.query_heads = _normalize_query_heads(query_heads)
        self.head_losses = nn.ModuleDict()
        for head in self.query_heads:
            self.head_losses[head["name"]] = LOSSES.build(head["criterion"])
        self.aux_component_keys = {
            "focal",
            "dice",
            "cls_matched",
            "cls_noobj",
            *[head["name"] for head in self.query_heads],
        }

    # ------------------------------------------------------------------ #
    # target building                                                    #
    # ------------------------------------------------------------------ #
    def build_targets(self, pred, input_dict):
        pred_masks_list = pred["pred_masks"]
        if self.overlap:
            return self._build_overlap_target_cache(pred_masks_list, input_dict)
        return self._build_target_cache(pred_masks_list, input_dict)

    def _build_overlap_target_cache(self, pred_masks_list, input_dict):
        """Per-event multi-hot masks + per-instance head targets from a
        membership matrix. Mirrors RingPanopticDetector._event_targets."""
        assert self.membership_key in input_dict, (
            f"overlap loss needs membership matrix input_dict[{self.membership_key!r}]"
        )
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
            meta = {
                "num_instances": int(masks.shape[0]),
                "valid_mask": torch.ones(P_b, dtype=torch.bool, device=device),
                "target_masks": masks,
                "target_sizes": masks.sum(1),
                "ring_slice": (r0, r1),
                "valid_rings": valid_b,
            }
            cache.append(meta)
        return cache

    # ------------------------------------------------------------------ #
    # per-instance target aggregation                                    #
    # ------------------------------------------------------------------ #
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
            pred_key = head["pred_key"]
            target_key = head["target_key"]
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
                    input_dict[target_key], batch_idx, counts, device, meta["valid_mask"]
                )
                target_per_inst = self._aggregate_target(
                    target, meta["inverse"], meta["num_instances"], head["aggregation"]
                )

            pred_b = pred[pred_key][batch_idx].to(device)
            pred_matched = pred_b[idx_q.long()]
            target_matched = target_per_inst[idx_gt.long()]
            if head["kind"] == "continuous":
                target_matched = target_matched.to(pred_matched.dtype)
                pred_matched, target_matched = self._align_regression_shapes(
                    pred_matched, target_matched
                )
            losses[name] = weight * self.head_losses[name](pred_matched, target_matched)
        return losses

    # ------------------------------------------------------------------ #
    # overlap matcher (per-event scipy Hungarian over float masks)       #
    # ------------------------------------------------------------------ #
    def _overlap_match(self, pred_masks_b, pred_logits_b, gt_masks):
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
            neg = (1 - a) * prob ** g * (-F.logsigmoid(-ml))
            cost_mask = (pos @ gt_masks.T + neg @ (1 - gt_masks).T) / max(N, 1)
            num = 2 * (prob @ gt_masks.T)
            den = prob.sum(1, keepdim=True) + gt_masks.sum(1)[None, :]
            cost_dice = 1 - (num + 1) / (den + 1)
            C = self.cost_mask * cost_mask + self.cost_dice * cost_dice
            if pred_logits_b is not None and self.cost_class > 0:
                # no per-ring class cost without a class target here; skip
                pass
            qi, ri = linear_sum_assignment(C.detach().cpu().numpy())
        dev = ml.device
        return (
            torch.as_tensor(qi, dtype=torch.long, device=dev),
            torch.as_tensor(ri, dtype=torch.long, device=dev),
        )

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, pred, input_dict, target_cache=None):
        assert isinstance(pred, dict) and "pred_masks" in pred
        pred_masks_list = pred["pred_masks"]
        counts = [pm.shape[1] for pm in pred_masks_list]
        if target_cache is None:
            target_cache = self.build_targets(pred, input_dict)

        if self.overlap:
            indices = []
            for b, (pm_b, meta) in enumerate(zip(pred_masks_list, target_cache)):
                logits_b = (
                    pred["pred_logits"][b] if "pred_logits" in pred else None
                )
                indices.append(
                    self._overlap_match(pm_b, logits_b, meta["target_masks"])
                )
        else:
            assert self.truth_label in input_dict
            indices = self.matcher(
                {
                    "pred_masks": pred_masks_list,
                    "pred_logits": pred.get("pred_logits", None),
                },
                target_cache,
            )

        z = pred_masks_list[0].new_tensor(0.0)
        total_loss_focal = z.clone()
        total_loss_dice = z.clone()
        total_loss_cls = z.clone()
        total_loss_noobj_mask = z.clone()
        total_loss_head = {h["name"]: z.clone() for h in self.query_heads}
        num_batches_with_head = {h["name"]: 0 for h in self.query_heads}
        num_batches_with_loss = 0

        total_focal = z.clone()
        total_dice = z.clone()
        total_pairs = 0
        total_ce_matched = z.clone()
        count_ce_matched = z.clone()
        total_ce_noobj = z.clone()
        count_ce_noobj = z.clone()
        total_head = {h["name"]: z.clone() for h in self.query_heads}
        queries_total = z.clone()
        gt_instances_total = z.clone()

        for batch_idx, (pm_b_full, meta, (idx_q, idx_gt)) in enumerate(
            zip(pred_masks_list, target_cache, indices)
        ):
            queries_total = queries_total + pm_b_full.shape[0]
            if idx_q.numel() == 0 or meta["num_instances"] == 0:
                continue

            pm_b = pm_b_full[:, meta["valid_mask"]]
            gt_instances_total = gt_instances_total + meta["num_instances"]
            if self.overlap:
                target_masks = meta["target_masks"].to(pm_b.dtype)
            else:
                target_masks = get_target_masks(meta, pm_b.dtype)

            idx_q = idx_q.to(pm_b.device).long()
            idx_gt = idx_gt.to(pm_b.device).long()
            pred_sel = pm_b[idx_q]
            num_pairs_b = pred_sel.shape[0]
            if num_pairs_b == 0:
                continue

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
                    total_loss_noobj_mask = total_loss_noobj_mask + F.binary_cross_entropy_with_logits(
                        nm, torch.zeros_like(nm)
                    )

            # primary classification (matched -> pid, unmatched -> no-obj)
            if (
                not self.overlap
                and "pred_logits" in pred
                and "segment" in input_dict
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
                total_loss_head[name] = total_loss_head[name] + value
                total_head[name] = total_head[name] + value
                num_batches_with_head[name] += 1

        denom = max(num_batches_with_loss, 1)
        loss = (
            self.loss_weight_focal * (total_loss_focal / denom)
            + self.loss_weight_dice * (total_loss_dice / denom)
            + total_loss_cls / denom
        )
        if self.noobj_mask_loss_weight > 0:
            loss = loss + self.noobj_mask_loss_weight * (total_loss_noobj_mask / denom)
        for name, value in total_loss_head.items():
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
