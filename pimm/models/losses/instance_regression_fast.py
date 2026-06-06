from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn as nn

from pimm.models.losses.builder import LOSSES
from pimm.models.losses.instance_fast import FastSingleLayerInstanceLoss
from pimm.models.panda_detector.matcher_fast import get_target_masks


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, dict):
        if "name" in value:
            return [value]
        return [dict(cfg, name=name) for name, cfg in value.items()]
    return list(value)


def _normalize_regression_targets(regression_targets):
    targets = []
    seen = set()
    for cfg in _as_list(regression_targets):
        cfg = deepcopy(cfg)
        name = cfg.get("name")
        if not name:
            raise ValueError("Each regression target needs a non-empty 'name'")
        if name in seen:
            raise ValueError(f"Duplicate regression target name: {name!r}")
        seen.add(name)
        cfg.setdefault("target_key", name)
        cfg.setdefault("pred_key", f"pred_{name}")
        cfg.setdefault("aggregation", "mean")
        cfg.setdefault("loss_weight", 1.0)
        cfg.setdefault(
            "criterion",
            dict(type="SmoothL1RegressionLoss", beta=1.0, reduction="mean"),
        )
        cfg.setdefault("required", True)
        targets.append(cfg)
    return targets


@LOSSES.register_module()
class FastInstanceSegmentationRegressionLoss(nn.Module):
    """Fast instance segmentation loss with configurable matched regressions."""

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
        regression_targets=None,
    ):
        super().__init__()
        self.aux_loss_weight = aux_loss_weight
        self.criterion = FastSingleLayerInstanceRegressionLoss(
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
            regression_targets=regression_targets,
        )

    def _target_cache(self, pred: Dict, input_dict: Dict) -> List[Dict]:
        return self.criterion._build_target_cache(pred["pred_masks"], input_dict)

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
                for key, value in aux_comp.items():
                    if key in self.criterion.aux_component_keys:
                        components[f"aux_{key}_L{layer_idx}"] = value

            final_loss = final_loss + self.aux_loss_weight * aux_loss

        return final_loss, components


class FastSingleLayerInstanceRegressionLoss(FastSingleLayerInstanceLoss):
    def __init__(self, *args, regression_targets=None, **kwargs):
        kwargs.pop("momentum_loss_weight", None)
        kwargs.pop("iou_loss_weight", None)
        super().__init__(*args, momentum_loss_weight=0.0, iou_loss_weight=0.0, **kwargs)
        self.regression_targets = _normalize_regression_targets(regression_targets)
        self.regression_losses = nn.ModuleDict()
        for cfg in self.regression_targets:
            self.regression_losses[cfg["name"]] = LOSSES.build(cfg["criterion"])
        self.aux_component_keys = {
            "focal",
            "dice",
            "cls_matched",
            "cls_noobj",
            *[cfg["name"] for cfg in self.regression_targets],
        }

    @staticmethod
    def _align_regression_shapes(pred, target):
        if pred.dim() == 1 and target.dim() == 2 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        if pred.dim() == 2 and pred.shape[-1] == 1 and target.dim() == 1:
            pred = pred.squeeze(-1)
        return pred, target

    def _aggregate_target(self, values, inverse, num_instances, aggregation):
        values = values.float()
        if aggregation == "mean":
            return self._mean_per_instance(values, inverse, num_instances)
        if aggregation == "first":
            out = values.new_zeros((num_instances,) + values.shape[1:])
            for inst_idx in range(num_instances):
                out[inst_idx] = values[inverse == inst_idx][0]
            return out
        raise ValueError(f"Unknown regression target aggregation: {aggregation!r}")

    def _compute_regression_losses(
        self,
        pred,
        input_dict,
        batch_idx,
        counts,
        meta,
        idx_q,
        idx_gt,
        device,
    ):
        losses = {}
        for cfg in self.regression_targets:
            name = cfg["name"]
            pred_key = cfg["pred_key"]
            target_key = cfg["target_key"]
            weight = float(cfg.get("loss_weight", 1.0))
            if weight == 0.0:
                continue
            missing = pred_key not in pred or target_key not in input_dict
            if missing:
                if cfg.get("required", True):
                    raise KeyError(
                        f"Missing regression target {name!r}: expected "
                        f"prediction {pred_key!r} and input {target_key!r}"
                    )
                continue

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
                cfg["aggregation"],
            )
            pred_b = pred[pred_key][batch_idx].to(device)
            pred_matched = pred_b[idx_q.long()]
            target_matched = target_per_inst[idx_gt.long()].to(pred_matched.dtype)
            pred_matched, target_matched = self._align_regression_shapes(
                pred_matched, target_matched
            )
            losses[name] = weight * self.regression_losses[name](
                pred_matched, target_matched
            )
        return losses

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
        total_loss_regression = {
            cfg["name"]: pred_masks_list[0].new_tensor(0.0)
            for cfg in self.regression_targets
        }
        num_batches_with_regression = {cfg["name"]: 0 for cfg in self.regression_targets}
        num_batches_with_loss = 0

        total_focal = pred_masks_list[0].new_tensor(0.0)
        total_dice = pred_masks_list[0].new_tensor(0.0)
        total_pairs = 0
        total_ce_matched = pred_masks_list[0].new_tensor(0.0)
        count_ce_matched = pred_masks_list[0].new_tensor(0.0)
        total_ce_noobj = pred_masks_list[0].new_tensor(0.0)
        count_ce_noobj = pred_masks_list[0].new_tensor(0.0)
        total_regression = {
            cfg["name"]: pred_masks_list[0].new_tensor(0.0)
            for cfg in self.regression_targets
        }
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

            if "pred_logits" in pred and "segment" in input_dict:
                logits_b = pred["pred_logits"][batch_idx]
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

            regression_losses = self._compute_regression_losses(
                pred,
                input_dict,
                batch_idx,
                counts,
                meta,
                idx_q,
                idx_gt,
                pm_b.device,
            )
            for name, value in regression_losses.items():
                total_loss_regression[name] = total_loss_regression[name] + value
                total_regression[name] = total_regression[name] + value
                num_batches_with_regression[name] += 1

        denom = max(num_batches_with_loss, 1)
        loss_masks = self.loss_weight_focal * (
            total_loss_focal / denom
        ) + self.loss_weight_dice * (total_loss_dice / denom)
        loss_cls = total_loss_cls / denom
        loss = loss_masks + loss_cls
        for name, value in total_loss_regression.items():
            count = max(num_batches_with_regression[name], 1)
            loss = loss + value / count

        unmatched_queries = queries_total - total_pairs
        unmatched_gt = gt_instances_total - total_pairs
        components = {
            "focal": total_focal / max(total_pairs, 1),
            "dice": total_dice / max(total_pairs, 1),
            "cls_matched": total_ce_matched / count_ce_matched.clamp_min(1),
            "cls_noobj": total_ce_noobj / count_ce_noobj.clamp_min(1),
            "num_pairs": total_pairs,
            "queries_total": queries_total,
            "gt_instances_total": gt_instances_total,
            "unmatched_queries": unmatched_queries,
            "unmatched_gt": unmatched_gt,
            "num_cls_matched": count_ce_matched,
            "num_cls_noobj": count_ce_noobj,
        }
        for name, value in total_regression.items():
            components[name] = value / max(num_batches_with_regression[name], 1)
        return loss, components
