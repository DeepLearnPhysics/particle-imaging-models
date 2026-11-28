from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


def _as_1d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    return tensor


def split_by_counts(
    tensor_or_list: Union[torch.Tensor, List[torch.Tensor]], counts: List[int]
) -> List[torch.Tensor]:
    if isinstance(tensor_or_list, torch.Tensor):
        tensor_or_list = _as_1d(tensor_or_list)
        splits = []
        start = 0
        for count in counts:
            splits.append(tensor_or_list[start : start + count])
            start += count
        return splits

    return [_as_1d(tensor) for tensor in tensor_or_list]


def _first_per_instance(
    values: torch.Tensor, inverse: torch.Tensor, num_instances: int
) -> torch.Tensor:
    out = values.new_zeros((num_instances,))
    for inst_idx in range(num_instances):
        mask = inverse == inst_idx
        if mask.any():
            out[inst_idx] = values[mask][0]
    return out


def build_target_cache(
    targets: Union[torch.Tensor, List[torch.Tensor]],
    counts: List[int],
    *,
    segment: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    inst_classes: Optional[List[torch.Tensor]] = None,
    ignore_index: Optional[int] = -1,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    labels_list = split_by_counts(targets, counts)
    segment_list = split_by_counts(segment, counts) if segment is not None else None
    cache = []

    for batch_idx, labels_full in enumerate(labels_list):
        if device is not None:
            labels_full = labels_full.to(device)
        labels_full = _as_1d(labels_full)

        if ignore_index is not None:
            valid_mask = labels_full != ignore_index
        else:
            valid_mask = torch.ones_like(labels_full, dtype=torch.bool)

        labels = labels_full[valid_mask]
        if labels.numel() == 0:
            cache.append(
                dict(
                    labels_full=labels_full,
                    labels=labels,
                    valid_mask=valid_mask,
                    uniq_ids=None,
                    inverse=None,
                    num_instances=0,
                    target_sizes=None,
                    target_masks_by_dtype={},
                    segment=None,
                    inst_class=None,
                )
            )
            continue

        uniq_ids, inverse = torch.unique(labels, sorted=True, return_inverse=True)
        num_instances = int(uniq_ids.numel())
        target_sizes = torch.bincount(inverse, minlength=num_instances)

        segment_valid = None
        inst_class = None
        if inst_classes is not None and inst_classes[batch_idx] is not None:
            inst_class = inst_classes[batch_idx].to(labels_full.device)
        elif segment_list is not None:
            segment_full = segment_list[batch_idx]
            if device is not None:
                segment_full = segment_full.to(device)
            segment_valid = _as_1d(segment_full)[valid_mask]
            inst_class = _first_per_instance(segment_valid, inverse, num_instances)

        cache.append(
            dict(
                labels_full=labels_full,
                labels=labels,
                valid_mask=valid_mask,
                uniq_ids=uniq_ids,
                inverse=inverse,
                num_instances=num_instances,
                target_sizes=target_sizes,
                target_masks_by_dtype={},
                segment=segment_valid,
                inst_class=inst_class,
            )
        )

    return cache


def get_target_masks(meta: Dict, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if meta["num_instances"] == 0:
        return None

    key = str(dtype)
    target_masks = meta["target_masks_by_dtype"].get(key)
    if target_masks is None:
        target_masks = F.one_hot(
            meta["inverse"], num_classes=meta["num_instances"]
        ).to(dtype=dtype).T
        meta["target_masks_by_dtype"][key] = target_masks.contiguous()
    return target_masks


class FastHungarianMatcher(nn.Module):
    """Hungarian matcher with cached targets and non-expanded focal mask cost."""

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
        ignore_index: int = -1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_points = num_points
        self.ignore_index = ignore_index
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, (
            "all costs cant be 0"
        )

    def _target_cache_from_legacy(self, outputs, targets):
        pred_masks_list = outputs["pred_masks"]
        counts = [pm.shape[1] for pm in pred_masks_list]

        if isinstance(targets, dict):
            labels = targets["labels"]
            segment = targets.get("segment", None)
            inst_classes = None
        else:
            labels = [target["labels"] for target in targets]
            segment_items = [target.get("segment", None) for target in targets]
            segment = segment_items if all(item is not None for item in segment_items) else None
            inst_classes = (
                [target.get("inst_classes", None) for target in targets]
                if any("inst_classes" in target for target in targets)
                else None
            )

        return build_target_cache(
            labels,
            counts,
            segment=segment,
            inst_classes=inst_classes,
            ignore_index=self.ignore_index,
            device=pred_masks_list[0].device if pred_masks_list else None,
        )

    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_masks_list = outputs["pred_masks"]
        assert isinstance(pred_masks_list, (list, tuple)), (
            "pred_masks must be a list per batch"
        )
        if isinstance(targets, list) and len(targets) > 0 and "valid_mask" in targets[0]:
            target_cache = targets
        else:
            target_cache = self._target_cache_from_legacy(outputs, targets)

        indices = []
        for batch_idx, out_mask_logits_full in enumerate(pred_masks_list):
            meta = target_cache[batch_idx]
            Q_b = out_mask_logits_full.shape[0]
            device = out_mask_logits_full.device

            if meta["num_instances"] == 0 or Q_b == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.int64),
                        torch.empty(0, dtype=torch.int64),
                    )
                )
                continue

            out_mask_logits = out_mask_logits_full[:, meta["valid_mask"]]
            P_b = out_mask_logits.shape[1]
            if P_b == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.int64),
                        torch.empty(0, dtype=torch.int64),
                    )
                )
                continue

            if self.num_points and self.num_points > 0 and P_b > self.num_points:
                sel_idx = torch.randint(0, P_b, (self.num_points,), device=device)
            else:
                sel_idx = torch.arange(P_b, device=device)

            pred_samples = out_mask_logits[:, sel_idx]
            inv_sel = meta["inverse"][sel_idx]
            num_instances = meta["num_instances"]

            with torch.amp.autocast(device_type=device.type, enabled=False):
                pred_samples_f = pred_samples.float()
                target_masks_f = F.one_hot(
                    inv_sel, num_classes=num_instances
                ).to(dtype=torch.float32).T

                prob = pred_samples_f.sigmoid()
                neg_ce = F.softplus(pred_samples_f)
                pos_ce = F.softplus(-pred_samples_f)
                neg_loss = (
                    (1 - self.focal_alpha)
                    * prob.pow(self.focal_gamma)
                    * neg_ce
                )
                pos_loss = (
                    self.focal_alpha
                    * (1 - prob).pow(self.focal_gamma)
                    * pos_ce
                )
                cost_mask = (
                    neg_loss.sum(dim=-1, keepdim=True)
                    + torch.einsum(
                        "qp,jp->qj", pos_loss - neg_loss, target_masks_f
                    )
                ) / pred_samples_f.shape[1]

                numerator = 2 * torch.einsum("qk,jk->qj", prob, target_masks_f)
                denominator = prob.sum(-1)[:, None] + target_masks_f.sum(-1)[None, :]
                cost_dice = 1 - (numerator + 1) / (denominator + 1)

            cost_class = 0.0
            if "pred_logits" in outputs and self.cost_class > 0:
                pred_logits_b = outputs["pred_logits"][batch_idx]
                C = pred_logits_b.shape[-1] - 1
                if meta.get("inst_class") is not None and sel_idx.numel() == P_b:
                    inst_class = meta["inst_class"].to(device)
                elif meta.get("segment") is not None:
                    seg_sel = meta["segment"][sel_idx]
                    inst_class = _first_per_instance(seg_sel, inv_sel, num_instances)
                else:
                    inst_class = None

                if inst_class is not None:
                    out_prob = F.softmax(pred_logits_b[:, :C], dim=-1)
                    cost_class = -torch.log(
                        out_prob[:, torch.clamp(inst_class, 0, C - 1)] + 1e-8
                    )

            C_mat = self.cost_mask * cost_mask + self.cost_dice * cost_dice
            if isinstance(cost_class, torch.Tensor):
                C_mat = C_mat + self.cost_class * cost_class

            match = linear_sum_assignment(C_mat.reshape(Q_b, -1).cpu())
            indices.append(
                (
                    torch.as_tensor(match[0], dtype=torch.int64),
                    torch.as_tensor(match[1], dtype=torch.int64),
                )
            )

        return indices
