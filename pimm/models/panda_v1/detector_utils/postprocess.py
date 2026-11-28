from __future__ import annotations

import torch
from typing import Dict, List, Optional, Tuple, Set


@torch.no_grad()
def mask_matrix_nms(
    masks: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    filter_thr: float = -1,
    nms_pre: int = -1,
    max_num: int = -1,
    kernel: str = "gaussian",
    sigma: float = 2.0,
    mask_area: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated matrix NMS for instance segmentation masks.
    """
    device = masks.device
    assert len(labels) == len(masks) == len(scores)

    if len(labels) == 0:
        return (
            scores.new_zeros(0),
            labels.new_zeros(0),
            masks.new_zeros(0, masks.shape[-1] if masks.dim() > 1 else 0),
            torch.zeros(0, dtype=torch.long, device=device),
        )

    if mask_area is None:
        mask_area = masks.sum(1).float()

    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = sort_inds.clone()

    if nms_pre > 0 and len(sort_inds) > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]

    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)
    flatten_masks = masks.reshape(num_masks, -1).float()
    inter_matrix = torch.mm(flatten_masks, flatten_masks.t())

    expanded_mask_area = mask_area.unsqueeze(0).expand(num_masks, num_masks)
    union = expanded_mask_area + expanded_mask_area.t() - inter_matrix
    iou_matrix = (inter_matrix / union.clamp(min=1e-6)).triu(diagonal=1)

    expanded_labels = labels.unsqueeze(0).expand(num_masks, num_masks)
    label_matrix = (expanded_labels == expanded_labels.t()).triu(diagonal=1)

    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.unsqueeze(0).expand(num_masks, num_masks).t()
    decay_iou = iou_matrix * label_matrix

    if kernel == "gaussian":
        decay_matrix = torch.exp(-sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix.clamp(min=1e-6)).min(0)
    elif kernel == "linear":
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou).clamp(min=1e-6)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(f"{kernel} kernel is not supported!")

    scores = scores * decay_coefficient

    if filter_thr > 0:
        keep = scores >= filter_thr
        if not keep.any():
            return (
                scores.new_zeros(0),
                labels.new_zeros(0),
                masks.new_zeros(0, masks.shape[-1]),
                torch.zeros(0, dtype=torch.long, device=device),
            )
        keep_inds = keep_inds[keep]
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]

    if max_num > 0 and len(sort_inds) > max_num:
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]

    masks = masks[sort_inds]
    labels = labels[sort_inds]
    return scores, labels, masks, keep_inds


@torch.no_grad()
def postprocess_panoptic(
    query_masks: torch.Tensor,
    query_classes: torch.Tensor,
    num_points: int,
    stuff_probs: Optional[torch.Tensor] = None,
    pred_momentum: Optional[torch.Tensor] = None,
    pred_regression: Optional[Dict[str, torch.Tensor]] = None,
    pred_iou: Optional[torch.Tensor] = None,
    stuff_classes: Optional[Set[int]] = None,
    stuff_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    conf_threshold: float = 0.5,
    nms_kernel: str = "gaussian",
    nms_sigma: float = 2.0,
    nms_pre: int = -1,
    nms_max: int = -1,
    min_points: int = 20,
    background_class_label: int = -1,
    fill_uncovered: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Postprocess panoptic segmentation outputs.
    """
    device = query_masks.device
    stuff_classes = stuff_classes or set()
    num_classes = query_classes.shape[1] - 1

    # threshold masks
    mask_sigmoid = query_masks.sigmoid()
    thresholded_masks = mask_sigmoid > mask_threshold

    if stuff_probs is not None:
        thing_mask = stuff_probs < stuff_threshold
        thresholded_masks = thresholded_masks & thing_mask.unsqueeze(0)

    # class predictions
    probs = torch.softmax(query_classes, dim=-1)
    pred_labels = probs.argmax(dim=-1)
    pred_scores = probs.max(dim=-1).values

    # filter candidates: exclude bg class, stuff classes, low-conf, tiny masks
    keep = pred_labels != num_classes
    if stuff_classes:
        stuff_tensor = pred_labels.new_tensor(list(stuff_classes))
        keep = keep & ~torch.isin(pred_labels, stuff_tensor)
    mask_sizes = thresholded_masks.sum(dim=1)
    keep = keep & (pred_scores >= conf_threshold) & (mask_sizes >= min_points)

    # initialize outputs
    pred_instance_labels = torch.full(
        (num_points,), -1, dtype=torch.long, device=device
    )
    pred_class_labels = torch.full(
        (num_points,), background_class_label, dtype=torch.long, device=device
    )
    pred_confidences = torch.zeros(num_points, dtype=torch.float, device=device)
    pred_query_labels = torch.full(
        (num_points,), -1, dtype=torch.long, device=device
    )
    
    regression_inputs = dict(pred_regression or {})
    if pred_momentum is not None:
        regression_inputs.setdefault("momentum", pred_momentum)
    pred_instance_regression = {}
    for name, values in regression_inputs.items():
        value_dim = values.shape[1] if values.dim() > 1 else 1
        out = torch.full(
            (num_points, value_dim), float("nan"), dtype=torch.float, device=device
        )
        if values.dim() == 1:
            out = out.squeeze(-1)
        pred_instance_regression[name] = out

    if keep.any():
        num_queries = query_classes.shape[0]
        query_indices = torch.arange(num_queries, device=device)
        sel_query_indices = query_indices[keep]

        sel_masks = thresholded_masks[keep]
        sel_labels = pred_labels[keep]
        sel_scores = pred_scores[keep]
        sel_sigmoid = mask_sigmoid[keep]
        
        sel_regression = {
            name: values[keep] for name, values in regression_inputs.items()
        }
        
        sel_iou = None
        if pred_iou is not None:
            sel_iou = pred_iou[keep]

        # mask confidence: mean of sigmoid values within each mask
        mask_sums = (sel_sigmoid * sel_masks.float()).sum(dim=1)
        mask_counts = sel_masks.sum(dim=1).float().clamp(min=1)
        mask_confidences = mask_sums / mask_counts
        
        # combine class score with mask confidence and predicted IoU (if available)
        if sel_iou is not None:
            combined_scores = mask_confidences * sel_scores * sel_iou
        else:
            combined_scores = mask_confidences * sel_scores

        # apply matrix NMS
        nms_scores, nms_labels, nms_masks, keep_inds = mask_matrix_nms(
            sel_masks,
            sel_labels,
            combined_scores,
            filter_thr=conf_threshold,
            nms_pre=nms_pre,
            max_num=nms_max,
            kernel=nms_kernel,
            sigma=nms_sigma,
        )
        
        nms_query_indices = sel_query_indices[keep_inds]
        nms_regression = {
            name: values[keep_inds] for name, values in sel_regression.items()
        }

        # greedy assignment: process masks in order, mark points as taken
        taken = torch.zeros(num_points, dtype=torch.bool, device=device)
        for i in range(nms_masks.shape[0]):
            mask = nms_masks[i]
            if not mask.any():
                continue
            unique_mask = mask & ~taken
            if unique_mask.sum().item() < min_points:
                continue
            inst_id = i + 1
            pred_instance_labels[unique_mask] = inst_id
            pred_class_labels[unique_mask] = nms_labels[i]
            pred_confidences[unique_mask] = nms_scores[i]
            pred_query_labels[unique_mask] = nms_query_indices[i]
            
            for name, values in nms_regression.items():
                pred_instance_regression[name][unique_mask] = values[i]
                
            taken = taken | unique_mask

        # fill uncovered points using remaining low-confidence masks
        if fill_uncovered and thresholded_masks.shape[0] > 0:
            uncovered = ~taken
            if uncovered.any():
                # consider all candidate masks sorted by score descending
                order = torch.argsort(combined_scores, descending=True)
                for q in order.tolist():
                    if pred_labels[keep][q].item() == num_classes:
                        continue
                    if (
                        stuff_classes
                        and int(pred_labels[keep][q].item()) in stuff_classes
                    ):
                        continue
                    mask = sel_masks[q]
                    if not mask.any():
                        continue
                    candidate = mask & uncovered
                    if candidate.sum().item() < min_points:
                        continue
                    inst_id = int(pred_instance_labels.max().item()) + 1
                    pred_instance_labels[candidate] = inst_id
                    pred_class_labels[candidate] = sel_labels[q]
                    pred_confidences[candidate] = combined_scores[q]
                    pred_query_labels[candidate] = sel_query_indices[q]
                    
                    for name, values in sel_regression.items():
                        pred_instance_regression[name][candidate] = values[q]
                        
                    taken = taken | candidate
                    uncovered = ~taken
                    if not uncovered.any():
                        break

    # handle stuff points (overrides everything)
    if stuff_probs is not None:
        stuff_mask = stuff_probs >= stuff_threshold
        pred_instance_labels[stuff_mask] = -1
        pred_class_labels[stuff_mask] = background_class_label
        pred_confidences[stuff_mask] = stuff_probs[stuff_mask]
        pred_query_labels[stuff_mask] = -1
        
        for values in pred_instance_regression.values():
            values[stuff_mask] = float("nan")

    output = {
        "instance_labels": pred_instance_labels,
        "class_labels": pred_class_labels,
        "confidences": pred_confidences,
        "query_labels": pred_query_labels,
        "instance_regression": pred_instance_regression,
        "pred_iou": pred_iou,  # original per-query IoU predictions
    }
    for name, values in pred_instance_regression.items():
        output[f"instance_{name}"] = values
        output[f"pred_{name}"] = values
    output["instance_momentum"] = pred_instance_regression.get("momentum")
    return output


@torch.no_grad()
def postprocess_batch(
    pred_masks: List[torch.Tensor],
    pred_logits: List[torch.Tensor],
    stuff_probs: Optional[torch.Tensor] = None,
    pred_momentum: Optional[List[torch.Tensor]] = None,
    pred_regression: Optional[Dict[str, List[torch.Tensor]]] = None,
    pred_iou: Optional[List[torch.Tensor]] = None,
    point_counts: Optional[torch.Tensor] = None,
    stuff_classes: Optional[Set[int]] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Postprocess a batch of panoptic segmentation outputs.

    Args:
        pred_masks: list of (Q, N_b) mask logits per batch
        pred_logits: list of (Q, C+1) class logits per batch
        stuff_probs: (N_total,) stuff probabilities (optional)
        pred_momentum: list of (Q,) or (Q, D) momentum predictions per batch (optional)
        pred_regression: dict mapping target name to per-batch query predictions
        pred_iou: list of (Q,) predicted IoU scores per batch (optional)
        point_counts: (B,) number of points per batch element
        stuff_classes: set of stuff class indices
        **kwargs: additional arguments for postprocess_panoptic

    Returns:
        dict with concatenated 1D tensors:
            instance_labels: (N_total,) instance IDs (-1 for stuff/uncovered)
            class_labels: (N_total,) class predictions
            confidences: (N_total,) confidence scores
            instance_regression: dict of point-wise regression predictions
            pred_iou: list of (Q,) original IoU predictions per batch (optional)
    """
    batch_size = len(pred_masks)

    all_instance_labels = []
    all_class_labels = []
    all_confidences = []
    all_query_labels = []
    all_instance_regression = {}
    all_pred_iou = []

    if stuff_probs is not None and point_counts is not None:
        point_offsets = torch.cat([point_counts.new_zeros(1), point_counts.cumsum(dim=0)])
    else:
        point_offsets = None

    instance_offset = 0
    for b in range(batch_size):
        query_masks_b = pred_masks[b]
        query_classes_b = pred_logits[b]
        num_points_b = query_masks_b.shape[1]
        
        pred_momentum_b = None
        if pred_momentum is not None:
            pred_momentum_b = pred_momentum[b]
        pred_regression_b = None
        if pred_regression is not None:
            pred_regression_b = {
                name: values[b] for name, values in pred_regression.items()
            }
        
        pred_iou_b = None
        if pred_iou is not None:
            pred_iou_b = pred_iou[b]

        stuff_probs_b = None
        if stuff_probs is not None and point_offsets is not None:
            start = point_offsets[b].item()
            end = point_offsets[b + 1].item()
            stuff_probs_b = stuff_probs[start:end]

        result = postprocess_panoptic(
            query_masks=query_masks_b,
            query_classes=query_classes_b,
            num_points=num_points_b,
            stuff_probs=stuff_probs_b,
            pred_momentum=pred_momentum_b,
            pred_regression=pred_regression_b,
            pred_iou=pred_iou_b,
            stuff_classes=stuff_classes,
            **kwargs,
        )

        # offset instance labels to be unique across batch
        inst_labels = result["instance_labels"]
        valid_inst = inst_labels >= 0
        if valid_inst.any():
            inst_labels[valid_inst] += instance_offset
            instance_offset = inst_labels.max().item() + 1

        all_instance_labels.append(inst_labels)
        all_class_labels.append(result["class_labels"])
        all_confidences.append(result["confidences"])
        all_query_labels.append(result["query_labels"])
        for name, values in result.get("instance_regression", {}).items():
            all_instance_regression.setdefault(name, []).append(values)
        if pred_iou_b is not None:
            all_pred_iou.append(pred_iou_b)

    output = {
        "instance_labels": torch.cat(all_instance_labels, dim=0),
        "class_labels": torch.cat(all_class_labels, dim=0),
        "confidences": torch.cat(all_confidences, dim=0),
        "query_labels": torch.cat(all_query_labels, dim=0),
    }
    
    if all_instance_regression:
        output["instance_regression"] = {}
        for name, chunks in all_instance_regression.items():
            values = torch.cat(chunks, dim=0)
            output["instance_regression"][name] = values
            output[f"instance_{name}"] = values
            output[f"pred_{name}"] = values
        output["instance_momentum"] = output["instance_regression"].get("momentum")
    
    # keep pred_iou as list for loss computation
    if all_pred_iou:
        output["pred_iou"] = all_pred_iou

    return output


def _mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """IoU of two boolean point masks."""
    inter = (a & b).sum().item()
    if inter == 0:
        return 0.0
    union = (a | b).sum().item()
    return inter / union if union else 0.0


@torch.no_grad()
def nms_overlap_instances(rings, iou_threshold: float = 0.9, same_pid: bool = True):
    """Confidence/mask-IoU NMS for OVERLAP-PRESERVING instances (rings).

    Greedy by descending ``score``; a lower-scored instance is dropped ONLY when its
    mask IoU with an already-kept instance is ``>= iou_threshold`` (and, if
    ``same_pid``, they share a class). Purely mask-confidence/class based -- NO
    physics gates (direction/vertex/energy) -- so genuinely-overlapping distinct
    instances are preserved. ``iou_threshold >= 1`` disables suppression.
    """
    if not rings or iou_threshold >= 1.0:
        return sorted(rings, key=lambda r: r["score"], reverse=True)
    order = sorted(range(len(rings)), key=lambda i: rings[i]["score"], reverse=True)
    kept = []
    for i in order:
        r = rings[i]
        dup = False
        for k in kept:
            if _mask_iou(r["mask"], k["mask"]) < iou_threshold:
                continue
            if same_pid and r["pid"] != k["pid"]:
                continue
            dup = True
            break
        if not dup:
            kept.append(r)
    return kept


@torch.no_grad()
def postprocess_overlap_batch(
    pred_masks: List[torch.Tensor],
    pred_logits: List[torch.Tensor],
    num_classes: int,
    pred_regression: Optional[Dict[str, List[torch.Tensor]]] = None,
    conf_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_points: int = 1,
    dedup_iou: float = 0.9,
    dedup_same_pid: bool = True,
    **kwargs,
) -> List[List[Dict]]:
    """Overlap-preserving counterpart to :func:`postprocess_batch`.

    Each kept query yields its OWN thresholded sigmoid mask -- no cross-query argmax
    fusion -- so a point can belong to several instances (e.g. overlapping Cherenkov
    rings). Selection is mask-confidence / class-confidence based ONLY: keep a query
    if its top class score (softmax over the ``num_classes`` real classes, excluding
    the +1 no-object) exceeds ``conf_threshold`` and its mask has ``>= min_points``
    points above ``mask_threshold``; a generic mask-IoU NMS (:func:`nms_overlap_instances`)
    removes near-duplicate queries. Any per-query regression heads in ``pred_regression``
    (e.g. energy/vertex/direction) pass through generically -- they are NOT used for
    selection or dedup. Unknown ``**kwargs`` (nms_kernel/sigma/etc. from the shared
    postprocess cfg) are ignored.

    Returns a per-event list; each event is a list of instance dicts with keys
    ``query, mask (bool, CPU), score, pid, cls_probs (CPU)`` plus one entry per
    regression head (CPU tensor).
    """
    batch_size = len(pred_masks)
    events: List[List[Dict]] = []
    for b in range(batch_size):
        cls = pred_logits[b].float()
        probs = cls.softmax(-1)
        scores, labels = probs[:, :num_classes].max(-1)
        prob_mask = pred_masks[b].float().sigmoid()
        reg_b = {name: vals[b] for name, vals in (pred_regression or {}).items()}
        rings: List[Dict] = []
        keep = (scores > conf_threshold).nonzero(as_tuple=True)[0]
        for q in keep.tolist():
            m = prob_mask[q] > mask_threshold
            if int(m.sum()) < min_points:
                continue
            ring = dict(
                query=int(q),
                mask=m.cpu(),
                score=float(scores[q]),
                pid=int(labels[q]),
                cls_probs=probs[q, :num_classes].detach().cpu(),
            )
            for name, vals in reg_b.items():
                ring[name] = vals[q].detach().float().cpu()
            rings.append(ring)
        events.append(nms_overlap_instances(
            rings, iou_threshold=dedup_iou, same_pid=dedup_same_pid))
    return events
