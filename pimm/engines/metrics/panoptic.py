"""Per-event panoptic and instance scores shared by Panda evaluators."""

from __future__ import annotations

import numpy as np
import torch

from .instance import eval_instances
from .fm4npp import fm4npp_event_ari


def scored_points(batch: dict, instance_key: str, segment_key: str):
    """The scored points: each stored hit when the batch keeps its truth (``origin_<key>``, copied
    before voxelization) and its voxel (GridSample's ``inverse``), else each voxel. Returns the points' truth
    instance and class (flat int64) and the index that takes per-voxel predictions onto them.
    """

    def flat(value):
        value = (
            value.detach().cpu().numpy()
            if torch.is_tensor(value)
            else np.asarray(value)
        )
        return value.reshape(-1).astype(np.int64)

    hit_keys = (f"origin_{instance_key}", f"origin_{segment_key}")
    if "inverse" in batch and all(key in batch for key in hit_keys):
        return (
            flat(batch[hit_keys[0]]),
            flat(batch[hit_keys[1]]),
            flat(batch["inverse"]),
        )
    return flat(batch[instance_key]), flat(batch[segment_key]), slice(None)


def noise_class(spec: dict) -> int:
    """The noise class for PQ: the configured stuff class, else one past the classes."""
    return (
        int(spec["stuff_classes"][0])
        if spec.get("stuff_classes")
        else int(spec["num_classes"])
    )


def noise_points(
    batch: dict, spec: dict, results: dict, gt_instance, gt_class, voxel, pred_instance
):
    """Per scored point, whether it is true noise and whether it is predicted noise (bool arrays).

    True noise: unlabeled points (instance -1), points of the noise class, and points the label's ``noise`` query
    head's target flags (on the hits when they are scored). Predicted noise: with a
    ``noise`` head, the points of instances whose query it calls noise (``results`` is postprocess's output, whose
    instance ids ``pred_instance`` (per voxel) keeps); without one, every point no instance claims (stuff and unassigned points).
    """
    head = next(
        (h for h in spec.get("query_heads") or [] if h["name"] == "noise"), None
    )
    truth = (gt_instance < 0) | (gt_class == noise_class(spec))
    if head is None:
        return truth, pred_instance[voxel] < 0
    hits = not isinstance(voxel, slice)
    flag = batch[f"origin_{head['target_key']}" if hits else head["target_key"]]
    truth = truth | (
        np.asarray(flag.cpu() if torch.is_tensor(flag) else flag).reshape(-1) > 0
    )
    instance = results["instance_labels"].reshape(-1).cpu().numpy()
    logits = results[f"instance_{head['name']}"].float().cpu().numpy()
    ids, first = np.unique(instance, return_index=True)
    noisy = ids[(ids >= 0) & (logits[first].argmax(-1) == 1)]
    return truth, np.isin(pred_instance, noisy)[voxel]


def _panoptic_counts(gt_inst, pr_inst, gt_cls, pr_cls, num_classes, iou_threshold):
    """Class-aware PQ sufficient counts for one event."""
    gt_table = _instance_class_table(gt_inst, gt_cls, num_classes)
    pr_table = _instance_class_table(pr_inst, pr_cls, num_classes)
    stats = eval_instances(
        gt_inst,
        pr_inst,
        gt_table,
        pr_table,
        class_names=tuple(map(str, range(num_classes))),
        iou_thresh=iou_threshold,
        require_class_for_match=True,
        pid_tables=True,
    )
    gt_ids = np.unique(gt_inst[gt_inst >= 0]).astype(np.int64)
    pr_ids = np.unique(pr_inst[pr_inst >= 0]).astype(np.int64)
    truth_count = np.bincount(
        gt_table[gt_ids][gt_table[gt_ids] >= 0], minlength=num_classes
    )
    pred_count = np.bincount(
        pr_table[pr_ids][pr_table[pr_ids] >= 0], minlength=num_classes
    )
    true_positive = np.zeros(num_classes, dtype=np.int64)
    iou_sum = np.zeros(num_classes, dtype=np.float64)
    for match in stats["matches"]:
        class_index = int(match["gt_cls"])
        if 0 <= class_index < num_classes:
            true_positive[class_index] += 1
            iou_sum[class_index] += float(match["iou"])
    return {
        "pid_pq_truth": truth_count.tolist(),
        "pid_pq_prediction": pred_count.tolist(),
        "pid_pq_true_positive": true_positive.tolist(),
        "pid_pq_iou_sum": iou_sum.tolist(),
    }


def _with_noise_segment(instance, classes, noise, noise_id_class):
    """Instances and classes with the noise points made one extra instance of the noise class."""
    noise_id = instance.max(initial=-1) + 1
    return np.where(noise, noise_id, instance), np.where(noise, noise_id_class, classes)


def event_row(
    gt_instance,
    gt_class,
    pred_instance,
    pred_class,
    gt_noise,
    pred_noise,
    num_classes: int,
    noise_class: int,
    iou_threshold: float = 0.5,
) -> dict:
    """All-point ARI and class-aware PQ counts with noise as one segment."""
    unlabeled = (gt_instance < 0) | (gt_class == noise_class)
    row = {"ari": fm4npp_event_ari(np.where(unlabeled, -1, gt_instance), pred_instance)}
    gt, gc = _with_noise_segment(gt_instance, gt_class, gt_noise, noise_class)
    pr, pc = _with_noise_segment(pred_instance, pred_class, pred_noise, noise_class)
    row.update(
        _panoptic_counts(
            gt, pr, gc, pc, max(num_classes, noise_class + 1), iou_threshold
        )
    )
    return row


# Panoptic quality: Kirillov et al., Panoptic Segmentation (CVPR 2019).
def summarize_rows(rows: list[dict], num_classes: int, noise_class: int) -> dict:
    """Reduce event ARI and pooled macro PQ; select their harmonic mean."""
    classes = max(num_classes, noise_class + 1)
    totals = {
        field: np.sum(
            [np.asarray(row[f"pid_pq_{field}"], dtype=np.float64) for row in rows],
            axis=0,
        )
        for field in ("truth", "prediction", "true_positive", "iou_sum")
    }
    false_positive = totals["prediction"] - totals["true_positive"]
    false_negative = totals["truth"] - totals["true_positive"]
    denominator = totals["true_positive"] + 0.5 * false_positive + 0.5 * false_negative
    rq = np.divide(
        totals["true_positive"],
        denominator,
        out=np.zeros(classes),
        where=denominator > 0,
    )
    sq = np.divide(
        totals["iou_sum"],
        totals["true_positive"],
        out=np.zeros(classes),
        where=totals["true_positive"] > 0,
    )
    pq = np.divide(
        totals["iou_sum"], denominator, out=np.zeros(classes), where=denominator > 0
    )
    active = denominator > 0
    micro_denominator = float(denominator.sum())
    micro_tp = float(totals["true_positive"].sum())
    micro_iou = float(totals["iou_sum"].sum())
    summary = dict(
        events=len(rows),
        ari=float(np.mean([row["ari"] for row in rows])),
        pid_pq=float(pq[active].mean()) if np.any(active) else 0.0,
        pid_rq=float(rq[active].mean()) if np.any(active) else 0.0,
        pid_sq=float(sq[active].mean()) if np.any(active) else 0.0,
        pid_pq_micro=micro_iou / micro_denominator if micro_denominator else 0.0,
        pid_rq_micro=micro_tp / micro_denominator if micro_denominator else 0.0,
        pid_sq_micro=micro_iou / micro_tp if micro_tp else 0.0,
        pid_panoptic_truth_instances=int(totals["truth"].sum()),
        pid_panoptic_pred_instances=int(totals["prediction"].sum()),
        pid_panoptic_matched_instances=int(micro_tp),
    )
    ari, pq = summary["ari"], summary["pid_pq"]
    summary["headline"] = 2 * ari * pq / (ari + pq) if ari + pq > 0 else 0.0
    return summary


def _instance_class_table(
    instance: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Return a mode class table indexed by non-negative instance id."""

    instance = np.asarray(instance, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    if len(instance) != len(target):
        raise ValueError("Instance and categorical target arrays must align.")
    valid_instance = instance >= 0
    if not np.any(valid_instance):
        return np.empty(0, dtype=np.int64)
    table = np.full(int(instance[valid_instance].max()) + 1, -1, dtype=np.int64)
    valid = valid_instance & (target >= 0) & (target < num_classes)
    if not np.any(valid):
        return table
    pair, count = np.unique(
        np.stack([instance[valid], target[valid]], axis=1),
        axis=0,
        return_counts=True,
    )
    for instance_id in np.unique(pair[:, 0]):
        selected = pair[:, 0] == instance_id
        classes = pair[selected, 1]
        table[int(instance_id)] = int(classes[np.argmax(count[selected])])
    return table
