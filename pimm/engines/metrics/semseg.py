"""Semantic segmentation metric computation shared across eval and test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SemSegMetrics:
    """Per-class and macro semantic segmentation metrics."""

    intersection: np.ndarray
    union: np.ndarray
    target: np.ndarray
    iou_class: np.ndarray
    acc_class: np.ndarray
    precision_class: np.ndarray
    recall_class: np.ndarray
    f1_class: np.ndarray
    macro_mask: np.ndarray
    m_iou: float
    m_acc: float
    all_acc: float
    m_precision: float
    m_recall: float
    m_f1: float


def macro_class_mask(num_classes: int, ignore_class_ids=()) -> np.ndarray:
    """Return the class mask used for macro metrics."""
    mask = np.ones(num_classes, dtype=bool)
    for idx in ignore_class_ids or ():
        if 0 <= idx < num_classes:
            mask[idx] = False
    return mask


def _valid_or_all(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask]
    return values if selected.size == 0 else selected


def _precision_recall_f1(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    for idx in range(num_classes):
        pred_i = pred == idx
        gt_i = target == idx
        if gt_i.sum() > 0 or pred_i.sum() > 0:
            tp = np.logical_and(pred_i, gt_i).sum()
            fp = np.logical_and(pred_i, np.logical_not(gt_i)).sum()
            fn = np.logical_and(np.logical_not(pred_i), gt_i).sum()
            precision[idx] = tp / (tp + fp + 1e-10)
            recall[idx] = tp / (tp + fn + 1e-10)
            f1[idx] = 2 * precision[idx] * recall[idx] / (
                precision[idx] + recall[idx] + 1e-10
            )
    return precision, recall, f1


def _intersection_union_target(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    target_count = np.zeros(num_classes)
    for idx in range(num_classes):
        pred_i = pred == idx
        gt_i = target == idx
        intersection[idx] = np.logical_and(pred_i, gt_i).sum()
        union[idx] = np.logical_or(pred_i, gt_i).sum()
        target_count[idx] = gt_i.sum()
    return intersection, union, target_count


def compute_semseg_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    *,
    macro_ignore_class_ids=(),
    intersection: np.ndarray | None = None,
    union: np.ndarray | None = None,
    target_count: np.ndarray | None = None,
) -> SemSegMetrics:
    """Compute the semantic segmentation metrics used by evaluator/tester."""
    precision, recall, f1 = _precision_recall_f1(pred, target, num_classes)
    if intersection is None or union is None or target_count is None:
        intersection, union, target_count = _intersection_union_target(
            pred, target, num_classes
        )

    iou = intersection / (union + 1e-10)
    acc = intersection / (target_count + 1e-10)
    macro_mask = macro_class_mask(num_classes, macro_ignore_class_ids)

    precision_valid = _valid_or_all(precision, macro_mask)
    recall_valid = _valid_or_all(recall, macro_mask)
    f1_valid = _valid_or_all(f1, macro_mask)
    iou_valid = _valid_or_all(iou, macro_mask)
    acc_valid = _valid_or_all(acc, macro_mask)

    return SemSegMetrics(
        intersection=intersection,
        union=union,
        target=target_count,
        iou_class=iou,
        acc_class=acc,
        precision_class=precision,
        recall_class=recall,
        f1_class=f1,
        macro_mask=macro_mask,
        m_iou=float(np.mean(iou_valid)),
        m_acc=float(np.mean(acc_valid)),
        all_acc=sum(intersection) / (sum(target_count) + 1e-10),
        m_precision=float(np.mean(precision_valid)),
        m_recall=float(np.mean(recall_valid)),
        m_f1=float(np.mean(f1_valid)),
    )
