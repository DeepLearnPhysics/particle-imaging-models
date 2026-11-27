"""Evaluate sphere-prior decoder pretraining on held-out synthetic episodes.

A query is kept by the Panda detector confidence rule (best real class has
probability >= 0.5 and its mask, logit > 0, has >= ``min_points`` points);
NMS/deduplication is not applied.  Kept masks are matched one-to-one to the
true instances by IoU (Hungarian, IoU >= 0.5) and scored with micro-averaged
recall, precision and F1, overall and by the scene's group count, with ARI from a greedy
partition of the kept masks.  Next to them the evaluator reports a held-out cap
oracle: the fraction of true instances that the best single cap (fit on half
of the points, scored on the other half) isolates at IoU >= 0.5, an estimate of
what a Panda mask head can reach on those features.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

import pimm.utils.comm as comm
from pimm.datasets.sphere_prior.geometry import cap_oracle_iou
from pimm.distributed import move_batch_to_device, unwrap_model
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase
from pimm.utils.checkpoints import checkpoint_model_state_dict

GROUP_METRICS = (
    "mask_recall",
    "recall",
    "precision",
    "f1",
    "matched_iou",
    "oracle_recall",
    "recall_over_oracle",
    "cap_threshold_median",
    "cap_sharpness_median",
)


@dataclass(frozen=True)
class OperatingPoint:
    confidence: float = 0.5
    min_points: int = 3
    match_iou: float = 0.5


@dataclass
class EventTruth:
    masks: torch.Tensor  # (R, N) bool membership
    owner: torch.Tensor  # (N,) owner, -1 for background
    name: str


def split_truth(input_dict: dict) -> list[EventTruth]:
    """Per-event ground truth from a collated sphere-prior batch."""
    offsets = [0] + input_dict["offset"].long().tolist()
    ring_slots = input_dict["inst_pe"].shape[1]
    events = []
    for b, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        valid = input_dict["ring_valid"][b * ring_slots : (b + 1) * ring_slots].bool()
        events.append(
            EventTruth(
                masks=(input_dict["inst_pe"][start:end][:, valid] > 0).T,
                owner=input_dict["instance"][start:end],
                name=input_dict["name"][b],
            )
        )
    return events


def real_class_scores(class_probs: torch.Tensor) -> torch.Tensor:
    """Best real-class probability per query (the last class is no-object)."""
    return class_probs[:, :-1].max(dim=1).values


def kept_queries(
    class_probs: torch.Tensor, mask_logits: torch.Tensor, op: OperatingPoint
) -> torch.Tensor:
    is_object = class_probs.argmax(dim=1) != class_probs.shape[1] - 1
    confident = real_class_scores(class_probs) >= op.confidence
    large_enough = (mask_logits > 0).sum(dim=1) >= op.min_points
    return is_object & confident & large_enough


def best_mask_iou_per_instance(
    masks: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    """(R,) best IoU of any query's mask (logit > 0) with each true instance."""
    if not len(masks) or not len(truth):
        return torch.zeros(len(truth))
    p, t = (masks > 0).float(), truth.float()
    intersection = p @ t.T
    union = p.sum(1)[:, None] + t.sum(1)[None, :] - intersection
    return (intersection / union.clamp_min(1)).max(dim=0).values


def match_masks(
    predicted: torch.Tensor, truth: torch.Tensor, min_iou: float
) -> list[float]:
    """IoUs of one-to-one matched (predicted, true) pairs with IoU >= min_iou."""
    if not len(predicted) or not len(truth):
        return []
    p, t = predicted.float(), truth.float()
    intersection = p @ t.T
    union = p.sum(1)[:, None] + t.sum(1)[None, :] - intersection
    iou = (intersection / union.clamp_min(1)).cpu().numpy()
    rows, cols = linear_sum_assignment(-iou)
    return [float(iou[r, c]) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]


def greedy_partition(
    mask_logits: torch.Tensor, class_probs: torch.Tensor, kept: torch.Tensor
) -> torch.Tensor:
    """Exclusive assignment: higher-scoring kept masks claim their points first."""
    labels = torch.full(
        (mask_logits.shape[1],), -1, dtype=torch.long, device=mask_logits.device
    )
    order = torch.argsort(real_class_scores(class_probs), descending=True)
    for rank, query in enumerate(order.tolist()):
        if kept[query]:
            labels[(mask_logits[query] > 0) & (labels < 0)] = rank
    return labels


def score_event(
    prediction: dict, truth: EventTruth, op: OperatingPoint, oracle_ious: list[float]
) -> dict:
    masks, probs = prediction["mask_logits"], prediction["class_probs"]
    kept = kept_queries(probs, masks, op)
    predicted = masks[kept] > 0
    matched = match_masks(predicted, truth.masks, op.match_iou)
    defined_oracle = [iou for iou in oracle_ious if not math.isnan(iou)]
    row = {
        "name": truth.name,
        "num_true": len(truth.masks),
        "num_kept": int(kept.sum()),
        "num_matched": len(matched),
        # confidence-free: does any query's mask isolate the instance?
        "num_mask_recalled": int(
            (best_mask_iou_per_instance(masks, truth.masks) >= op.match_iou).sum()
        ),
        "matched_iou_sum": float(sum(matched)),
        "num_oracle_defined": len(defined_oracle),
        "num_oracle_isolated": sum(iou >= op.match_iou for iou in defined_oracle),
        "cap_threshold": prediction["cap_threshold"][kept].tolist(),
        "cap_sharpness": prediction["cap_sharpness"][kept].tolist(),
    }
    labels = greedy_partition(masks, probs, kept)
    row["ari"] = float(
        adjusted_rand_score(truth.owner.cpu().numpy(), labels.cpu().numpy())
    )
    return row


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def _group_metrics(rows: list[dict]) -> dict:
    true = sum(r["num_true"] for r in rows)
    kept = sum(r["num_kept"] for r in rows)
    matched = sum(r["num_matched"] for r in rows)
    recall = _ratio(matched, true)
    oracle_recall = _ratio(
        sum(r["num_oracle_isolated"] for r in rows),
        sum(r["num_oracle_defined"] for r in rows),
    )
    return {
        "mask_recall": _ratio(sum(r["num_mask_recalled"] for r in rows), true),
        "recall": recall,
        "precision": _ratio(matched, kept),
        "f1": _ratio(2 * matched, true + kept),
        "matched_iou": _ratio(sum(r["matched_iou_sum"] for r in rows), matched),
        "oracle_recall": oracle_recall,
        "recall_over_oracle": _ratio(recall, oracle_recall),
        "cap_threshold_median": _median([t for r in rows for t in r["cap_threshold"]]),
        "cap_sharpness_median": _median([k for r in rows for k in r["cap_sharpness"]]),
    }


INSTANCE_COUNT_BINS = ((0, 2), (3, 5), (6, 10), (11, 20), (21, 1000))


def instance_count_bin(count: int) -> str:
    """Label of the instance-count bin, e.g. '6-10' (micro averages hide crowding)."""
    for low, high in INSTANCE_COUNT_BINS:
        if low <= count <= high:
            return f"{low}-{high}" if high < 1000 else f"{low}+"
    return "unknown"


def summarize(rows: list[dict]) -> dict:
    """Micro-averaged metrics over all scenes and per group-count bin, plus the mean ARI."""
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[f"instances_{instance_count_bin(row['num_true'])}"].append(row)

    metrics = {}
    for group, members in groups.items():
        metrics.update(
            {
                f"{group}/{name}": value
                for name, value in _group_metrics(members).items()
            }
        )
    metrics["f1"] = metrics.get("all/f1", 0.0)
    metrics["ari"] = float(np.mean([r["ari"] for r in rows])) if rows else float("nan")
    metrics["events"] = len(rows)
    return metrics


def _strip_ddp_prefix(state: dict) -> dict:
    return {key.removeprefix("module."): value for key, value in state.items()}


@HOOKS.register_module()
class SpherePriorEvaluator(HookBase):
    """Score the decoder on independent scenes from its training prior.

    ``f1`` (over all scenes) selects the best checkpoint.
    The cap oracle depends only on the episode, so it is computed once per
    episode name and cached.
    """

    def __init__(
        self,
        every_n_steps: int = 5000,
        prefix: str = "val",
        test_after_train: bool = True,
        load_best_for_test: bool = True,
        selection_metric: str = "f1",
        operating_point: dict | None = None,
    ) -> None:
        self.every_n_steps = int(every_n_steps)
        self.prefix = prefix
        self.test_after_train = test_after_train
        self.load_best_for_test = load_best_for_test
        self.selection_metric = selection_metric
        self.operating_point = OperatingPoint(**(operating_point or {}))
        self._oracle_cache: dict[str, list[float]] = {}

    def _global_step(self) -> int:
        info = self.trainer.comm_info
        return info["iter"] + info["iter_per_epoch"] * info["epoch"] + 1

    def after_step(self) -> None:
        if (
            self.trainer.cfg.evaluate
            and self.every_n_steps
            and self._global_step() % self.every_n_steps == 0
        ):
            model = unwrap_model(self.trainer.model)
            self._run(
                model,
                self.trainer.val_loader,
                prefix=self.prefix,
                set_current_metric=True,
            )

    def after_train(self) -> None:
        if not (
            self.trainer.cfg.evaluate
            and self.test_after_train
            and self.trainer.test_loader
        ):
            return
        # score a copy so the live model (and its final checkpoint) is untouched
        model = copy.deepcopy(unwrap_model(self.trainer.model))
        if self.load_best_for_test:
            path = os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(
                _strip_ddp_prefix(checkpoint_model_state_dict(checkpoint)), strict=True
            )
            self.trainer.logger.info(
                "SpherePriorEvaluator: scoring model_best on the test split."
            )
        self._run(
            model, self.trainer.test_loader, prefix="test", set_current_metric=False
        )

    def _oracle(self, prediction: dict, truth: EventTruth) -> list[float]:
        if truth.name not in self._oracle_cache:
            sphere = prediction["sphere_points"].double()
            self._oracle_cache[truth.name] = [
                cap_oracle_iou(sphere, members) for members in truth.masks
            ]
        return self._oracle_cache[truth.name]

    @torch.inference_mode()
    def _score(self, model, loader) -> list[dict]:
        rows = []
        for input_dict in loader:
            input_dict = move_batch_to_device(
                input_dict, self.trainer.parallel_context.device
            )
            for prediction, truth in zip(
                model.predict(input_dict), split_truth(input_dict)
            ):
                rows.append(
                    score_event(
                        prediction,
                        truth,
                        self.operating_point,
                        self._oracle(prediction, truth),
                    )
                )
        return rows

    def _run(
        self, model, loader, *, prefix: str, set_current_metric: bool
    ) -> dict | None:
        if loader is None:
            return None
        was_training = model.training
        model.eval()
        try:
            rows = self._score(model, loader)
        finally:
            model.train(was_training)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if comm.get_world_size() > 1:
            gathered = comm.gather(rows, dst=0)
            if not comm.is_main_process():
                return None
            rows = [row for rank_rows in gathered for row in rank_rows]

        metrics = summarize(rows)
        self._report(metrics, rows, prefix)
        if set_current_metric:
            self.trainer.comm_info["current_metric_value"] = metrics[
                self.selection_metric
            ]
            self.trainer.comm_info["current_metric_name"] = (
                f"{prefix}/{self.selection_metric}"
            )
        return metrics

    def _report(self, metrics: dict, rows: list[dict], prefix: str) -> None:
        log = self.trainer.logger
        m = {name: metrics[f"all/{name}"] for name in GROUP_METRICS}
        log.info(
            "[%s] F1=%.4f ARI=%.4f mask-recall=%.4f recall=%.4f precision=%.4f matched-IoU=%.3f oracle-recall=%.3f "
            "recall/oracle=%.3f cap t=%.3f kappa=%.1f",
            prefix,
            m["f1"],
            metrics["ari"],
            m["mask_recall"],
            m["recall"],
            m["precision"],
            m["matched_iou"],
            m["oracle_recall"],
            m["recall_over_oracle"],
            m["cap_threshold_median"],
            m["cap_sharpness_median"],
        )
        per_count = ", ".join(
            f"{name}={metrics[f'instances_{name}/mask_recall']:.2f}"
            for name in [
                f"{lo}-{hi}" if hi < 1000 else f"{lo}+"
                for lo, hi in INSTANCE_COUNT_BINS
            ]
            if f"instances_{name}/mask_recall" in metrics
        )
        log.info("[%s] mask recall by instance count: %s", prefix, per_count)
        if self.trainer.writer is not None:
            step = self._global_step()
            for name, value in metrics.items():
                if np.isfinite(value):
                    self.trainer.writer.add_scalar(f"{prefix}/{name}", value, step)
        path = os.path.join(
            self.trainer.cfg.save_path, f"{prefix}_sphere_prior_metrics.json"
        )
        with open(path, "w") as file:
            json.dump({"summary": _finite(metrics), "per_event": rows}, file, indent=1)


def _finite(metrics: dict) -> dict:
    """Replace NaN with None so the JSON stays standard."""
    return {
        name: (value if np.isfinite(value) else None) for name, value in metrics.items()
    }


__all__ = [
    "SpherePriorEvaluator",
    "OperatingPoint",
    "summarize",
    "score_event",
    "split_truth",
]
