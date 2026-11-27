"""Geometry statistics of labelled point sets on the decoder's LayerNorm sphere.

A Panda mask is a single spherical cap on this sphere, so how well one cap can
isolate an instance indicates what a decoder can reach on given features.
Used by the pretraining evaluator and prior coverage statistics.
"""

from __future__ import annotations

import torch


def to_sphere(points: torch.Tensor) -> torch.Tensor:
    """Pre-affine LayerNorm direction of each row: centre, then unit norm."""
    centred = points - points.mean(dim=-1, keepdim=True)
    return centred / centred.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def lda_axis(
    sphere: torch.Tensor, members: torch.Tensor, shrinkage: float
) -> torch.Tensor | None:
    """Shrunk two-class LDA axis separating ``members`` from the rest."""
    inside, outside = sphere[members], sphere[~members]
    if len(inside) < 2 or len(outside) < 2:
        return None
    scatter = torch.cov(inside.T) * (len(inside) - 1) + torch.cov(outside.T) * (
        len(outside) - 1
    )
    scatter = scatter / (len(sphere) - 2)
    ridge = shrinkage * scatter.trace() / scatter.shape[0]
    scatter = scatter + ridge * torch.eye(
        scatter.shape[0], dtype=scatter.dtype, device=scatter.device
    )
    return torch.linalg.solve(scatter, inside.mean(dim=0) - outside.mean(dim=0))


def best_cap_iou(scores: torch.Tensor, members: torch.Tensor) -> tuple[float, float]:
    """Best IoU of ``scores >= threshold`` against ``members``; returns (iou, threshold)."""
    order = torch.argsort(scores, descending=True)
    hits = members[order].double().cumsum(dim=0)
    taken = torch.arange(1, len(scores) + 1, device=scores.device, dtype=torch.float64)
    iou = hits / (members.sum() + taken - hits)
    best = int(torch.argmax(iou))
    return float(iou[best]), float(scores[order][best])


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else float("nan")


def candidate_axes(sphere: torch.Tensor, members: torch.Tensor) -> list[torch.Tensor]:
    """Unit cap axes worth trying for one instance: its centroid, its centroid
    relative to the event mean, and LDA at two shrinkage strengths."""
    centroid = sphere[members].mean(dim=0)
    axes = [centroid, centroid - sphere.mean(dim=0)]
    for shrinkage in (0.1, 1.0):
        axis = lda_axis(sphere, members, shrinkage)
        if axis is not None:
            axes.append(axis)
    return [axis / axis.norm().clamp_min(1e-12) for axis in axes]


def cap_oracle_iou(sphere: torch.Tensor, members: torch.Tensor, seed: int = 0) -> float:
    """Held-out IoU of the best single cap isolating ``members``.

    The axis (among :func:`candidate_axes`) and its threshold are chosen on a
    random half of the points and scored on the other half, so the estimate is
    not inflated by fitting noise in high dimension.  Returns NaN when either
    half has too few members to fit or score.
    """
    rng = torch.Generator(device=sphere.device).manual_seed(seed)
    fit = torch.rand(len(sphere), generator=rng, device=sphere.device) < 0.5
    if members[fit].sum() < 2 or members[~fit].sum() == 0:
        return float("nan")
    best_iou, best_threshold, best_axis = -1.0, 0.0, None
    for axis in candidate_axes(sphere[fit], members[fit]):
        iou, threshold = best_cap_iou(sphere[fit] @ axis, members[fit])
        if iou > best_iou:
            best_iou, best_threshold, best_axis = iou, threshold, axis
    return _iou(sphere[~fit] @ best_axis >= best_threshold, members[~fit])
