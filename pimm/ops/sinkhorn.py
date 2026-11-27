"""Sinkhorn distillation loss using dense PyTorch operations or tiled Triton kernels."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from pimm.utils.comm import get_world_size


# Sinkhorn-Knopp balancing follows DINO (facebookresearch/dino).
@torch.no_grad()
def _sinkhorn_assignments(logits, temperature, *, iterations, marginal, distributed):
    q = logits.to(torch.float32, copy=True).div_(temperature).exp_().t()
    prototypes, rows = q.shape
    normalizers = torch.stack((q.new_tensor(rows), q.sum()))
    distributed = distributed and get_world_size() > 1
    if distributed:
        dist.all_reduce(normalizers)
    rows = normalizers[0]
    q.div_(normalizers[1])
    if marginal is not None:
        marginal = marginal.to(device=q.device, dtype=q.dtype)
    for _ in range(iterations):
        row_sum = q.sum(dim=1)
        if distributed:
            dist.all_reduce(row_sum)
        q.div_(row_sum.unsqueeze(1))
        if marginal is None:
            q.div_(prototypes)
        else:
            q.mul_(marginal.unsqueeze(1))
        q.div_(q.sum(dim=0, keepdim=True)).div_(rows)
    return q.mul_(rows).t()


def sinkhorn_distillation_loss(
    student_features: torch.Tensor,
    student_prototypes: torch.Tensor,
    teacher_features: torch.Tensor,
    teacher_prototypes: torch.Tensor,
    student_temperature: float,
    teacher_temperature: float,
    *,
    iterations: int = 3,
    marginal: torch.Tensor | None = None,
    fused: bool = False,
    distributed: bool = True,
) -> torch.Tensor:
    """Per-row Sinkhorn distillation from matched ``[N, D]`` features and ``[K, D]`` prototypes.

    The reference projects under the caller's autocast, then uses the dense Sinkhorn and CE operations. Teacher
    inputs are detached. ``fused`` runs a Triton kernel, padding the embeddings to a power-of-two width. It rounds
    operands directly to bfloat16 even under float16 autocast, then uses fp32 logits, potentials, exponentials
    and accumulation. Its rounding differs from the reference, it stores no dense score tables, and it supports
    first derivatives only. With ``distributed`` (the default) balancing spans the ranks of the default process
    group, like SyncBN; otherwise each rank balances its own rows. Prototype gradients stay local for DDP. An empty local input returns an empty loss.
    """
    tensors = (
        student_features,
        student_prototypes,
        teacher_features,
        teacher_prototypes,
    )
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("features and prototypes must be matrices")
    if (
        student_features.shape[0] != teacher_features.shape[0]
        or student_prototypes.shape[0] != teacher_prototypes.shape[0]
        or student_features.shape[1] != student_prototypes.shape[1]
        or teacher_features.shape[1] != teacher_prototypes.shape[1]
    ):
        raise ValueError(
            "matched rows, prototype counts and feature dimensions must agree"
        )
    if any(t.device != student_features.device for t in tensors):
        raise ValueError("all features and prototypes must be on the same device")
    if iterations < 1 or student_temperature <= 0 or teacher_temperature <= 0:
        raise ValueError(
            "positive temperatures and at least one iteration are required"
        )
    if marginal is not None and marginal.shape != (student_prototypes.shape[0],):
        raise ValueError("marginal must have shape [K]")

    if not fused:
        with torch.no_grad():
            teacher_logits = teacher_features @ teacher_prototypes.t()
            target = _sinkhorn_assignments(
                teacher_logits,
                teacher_temperature,
                iterations=iterations,
                marginal=marginal,
                distributed=distributed,
            )
        student_logits = student_features @ student_prototypes.t()
        return -torch.sum(
            target * F.log_softmax(student_logits / student_temperature, dim=-1), dim=-1
        )
    from .triton.sinkhorn import sinkhorn_distillation_loss as triton_loss

    if marginal is not None:
        marginal = marginal.to(device=student_features.device, dtype=torch.float32)
    return triton_loss(
        student_features,
        student_prototypes,
        teacher_features,
        teacher_prototypes,
        student_temperature,
        teacher_temperature,
        iterations,
        marginal,
        distributed,
    )
