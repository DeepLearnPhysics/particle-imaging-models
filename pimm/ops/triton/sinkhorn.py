"""Tiled bf16 Sinkhorn distillation with fp32 reductions and no dense score tables.

Why this exists: Panda V2 self-distillation (``pimm/models/panda/pretrain/panda_ssl_v2m1.py``, ``fused_sinkhorn``)
balances the teacher's point-to-prototype scores with Sinkhorn-Knopp and trains the student on a cross-entropy against
them, over many points and prototypes. The PyTorch path (``pimm.ops.sinkhorn_distillation_loss`` with ``fused=False``)
materializes the fp32 ``[points, prototypes]`` teacher and student score tables and the assignment table, so its memory
grows with their product. These kernels keep only per-point and per-prototype log-normalizers (Sinkhorn potentials)
and recompute score tiles from the features and prototypes in the forward and the backward.

Row counts and temperatures remain runtime arguments as masks and schedules change.

Reached through ``pimm.ops.sinkhorn_distillation_loss(..., fused=True)``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

_SPLITS = 16


def _configuration(operation, architecture):
    """Private batch/prototype tiles, warps and dot width for each GPU family."""
    if architecture == 90:
        return {
            "col": (32, 128, 4, 256),
            "row": (64, 128, 4, 256),
            "fwd": (32, 128, 8, 64),
            "dx": (32, 64, 8, 64),
            "dw": (64, 32, 4, 64),
        }[operation]
    return {
        "col": (32, 128, 4, 256),
        "row": (64, 128, 4, 256),
        "fwd": (64, 128, 4, 256),
        "dx": (32, 64, 4, 256),
        "dw": (64, 32, 4, 64),
    }[operation]


@triton.jit
def _logit_tile(
    X,
    W,
    b,
    k,
    B,
    K,
    D: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    DOT_D: tl.constexpr = 64,
):
    """Bound the live dot operands rather than staging the whole embedding."""
    DD: tl.constexpr = min(D, DOT_D)
    d = tl.arange(0, DD)
    acc = tl.zeros((BB, BK), tl.float32)
    for block in range(tl.cdiv(D, DD)):
        columns = block * DD + d
        x = tl.load(X + b[:, None] * D + columns[None, :], b[:, None] < B, 0)
        w = tl.load(W + k[:, None] * D + columns[None, :], k[:, None] < K, 0)
        acc = tl.dot(x, tl.trans(w), acc)
    return acc


@triton.jit(do_not_specialize=["B", "INV_T"])
def _col_lse_kernel(
    X,
    W,
    U,
    PART,
    B,
    K: tl.constexpr,
    INV_T,
    D: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    DOT_D: tl.constexpr = 64,
):
    """Partial column log-sum-exp ``PART[split, k]`` of ``x . w / T + U`` for ``BK`` prototypes over one of ``SPLITS``
    shares of the row tiles, where ``U`` is the row potential."""
    k = tl.program_id(0) * BK + tl.arange(0, BK)
    split = tl.program_id(1)
    maximum = tl.full((BK,), -float("inf"), tl.float32)
    total = tl.zeros((BK,), tl.float32)
    tiles_per_split = tl.cdiv(tl.cdiv(B, BB), SPLITS)
    for tile in range(tiles_per_split):
        b = (split * tiles_per_split + tile) * BB + tl.arange(0, BB)
        u = tl.load(U + b, b < B, -float("inf"))
        logits = _logit_tile(W, X, k, b, K, B, D, BK, BB, DOT_D) * INV_T + u[None, :]
        new_maximum = tl.maximum(maximum, tl.max(logits, 1))
        safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
        total = total * tl.exp(maximum - safe_maximum) + tl.sum(
            tl.exp(logits - safe_maximum[:, None]), 1
        )
        maximum = new_maximum
    tl.store(PART + split * K + k, maximum + tl.log(total), k < K)


@triton.jit
def _combine_lse_kernel(
    PART, OUT, K: tl.constexpr, SPLITS: tl.constexpr, BK: tl.constexpr
):
    """Merge the ``SPLITS`` partial log-sum-exps of ``BK`` prototypes into ``OUT``."""
    k = tl.program_id(0) * BK + tl.arange(0, BK)
    split = tl.arange(0, SPLITS)
    values = tl.load(
        PART + split[:, None] * K + k[None, :], k[None, :] < K, -float("inf")
    )
    maximum = tl.max(values, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    total = tl.sum(tl.exp(values - safe_maximum[None, :]), 0)
    tl.store(OUT + k, maximum + tl.log(total), k < K)


@triton.jit(do_not_specialize=["B", "INV_T"])
def _row_lse_kernel(
    X,
    W,
    V,
    U,
    B,
    K: tl.constexpr,
    INV_T,
    D: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    DOT_D: tl.constexpr = 64,
):
    """Row potential ``U = -logsumexp_k(x . w / T + V)`` of ``BB`` rows, looping over all prototypes in tiles of
    ``BK``."""
    b = tl.program_id(0) * BB + tl.arange(0, BB)
    maximum = tl.full((BB,), -float("inf"), tl.float32)
    total = tl.zeros((BB,), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + tl.arange(0, BK)
        v = tl.load(V + k, k < K, -float("inf"))
        logits = _logit_tile(X, W, b, k, B, K, D, BB, BK, DOT_D) * INV_T + v[None, :]
        new_maximum = tl.maximum(maximum, tl.max(logits, 1))
        total = total * tl.exp(maximum - new_maximum) + tl.sum(
            tl.exp(logits - new_maximum[:, None]), 1
        )
        maximum = new_maximum
    tl.store(U + b, -(maximum + tl.log(total)), b < B)


@triton.jit(do_not_specialize=["B", "INV_T", "INV_S"])
def _loss_fwd_kernel(
    XT,
    WT,
    XS,
    WS,
    V,
    LOSS,
    LT,
    LS,
    B,
    K: tl.constexpr,
    INV_T,
    INV_S,
    D: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    DOT_D: tl.constexpr = 64,
):
    """Per-row loss ``LOSS`` of ``BB`` rows: the cross-entropy of the student softmax against the balanced teacher
    target ``softmax_k(teacher logits / T + V)``, streaming prototype tiles. Also stores both rows' log-normalizers,
    ``LT`` and ``LS``, for the backward."""
    b = tl.program_id(0) * BB + tl.arange(0, BB)
    mt = tl.full((BB,), -float("inf"), tl.float32)
    ms = tl.full((BB,), -float("inf"), tl.float32)
    zt = tl.zeros((BB,), tl.float32)
    zs = tl.zeros((BB,), tl.float32)
    weighted_student = tl.zeros((BB,), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + tl.arange(0, BK)
        v = tl.load(V + k, k < K, -float("inf"))
        t = _logit_tile(XT, WT, b, k, B, K, D, BB, BK, DOT_D) * INV_T + v[None, :]
        s = _logit_tile(XS, WS, b, k, B, K, D, BB, BK, 256) * INV_S
        student_finite = tl.where(k[None, :] < K, s, 0.0)
        s = tl.where(k[None, :] < K, s, -float("inf"))
        next_mt = tl.maximum(mt, tl.max(t, 1))
        scale = tl.exp(mt - next_mt)
        probabilities = tl.exp(t - next_mt[:, None])
        zt = zt * scale + tl.sum(probabilities, 1)
        weighted_student = weighted_student * scale + tl.sum(
            probabilities * student_finite, 1
        )
        mt = next_mt
        next_ms = tl.maximum(ms, tl.max(s, 1))
        zs = zs * tl.exp(ms - next_ms) + tl.sum(tl.exp(s - next_ms[:, None]), 1)
        ms = next_ms
    lse_t = mt + tl.log(zt)
    lse_s = ms + tl.log(zs)
    tl.store(LT + b, lse_t, b < B)
    tl.store(LS + b, lse_s, b < B)
    tl.store(LOSS + b, lse_s - weighted_student / zt, b < B)


@triton.jit
def _ds_tile(
    XT,
    WT,
    XS,
    WS,
    b,
    k,
    v,
    lt,
    ls,
    g,
    B,
    K: tl.constexpr,
    D: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    INV_T,
    INV_S,
    DOT_D: tl.constexpr,
):
    """Loss gradient with respect to a ``[BB, BK]`` tile of student logits, ``(student - teacher probabilities) * g /
    T_s``, recomputing both tiles of logits."""
    t = _logit_tile(XT, WT, b, k, B, K, D, BB, BK, DOT_D) * INV_T + v[None, :]
    s = _logit_tile(XS, WS, b, k, B, K, D, BB, BK, 256) * INV_S
    return (tl.exp(s - ls[:, None]) - tl.exp(t - lt[:, None])) * (g * INV_S)[:, None]


@triton.jit
def _gradient_dot(probability_gradient, operands, accumulator):
    """Multiply an fp32 difference by bf16 operands without one-term rounding."""
    high = probability_gradient.to(tl.bfloat16)
    residual = probability_gradient - high.to(tl.float32)
    middle = residual.to(tl.bfloat16)
    low = (residual - middle.to(tl.float32)).to(tl.bfloat16)
    accumulator = tl.dot(low, operands, accumulator)
    accumulator = tl.dot(middle, operands, accumulator)
    return tl.dot(high, operands, accumulator)


@triton.jit(do_not_specialize=["B", "INV_T", "INV_S"])
def _bwd_dx_kernel(
    XT,
    WT,
    XS,
    WS,
    V,
    LT,
    LS,
    G,
    DX,
    B,
    K: tl.constexpr,
    INV_T,
    INV_S,
    D: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    DOT_D: tl.constexpr = 64,
):
    """Student feature gradient ``DX = dS @ WS`` of ``BB`` rows at full width ``D``, looping over prototype tiles."""
    b = tl.program_id(0) * BB + tl.arange(0, BB)
    d = tl.arange(0, D)
    lt = tl.load(LT + b, b < B, float("inf"))
    ls = tl.load(LS + b, b < B, float("inf"))
    g = tl.load(G + b, b < B, 0)
    acc = tl.zeros((BB, D), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + tl.arange(0, BK)
        v = tl.load(V + k, k < K, -float("inf"))
        ds = _ds_tile(
            XT, WT, XS, WS, b, k, v, lt, ls, g, B, K, D, BB, BK, INV_T, INV_S, DOT_D
        )
        ds = tl.where(k[None, :] < K, ds, 0.0)
        ws = tl.load(WS + k[:, None] * D + d[None, :], k[:, None] < K, 0)
        acc = _gradient_dot(ds, ws, acc)
    tl.store(DX + b[:, None] * D + d[None, :], acc, b[:, None] < B)


@triton.jit(do_not_specialize=["B", "INV_T", "INV_S"])
def _bwd_dw_kernel(
    XT,
    WT,
    XS,
    WS,
    V,
    LT,
    LS,
    G,
    PART,
    B,
    K: tl.constexpr,
    INV_T,
    INV_S,
    D: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    DOT_D: tl.constexpr = 64,
):
    """Partial prototype gradient ``PART[split] = dS.T @ XS`` of ``BK`` prototypes at full width ``D`` over one of
    ``SPLITS`` shares of the row tiles."""
    k = tl.program_id(0) * BK + tl.arange(0, BK)
    split = tl.program_id(1)
    d = tl.arange(0, D)
    v = tl.load(V + k, k < K, -float("inf"))
    acc = tl.zeros((BK, D), tl.float32)
    tiles_per_split = tl.cdiv(tl.cdiv(B, BB), SPLITS)
    for tile in range(tiles_per_split):
        b = (split * tiles_per_split + tile) * BB + tl.arange(0, BB)
        lt = tl.load(LT + b, b < B, float("inf"))
        ls = tl.load(LS + b, b < B, float("inf"))
        g = tl.load(G + b, b < B, 0)
        t = _logit_tile(WT, XT, k, b, K, B, D, BK, BB, DOT_D) * INV_T + v[:, None]
        s = _logit_tile(WS, XS, k, b, K, B, D, BK, BB, 256) * INV_S
        ds = (tl.exp(s - ls[None, :]) - tl.exp(t - lt[None, :])) * (g * INV_S)[None, :]
        ds = tl.where((k[:, None] < K) & (b[None, :] < B), ds, 0.0)
        xs = tl.load(XS + b[:, None] * D + d[None, :], b[:, None] < B, 0)
        acc = _gradient_dot(ds, xs, acc)
    tl.store(PART + split * K * D + k[:, None] * D + d[None, :], acc, k[:, None] < K)


@triton.jit
def _combine_dw_kernel(
    PART, DW, N: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    """Sum the ``SPLITS`` partial prototype gradients over ``BLOCK`` flat elements of ``DW``."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for split in tl.static_range(SPLITS):
        acc += tl.load(PART + split * N + index, index < N, 0)
    tl.store(DW + index, acc, index < N)


def _distributed_lse(local_lse):
    """Combine prototype normalizers, including empty ranks and globally empty inputs."""
    maximum = local_lse.clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    finite_maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    total = (local_lse - finite_maximum).exp()
    dist.all_reduce(total)
    return finite_maximum + total.log()


@torch.no_grad()
def _sinkhorn_potentials(
    features, prototypes, temperature, iterations, marginal, architecture, distributed
):
    """Column potential ``[K]`` (fp32) after ``iterations`` log-space Sinkhorn-Knopp rounds on the teacher's ``[N, D]``
    features and ``[K, D]`` prototypes; the balanced target is ``softmax_k(x . w / temperature + potential)``. Column
    normalizers are summed across ranks when ``distributed``, and ``marginal`` replaces the uniform prototype
    marginal."""
    rows, dim = features.shape
    count = prototypes.shape[0]
    distributed = (
        distributed
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
    row_potential = torch.zeros(rows, device=features.device, dtype=torch.float32)
    potential = torch.empty(count, device=features.device, dtype=torch.float32)
    partial = torch.empty((_SPLITS, count), device=features.device, dtype=torch.float32)
    log_marginal = marginal.float().log() if marginal is not None else None
    for iteration in range(iterations):
        if rows:
            batch_tile, prototype_tile, warps, dot_dim = _configuration(
                "col", architecture
            )
            _col_lse_kernel[(triton.cdiv(count, prototype_tile), _SPLITS)](
                features,
                prototypes,
                row_potential,
                partial,
                rows,
                count,
                1.0 / temperature,
                dim,
                batch_tile,
                prototype_tile,
                _SPLITS,
                DOT_D=dot_dim,
                num_warps=warps,
                num_stages=1,
            )
            _combine_lse_kernel[(triton.cdiv(count, 128),)](
                partial, potential, count, _SPLITS, 128, num_warps=4
            )
        else:
            potential.fill_(-torch.inf)
        if distributed:
            potential = _distributed_lse(potential)
        potential = torch.where(torch.isfinite(potential), -potential, 0.0)
        if log_marginal is not None:
            potential = potential + log_marginal
        if iteration < iterations - 1 and rows:
            batch_tile, prototype_tile, warps, dot_dim = _configuration(
                "row", architecture
            )
            _row_lse_kernel[(triton.cdiv(rows, batch_tile),)](
                features,
                prototypes,
                potential,
                row_potential,
                rows,
                count,
                1.0 / temperature,
                dim,
                batch_tile,
                prototype_tile,
                DOT_D=dot_dim,
                num_warps=warps,
                num_stages=1,
            )
    return potential


class _SinkhornDistillation(torch.autograd.Function):
    """Fused Sinkhorn distillation loss with gradients for the student features and prototypes only."""

    @staticmethod
    def forward(
        ctx,
        student_features,
        student_prototypes,
        teacher_features,
        teacher_prototypes,
        student_temperature,
        teacher_temperature,
        iterations,
        marginal,
        distributed,
    ):
        """Per-row fp32 loss ``[N]`` from ``[N, D]`` features and ``[K, D]`` prototypes, after rounding all four to bf16
        and zero-padding them to a common power-of-two width."""
        ctx.student_dtypes = student_features.dtype, student_prototypes.dtype
        ctx.student_dim = student_features.shape[1]
        teacher_dim = teacher_features.shape[1]
        padded_dim = max(32, triton.next_power_of_2(max(ctx.student_dim, teacher_dim)))
        student_features = student_features.to(torch.bfloat16).contiguous()
        student_prototypes = student_prototypes.to(torch.bfloat16).contiguous()
        teacher_features = teacher_features.detach().to(torch.bfloat16).contiguous()
        teacher_prototypes = teacher_prototypes.detach().to(torch.bfloat16).contiguous()
        if ctx.student_dim != padded_dim:
            student_features = F.pad(
                student_features, (0, padded_dim - ctx.student_dim)
            )
            student_prototypes = F.pad(
                student_prototypes, (0, padded_dim - ctx.student_dim)
            )
        if teacher_dim != padded_dim:
            teacher_features = F.pad(teacher_features, (0, padded_dim - teacher_dim))
            teacher_prototypes = F.pad(
                teacher_prototypes, (0, padded_dim - teacher_dim)
            )
        rows, dim = student_features.shape
        count = student_prototypes.shape[0]
        capability = torch.cuda.get_device_capability(student_features.device)
        architecture = capability[0] * 10 + capability[1]
        potential = _sinkhorn_potentials(
            teacher_features,
            teacher_prototypes,
            teacher_temperature,
            iterations,
            marginal,
            architecture,
            distributed,
        )
        loss = torch.empty(rows, device=student_features.device, dtype=torch.float32)
        teacher_lse, student_lse = torch.empty_like(loss), torch.empty_like(loss)
        if rows:
            batch_tile, prototype_tile, warps, dot_dim = _configuration(
                "fwd", architecture
            )
            _loss_fwd_kernel[(triton.cdiv(rows, batch_tile),)](
                teacher_features,
                teacher_prototypes,
                student_features,
                student_prototypes,
                potential,
                loss,
                teacher_lse,
                student_lse,
                rows,
                count,
                1.0 / teacher_temperature,
                1.0 / student_temperature,
                dim,
                batch_tile,
                prototype_tile,
                DOT_D=dot_dim,
                num_warps=warps,
                num_stages=1,
            )
        ctx.save_for_backward(
            teacher_features,
            teacher_prototypes,
            student_features,
            student_prototypes,
            potential,
            teacher_lse,
            student_lse,
        )
        ctx.temperatures = teacher_temperature, student_temperature
        ctx.architecture = architecture
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        """Student feature and prototype gradients in their input dtypes and widths; the other inputs get none."""
        (
            teacher_features,
            teacher_prototypes,
            student_features,
            student_prototypes,
            potential,
            teacher_lse,
            student_lse,
        ) = ctx.saved_tensors
        rows, dim = student_features.shape
        count = student_prototypes.shape[0]
        teacher_temperature, student_temperature = ctx.temperatures
        grad_features = (
            torch.empty_like(student_features, dtype=ctx.student_dtypes[0])
            if ctx.needs_input_grad[0]
            else None
        )
        grad_prototypes = (
            torch.empty_like(student_prototypes, dtype=ctx.student_dtypes[1])
            if ctx.needs_input_grad[1]
            else None
        )
        if rows:
            common = (
                teacher_features,
                teacher_prototypes,
                student_features,
                student_prototypes,
                potential,
                teacher_lse,
                student_lse,
                grad_loss.float().contiguous(),
            )
            if grad_features is not None:
                batch_tile, prototype_tile, warps, dot_dim = _configuration(
                    "dx", ctx.architecture
                )
                _bwd_dx_kernel[(triton.cdiv(rows, batch_tile),)](
                    *common,
                    grad_features,
                    rows,
                    count,
                    1.0 / teacher_temperature,
                    1.0 / student_temperature,
                    dim,
                    batch_tile,
                    prototype_tile,
                    DOT_D=dot_dim,
                    num_warps=warps,
                    num_stages=1,
                )
            if grad_prototypes is not None:
                batch_tile, prototype_tile, warps, dot_dim = _configuration(
                    "dw", ctx.architecture
                )
                partial = torch.empty(
                    (_SPLITS, count, dim),
                    device=student_prototypes.device,
                    dtype=torch.float32,
                )
                _bwd_dw_kernel[(triton.cdiv(count, prototype_tile), _SPLITS)](
                    *common,
                    partial,
                    rows,
                    count,
                    1.0 / teacher_temperature,
                    1.0 / student_temperature,
                    dim,
                    batch_tile,
                    prototype_tile,
                    _SPLITS,
                    DOT_D=dot_dim,
                    num_warps=warps,
                    num_stages=1,
                )
                _combine_dw_kernel[(triton.cdiv(count * dim, 256),)](
                    partial, grad_prototypes, count * dim, _SPLITS, 256, num_warps=4
                )
        elif grad_prototypes is not None:
            grad_prototypes.zero_()
        if ctx.student_dim != dim:
            if grad_features is not None:
                grad_features = grad_features[:, : ctx.student_dim]
            if grad_prototypes is not None:
                grad_prototypes = grad_prototypes[:, : ctx.student_dim]
        return grad_features, grad_prototypes, None, None, None, None, None, None, None


sinkhorn_distillation_loss = _SinkhornDistillation.apply
