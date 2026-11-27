"""Masked variable-length attention with FP32 softmax accumulation.

Why this exists: the Panda V2 detector's decoder (``pimm/models/panda/detector/layers.py``) uses masked
cross-attention: each event's queries attend to that event's points, with an additive mask taken from the previous
block's mask prediction. FlashAttention's variable-length kernels accept no additive mask, and PyTorch's fallback
(``pimm.ops.varlen_attention`` with ``fused=False``) pads every event to the longest one and builds a dense mask, so its
memory and time follow the largest event in the batch. This kernel reads each event's ``[queries, points]`` mask block
straight from the jagged layout (each event's block, row-major, concatenated), never pads, and accumulates the softmax in FP32 with
FlashAttention's tiled online softmax (Dao et al., 2022). It has a forward and a backward; the mask gets no gradient.

Reached through ``pimm.ops.varlen_attention(..., attn_mask=mask, fused=True)``
"""

import torch
import triton
import triton.language as tl

BATCH = 64  # sequences summed per step when locating a block


@triton.jit
def block_start(cu_a, cu_b, seq, BATCH: tl.constexpr):
    """Where sequence ``seq``'s ``(rows_a, rows_b)`` block starts in the jagged layout: the earlier blocks' total size."""
    total = tl.zeros([BATCH], tl.int64)
    for first in range(0, seq, BATCH):
        index = first + tl.arange(0, BATCH)
        valid = index < seq
        rows = tl.load(cu_a + index + 1, valid, other=0) - tl.load(
            cu_a + index, valid, other=0
        )
        cols = tl.load(cu_b + index + 1, valid, other=0) - tl.load(
            cu_b + index, valid, other=0
        )
        total += rows.to(tl.int64) * cols
    return tl.sum(total, 0)


def precision(dtype):
    """tl.dot's float32 rounding: TF32 where torch's matmuls take it, else exact."""
    return (
        "tf32"
        if dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32
        else "ieee"
    )

# FlashAttention's tiled online softmax (Dao et al., 2022).
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _masked_scores(
    query,
    key,
    mask,
    start,
    row,
    col,
    rows_valid,
    cols_valid,
    keys,
    scale,
    PRECISION: tl.constexpr,
):
    """Base-2 logits of a [rows, keys] tile, the mask rounded to the inputs' dtype as the reference rounds it."""
    bias = tl.load(
        mask + start + row[:, None] * keys + col[None, :],
        rows_valid[:, None] & cols_valid[None, :],
        other=float("-inf"),
    )
    bias = bias.to(query.dtype).to(tl.float32)
    scores = tl.dot(query, tl.trans(key), input_precision=PRECISION) * scale + bias
    return tl.where(cols_valid[None, :], scores * LOG2E, float("-inf"))


@triton.jit
def _load_rows(
    base, begin, index, valid, stride, head, DIM: tl.constexpr, BLOCK_D: tl.constexpr
):
    """One head's ``[len(index), BLOCK_D]`` tile of rows ``begin + index`` from a ``[rows, heads * DIM]`` tensor with
    row stride ``stride``, zero on invalid rows and past ``DIM``."""
    dim = tl.arange(0, BLOCK_D)
    offsets = (begin + index).to(tl.int64)[:, None] * stride + head * DIM + dim[None, :]
    return tl.load(base + offsets, valid[:, None] & (dim < DIM)[None, :], other=0)


@triton.jit
def _masked_forward(
    q,
    k,
    v,
    mask,
    cu_q,
    cu_k,
    out,
    lse,
    scale,
    total_q,
    stride_q,
    stride_k,
    stride_v,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BATCH: tl.constexpr,
    PRECISION: tl.constexpr,
):
    """Attention output and base-2 log-sum-exp (``lse``, ``[heads, total_q]``) of one block of ``BLOCK_M`` queries of
    one sequence and head, streaming that sequence's keys in tiles of ``BLOCK_N``. The softmax starts from one
    zero-valued key at logit -1e4, so a query whose every key is masked to -inf gets zeros."""
    seq, head = tl.program_id(1), tl.program_id(2)
    q_begin, k_begin = tl.load(cu_q + seq), tl.load(cu_k + seq)
    rows, keys = tl.load(cu_q + seq + 1) - q_begin, tl.load(cu_k + seq + 1) - k_begin
    if tl.program_id(0) * BLOCK_M >= rows:
        return
    start = block_start(cu_q, cu_k, seq, BATCH)
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_valid = row < rows
    query = _load_rows(q, q_begin, row, rows_valid, stride_q, head, DIM, BLOCK_D)
    peak = tl.full([BLOCK_M], -1e4 * LOG2E, tl.float32)  # the zero key
    norm = tl.full([BLOCK_M], 1.0, tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    for first in range(0, keys, BLOCK_N):
        col = first + tl.arange(0, BLOCK_N)
        cols_valid = col < keys
        key = _load_rows(k, k_begin, col, cols_valid, stride_k, head, DIM, BLOCK_D)
        scores = _masked_scores(
            query,
            key,
            mask,
            start,
            row,
            col,
            rows_valid,
            cols_valid,
            keys,
            scale,
            PRECISION,
        )
        new_peak = tl.maximum(peak, tl.max(scores, 1))
        p = tl.exp2(scores - new_peak[:, None])
        decay = tl.exp2(peak - new_peak)
        norm = norm * decay + tl.sum(p, 1)
        value = _load_rows(v, k_begin, col, cols_valid, stride_v, head, DIM, BLOCK_D)
        acc = acc * decay[:, None] + tl.dot(
            p.to(value.dtype), value, input_precision=PRECISION
        )
        peak = new_peak
    dim = tl.arange(0, BLOCK_D)
    offsets = (
        (q_begin + row).to(tl.int64)[:, None] * (HEADS * DIM)
        + head * DIM
        + dim[None, :]
    )
    tl.store(
        out + offsets,
        (acc / norm[:, None]).to(out.dtype.element_ty),
        rows_valid[:, None] & (dim < DIM)[None, :],
    )
    tl.store(lse + head * total_q + q_begin + row, peak + tl.log2(norm), rows_valid)


@triton.jit
def _masked_backward(
    q,
    k,
    v,
    mask,
    cu_q,
    cu_k,
    out,
    dout,
    lse,
    dq,
    dk,
    dv,
    scale,
    total_q,
    stride_q,
    stride_k,
    stride_v,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BATCH: tl.constexpr,
    FOR_KEYS: tl.constexpr,
    PRECISION: tl.constexpr,
):
    """dK and dV of a block of keys, looping over the queries (FOR_KEYS), or dQ of a block of queries."""
    seq, head = tl.program_id(1), tl.program_id(2)
    q_begin, k_begin = tl.load(cu_q + seq), tl.load(cu_k + seq)
    rows, keys = tl.load(cu_q + seq + 1) - q_begin, tl.load(cu_k + seq + 1) - k_begin
    start = block_start(cu_q, cu_k, seq, BATCH)
    dim = tl.arange(0, BLOCK_D)
    stride_o = HEADS * DIM
    if FOR_KEYS:
        if tl.program_id(0) * BLOCK_N >= keys:
            return
        col = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        cols_valid = col < keys
        key = _load_rows(k, k_begin, col, cols_valid, stride_k, head, DIM, BLOCK_D)
        value = _load_rows(v, k_begin, col, cols_valid, stride_v, head, DIM, BLOCK_D)
        acc_k = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
        acc_v = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
        for first in range(0, rows, BLOCK_M):
            row = first + tl.arange(0, BLOCK_M)
            rows_valid = row < rows
            query = _load_rows(
                q, q_begin, row, rows_valid, stride_q, head, DIM, BLOCK_D
            )
            grad = _load_rows(
                dout, q_begin, row, rows_valid, stride_o, head, DIM, BLOCK_D
            )
            output = _load_rows(
                out, q_begin, row, rows_valid, stride_o, head, DIM, BLOCK_D
            )
            delta = tl.sum(grad.to(tl.float32) * output.to(tl.float32), 1)
            log_norm = tl.load(
                lse + head * total_q + q_begin + row, rows_valid, other=0
            )
            scores = _masked_scores(
                query,
                key,
                mask,
                start,
                row,
                col,
                rows_valid,
                cols_valid,
                keys,
                scale,
                PRECISION,
            )
            p = tl.exp2(scores - log_norm[:, None])
            acc_v += tl.dot(tl.trans(p).to(grad.dtype), grad, input_precision=PRECISION)
            ds = p * (
                tl.dot(grad, tl.trans(value), input_precision=PRECISION)
                - delta[:, None]
            )
            acc_k += tl.dot(
                tl.trans(ds).to(query.dtype), query, input_precision=PRECISION
            )
        offsets = (
            (k_begin + col).to(tl.int64)[:, None] * stride_o + head * DIM + dim[None, :]
        )
        store = cols_valid[:, None] & (dim < DIM)[None, :]
        tl.store(dk + offsets, (acc_k * scale).to(dk.dtype.element_ty), store)
        tl.store(dv + offsets, acc_v.to(dv.dtype.element_ty), store)
    else:
        if tl.program_id(0) * BLOCK_M >= rows:
            return
        row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        rows_valid = row < rows
        query = _load_rows(q, q_begin, row, rows_valid, stride_q, head, DIM, BLOCK_D)
        grad = _load_rows(dout, q_begin, row, rows_valid, stride_o, head, DIM, BLOCK_D)
        output = _load_rows(out, q_begin, row, rows_valid, stride_o, head, DIM, BLOCK_D)
        delta = tl.sum(grad.to(tl.float32) * output.to(tl.float32), 1)
        log_norm = tl.load(lse + head * total_q + q_begin + row, rows_valid, other=0)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        for first in range(0, keys, BLOCK_N):
            col = first + tl.arange(0, BLOCK_N)
            cols_valid = col < keys
            key = _load_rows(k, k_begin, col, cols_valid, stride_k, head, DIM, BLOCK_D)
            value = _load_rows(
                v, k_begin, col, cols_valid, stride_v, head, DIM, BLOCK_D
            )
            scores = _masked_scores(
                query,
                key,
                mask,
                start,
                row,
                col,
                rows_valid,
                cols_valid,
                keys,
                scale,
                PRECISION,
            )
            p = tl.exp2(scores - log_norm[:, None])
            ds = p * (
                tl.dot(grad, tl.trans(value), input_precision=PRECISION)
                - delta[:, None]
            )
            acc += tl.dot(ds.to(key.dtype), key, input_precision=PRECISION)
        offsets = (
            (q_begin + row).to(tl.int64)[:, None] * stride_o + head * DIM + dim[None, :]
        )
        tl.store(
            dq + offsets,
            (acc * scale).to(dq.dtype.element_ty),
            rows_valid[:, None] & (dim < DIM)[None, :],
        )


def _masked_launch(q, cu_q, max_q):
    """Tile sizes (queries, keys, head dim), smaller for wider rows so the tiles fit in shared memory, and the
    grid's sequence and head axes."""
    block_d = max(16, triton.next_power_of_2(q.shape[-1]))
    row = block_d * q.element_size()
    block = 64 if row <= 64 else 32 if row <= 128 else 16
    block_m = min(block, 16 if max_q <= 16 else 32 if max_q <= 32 else 64)
    return block_m, block, block_d, (cu_q.numel() - 1, q.shape[1])


def masked_forward(q, k, v, mask, cu_q, cu_k, max_q, max_k, scale):
    """``q`` ``[total_q, heads, dim]``, ``k`` and ``v`` ``[total_k, heads, dim]`` (row-contiguous), ``mask`` jagged."""
    out = torch.empty(
        (q.shape[0], q.shape[1] * q.shape[2]), device=q.device, dtype=q.dtype
    )
    lse = torch.empty((q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32)
    block_m, block_n, block_d, axes = _masked_launch(q, cu_q, max_q)
    if q.shape[0] and axes[0]:
        _masked_forward[(triton.cdiv(max_q, block_m), *axes)](
            q,
            k,
            v,
            mask,
            cu_q,
            cu_k,
            out,
            lse,
            scale,
            q.shape[0],
            q.stride(0),
            k.stride(0),
            v.stride(0),
            q.shape[1],
            q.shape[2],
            block_d,
            block_m,
            block_n,
            BATCH,
            precision(q.dtype),
        )
    return out.view(q.shape), lse


def masked_backward(grad, q, k, v, mask, out, lse, cu_q, cu_k, max_q, max_k, scale):
    """``dq``, ``dk`` and ``dv`` in the shapes of ``q``, ``k`` and ``v`` from the output gradient ``grad`` and the
    forward's ``out`` and ``lse``: one launch over key blocks for ``dk`` and ``dv``, one over query blocks for
    ``dq``."""
    grad = grad.contiguous()
    dq = torch.zeros(
        (q.shape[0], q.shape[1] * q.shape[2]), device=q.device, dtype=q.dtype
    )
    dk, dv = (
        torch.zeros(
            (t.shape[0], q.shape[1] * q.shape[2]), device=q.device, dtype=t.dtype
        )
        for t in (k, v)
    )
    block_m, block_n, block_d, axes = _masked_launch(q, cu_q, max_q)
    if q.shape[0] and k.shape[0] and axes[0]:
        for length, block, for_keys in (
            (max_k, block_n, True),
            (max_q, block_m, False),
        ):
            _masked_backward[(triton.cdiv(max(length, 1), block), *axes)](
                q,
                k,
                v,
                mask,
                cu_q,
                cu_k,
                out,
                grad,
                lse,
                dq,
                dk,
                dv,
                scale,
                q.shape[0],
                q.stride(0),
                k.stride(0),
                v.stride(0),
                q.shape[1],
                q.shape[2],
                block_d,
                block_m,
                block_n,
                BATCH,
                for_keys,
                precision(q.dtype),
            )
    return dq.view(q.shape), dk.view(k.shape), dv.view(v.shape)
