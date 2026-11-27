"""Gather and rotate serialized windows around dropout-free FlashAttention."""

import torch
import triton
import triton.language as tl

from ..attention import _flash


# Rotary embeddings from RoFormer (Su et al., 2021).
@triton.jit
def _rotate(
    value, partner, cosine, sine, dim, DIM: tl.constexpr, BACKWARD: tl.constexpr
):
    sign = tl.where(dim % (DIM // 3) < DIM // 6, -1.0, 1.0)
    if BACKWARD:
        sign = -sign
    return value * cosine + (sign * partner) * sine


@triton.jit(do_not_specialize=["ROWS", "SLOTS"])
def _pack(
    qkv,
    cosine,
    sine,
    gather,
    scatter,
    query_offsets,
    offsets,
    query,
    kv,
    query_map,
    duplicates,
    ROWS,
    SLOTS,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    window = tl.program_id(1)
    begin = tl.load(offsets + window).to(tl.int64)
    end = tl.load(offsets + window + 1)
    query_end = tl.load(query_offsets + window + 1).to(tl.int64)
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot, channel = begin + index // (HEADS * PAD), index % (HEADS * PAD)
    head, dim = channel // PAD, channel % PAD
    valid = slot < end
    point = tl.load(gather + slot, valid, other=0).to(tl.int64)
    primary = tl.load(scatter + point, valid, other=-1) == slot
    # A shifted final window duplicates its prefix; its primary queries form the suffix.
    query_index = query_end + slot - end
    source = point * (3 * HEADS * DIM) + head * DIM + dim
    mask = valid & (dim < DIM)
    q = tl.load(qkv + source, mask & primary, other=0).to(tl.float32)
    k = tl.load(qkv + source + HEADS * DIM, mask, other=0).to(tl.float32)
    v = tl.load(qkv + source + 2 * HEADS * DIM, mask, other=0)
    shift = tl.where(dim % (DIM // 3) < DIM // 6, DIM // 6, -DIM // 6)
    qp = tl.load(qkv + source + shift, mask & primary, other=0).to(tl.float32)
    kp = tl.load(qkv + source + HEADS * DIM + shift, mask, other=0).to(tl.float32)
    cos = tl.load(cosine + slot * DIM + dim, mask, other=0)
    sin = tl.load(sine + slot * DIM + dim, mask, other=0)
    q = _rotate(q, qp, cos, sin, dim, DIM, False)
    k = _rotate(k, kp, cos, sin, dim, DIM, False)
    tl.store(query + query_index * (HEADS * PAD) + channel, q, valid & primary)
    plane = SLOTS.to(tl.int64) * (HEADS * PAD)
    tl.store(kv + slot * (HEADS * PAD) + channel, k, valid)
    tl.store(
        kv + plane + slot * (HEADS * PAD) + channel, tl.where(dim < DIM, v, 0), valid
    )
    tl.store(query_map + point, query_index, valid & primary & (channel == 0))
    tl.store(duplicates + point, slot, valid & ~primary & (channel == 0))


@triton.jit(do_not_specialize=["ROWS"])
def _scatter_output(
    attention,
    query_map,
    result,
    ROWS,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, channel = index // (HEADS * DIM), index % (HEADS * DIM)
    query = tl.load(query_map + row, row < ROWS, other=0)
    source = query * (HEADS * PAD) + (channel // DIM) * PAD + channel % DIM
    value = tl.load(attention + source, row < ROWS, other=0)
    tl.store(result + index, value, row < ROWS)


@triton.jit(do_not_specialize=["ROWS"])
def _gather_grad(
    grad,
    query_map,
    result,
    ROWS,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, channel = index // (HEADS * PAD), index % (HEADS * PAD)
    query = tl.load(query_map + row, row < ROWS, other=0)
    source = row * (HEADS * DIM) + (channel // PAD) * DIM + channel % PAD
    value = tl.load(grad + source, (row < ROWS) & (channel % PAD < DIM), other=0)
    tl.store(result + query * (HEADS * PAD) + channel, value, row < ROWS)


@triton.jit(do_not_specialize=["ROWS", "SLOTS"])
def _unpack_grad(
    dq,
    dkv,
    query_map,
    duplicates,
    cosine,
    sine,
    scatter,
    result,
    ROWS,
    SLOTS,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = index // (3 * HEADS * DIM)
    part = index // (HEADS * DIM) % 3
    channel = index % (HEADS * DIM)
    head, dim = channel // DIM, channel % DIM
    valid = row < ROWS
    slot = tl.load(scatter + row, valid, other=0).to(tl.int64)
    query = tl.load(query_map + row, valid, other=0)
    compact = query * (HEADS * PAD) + head * PAD + dim
    source = slot * (HEADS * PAD) + head * PAD + dim
    plane = SLOTS.to(tl.int64) * (HEADS * PAD)
    q = tl.load(dq + compact, valid & (part == 0), other=0).to(tl.float32)
    kv = tl.load(
        dkv + tl.maximum(part - 1, 0) * plane + source, valid & (part != 0), other=0
    ).to(tl.float32)
    value = tl.where(part == 0, q, kv)
    shift = tl.where(dim % (DIM // 3) < DIM // 6, DIM // 6, -DIM // 6)
    qp = tl.load(dq + compact + shift, valid & (part == 0), other=0).to(tl.float32)
    kp = tl.load(dkv + source + shift, valid & (part == 1), other=0).to(tl.float32)
    cos = tl.load(cosine + slot * DIM + dim, valid, other=0)
    sin = tl.load(sine + slot * DIM + dim, valid, other=0)
    rotated = _rotate(value, tl.where(part == 0, qp, kp), cos, sin, dim, DIM, True)
    value = tl.where(part != 2, rotated, value)
    # Each slot crosses the input-dtype boundary before duplicate gradients add.
    value = value.to(result.dtype.element_ty).to(tl.float32)
    duplicate = tl.load(duplicates + row, valid, other=-1).to(tl.int64)
    has_duplicate = valid & (duplicate >= 0) & (part != 0)
    other_source = duplicate * (HEADS * PAD) + head * PAD + dim
    other = tl.load(
        dkv + tl.maximum(part - 1, 0) * plane + other_source, has_duplicate, other=0
    ).to(tl.float32)
    partner = tl.load(
        dkv + other_source + shift, has_duplicate & (part == 1), other=0
    ).to(tl.float32)
    rotated = _rotate(other, partner, cos, sin, dim, DIM, True)
    other = tl.where(part == 1, rotated, other)
    other = other.to(result.dtype.element_ty).to(tl.float32)
    tl.store(result + index, value + other, valid)


# FlashAttention's variable-length kernels (Dao et al., 2022).
def forward(
    qkv, gather, scatter, cosine, sine, query_offsets, offsets, max_seqlen, heads
):
    rows, slots = qkv.shape[0], gather.numel()
    dim = qkv.shape[1] // (3 * heads)
    padded = triton.cdiv(dim, 8) * 8
    query = qkv.new_empty((rows, heads, padded), dtype=torch.float16)
    kv = qkv.new_empty((2, slots, heads, padded), dtype=torch.float16)
    query_map = torch.empty_like(scatter)
    duplicates = torch.full_like(scatter, -1)
    _pack[(triton.cdiv(max_seqlen * heads * padded, 512), offsets.numel() - 1)](
        qkv,
        cosine,
        sine,
        gather,
        scatter,
        query_offsets,
        offsets,
        query,
        kv,
        query_map,
        duplicates,
        rows,
        slots,
        heads,
        dim,
        padded,
        512,
        enable_fp_fusion=False,
    )
    output, lse, _, _ = _flash(qkv)._flash_attn_varlen_forward(
        query,
        kv[0],
        kv[1],
        query_offsets,
        offsets,
        max_seqlen,
        max_seqlen,
        0.0,
        dim**-0.5,
        False,
    )
    result = qkv.new_empty((rows, heads * dim))
    _scatter_output[(triton.cdiv(result.numel(), 512),)](
        output,
        query_map,
        result,
        rows,
        heads,
        dim,
        padded,
        512,
    )
    return result, query, kv, output, lse, query_map, duplicates


def backward(
    grad,
    query,
    kv,
    output,
    lse,
    query_map,
    duplicates,
    scatter,
    cosine,
    sine,
    query_offsets,
    offsets,
    max_seqlen,
    heads,
):
    rows, dim = grad.shape[0], grad.shape[1] // heads
    padded, slots = query.shape[-1], kv.shape[1]
    dout = torch.empty_like(query)
    _gather_grad[(triton.cdiv(dout.numel(), 512),)](
        grad.contiguous(),
        query_map,
        dout,
        rows,
        heads,
        dim,
        padded,
        512,
    )
    dq, dkv = torch.empty_like(query), torch.empty_like(kv)
    _flash(grad)._flash_attn_varlen_backward(
        dout,
        query,
        kv[0],
        kv[1],
        output,
        lse,
        dq,
        dkv[0],
        dkv[1],
        query_offsets,
        offsets,
        max_seqlen,
        max_seqlen,
        0.0,
        dim**-0.5,
        False,
        -1,
        -1,
        0.0,
        None,
        False,
    )
    result = grad.new_empty((rows, 3 * heads * dim))
    _unpack_grad[(triton.cdiv(result.numel(), 256),)](
        dq,
        dkv,
        query_map,
        duplicates,
        cosine,
        sine,
        scatter,
        result,
        rows,
        slots,
        heads,
        dim,
        padded,
        256,
        enable_fp_fusion=False,
    )
    return result
