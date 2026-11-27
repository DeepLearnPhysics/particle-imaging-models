"""fp16 submanifold convolution for spconv's Native algorithm with odd kernels.

Why this exists: spconv's Native algorithm reads the kernel offsets' pair counts on the host and runs a gather, GEMM
and scatter per offset, so a large kernel such as LitePT's 5x5x5 stem costs a device sync and 124 such rounds per
call. Here the forward runs in Triton: it marks each output site's active offsets in a bitmap and accumulates all of
them in one kernel, after one dense matmul for the central offset. The backward computes the weight gradient in
Triton when the input needs no gradient (an input layer, such as that stem) and calls spconv's Native backward
otherwise. Accumulation is rounded to fp16 after each kernel offset, as spconv's fp16 Native path does.

Reached through ``pimm.ops.SubMConv3d(..., algo=ConvAlgo.Native, fused=True)`` in training.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch.autograd.function import once_differentiable

from cumm.gemm.algospec.core import ShuffleStrideType
from spconv.core import AlgoHint, ConvAlgo
from spconv.pytorch import ops
from spconv.pytorch.cppcore import get_arch, torch_tensor_to_tv

_SPLIT_COUNTS = {}


# spconv's Native algorithm (traveller59/spconv).
class NativeSubmConvFp16(torch.autograd.Function):
    """spconv's fp16 Native submanifold convolution: same AMP boundary (inputs cast to fp16) and rounding."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(ctx, features, filters, pairs, pair_num):
        """``[N, out]`` output from ``[N, in]`` features, KRSC filters ``[out, *kernel, in]``, spconv's Native pairs
        ``[2, K, N]`` (input, output site per offset) and per-offset pair counts ``[K]``.
        """
        kernel_volume = pairs.shape[1]
        words = triton.cdiv(kernel_volume, 32)
        dense = torch.empty(
            (len(features), words * 32), dtype=torch.int32, device=pairs.device
        )
        bits = torch.zeros(
            (len(features), words), dtype=torch.int32, device=pairs.device
        )
        _native_bitmap[(kernel_volume, 4)](
            pairs,
            pair_num,
            dense,
            bits,
            len(features),
            kernel_volume,
            words * 32,
            words,
            256,
        )
        in_channels, out_channels = features.shape[1], filters.shape[0]
        flat = filters.reshape(out_channels, -1, in_channels)
        kernel_volume = flat.shape[1]
        out = torch.mm(features, flat[:, kernel_volume // 2].T)
        words = triton.cdiv(kernel_volume, 32)
        _native_bitmap_accumulate[
            (triton.cdiv(len(features), 16), triton.cdiv(out_channels, 64))
        ](
            features,
            flat,
            dense,
            bits,
            out,
            len(features),
            in_channels,
            out_channels,
            kernel_volume,
            words,
            16,
            num_warps=4,
        )
        ctx.save_for_backward(features, filters, pairs, pair_num)
        return out

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        """Filter gradient in Triton, split over pairs as spconv's tuned split-K, when the input needs no gradient and
        the tuner has this shape; otherwise spconv's Native backward for both gradients.
        """
        features, filters, pairs, counts = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        splits = None
        # An untuned shape first uses Native's backward to populate its tuner.
        if (
            not ctx.needs_input_grad[0]
            and grad_output.dtype == torch.float16
            and features.dtype == filters.dtype == torch.float16
        ):
            tuner = ops.GEMM_CPP
            n, in_channels = features.shape
            out_channels = filters.shape[0]
            key = (
                id(tuner),
                features.device,
                torch.cuda.current_device(),
                in_channels,
                out_channels,
            )
            splits = _SPLIT_COUNTS.get(key)
            if splits is None:
                dtype = torch_tensor_to_tv(features).dtype
                tuned, exists = tuner.get_tuned_algo(
                    dtype,
                    dtype,
                    dtype,
                    [n, out_channels],
                    [n, in_channels],
                    [out_channels, in_channels],
                    True,
                    False,
                    False,
                    get_arch(),
                    ShuffleStrideType.ShuffleAB.value,
                    [1],
                    [1],
                    [],
                    AlgoHint.BackwardWeight.value,
                )
                if exists and tuned.splitk in (1, 2, 4, 8, 16):
                    splits = _SPLIT_COUNTS[key] = tuned.splitk
        if splits is not None:
            din = None
            in_channels, out_channels = features.shape[1], filters.shape[0]
            flat = filters.reshape(out_channels, -1, in_channels)
            kernel_volume = flat.shape[1]
            dw = torch.empty_like(flat)
            torch.mm(grad_output.T, features, out=dw[:, kernel_volume // 2])
            part = torch.empty(
                (splits, kernel_volume, out_channels, in_channels),
                dtype=torch.float32,
                device=features.device,
            )
            input_tile = min(32, triton.next_power_of_2(in_channels))
            _native_wgrad_part[
                (
                    kernel_volume - 1,
                    triton.cdiv(out_channels, 32)
                    * triton.cdiv(in_channels, input_tile),
                    splits,
                )
            ](
                features,
                grad_output,
                pairs,
                counts,
                part,
                len(features),
                in_channels,
                out_channels,
                kernel_volume,
                input_tile,
                splits,
                32,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _native_wgrad_finish[(triton.cdiv(dw.numel(), 256),)](
                part, dw, in_channels, out_channels, kernel_volume, splits, 256
            )
            dw = dw.reshape_as(filters)
        else:
            din, dw = ops.indice_conv_backward(
                features,
                filters,
                grad_output,
                pairs,
                counts.cpu(),
                False,
                True,
                algo=ConvAlgo.Native,
            )
        return din, dw, None, None


@triton.jit(do_not_specialize=["n"])
def _native_bitmap(
    Pairs,
    Counts,
    Dense,
    Bits,
    n,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    WORDS: tl.constexpr,
    BS: tl.constexpr,
):
    """For one non-central kernel offset, scatter its pairs into ``Dense[output, offset] = input`` and set the
    offset's bit in ``Bits[output]``; each of four programs per offset takes every fourth chunk of ``BS`` pairs.
    """
    n = tl.cast(n, tl.int64)
    off = tl.program_id(0).to(tl.int64)
    count = tl.minimum(tl.load(Counts + tl.minimum(off, K - 1 - off)), n)
    count = tl.where(off == K // 2, 0, count)
    bit = (1 << (off % 32)).to(tl.int32)
    for p0 in range(tl.program_id(1).to(tl.int64) * BS, count, 4 * BS):
        p = p0 + tl.arange(0, BS)
        valid = p < count
        src = (tl.load(Pairs + off * n + p, valid, other=0)).to(tl.int64)
        dst = (tl.load(Pairs + (K + off) * n + p, valid, other=0)).to(tl.int64)
        tl.store(Dense + dst * STRIDE + off, src, valid)
        tl.atomic_or(Bits + dst * WORDS + off // 32, bit, valid, sem="relaxed")


@triton.jit(do_not_specialize=["n"])
def _native_bitmap_accumulate(
    X,
    W,
    Dense,
    Bits,
    Y,
    n,
    CI: tl.constexpr,
    CO: tl.constexpr,
    K: tl.constexpr,
    WORDS: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Add every non-central offset's contribution to ``ROWS`` output sites by 64 output channels of ``Y``, which
    holds the central offset's product. Each site's set bits are visited in increasing offset order, rounding to fp16
    after each offset."""
    n = tl.cast(n, tl.int64)
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    ch = tl.program_id(1).to(tl.int64) * 64 + tl.arange(0, 64)
    h = tl.load(
        Y + row[:, None] * CO + ch[None, :],
        (row[:, None] < n) & (ch[None, :] < CO),
        other=0,
    )
    for group in range(tl.cdiv(WORDS, 4)):
        word_base = group * 4
        m0 = tl.load(
            Bits + row * WORDS + word_base, (row < n) & (word_base < WORDS), other=0
        )
        m1 = tl.load(
            Bits + row * WORDS + word_base + 1,
            (row < n) & (word_base + 1 < WORDS),
            other=0,
        )
        m2 = tl.load(
            Bits + row * WORDS + word_base + 2,
            (row < n) & (word_base + 2 < WORDS),
            other=0,
        )
        m3 = tl.load(
            Bits + row * WORDS + word_base + 3,
            (row < n) & (word_base + 3 < WORDS),
            other=0,
        )
        while tl.sum(((m0 | m1 | m2 | m3) != 0).to(tl.int32), 0) > 0:
            has = ((m0 | m1 | m2 | m3) != 0) & (row < n)
            word = tl.where(m0 != 0, 0, tl.where(m1 != 0, 1, tl.where(m2 != 0, 2, 3)))
            value = tl.where(
                m0 != 0, m0, tl.where(m1 != 0, m1, tl.where(m2 != 0, m2, m3))
            )
            off = (word_base + word) * 32 + libdevice.ffs(value) - 1
            src = (tl.load(Dense + row * (WORDS * 32) + off, has, other=0)).to(tl.int64)
            wbase = ch[None, :] * (K * CI) + off[:, None] * CI
            mask = has[:, None] & (ch[None, :] < CO)
            p = tl.zeros((ROWS, 64), tl.float32)
            for ci in tl.static_range(4):
                x = tl.load(X + src * CI + ci, has, other=0).to(tl.float32)
                w = tl.load(W + wbase + ci, mask, other=0).to(tl.float32)
                if ci == 0:
                    p = x[:, None] * w
                else:
                    p = tl.fma(x[:, None], w, p)
            h = tl.where(has[:, None], (p + h.to(tl.float32)).to(tl.float16), h)
            m0 = tl.where(word == 0, m0 & (m0 - 1), m0)
            m1 = tl.where(word == 1, m1 & (m1 - 1), m1)
            m2 = tl.where(word == 2, m2 & (m2 - 1), m2)
            m3 = tl.where(word == 3, m3 & (m3 - 1), m3)
    tl.store(
        Y + row[:, None] * CO + ch[None, :], h, (row[:, None] < n) & (ch[None, :] < CO)
    )


@triton.jit(do_not_specialize=["n"])
def _native_wgrad_part(
    X,
    DY,
    Pairs,
    Counts,
    Partial,
    n,
    CI: tl.constexpr,
    CO: tl.constexpr,
    K: tl.constexpr,
    CI_TILE: tl.constexpr,
    SPLITS: tl.constexpr,
    CM: tl.constexpr,
):
    """fp32 partial filter gradient ``dy.T @ x`` of one non-central offset, a ``[CM, CI_TILE]`` tile of output by input
    channels, over one of ``SPLITS`` shares of that offset's pairs, reduced over four partitions in cumm's order.
    """
    n = tl.cast(n, tl.int64)
    off0 = tl.program_id(0).to(tl.int64)
    off = tl.where(off0 < K // 2, off0, off0 + 1)
    ci_tiles = tl.cdiv(CI, CI_TILE)
    co = (tl.program_id(1).to(tl.int64) // ci_tiles) * CM + tl.arange(0, CM)
    ci = (tl.program_id(1).to(tl.int64) % ci_tiles) * CI_TILE + tl.arange(0, CI_TILE)
    part = tl.arange(0, 4)
    split = tl.program_id(2).to(tl.int64)
    count = tl.minimum(tl.load(Counts + tl.minimum(off, K - 1 - off)), n)
    span = tl.cdiv(tl.cdiv(count, 32), SPLITS) * 32
    begin = split * span
    end = tl.minimum(count, begin + span)
    acc = tl.zeros((4, CM, CI_TILE), tl.float32)
    for base in range(begin, end, 32):
        for k in tl.static_range(8):
            p = base + part * 8 + k
            src = (tl.load(Pairs + off * n + p, p < end, other=0)).to(tl.int64)
            dst = (tl.load(Pairs + (K + off) * n + p, p < end, other=0)).to(tl.int64)
            x = tl.load(
                X + src[:, None] * CI + ci[None, :],
                (p[:, None] < end) & (ci[None, :] < CI),
                other=0,
            ).to(tl.float32)
            y = tl.load(
                DY + dst[:, None] * CO + co[None, :],
                (p[:, None] < end) & (co[None, :] < CO),
                other=0,
            ).to(tl.float32)
            acc = tl.fma(y[:, :, None], x[:, None, :], acc)
    # cumm reduces the four warp-K partitions in increasing order.
    s = tl.sum(tl.where(part[:, None, None] == 0, acc, 0.0), 0)
    s += tl.sum(tl.where(part[:, None, None] == 1, acc, 0.0), 0)
    s += tl.sum(tl.where(part[:, None, None] == 2, acc, 0.0), 0)
    s += tl.sum(tl.where(part[:, None, None] == 3, acc, 0.0), 0)
    tl.store(
        Partial + ((split * K + off) * CO + co[:, None]) * CI + ci[None, :],
        s,
        (co[:, None] < CO) & (ci[None, :] < CI),
    )


@triton.jit
def _native_wgrad_finish(
    Partial,
    DW,
    CI: tl.constexpr,
    CO: tl.constexpr,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sum the ``SPLITS`` partials into ``BLOCK`` flat elements of the fp16 filter gradient ``DW`` ``[CO, K, CI]``,
    rounding after each split; the central offset is left to the dense matmul."""
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    co = i // (K * CI)
    off = i // CI % K
    ci = i % CI
    valid = (co < CO) & (off != K // 2)
    h = tl.zeros((BLOCK,), tl.float16)
    for split in tl.static_range(SPLITS):
        p = tl.load(Partial + ((split * K + off) * CO + co) * CI + ci, valid, other=0)
        h = (h.to(tl.float32) + p).to(tl.float16)
    tl.store(DW + i, h, valid)
