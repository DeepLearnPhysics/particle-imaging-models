"""Triton Hilbert codes: Skilling's transpose algorithm, bit interleave and Gray decode, one point per lane.

The Panda V2 encoder (``pimm/models/panda_v2/panda_encoder_v2m1.py``) serializes every batch along Hilbert curves.
The PyTorch encoder in ``pimm.models.utils.serialization`` runs several elementwise operations per bit and axis, each
a kernel launch over all points, and the optional CUDA extension that replaces it needs a separate build. This kernel
computes each code in one launch, bit for bit equal to the PyTorch encoder.

Reached through ``pimm.ops.hilbert_encode``.
"""

import torch
import triton
import triton.language as tl


# Skilling, Programming the Hilbert curve (2004).
@triton.jit(do_not_specialize=["n", "bits"])
def _hilbert_kernel(grid_ptr, code_ptr, n, bits, BLOCK: tl.constexpr):
    """Int64 Hilbert codes of ``BLOCK`` points, read as row-major ``[n, 3]`` coordinates with ``bits`` bits per
    axis."""
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    x = tl.load(grid_ptr + 3 * i, mask=valid, other=0).to(tl.int64)
    y = tl.load(grid_ptr + 3 * i + 1, mask=valid, other=0).to(tl.int64)
    z = tl.load(grid_ptr + 3 * i + 2, mask=valid, other=0).to(tl.int64)
    for b in range(bits):
        shift = (bits - 1 - b).to(tl.int64)
        low = (tl.full((), 1, tl.int64) << shift) - 1
        x = tl.where((x >> shift) & 1 == 1, x ^ low, x)
        for_y = (y >> shift) & 1 == 1
        x = tl.where(for_y, x ^ low, x)
        swap = tl.where(for_y, 0, (x ^ y) & low)
        x, y = x ^ swap, y ^ swap
        for_z = (z >> shift) & 1 == 1
        x = tl.where(for_z, x ^ low, x)
        swap = tl.where(for_z, 0, (x ^ z) & low)
        x, z = x ^ swap, z ^ swap
    code = tl.zeros_like(x)
    for b in range(bits):
        shift = (bits - 1 - b).to(tl.int64)
        top = 3 * (bits - b).to(tl.int64) - 1
        code |= ((x >> shift) & 1) << top
        code |= ((y >> shift) & 1) << (top - 1)
        code |= ((z >> shift) & 1) << (top - 2)
    for k in tl.static_range(6):  # Gray decode; shifts >= 3 * bits are no-ops
        code ^= code >> (1 << k)
    tl.store(code_ptr + i, code, mask=valid)


def hilbert_codes(grid, bits):
    """``[N, 3]`` non-negative integer coordinates below ``2**bits``, on a CUDA device, to ``[N]`` int64 codes."""
    code = torch.empty(len(grid), dtype=torch.long, device=grid.device)
    _hilbert_kernel[(triton.cdiv(len(grid), 256),)](
        grid.contiguous(), code, len(grid), bits, BLOCK=256
    )
    return code
