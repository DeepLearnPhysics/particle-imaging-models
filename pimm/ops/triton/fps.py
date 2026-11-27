"""Farthest point sampling on padded events, one program per event.

Why this exists: in the PoLAr-MAE tokenizer (``pimm/models/polarmae/layers.py``, ``PointcloudGrouping``), the groups
of every event are seeded with its 2048 farthest points, and every group is thinned to its farthest points. The picks
are sequential, and in the PyTorch version several kernels are launched per pick. Here all the picks of one event are
made by one program: for every pick, the event's points are scanned in tiles, each point's squared distance to the
nearest pick so far is kept in a scratch row, and the farthest point is reduced over the tiles.

Reached through ``pimm.models.polarmae.ops.sample_farthest_points`` on CUDA tensors.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fps_kernel(points, stride_b, stride_p, lengths, start, closest, index, P, K, BLOCK: tl.constexpr):
    """The first ``min(lengths[b], K)`` picks of event ``b = program_id(0)``, written to ``index[b]``.

    ``points`` is ``[B, P, 3]`` FP32 with the given event and point strides, and ``closest`` a ``[B, P]`` FP32 scratch
    filled with inf. Squared distances are summed from the coordinate differences with fused multiply-adds, as in
    pytorch3d's CUDA kernel, and of equally far points the lowest index is picked."""
    b = tl.program_id(0).to(tl.int64)
    n = tl.minimum(tl.load(lengths + b), P).to(tl.int32)
    picks = tl.minimum(n, K)
    if picks == 0:
        return
    row = points + b * stride_b
    scratch = closest + b * P
    sel = tl.load(start + b).to(tl.int32)
    tl.store(index + b * K, sel)
    column = tl.arange(0, BLOCK)
    for i in range(1, picks):
        at_sel = row + sel * stride_p
        sx, sy, sz = tl.load(at_sel), tl.load(at_sel + 1), tl.load(at_sel + 2)
        best = tl.full([], -1.0, tl.float32)
        arg = tl.full([], 0, tl.int32)
        for first in range(0, n, BLOCK):
            p = first + column
            inside = p < n
            dx = sx - tl.load(row + p * stride_p, inside, other=0.0)
            dy = sy - tl.load(row + p * stride_p + 1, inside, other=0.0)
            dz = sz - tl.load(row + p * stride_p + 2, inside, other=0.0)
            squared = tl.fma(dz, dz, tl.fma(dy, dy, dx * dx))
            near = tl.where(inside, tl.minimum(squared, tl.load(scratch + p, inside)), -1.0)
            tl.store(scratch + p, near, inside)
            far, at = tl.max(near, 0, return_indices=True, return_indices_tie_break_left=True)
            arg = tl.where(far > best, first + at, arg)  # strict, so an earlier tile keeps a tie
            best = tl.maximum(far, best)
        sel = arg
        tl.store(index + b * K + i, sel)
        # The scratch row written for this pick is read back for the next.
        tl.debug_barrier()


def farthest_point_sampling(points, lengths, K, start):
    """``[B, K]`` int64 indices of the farthest point samples of every event, -1 past ``min(lengths, K)``.

    ``points`` is a ``[B, P, 3]`` FP32 CUDA tensor with contiguous coordinates, ``lengths`` the ``[B]`` int64 valid
    counts, and ``start`` the ``[B]`` int64 index of each event's first pick."""
    B, P, _ = points.shape
    index = torch.full((B, K), -1, dtype=torch.long, device=points.device)
    if B == 0 or P == 0 or K == 0:
        return index
    closest = torch.full((B, P), float("inf"), device=points.device)
    BLOCK = min(triton.next_power_of_2(P), 4096)
    _fps_kernel[(B,)](
        points, points.stride(0), points.stride(1), lengths, start, closest, index, P, K,
        BLOCK=BLOCK, num_warps=max(1, min(16, BLOCK // 256)),
    )
    return index
