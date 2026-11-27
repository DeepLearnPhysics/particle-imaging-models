"""The first ``K`` points inside a radius on padded events, in index order, one block of queries per program.

Why this exists: in the PoLAr-MAE tokenizer (``pimm/models/polarmae/layers.py``, ``PointcloudGrouping``), every group
is gathered by a ball query around its centre, and for ``cnms`` the candidate centres inside each centre's ball are
counted. In the PyTorch version, a distance table and its running count are stored for every pair. Here a block of
queries of one event is assigned to each program, the event's points are scanned in tiles, and the members found in
each tile are written after those of the earlier tiles, so no table is stored and the scan stops once every query has
``K`` members.

Reached through ``pimm.models.polarmae.ops.ball_query`` on CUDA tensors.
"""

import torch
import triton
import triton.language as tl

BLOCK_Q = 32
BLOCK_P = 64


@triton.jit
def _ball_query_kernel(
    p1, s1b, s1p, p2, s2b, s2p, lengths1, lengths2, index, P1, K, radius2,
    D: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_P: tl.constexpr,
):
    """Members of the balls around a block of ``BLOCK_Q`` queries of event ``program_id(0)``.

    ``p1`` and ``p2`` are ``[B, P, D]`` FP32 with the given event and point strides, and ``index`` is ``[B, P1, K]``
    int64, filled with -1. Squared distances are summed from the coordinate differences with fused multiply-adds, as in
    pytorch3d's CUDA kernel, and compared strictly with ``radius2``."""
    b = tl.program_id(0).to(tl.int64)
    n1, n2 = tl.load(lengths1 + b).to(tl.int32), tl.load(lengths2 + b).to(tl.int32)
    if tl.program_id(1) * BLOCK_Q >= n1:
        return
    q = tl.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    valid = q < n1
    query = p1 + b * s1b + q * s1p
    points = p2 + b * s2b
    out = index + (b * P1 + q)[:, None] * K
    count = tl.where(valid, 0, K)  # rows past the event count as full
    column = tl.arange(0, BLOCK_P)
    first = tl.full([], 0, tl.int32)
    while (first < n2) & (tl.min(count) < K):
        p = first + column
        inside = p < n2
        squared = tl.zeros([BLOCK_Q, BLOCK_P], tl.float32)
        for c in tl.static_range(D):
            point = tl.load(points + p * s2p + c, inside, other=0.0)
            diff = tl.load(query + c, valid, other=0.0)[:, None] - point[None, :]
            squared = tl.fma(diff, diff, squared)
        within = (squared < radius2) & inside[None, :]
        slot = count[:, None] + tl.cumsum(within.to(tl.int32), 1) - 1
        tl.store(out + slot, p[None, :].to(tl.int64), within & (slot < K))
        count += tl.sum(within.to(tl.int32), 1)
        first += BLOCK_P


def ball_query(p1, p2, lengths1, lengths2, K, radius2):
    """``[B, P1, K]`` int64 indices into ``p2`` of the first ``K`` points with squared distance below ``radius2`` from
    each point of ``p1``, in ascending index order, -1 where fewer were found and on rows past ``lengths1``.

    ``p1`` and ``p2`` are ``[B, P, D]`` FP32 CUDA tensors with contiguous coordinates, and ``lengths1`` and ``lengths2``
    the ``[B]`` int64 valid counts."""
    B, P1, D = p1.shape
    index = torch.full((B, P1, K), -1, dtype=torch.long, device=p1.device)
    if B == 0 or P1 == 0 or K == 0:
        return index
    grid = (B, triton.cdiv(P1, BLOCK_Q))
    _ball_query_kernel[grid](
        p1, p1.stride(0), p1.stride(1), p2, p2.stride(0), p2.stride(1), lengths1, lengths2, index, P1, K, radius2,
        D=D, BLOCK_Q=BLOCK_Q, BLOCK_P=BLOCK_P,
    )
    return index
