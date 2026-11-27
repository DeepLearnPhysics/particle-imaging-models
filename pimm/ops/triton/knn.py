"""Nearest neighbour of every query point among the points of its event, one block of queries per program.

Why this exists: in Sonata pretraining (``pimm/models/sonata/sonata_v1m1_base.py``, ``match_neighbour``), every point
of one view is paired with the nearest point of another view of the same event. This was done by the pointops CUDA
extension, which had to be built separately. Here a block of queries of one event is assigned to each program, the
points of that event are scanned in tiles, and a running minimum and its index are kept, so no distance table is
stored.

Reached through ``pimm.ops.nearest_neighbor(..., fused=True)``.
"""

import torch
import triton
import triton.language as tl

BLOCK_Q = 64
BLOCK_P = 32


@triton.jit
def _nearest_kernel(
    query,
    points,
    query_bounds,
    point_bounds,
    index,
    distance,
    BLOCK_Q: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Index and distance of the nearest point of event ``program_id(1)`` for a block of ``BLOCK_Q`` of its queries.

    Coordinates are row-major ``[rows, 3]`` FP32, and the rows of event ``b`` are ``bounds[b]:bounds[b + 1]``. Squared
    distances are summed from the coordinate differences in pointops' order. A running minimum is kept for every
    (query, tile column) pair, with the start of the tile that set it, and is reduced over the columns once, after
    the last tile; of equally near points the lowest index is kept. Queries of an event without points get index -1
    and distance inf."""
    event = tl.program_id(1)
    q_begin, q_end = tl.load(query_bounds + event), tl.load(query_bounds + event + 1)
    if q_begin + tl.program_id(0) * BLOCK_Q >= q_end:
        return
    row = q_begin + tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    valid = row < q_end
    qx = tl.load(query + 3 * row, valid, other=0.0)[:, None]
    qy = tl.load(query + 3 * row + 1, valid, other=0.0)[:, None]
    qz = tl.load(query + 3 * row + 2, valid, other=0.0)[:, None]
    p_begin, p_end = tl.load(point_bounds + event), tl.load(point_bounds + event + 1)
    column = tl.arange(0, BLOCK_P)
    best = tl.full([BLOCK_Q, BLOCK_P], float("inf"), tl.float32)
    start = tl.zeros([BLOCK_Q, BLOCK_P], tl.int32)  # tile start, relative to the event's first point
    for first in range(p_begin, p_end, BLOCK_P):
        col = first + column
        inside = col < p_end  # points past the event load as inf, at an infinite distance
        dx = qx - tl.load(points + 3 * col, inside, other=float("inf"))[None, :]
        dy = qy - tl.load(points + 3 * col + 1, inside, other=float("inf"))[None, :]
        dz = qz - tl.load(points + 3 * col + 2, inside, other=float("inf"))[None, :]
        squared = dx * dx + dy * dy + dz * dz
        closer = squared < best  # strict, so an earlier tile keeps a tie
        best = tl.where(closer, squared, best)
        start = tl.where(closer, (first - p_begin).to(tl.int32), start)
    smallest = tl.min(best, 1)
    tied = tl.where(best == smallest[:, None], start + column[None, :], 2**31 - 1)
    found = smallest < float("inf")
    tl.store(index + row, tl.where(found, p_begin + tl.min(tied, 1), -1), valid)
    tl.store(distance + row, tl.sqrt_rn(smallest), valid)


def nearest(query, query_bounds, points, point_bounds):
    """``[M]`` int64 index into ``points`` and ``[M]`` FP32 distance of the nearest point of each query's event.

    ``query`` and ``points`` are contiguous ``[rows, 3]`` FP32 CUDA tensors, and ``query_bounds`` and
    ``point_bounds`` are ``[events + 1]`` int64 row boundaries starting at 0."""
    index = torch.full((len(query),), -1, dtype=torch.long, device=query.device)
    distance = torch.full((len(query),), float("inf"), device=query.device)
    if len(query) == 0:
        return index, distance
    longest = int(query_bounds.diff().max())  # one host sync, to size the grid
    grid = (triton.cdiv(longest, BLOCK_Q), len(query_bounds) - 1)
    _nearest_kernel[grid](
        query, points, query_bounds, point_bounds, index, distance, BLOCK_Q=BLOCK_Q, BLOCK_P=BLOCK_P
    )
    return index, distance
