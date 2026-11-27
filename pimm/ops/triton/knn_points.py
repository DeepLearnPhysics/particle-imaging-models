"""The ``K`` nearest neighbours on padded events, one block of queries per program.

Why this exists: in PoLAr-MAE, token features are interpolated back to the points from each point's 5 nearest group
centres (``pimm/models/polarmae/layers.py``, ``PointNetFeatureUpsampling``), and in the Chamfer loss every
reconstructed point is paired with its nearest true one. In the PyTorch version, the distance table is stored and
``topk`` is run over it. Here, as in pytorch3d's CUDA kernel, a block of queries of one event is assigned to each
program, the event's points are visited in index order, and the ``K`` nearest so far are kept in order for every
query, so no table is stored. The points are visited four at a time, and the insertion is skipped when none of the
four is nearer than any query's ``K``-th neighbour, which after the first points is most of the time.

Reached through ``pimm.models.polarmae.ops.knn_points`` on CUDA tensors.
"""

import torch
import triton
import triton.language as tl

MAX_K = 8
BLOCK_Q = 32


@triton.jit
def _order(near, at, d, j):
    """Two (distance, index) pairs, the one with the smaller distance first, and of equal distances the lower index."""
    swap = (d < near) | ((d == near) & (j < at))
    return tl.where(swap, d, near), tl.where(swap, j, at), tl.where(swap, near, d), tl.where(swap, at, j)


@triton.jit
def _insert(d0, i0, d1, i1, d2, i2, d3, i3, d4, i4, d5, i5, d6, i6, d7, i7, d, i, K: tl.constexpr):
    """``(d, i)`` inserted into the first ``K`` ordered slots; the pair pushed out of slot ``K - 1`` is dropped."""
    d0, i0, d, i = _order(d0, i0, d, i)
    if K > 1:
        d1, i1, d, i = _order(d1, i1, d, i)
    if K > 2:
        d2, i2, d, i = _order(d2, i2, d, i)
    if K > 3:
        d3, i3, d, i = _order(d3, i3, d, i)
    if K > 4:
        d4, i4, d, i = _order(d4, i4, d, i)
    if K > 5:
        d5, i5, d, i = _order(d5, i5, d, i)
    if K > 6:
        d6, i6, d, i = _order(d6, i6, d, i)
    if K > 7:
        d7, i7, d, i = _order(d7, i7, d, i)
    return d0, i0, d1, i1, d2, i2, d3, i3, d4, i4, d5, i5, d6, i6, d7, i7


@triton.jit
def _last(d0, d1, d2, d3, d4, d5, d6, d7, K: tl.constexpr):
    """The distance in slot ``K - 1``, that of each query's ``K``-th neighbour so far."""
    worst = d0
    if K == 2:
        worst = d1
    if K == 3:
        worst = d2
    if K == 4:
        worst = d3
    if K == 5:
        worst = d4
    if K == 6:
        worst = d5
    if K == 7:
        worst = d6
    if K == 8:
        worst = d7
    return worst


@triton.jit
def _distance(query, valid, points, j, D: tl.constexpr):
    """Squared distance of every query to the point at ``points + j``, summed over the coordinates with fused
    multiply-adds."""
    d = tl.zeros(valid.shape, tl.float32)
    for c in tl.static_range(D):
        diff = tl.load(query + c, valid, other=0.0) - tl.load(points + j + c)
        d = tl.fma(diff, diff, d)
    return d


@triton.jit
def _tile(x0, x1, x2, x3, x4, x5, x6, x7):
    """The eight ``[BLOCK_Q]`` slots as the columns of one ``[BLOCK_Q, 8]`` tile."""
    col = tl.arange(0, 8)[None, :]
    tile = tl.where(col == 6, x6[:, None], x7[:, None])
    tile = tl.where(col == 5, x5[:, None], tile)
    tile = tl.where(col == 4, x4[:, None], tile)
    tile = tl.where(col == 3, x3[:, None], tile)
    tile = tl.where(col == 2, x2[:, None], tile)
    tile = tl.where(col == 1, x1[:, None], tile)
    return tl.where(col == 0, x0[:, None], tile)


@triton.jit
def _knn_kernel(
    p1, s1b, s1p, p2, s2b, s2p, lengths1, lengths2, dists, index, P1,
    D: tl.constexpr, K: tl.constexpr, BLOCK_Q: tl.constexpr,
):
    """Squared distances and indices of the ``K`` nearest points of event ``program_id(0)`` for a block of queries.

    ``p1`` and ``p2`` are ``[B, P, D]`` FP32 with the given event and point strides, and ``dists`` and ``index`` are
    ``[B, P1, K]``. Squared distances are summed from the coordinate differences with fused multiply-adds, as in
    pytorch3d's CUDA kernel. Points are inserted into slots ordered by distance and then by index, so of equally near
    points the lowest indices are kept. Slots past the event's points get distance inf and index 0, and rows past the
    event's queries get distance 0 and index 0."""
    b = tl.program_id(0).to(tl.int64)
    n1, n2 = tl.load(lengths1 + b).to(tl.int32), tl.load(lengths2 + b).to(tl.int32)
    q = tl.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    valid = (q < n1) & (q < P1)
    query = p1 + b * s1b + q * s1p
    points = p2 + b * s2b
    inf = tl.full([BLOCK_Q], float("inf"), tl.float32)
    zero = tl.zeros([BLOCK_Q], tl.int32)
    d0, d1, d2, d3, d4, d5, d6, d7 = inf, inf, inf, inf, inf, inf, inf, inf
    i0, i1, i2, i3, i4, i5, i6, i7 = zero, zero, zero, zero, zero, zero, zero, zero
    stop = tl.where(tl.program_id(1) * BLOCK_Q < n1, n2, 0)  # no scan for a block past the event's queries
    for j in range(0, stop // 4 * 4, 4):
        da = _distance(query, valid, points, j * s2p, D)
        db = _distance(query, valid, points, (j + 1) * s2p, D)
        dc = _distance(query, valid, points, (j + 2) * s2p, D)
        dd = _distance(query, valid, points, (j + 3) * s2p, D)
        nearest = tl.minimum(tl.minimum(da, db), tl.minimum(dc, dd))
        worst = _last(d0, d1, d2, d3, d4, d5, d6, d7, K)
        if tl.min(nearest - worst) < 0:  # a later point never displaces an equally near one
            d0, i0, d1, i1, d2, i2, d3, i3, d4, i4, d5, i5, d6, i6, d7, i7 = _insert(
                d0, i0, d1, i1, d2, i2, d3, i3, d4, i4, d5, i5, d6, i6, d7, i7, da, zero + j, K
            )
            d0, i0, d1, i1, d2, i2, d3, i3, d4, i4, d5, i5, d6, i6, d7, i7 = _insert(
                d0, i0, d1, i1, d2, i2, d3, i3, d4, i4, d5, i5, d6, i6, d7, i7, db, zero + j + 1, K
            )
            d0, i0, d1, i1, d2, i2, d3, i3, d4, i4, d5, i5, d6, i6, d7, i7 = _insert(
                d0, i0, d1, i1, d2, i2, d3, i3, d4, i4, d5, i5, d6, i6, d7, i7, dc, zero + j + 2, K
            )
            d0, i0, d1, i1, d2, i2, d3, i3, d4, i4, d5, i5, d6, i6, d7, i7 = _insert(
                d0, i0, d1, i1, d2, i2, d3, i3, d4, i4, d5, i5, d6, i6, d7, i7, dd, zero + j + 3, K
            )
    for j in range(stop // 4 * 4, stop):
        d0, i0, d1, i1, d2, i2, d3, i3, d4, i4, d5, i5, d6, i6, d7, i7 = _insert(
            d0, i0, d1, i1, d2, i2, d3, i3, d4, i4, d5, i5, d6, i6, d7, i7, _distance(query, valid, points, j * s2p, D),
            zero + j, K,
        )
    # Each query row is written as K consecutive slots; rows past the event get distance 0 and index 0.
    col = tl.arange(0, 8)[None, :]
    out = (b * P1 + q)[:, None] * K + col
    keep = (q < P1)[:, None] & (col < K)
    near = tl.where(valid[:, None], _tile(d0, d1, d2, d3, d4, d5, d6, d7), 0.0)
    tl.store(dists + out, near, keep)
    tl.store(index + out, tl.where(valid[:, None], _tile(i0, i1, i2, i3, i4, i5, i6, i7), 0).to(tl.int64), keep)

def knn_points(p1, p2, lengths1, lengths2, K):
    """``([B, P1, K]`` FP32 squared distances, ``[B, P1, K]`` int64 indices``)`` of the ``K`` nearest points of ``p2``
    for every point of ``p1``, nearest first, for ``K`` from 1 to ``MAX_K``.

    ``p1`` and ``p2`` are ``[B, P, D]`` FP32 CUDA tensors with contiguous coordinates, and ``lengths1`` and ``lengths2``
    the ``[B]`` int64 valid counts. Where an event of ``p2`` has fewer than ``K`` points, the tail gets distance inf and
    index 0; rows past ``lengths1`` get distance 0 and index 0, as in pytorch3d. No gradient is computed."""
    if not 1 <= K <= MAX_K:
        raise ValueError(f"the knn_points kernel keeps 1 to {MAX_K} neighbours, not K={K}")
    B, P1, D = p1.shape
    dists = torch.empty((B, P1, K), device=p1.device)
    index = torch.empty((B, P1, K), dtype=torch.long, device=p1.device)
    if B == 0 or P1 == 0:
        return dists, index
    grid = (B, triton.cdiv(P1, BLOCK_Q))
    _knn_kernel[grid](
        p1, p1.stride(0), p1.stride(1), p2, p2.stride(0), p2.stride(1), lengths1, lengths2, dists, index, P1,
        D=D, K=K, BLOCK_Q=BLOCK_Q, num_warps=1,
    )
    return dists, index
