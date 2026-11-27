"""Nearest neighbour within packed events, using a tiled Triton kernel or chunked PyTorch."""

import torch

from ._fused import require_fused

__all__ = ["nearest_neighbor"]

_PAIRS = 2**22  # query-point pairs per step of the PyTorch version


def _bounds(offset):
    """Cumulative row counts ``[events]`` to int64 row boundaries ``[events + 1]`` starting at 0."""
    return torch.cat([offset.new_zeros(1), offset]).long()


def _reference(query, query_bounds, points, point_bounds):
    """The PyTorch version of ``nearest_neighbor``, one event and one chunk of its queries at a time."""
    index = torch.full((len(query),), -1, dtype=torch.long, device=query.device)
    distance = torch.full((len(query),), float("inf"), device=query.device)
    q, p = query_bounds.tolist(), point_bounds.tolist()
    for b in range(len(q) - 1):
        if p[b + 1] == p[b]:
            continue
        event = points[p[b] : p[b + 1]]
        step = max(1, _PAIRS // len(event))
        for begin in range(q[b], q[b + 1], step):
            end = min(begin + step, q[b + 1])
            diff = query[begin:end, None] - event[None]
            squared = diff[..., 0] ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2
            best, arg = squared.min(1)  # the first index on ties
            index[begin:end] = p[b] + arg
            distance[begin:end] = best.sqrt()
    return index, distance


@torch.no_grad()
def nearest_neighbor(query, query_offset, points, point_offset, *, fused=False):
    """Index and distance of the nearest point of the same event for every ``[M, 3]`` query among ``[N, 3]`` points.

    Events are packed rows: ``query_offset`` and ``point_offset`` are the cumulative row counts of the events (the
    ``offset`` of a pimm ``Point``), so the rows of event ``b`` are ``offset[b - 1]:offset[b]``. The ``[M]`` int64
    index is into ``points``, and the ``[M]`` FP32 distance is Euclidean, not squared, as in pointops' ``knn_query``.
    Squared distances are summed in FP32 from the coordinate differences, and of equally near points the lowest
    index is returned. Queries of an event without points get index -1 and distance inf, and no rows are
    returned for an event without queries. No gradient is computed.

    With ``fused``, a Triton kernel is run on CUDA tensors, and each event's points are scanned in tiles for a block
    of its queries. Otherwise the distances of each event are computed in PyTorch, in chunks of queries, on any
    device. Multiply-adds are fused in the kernel, as in pointops' CUDA build, and rounded separately in PyTorch, so
    of two points within one rounding of each other, a different one can be returned by each version.
    """
    if query.ndim != 2 or points.ndim != 2 or query.shape[1] != 3 or points.shape[1] != 3:
        raise ValueError("query and points must be [rows, 3] coordinates")
    if len(query_offset) != len(point_offset):
        raise ValueError("query_offset and point_offset must count the same events")
    query, points = query.float().contiguous(), points.float().contiguous()
    query_bounds, point_bounds = _bounds(query_offset), _bounds(point_offset)
    if not fused:
        return _reference(query, query_bounds, points, point_bounds)
    on_cuda = all(t.is_cuda for t in (query, query_offset, points, point_offset))
    require_fused("nearest_neighbor", None if on_cuda else "the Triton kernel takes CUDA tensors only")
    from .triton.knn import nearest

    return nearest(query, query_bounds, points, point_bounds)
