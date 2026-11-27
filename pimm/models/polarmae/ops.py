# Ported from https://github.com/WireCell/wire-cell-fm (wcfm/model/backbones/polarmae/ops.py).
"""Point-set operations for the PoLAr-MAE tokenizer and losses.

The conventions are pytorch3d's, whose compiled kernels PoLAr-MAE used before:

- distances are squared Euclidean over every coordinate passed;
- in `knn_points`, a short row is padded with index 0 and distance `inf` (pytorch3d pads with 0);
- in `ball_query`, the first `K` points strictly inside the radius are listed in ascending index
  order and padded with -1; `sample_farthest_points` is padded with -1 as well;
- the padded tail of each event in a `(B, N, D)` batch is masked by a `lengths` argument.

CUDA tensors are handled by the Triton kernels in `pimm/ops/triton/` (`fps.py`, `ball_query.py`,
`knn_points.py`, `cnms.py`), CPU tensors by the PyTorch code here; without Triton, an ImportError is
raised for CUDA tensors. In the PyTorch code, distance tables are computed in chunks of query rows,
at most `CHUNK_ELEMENTS` values per chunk.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Upper bound on the elements of one `(B, chunk, P)` distance block.
CHUNK_ELEMENTS = 1 << 26


def _chunk_rows(B: int, P: int) -> int:
    return max(1, CHUNK_ELEMENTS // max(1, B * P))


def _valid(lengths: Tensor | None, B: int, P: int, device) -> Tensor:
    if lengths is None:
        return torch.ones(B, P, dtype=torch.bool, device=device)
    return torch.arange(P, device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)


def _coords(points: Tensor) -> Tensor:
    """FP32 points whose coordinates are contiguous, as the Triton kernels read them."""
    points = points.float()
    return points if points.stride(-1) == 1 else points.contiguous()


def _lengths(lengths: Tensor | None, B: int, P: int, device) -> Tensor:
    if lengths is None:
        return torch.full((B,), P, dtype=torch.int64, device=device)
    return lengths.to(device=device, dtype=torch.int64).contiguous()


def sq_dists(p1: Tensor, p2: Tensor) -> Tensor:
    """`(B, c, D) x (B, P, D) -> (B, c, P)` squared Euclidean distances, formed as pytorch3d's
    kernels form them: the difference per coordinate, squared and summed in coordinate order, in
    fp32 under any autocast. With the square expanded through `bmm` instead, near neighbours swap
    and points that sit on the radius of a ball move in or out."""
    p1 = p1.float()
    p2 = p2.float()
    d2 = p1.new_zeros(p1.shape[0], p1.shape[1], p2.shape[1])
    for d in range(p1.shape[-1]):
        diff = p1[..., d].unsqueeze(2) - p2[..., d].unsqueeze(1)
        d2.addcmul_(diff, diff)
    return d2


def _radius2(radius: float) -> float:
    """`radius * radius` as pytorch3d forms it: the radius rounded to fp32, then squared in fp32."""
    return float(torch.tensor(float(radius), dtype=torch.float32) ** 2)


def knn_points(
    p1: Tensor,
    p2: Tensor,
    lengths1: Tensor | None = None,
    lengths2: Tensor | None = None,
    K: int = 1,
    return_sorted: bool = True,
) -> tuple[Tensor, Tensor]:
    """The `K` nearest points of `p2` for every point of `p1`: `(dists, idx)`, both `(B, P1, K)`.

    Where a row of `p2` is shorter than `K`, the tail is filled with distance `inf` and index 0;
    rows of `p1` beyond `lengths1` get distance 0 and index 0. Differentiable in the distances,
    which `chamfer_distance` relies on. On CUDA the neighbours are found by the Triton kernel, for
    `K` up to 8 (a larger `K` raises), nearest first and ties to the lowest index, whatever
    `return_sorted`.
    """
    B, P1, D = p1.shape
    P2 = p2.shape[1]
    K = int(K)
    if p1.is_cuda:
        from pimm.ops.triton.knn_points import knn_points as knn

        l1, l2 = _lengths(lengths1, B, P1, p1.device), _lengths(lengths2, B, P2, p1.device)
        dists, idx = knn(_coords(p1), _coords(p2), l1, l2, K)
        if torch.is_grad_enabled() and (p1.requires_grad or p2.requires_grad):
            # No gradient is computed by the kernel, so the distances to the neighbours are formed again.
            near = p2.float().gather(1, idx.flatten(1).unsqueeze(-1).expand(-1, -1, D))
            again = (p1.float().unsqueeze(2) - near.view(B, P1, K, D)).square().sum(-1)
            found = torch.isfinite(dists) & _valid(l1, B, P1, p1.device).unsqueeze(-1)
            dists = torch.where(found, again, dists)
        return dists, idx
    valid2 = _valid(lengths2, B, P2, p1.device)
    Keff = min(K, P2)
    dists_out = p1.new_full((B, P1, K), float("inf"), dtype=torch.float32)
    idx_out = torch.zeros((B, P1, K), dtype=torch.int64, device=p1.device)
    step = _chunk_rows(B, P2)
    for s in range(0, P1, step):
        e = min(s + step, P1)
        d2 = sq_dists(p1[:, s:e], p2).masked_fill(~valid2.unsqueeze(1), float("inf"))
        dists, idx = torch.topk(d2, Keff, dim=2, largest=False, sorted=return_sorted)
        dists_out[:, s:e, :Keff] = dists
        idx_out[:, s:e, :Keff] = idx.masked_fill(torch.isinf(dists), 0)
    if lengths1 is not None:
        padded = ~_valid(lengths1, B, P1, p1.device).unsqueeze(-1)
        dists_out, idx_out = dists_out.masked_fill(padded, 0.0), idx_out.masked_fill(padded, 0)
    return dists_out, idx_out


@torch.no_grad()
def ball_query(
    p1: Tensor,
    p2: Tensor,
    K: int,
    radius: float,
    lengths1: Tensor | None = None,
    lengths2: Tensor | None = None,
) -> Tensor:
    """Indices into `p2` of the first `K` points strictly within `radius` of each point of `p1`,
    in ascending `p2` index order, `(B, P1, K)`, -1 where fewer than `K` were found.

    With index order the result is reproducible: in a point cloud sorted by coordinate, every
    group is filled with its lowest-index members.
    """
    B, P1, _ = p1.shape
    P2 = p2.shape[1]
    device = p1.device
    K = int(K)
    r2 = _radius2(radius)
    if p1.is_cuda:
        from pimm.ops.triton.ball_query import ball_query as query

        l1, l2 = _lengths(lengths1, B, P1, device), _lengths(lengths2, B, P2, device)
        return query(_coords(p1), _coords(p2), l1, l2, K, r2)
    valid2 = _valid(lengths2, B, P2, device)
    members = torch.arange(P2, device=device)
    idx_out = torch.empty((B, P1, K), dtype=torch.int64, device=device)
    step = _chunk_rows(B, P2)
    for s in range(0, P1, step):
        e = min(s + step, P1)
        within = (sq_dists(p1[:, s:e], p2) < r2) & valid2.unsqueeze(1)
        # Member j of a ball is written to slot (members up to j) - 1, and members past K to
        # slot K, which is dropped.
        rank = within.cumsum(-1, dtype=torch.int32)
        slot = torch.where(within & (rank <= K), rank - 1, K).long()
        out = torch.full((B, e - s, K + 1), -1, dtype=torch.int64, device=device)
        out.scatter_(2, slot, members.expand(B, e - s, P2))
        idx_out[:, s:e] = out[..., :K]
    if lengths1 is not None:
        idx_out = idx_out.masked_fill(~_valid(lengths1, B, P1, device).unsqueeze(-1), -1)
    return idx_out


@torch.no_grad()
def sample_farthest_points(
    points: Tensor, lengths: Tensor, K: int, start_idxs: Tensor | None = None
) -> Tensor:
    """Farthest point sampling: `(B, K)` indices into `points`, -1 past `min(lengths, K)`.

    Position 0 is `start_idxs` (0 when not given); every following pick is the valid point
    farthest from the picks so far, ties going to the lowest index. On CUDA the picks are made by
    the Triton kernel.
    """
    points = points[..., :3].float()
    B, P, _ = points.shape
    device = points.device
    lengths = lengths.to(device=device, dtype=torch.int64)
    K = int(K)
    idx_out = torch.full((B, K), -1, dtype=torch.int64, device=device)
    if B == 0 or K == 0 or P == 0:
        return idx_out
    rows = torch.arange(B, device=device)
    if start_idxs is None:
        sel = torch.zeros(B, dtype=torch.int64, device=device)
    else:
        sel = start_idxs.to(device=device, dtype=torch.int64).clamp(min=0, max=P - 1)
    if points.is_cuda:
        from pimm.ops.triton.fps import farthest_point_sampling

        return farthest_point_sampling(_coords(points), lengths.contiguous(), K, sel)
    valid = _valid(lengths, B, P, device)
    # Padded points start at -inf and stay there through `minimum`, out of every argmax.
    closest = torch.full((B, P), float("inf"), device=device).masked_fill_(~valid, float("-inf"))
    idx_out[:, 0] = sel
    for i in range(1, K):
        d = (points - points[rows, sel].unsqueeze(1)).square_().sum(-1)
        torch.minimum(closest, d, out=closest)
        sel = closest.argmax(dim=1)
        idx_out[:, i] = sel
    keep = torch.arange(K, device=device).unsqueeze(0) < lengths.clamp(max=K).unsqueeze(1)
    return idx_out.masked_fill_(~keep, -1)


@torch.no_grad()
def cnms(
    centroids: Tensor, radius: float, overlap_factor: float, K: int, lengths: Tensor
) -> tuple[Tensor, Tensor]:
    """Centrality-based non-maximum suppression over candidate group centres.

    For every centroid, the candidates within `2 * radius * overlap_factor` of it are counted (at
    most `K`, in index order). Candidates are visited in descending count, ties by index; a
    candidate not yet suppressed is retained, and everything in its ball is suppressed. Returned:
    the centroids reordered retained-first, in ascending index order within each half, and the
    retained count per event.

    On CUDA the greedy pass is the Triton kernel `pimm.ops.triton.cnms.greedy_retain`, one
    candidate at a time per event; elsewhere it is `_greedy_rounds`, with the same set. In the
    compiled `cnms` package used before, the counts were sorted unstably, so its set could differ
    wherever tied counts were reordered; and with neighbour lists truncated below the largest
    ball, earlier centres that sat in a later one's list were un-retained. PoLAr-MAE passes
    `K = P`, where neither applies.
    """
    B, P, D = centroids.shape
    lengths = lengths.to(device=centroids.device, dtype=torch.int64)
    query_radius = 2.0 * float(radius) * float(overlap_factor)
    idx = ball_query(
        centroids, centroids, K=K, radius=query_radius, lengths1=lengths, lengths2=lengths
    )
    counts = idx.ge(0).sum(-1)  # (B, P)
    # Columns past the fullest neighbour list hold only -1.
    idx = idx[..., : max(int(counts.max()), 1)] if counts.numel() else idx
    _, order = counts.sort(dim=-1, descending=True, stable=True)
    if centroids.is_cuda:
        from pimm.ops.triton.cnms import greedy_retain

        retain = greedy_retain(order, idx, lengths)
    else:
        retain = _greedy_rounds(order, idx, lengths)
    reorder = torch.argsort((~retain).to(torch.int8), dim=1, stable=True)
    centroids = centroids.gather(1, reorder.unsqueeze(-1).expand(-1, -1, D))
    return centroids, retain.sum(dim=1)


def _greedy_rounds(order: Tensor, idx: Tensor, lengths: Tensor) -> Tensor:
    """`cnms`'s greedy pass in rounds instead of one candidate at a time: in each round, every
    unresolved candidate that no unresolved candidate earlier in `order` can still suppress is
    retained, and then their balls are suppressed. The number of rounds is the depth of the
    suppression chain rather than the number of centres."""
    B, P, K = idx.shape
    device = idx.device
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(P, device=device).expand(B, P))
    rank = rank.to(torch.int32)

    nb_valid = idx.ge(0)
    idx0 = idx.clamp(min=0).reshape(B, P * K)
    unresolved = _valid(lengths, B, P, device)
    retain = torch.zeros(B, P, dtype=torch.bool, device=device)
    none = torch.tensor(P + 1, dtype=torch.int32, device=device)
    while bool(unresolved.any()):
        r = torch.where(unresolved, rank, none)
        edge = torch.where(nb_valid, r.unsqueeze(-1).expand(B, P, K), none)
        best = torch.full((B, P), int(none), dtype=torch.int32, device=device)
        best.scatter_reduce_(1, idx0, edge.reshape(B, P * K), reduce="amin")
        now = unresolved & (best >= rank)
        retain |= now
        # The members of a retained centre's ball that come after it in the order are suppressed.
        # The ones before it were decided at their own turn, and with a truncated neighbour list
        # a centre can sit in a later centre's ball without the reverse holding.
        r_now = torch.where(now, rank, none)
        edge = torch.where(nb_valid, r_now.unsqueeze(-1).expand(B, P, K), none)
        hit = torch.full((B, P), int(none), dtype=torch.int32, device=device)
        hit.scatter_reduce_(1, idx0, edge.reshape(B, P * K), reduce="amin")
        unresolved &= ~(now | (hit < rank))
    return retain


def chamfer_distance(x: Tensor, y: Tensor, x_lengths: Tensor, y_lengths: Tensor) -> Tensor:
    """Bidirectional Chamfer distance between padded point sets, `(N, P1, D)` and `(N, P2, D)`.

    Each direction is the mean over valid points of the squared distance to the nearest valid
    point on the other side; the two are summed and averaged over the batch, which is pytorch3d's
    default reduction. A set with no valid points counts as zero.
    """

    def one_way(a, b, la, lb):
        d = knn_points(a, b, lengths1=la, lengths2=lb, K=1)[0][..., 0]
        valid = _valid(la, a.shape[0], a.shape[1], a.device) & torch.isfinite(d)
        return torch.where(valid, d, torch.zeros_like(d)).sum(1) / la.clamp(min=1)

    per_event = one_way(x, y, x_lengths, y_lengths) + one_way(y, x, y_lengths, x_lengths)
    return per_event.sum() / max(int(x.shape[0]), 1)
