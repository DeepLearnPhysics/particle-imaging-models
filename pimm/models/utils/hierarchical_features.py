"""Hierarchical (multi-resolution) feature sampling for pimm point backbones.

pimm's PTv3 / LitePT encoders build a pooling hierarchy: each ``GridPooling``
stage stores its (finer) input as ``pooling_parent`` and the cluster assignment
that maps finer points into the coarser voxel grid as ``pooling_inverse``. The
default way of turning that hierarchy into a point-resolution feature is the
"upcast" walk used throughout the codebase: pop the chain from coarse to fine
and, at each step, concatenate the coarse feature (broadcast back through
``pooling_inverse``) onto the finer feature. That broadcast is exactly a
*nearest-voxel* lookup -- each fine point copies the feature of the single
coarse voxel it pooled into.

This module adds the multi-resolution feature sampling of *Masked Scene
Modeling* (Hermosilla et al., https://arxiv.org/abs/2504.06719,
https://github.com/phermosilla/msm). For each selected hierarchy level it
samples that level's voxel grid at the original (finest) point locations and
concatenates the per-level results into a rich point-level descriptor. Two
sampling modes are provided:

- ``"nearest"``: take the feature of the voxel each point falls in. Computed for
  free from the composed ``pooling_inverse`` chain. With all levels selected
  this reproduces the default upcast concatenation exactly.
- ``"trilinear"``: trilinearly interpolate between the eight neighbouring voxel
  centres at each level (the differentiable sampling MSM uses), implemented as a
  sparse 8-neighbour gather so it scales to the large, sparse grids of particle
  imaging without densifying the volume.

``Point.upcast(feature_sampling=...)`` selects between the default upcast
(``feature_sampling=None``) and MSM sampling; both return a finest resolution
``Point`` whose ``feat`` is the concatenated multi-level feature.
"""

import torch

from pimm.models.utils.structure import Point

__all__ = ["hierarchical_feature_sampling", "normalize_feature_sampling"]


def _collect_levels(point):
    """Return hierarchy levels finest-first without consuming the chain.

    ``levels[0]`` is the finest (point-resolution) ``Point`` and ``levels[-1]``
    the coarsest. The chain is only read, so the caller's ``point`` is left
    intact.
    """
    chain = [point]
    cur = point
    while "pooling_parent" in cur.keys():
        cur = cur["pooling_parent"]
        chain.append(cur)
    return list(reversed(chain))


def _composed_inverses(levels):
    """Map finest-point indices to each level's voxel index.

    ``levels[k]`` (coarser) stores ``pooling_inverse`` mapping its parent
    ``levels[k - 1]`` (finer) onto itself. Composing these gives, for every
    finest point, the row index of the voxel it pooled into at each level.
    """
    inv = torch.arange(levels[0].feat.shape[0], device=levels[0].feat.device)
    inverses = [inv]
    for k in range(1, len(levels)):
        pinv = levels[k]["pooling_inverse"]
        inv = pinv[inv]
        inverses.append(inv)
    return inverses


def _recover_stride(fine_grid_coord, level_grid_coord, inverse):
    """Recover the per-axis cumulative grid stride between two levels.

    Each ``GridPooling`` does ``grid_coord //= stride``, so a level's grid
    coordinate is ``floor(fine_grid_coord / cum_stride)`` with ``cum_stride``
    the product of strides up to that level (exact for positive-integer
    division). We solve for ``cum_stride`` per axis as the smallest divisor that
    reproduces the level's voxel index, searching powers of two first (the usual
    case) and falling back to a full integer scan.
    """
    target = level_grid_coord[inverse]
    strides = []
    for axis in range(fine_grid_coord.shape[1]):
        g = fine_grid_coord[:, axis]
        t = target[:, axis]
        gmax = int(g.max().item())
        found = None
        s = 1
        while s <= gmax + 1:
            if torch.equal(torch.div(g, s, rounding_mode="floor"), t):
                found = s
                break
            s *= 2
        if found is None:
            for cand in range(1, gmax + 2):
                if torch.equal(torch.div(g, cand, rounding_mode="floor"), t):
                    found = cand
                    break
        strides.append(found if found is not None else 1)
    return torch.tensor(strides, device=fine_grid_coord.device, dtype=torch.long)


def _build_voxel_lookup(grid_coord, batch):
    """Build a sorted (batch, voxel) key table for O(log N) neighbour queries."""
    gc = grid_coord.long()
    b = batch.long()
    mn = gc.min(0).values
    gc0 = gc - mn
    dims = gc0.max(0).values + 1
    key = ((b * dims[0] + gc0[:, 0]) * dims[1] + gc0[:, 1]) * dims[2] + gc0[:, 2]
    order = torch.argsort(key)
    return key[order], order, mn, dims


def _query_voxel(lookup, query_coord, query_batch):
    """Look up which queried (batch, voxel) cells exist in the table.

    Returns a boolean ``found`` mask and, for the found entries, the row index
    into the original level features.
    """
    keys, order, mn, dims = lookup
    qc = query_coord.long() - mn
    valid = (qc >= 0).all(1) & (qc < dims.view(1, 3)).all(1)
    qcc = torch.minimum(
        torch.maximum(qc, torch.zeros_like(dims).view(1, 3)), (dims - 1).view(1, 3)
    )
    qkey = ((query_batch.long() * dims[0] + qcc[:, 0]) * dims[1] + qcc[:, 1]) * dims[
        2
    ] + qcc[:, 2]
    pos = torch.searchsorted(keys, qkey).clamp(max=keys.numel() - 1)
    found = valid & (keys[pos] == qkey)
    idx = order[pos]
    return found, idx


def _trilinear_sample(fine_grid_coord, fine_batch, level, inverse):
    """Sparse trilinear interpolation of a level's features at the finest points.

    Each finest point is placed in the level's voxel-centre coordinate frame and
    interpolated from the (up to) eight surrounding occupied voxel centres,
    renormalising by the total present weight. The voxel the point pooled into
    is always one of the eight corners, so the result is always well defined.
    """
    stride = _recover_stride(fine_grid_coord, level["grid_coord"], inverse)
    feat = level["feat"]

    # Continuous position in the level's voxel-centre frame.
    t = (fine_grid_coord.float() + 0.5) / stride.float() - 0.5
    base = torch.floor(t)
    frac = t - base
    base = base.long()

    lookup = _build_voxel_lookup(level["grid_coord"], level["batch"])
    out = torch.zeros(
        (fine_grid_coord.shape[0], feat.shape[1]), device=feat.device, dtype=feat.dtype
    )
    wsum = torch.zeros(
        (fine_grid_coord.shape[0], 1), device=feat.device, dtype=feat.dtype
    )
    for cx in (0, 1):
        wx = frac[:, 0] if cx else 1.0 - frac[:, 0]
        for cy in (0, 1):
            wy = frac[:, 1] if cy else 1.0 - frac[:, 1]
            for cz in (0, 1):
                wz = frac[:, 2] if cz else 1.0 - frac[:, 2]
                corner = base + torch.tensor([cx, cy, cz], device=base.device)
                found, idx = _query_voxel(lookup, corner, fine_batch)
                w = (wx * wy * wz).unsqueeze(1) * found.unsqueeze(1).to(feat.dtype)
                out = out + w * feat[idx]
                wsum = wsum + w
    return out / wsum.clamp_min(1e-8)


def _resolve_levels(requested, num_levels):
    """Normalise the requested level selection to a sorted, valid index list."""
    if requested is None:
        return list(range(num_levels))
    levels = sorted({int(l) for l in requested if 0 <= int(l) < num_levels})
    if not levels:
        raise ValueError(
            f"feature_sampling selected no valid levels from {requested!r}; "
            f"backbone exposes {num_levels} levels (0..{num_levels - 1})"
        )
    return levels


def hierarchical_feature_sampling(point, method="trilinear", levels=None):
    """MSM multi-resolution feature sampling over the pooling hierarchy.

    Args:
        point: Coarsest ``Point`` returned by a PTv3/LitePT backbone, still
            carrying the ``pooling_parent`` / ``pooling_inverse`` chain.
        method: ``"trilinear"`` (sparse 8-neighbour interpolation) or
            ``"nearest"`` (voxel the point pooled into; equals the default
            upcast when all levels are selected).
        levels: Iterable of level indices to include (``0`` = finest), or
            ``None`` for every level.

    Returns:
        The finest-resolution ``Point`` with ``feat`` set to the concatenation
        of the sampled per-level features (in increasing-level order). All other
        finest-level fields (``coord``, ``grid_coord``, ``offset``, ``batch``,
        ...) are preserved; the pooling chain is detached so downstream code does
        not re-walk it.
    """
    hierarchy = _collect_levels(point)
    sel = _resolve_levels(levels, len(hierarchy))
    inverses = _composed_inverses(hierarchy)

    finest = hierarchy[0]
    fine_grid_coord = finest["grid_coord"]
    fine_batch = finest["batch"]

    feats = []
    for k in sel:
        if k == 0:
            # Finest level: the query points *are* the voxels; both methods are
            # exact, so sample directly with no interpolation.
            feats.append(hierarchy[0]["feat"])
        elif method == "nearest":
            feats.append(hierarchy[k]["feat"][inverses[k]])
        elif method == "trilinear":
            feats.append(
                _trilinear_sample(
                    fine_grid_coord, fine_batch, hierarchy[k], inverses[k]
                )
            )
        else:
            raise ValueError(
                f"Unknown feature sampling method {method!r}; "
                "expected 'trilinear' or 'nearest'"
            )

    out = Point(finest)
    out.pop("pooling_parent", None)
    out.pop("pooling_inverse", None)
    out["feat"] = torch.cat(feats, dim=-1)
    return out


def normalize_feature_sampling(feature_sampling):
    """Normalise the ``feature_sampling`` kwarg to a ``(method, levels)`` config."""
    if feature_sampling is True or (
        isinstance(feature_sampling, str) and feature_sampling.lower() == "msm"
    ):
        return dict(method="trilinear", levels=None)
    if isinstance(feature_sampling, str):
        return dict(method=feature_sampling, levels=None)
    if isinstance(feature_sampling, dict):
        cfg = dict(method="trilinear", levels=None)
        cfg.update(feature_sampling)
        return cfg
    raise TypeError(
        "feature_sampling must be None, a method string ('trilinear'/'nearest'/"
        "'msm'), or a dict like {'method': ..., 'levels': ...}; "
        f"got {type(feature_sampling).__name__}"
    )
