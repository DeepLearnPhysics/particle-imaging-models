"""
Anchor mining utilities for LArTPC point clouds (pure SSL).

This module provides:
 - kNN graph + local PCA curvature
 - Endpoint detection with energy continuity (for shower-like)
 - Branch anchors for tracks and showers
 - Bragg peak detection near track endpoints
 - LED suppression via connected components on kNN graph

Returned anchors are unit-agnostic and rely only on geometry + energy.

Note: Implementation is vectorized with NumPy/Torch and uses SciPy for
nearest-neighbor queries and connected components. No labels are used.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# Default config as a plain dict (override from training config if needed)
ANCHOR_DEFAULT_CFG: Dict[str, float | int] = dict(
    k=16,  # kNN
    r_vox=6.0,  # relative neighbor radius (voxel-ish units)
    min_cluster_pts=8,  # LED suppression
    endpoint_contig=3,  # energy continuity window at endpoints
    branch_energy_frac=0.10,  # shower branch significance threshold
    branch_min_angle_deg=15.0,
    bragg_window=7,  # window length along estimated geodesic (odd preferred)
    bragg_sigma=2.0,  # std multiplier for prominence
    anchor_radius_scale=1.5,  # used by samplers (not here)
)


def _safe_quantile(x: np.ndarray, q: float, default: float) -> float:
    """Return a quantile with a fallback for empty inputs."""
    if x.size == 0:
        return default
    return float(np.quantile(x, q))


def _build_knn(xyz: np.ndarray, k: int):
    """Return (idx, dist) of kNN (excluding self) using cKDTree.

    Args:
        xyz: (N,3) float32
        k: int

    Returns:
        knn_idx: (N,k) neighbor indices
        knn_dist: (N,k) neighbor distances
        nn1: (N,) distance to 1-NN used for thresholds
    """
    tree = cKDTree(xyz)
    # Query k+1 to include self; cut self
    dist, idx = tree.query(xyz, k=min(k + 1, max(2, xyz.shape[0])), workers=-1)
    if dist.ndim == 1:  # degenerate case when N <= k
        dist = dist[:, None]
        idx = idx[:, None]
    # remove self (first column)
    dist = dist[:, 1: k + 1]
    idx = idx[:, 1: k + 1]
    nn1 = dist[:, 0].copy()
    return idx.astype(np.int64), dist.astype(np.float32), nn1.astype(np.float32)


def _local_pca_curvature(xyz: np.ndarray, knn_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute PCA eigenvalues and curvature per point from kNN.

    Returns:
        eigvals: (N,3) sorted descending
        curv: (N,) lam3 / sum(lam)
    """
    N, K = knn_idx.shape
    # Gather neighbor coords
    neigh = xyz[knn_idx]  # (N,K,3)
    mean = neigh.mean(axis=1, keepdims=True)  # (N,1,3)
    X = neigh - mean  # (N,K,3)
    # Covariance: (3,3) per point = X^T X / K
    # Compute batched covariance using torch for numerical stability
    Xt = torch.from_numpy(X).to(torch.float32)  # (N,K,3)
    cov = torch.matmul(Xt.transpose(1, 2), Xt) / float(K)  # (N,3,3)
    # Eigen-decomposition (symmetric)
    eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending
    eigvals = torch.flip(eigvals, dims=[1])  # descending
    # curvature
    s = torch.clamp(eigvals.sum(dim=1), min=1e-12)
    curv = (eigvals[:, -1] / s).contiguous()  # smallest / sum
    return eigvals.cpu().numpy(), curv.cpu().numpy()


def _degree_after_thinning(knn_dist: np.ndarray) -> np.ndarray:
    """Light radius-thinning: keep neighbors closer than median kNN distance.
    Degree is then the count of such neighbors per node.
    """
    med = np.median(knn_dist, axis=1, keepdims=True)
    return (knn_dist < med).sum(axis=1)


def _connected_components_led(xyz: np.ndarray, knn_idx: np.ndarray, knn_dist: np.ndarray,
                              energy: np.ndarray, min_cluster_pts: int) -> np.ndarray:
    """Label LEDs using connected components on sparsified kNN graph.

    We create edges for neighbor distances below a loose threshold based on
    global 1-NN statistics to avoid dense graphs. Components with size <
    min_cluster_pts and low total energy are LEDs.

    Returns:
        led_mask: (N,) bool
    """
    N, K = knn_idx.shape
    # Construct symmetric sparse graph with edges shorter than 1.5 * median 1NN
    # Use first neighbor distances as a proxy
    nn1 = knn_dist[:, 0]
    thr = _safe_quantile(nn1, 0.5, 0.0) * 1.5 + 1e-8
    sel = knn_dist < thr  # (N,K)
    rows = np.repeat(np.arange(N), K)
    cols = knn_idx.reshape(-1)
    mask = sel.reshape(-1)
    rows = rows[mask]
    cols = cols[mask]
    # Add symmetric edges
    r = np.concatenate([rows, cols])
    c = np.concatenate([cols, rows])
    data = np.ones_like(r, dtype=np.float32)
    G = coo_matrix((data, (r, c)), shape=(N, N))
    num_comp, labels = connected_components(G, directed=False)
    # Compute component sizes and energies
    comp_sizes = np.bincount(labels, minlength=num_comp)
    energy = energy.reshape(-1)
    comp_energy = np.bincount(labels, weights=energy, minlength=num_comp)
    median_e = float(np.median(energy)) if energy.size > 0 else 0.0
    # LED if tiny and low energy
    is_led_comp = (comp_sizes < int(min_cluster_pts)) & (comp_energy < median_e * comp_sizes)
    led_mask = is_led_comp[labels]
    return led_mask


def _endpoint_candidates(deg: np.ndarray, led_mask: np.ndarray) -> np.ndarray:
    """Select degree-one non-LED points as endpoint candidates."""
    return (deg == 1) & (~led_mask)


def _energy_continuity_filter(xyz: np.ndarray,
                              knn_idx: np.ndarray,
                              X_center: np.ndarray,
                              energy: np.ndarray,
                              endpoint_mask: np.ndarray,
                              eigvecs_dir: np.ndarray,
                              endpoint_contig: int) -> np.ndarray:
    """For endpoint candidates in showers, require last `endpoint_contig` neighbors
    along local direction to have above-median local energy.

    Args:
        eigvecs_dir: (N,3) principal directions (unit). If None, skip.
    """
    if eigvecs_dir is None:
        return endpoint_mask
    idx = knn_idx  # (N,K)
    neigh = X_center[idx]  # (N,K,3)
    # Projection of neighbor vectors onto local direction
    dirs = eigvecs_dir[:, None, :]  # (N,1,3)
    proj = ((neigh - X_center[:, None, :]) * dirs).sum(axis=2)  # (N,K)
    # Take neighbors with largest positive projection
    order = np.argsort(proj, axis=1)
    # pick last `endpoint_contig` (largest projections)
    # index energies
    local_e = energy[idx]
    med_local = np.median(local_e, axis=1, keepdims=True)
    last_e = np.take_along_axis(local_e, order[:, -endpoint_contig:], axis=1)
    cont_ok = (last_e >= med_local).all(axis=1)
    # Only enforce for endpoints; others remain unchanged
    out = endpoint_mask & cont_ok
    return out


def _principal_dirs_from_cov(eigvecs: torch.Tensor) -> np.ndarray:
    """Take principal direction (largest eigenvector) from eigvecs.
    Args:
        eigvecs: (N,3,3) from torch.linalg.eigh (ascending), columns are eigenvectors
    Returns:
        (N,3) numpy float32
    """
    # Columns correspond to eigenvectors; take the last column (largest eigval)
    dirs = eigvecs[:, :, -1]
    # Normalize (just in case)
    dirs = torch.nn.functional.normalize(dirs, dim=1)
    return dirs.cpu().numpy().astype(np.float32)


def _track_vs_shower_heuristic(curv: np.ndarray) -> np.ndarray:
    """Heuristic: shower-like if curvature above 60th percentile."""
    th = _safe_quantile(curv, 0.6, 0.0)
    return curv >= th


def compute_anchors(
    xyz: torch.Tensor | np.ndarray,
    energy: torch.Tensor | np.ndarray,
    is_shower_like: Optional[torch.Tensor | np.ndarray] = None,
    cfg: Optional[Dict[str, float | int]] = None,
) -> Dict[str, np.ndarray]:
    """Compute anchors for a single event.

    Args:
        xyz: (N,3)
        energy: (N,) or (N,1)
        is_shower_like: optional per-point boolean; if None, inferred via curvature
        cfg: dictionary of hyper-params; default ANCHOR_DEFAULT_CFG

    Returns:
        dict with keys: "endpoints", "branches_track", "branches_shower", "bragg", "led"
        values are float32 arrays of shape (M,3). Missing keys map to empty arrays.
    """
    if isinstance(xyz, torch.Tensor):
        xyz_np = xyz.detach().cpu().numpy()
    else:
        xyz_np = np.asarray(xyz)
    if isinstance(energy, torch.Tensor):
        energy_np = energy.detach().cpu().numpy().reshape(-1)
    else:
        energy_np = np.asarray(energy).reshape(-1)

    cfg = {**ANCHOR_DEFAULT_CFG, **(cfg or {})}
    k = int(cfg["k"]) if cfg.get("k", None) is not None else 16
    min_cluster_pts = int(cfg.get("min_cluster_pts", 8))
    endpoint_contig = int(cfg.get("endpoint_contig", 3))
    branch_energy_frac = float(cfg.get("branch_energy_frac", 0.10))
    branch_min_angle_deg = float(cfg.get("branch_min_angle_deg", 15.0))
    bragg_window = int(cfg.get("bragg_window", 7))
    bragg_sigma = float(cfg.get("bragg_sigma", 2.0))

    N = xyz_np.shape[0]
    if N == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return {
            "endpoints": empty,
            "branches_track": empty,
            "branches_shower": empty,
            "bragg": empty,
            "led": empty,
        }

    # kNN
    knn_idx, knn_dist, nn1 = _build_knn(xyz_np, k=k)
    # PCA curvature
    eigvals, curv = _local_pca_curvature(xyz_np, knn_idx)

    # LED suppression
    led_mask = _connected_components_led(xyz_np, knn_idx, knn_dist, energy_np, min_cluster_pts)

    # Degree after light thinning
    deg = _degree_after_thinning(knn_dist)

    # Heuristic shower-like if not given: use eigenvalue ratios (scattering)
    if is_shower_like is None:
        # eigvals from PCA are descending; compute scattering lam3/lam1.
        lam1 = eigvals[:, 0].copy()
        lam3 = eigvals[:, 2].copy()
        scattering = lam3 / (lam1 + 1e-12)
        th = np.quantile(scattering, 0.55) if scattering.size else 0.0
        shower_like = scattering >= th
    else:
        if isinstance(is_shower_like, torch.Tensor):
            shower_like = is_shower_like.detach().cpu().numpy().astype(bool)
        else:
            shower_like = np.asarray(is_shower_like).astype(bool)

    # Endpoint candidates
    endpoint_mask = _endpoint_candidates(deg, led_mask)

    # principal directions (largest eigenvector)
    # Recompute eigvecs from covariance (ascending order eigenvalues)
    K = knn_idx.shape[1]
    neigh = xyz_np[knn_idx]
    X = neigh - neigh.mean(axis=1, keepdims=True)
    Xt = torch.from_numpy(X.astype(np.float32))
    cov = torch.matmul(Xt.transpose(1, 2), Xt) / float(K)
    _, eigvecs_t = torch.linalg.eigh(cov)
    principal_dir = _principal_dirs_from_cov(eigvecs_t)  # (N,3)

    # Energy continuity at endpoints for shower-like points only
    endpoint_mask_refined = endpoint_mask.copy()
    if endpoint_mask.any():
        cont_ok = _energy_continuity_filter(
            xyz_np,
            knn_idx,
            xyz_np,
            energy_np,
            endpoint_mask,
            principal_dir,
            endpoint_contig,
        )
        # apply only to shower-like
        endpoint_mask_refined = np.where(shower_like, cont_ok, endpoint_mask)

    # Track branch anchors: deg>=3 OR high curvature (top 5%)
    curv_th = _safe_quantile(curv, 0.95, float(curv.max() if curv.size else 0.0))
    branch_track_mask = ((deg >= 3) | (curv >= curv_th)) & (~led_mask)

    # Shower branch anchors using branch significance
    # For each point, evaluate neighbors' deviation from the principal direction
    dirs = principal_dir  # (N,3)
    neigh_vec = neigh - xyz_np[:, None, :]  # (N,K,3)
    denom = np.linalg.norm(neigh_vec, axis=2) * np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    cosang = (neigh_vec * dirs[:, None, :]).sum(axis=2) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    sing = np.sqrt(1.0 - cosang ** 2)  # sin(theta)
    # energy fractions
    local_e = energy_np[knn_idx]  # (N,K)
    parent_e = (local_e.sum(axis=1, keepdims=True) + energy_np[:, None] + 1e-8)
    frac = local_e / parent_e
    score = frac * sing
    ang_min_sin = np.sin(np.deg2rad(branch_min_angle_deg))
    shower_signif = (score >= branch_energy_frac) & (sing >= ang_min_sin)
    branch_shower_mask = (shower_signif.any(axis=1)) & shower_like & (~led_mask)

    # Bragg peaks near track endpoints: significant local energy prominence
    bragg_mask = np.zeros(N, dtype=bool)
    if endpoint_mask_refined.any():
        L = max(3, int(bragg_window))
        proj = ((neigh - xyz_np[:, None, :]) * dirs[:, None, :]).sum(axis=2)  # (N,K)
        order = np.argsort(proj, axis=1)
        forward_idx = order[:, -L:]
        forward_e = np.take_along_axis(local_e, forward_idx, axis=1)
        mu = forward_e.mean(axis=1)
        sd = forward_e.std(axis=1) + 1e-8
        bragg_mask = (energy_np >= mu + bragg_sigma * sd) & endpoint_mask_refined & (~led_mask)

    # Collate coordinates
    def pick(mask: np.ndarray) -> np.ndarray:
        if mask.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        pts = xyz_np[mask]
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        return pts.astype(np.float32)

    anchors = dict(
        endpoints=pick(endpoint_mask_refined),
        branches_track=pick(branch_track_mask),
        branches_shower=pick(branch_shower_mask),
        bragg=pick(bragg_mask),
        led=pick(led_mask),
    )
    return anchors


__all__ = ["compute_anchors", "ANCHOR_DEFAULT_CFG"]
