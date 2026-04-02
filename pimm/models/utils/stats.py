"""
Encoder feature statistics for monitoring representation health.
"""

import torch
import torch.nn.functional as F

from pimm.models.utils.structure import Point


@torch.no_grad()
def compute_enc_feat_stats(encoder_output: Point, max_subsample: int = 5000) -> dict:
    """Compute encoder feature statistics for collapse detection and
    representation quality monitoring.

    Args:
        encoder_output: Point with .feat attribute (N, C)
        max_subsample: max points for covariance eigendecomposition

    Returns:
        dict of scalar stats with "enc/" prefix keys
    """
    enc_feat = encoder_output.feat.detach()
    device = enc_feat.device
    stats = {}

    # Global feature statistics
    stats["enc/feat_std"] = enc_feat.std().item()
    stats["enc/feat_mean"] = enc_feat.mean().item()
    stats["enc/feat_norm_mean"] = enc_feat.norm(dim=-1).mean().item()

    # Per-channel variance (collapse detection)
    channel_var = enc_feat.var(dim=0)
    stats["enc/channel_var_min"] = channel_var.min().item()
    stats["enc/channel_var_mean"] = channel_var.mean().item()
    stats["enc/channel_var_max"] = channel_var.max().item()

    # Feature rank approximation (effective dimensionality)
    feat_centered = enc_feat - enc_feat.mean(dim=0, keepdim=True)
    if feat_centered.shape[0] > max_subsample:
        idx = torch.randperm(feat_centered.shape[0], device=device)[:max_subsample]
        feat_centered = feat_centered[idx]

    # Covariance eigenvalues for effective rank
    cov = (feat_centered.T @ feat_centered) / feat_centered.shape[0]
    eigvals = torch.linalg.eigvalsh(cov.float())
    eigvals = eigvals.clamp(min=0)
    eigvals_norm = eigvals / (eigvals.sum() + 1e-10)
    # Effective rank = exp(entropy of normalized eigenvalues)
    entropy = -(eigvals_norm * (eigvals_norm + 1e-10).log()).sum()
    effective_rank = entropy.exp().item()
    stats["enc/effective_rank"] = effective_rank
    stats["enc/effective_rank_ratio"] = effective_rank / enc_feat.shape[1]

    # Top eigenvalue ratio (dominance = potential collapse)
    stats["enc/top_eigval_ratio"] = (
        eigvals[-1] / (eigvals.sum() + 1e-10)
    ).item()

    # Cosine similarity between random pairs (uniformity measure)
    if enc_feat.shape[0] > 100:
        idx1 = torch.randperm(enc_feat.shape[0], device=device)[:100]
        idx2 = torch.randperm(enc_feat.shape[0], device=device)[:100]
        feat1 = F.normalize(enc_feat[idx1], dim=-1)
        feat2 = F.normalize(enc_feat[idx2], dim=-1)
        cos_sim = (feat1 * feat2).sum(dim=-1).mean()
        stats["enc/random_pair_cos_sim"] = cos_sim.item()

    return stats
