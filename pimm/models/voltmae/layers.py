"""Building blocks for Volt-MAE.

Contains the Volt backbone transformer primitives (RoPE, RoPE_Attention, Block)
copied from libs/Volt/pointcept/models/volt/volt_base.py — kept verbatim to
preserve checkpoint compatibility with vanilla Volt downstream configs — plus
Volt-MAE-specific utilities: point<->token alignment, target construction,
random patch-level masking, and the reconstruction head.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import LayerScale

from pimm.models.losses.chamfer import chamfer_distance

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except ImportError:
    flash_attn_varlen_qkvpacked_func = None


# ---------------------------------------------------------------------------
# Volt backbone primitives (verbatim from libs/Volt/pointcept/models/volt/volt_base.py)
# ---------------------------------------------------------------------------
class RoPE(nn.Module):
    def __init__(
        self,
        theta: float = 100.0,
        freq_split: tuple = (12, 12, 8),
        max_grid_size: tuple = (1024, 1024, 512),
    ) -> None:
        super().__init__()
        freqs_x = 1.0 / theta ** torch.linspace(0, 1, freq_split[0])
        freqs_y = 1.0 / theta ** torch.linspace(0, 1, freq_split[1])
        freqs_z = 1.0 / theta ** torch.linspace(0, 1, freq_split[2])

        self.register_buffer(
            "cis_cache_x", self._precompute(freqs_x, max_grid_size[0]), persistent=False
        )
        self.register_buffer(
            "cis_cache_y", self._precompute(freqs_y, max_grid_size[1]), persistent=False
        )
        self.register_buffer(
            "cis_cache_z", self._precompute(freqs_z, max_grid_size[2]), persistent=False
        )

    def _precompute(self, freqs, max_pos):
        freqs_pos = torch.outer(torch.arange(max_pos).float(), freqs)
        return torch.polar(torch.ones_like(freqs_pos), freqs_pos)

    def compute_axial_cis_efficient(self, indices):
        cis_x = self.cis_cache_x[indices[:, 0]]
        cis_y = self.cis_cache_y[indices[:, 1]]
        cis_z = self.cis_cache_z[indices[:, 2]]
        return torch.cat([cis_x, cis_y, cis_z], dim=-1).unsqueeze(0)


class RoPE_Attention(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.h_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(self.h_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.h_dim) if qk_norm else nn.Identity()

    @staticmethod
    def apply_rotary_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        q_out = torch.view_as_real(q_ * freqs_cis).flatten(2)
        k_out = torch.view_as_real(k_ * freqs_cis).flatten(2)

        return q_out.type_as(q), k_out.type_as(k)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        N, C = x.shape
        qkv = self.qkv(x).view(N, 3, self.num_heads, self.h_dim)
        qkv = qkv.permute(1, 2, 0, 3)
        q, k, v = qkv.unbind(dim=0)

        q, k = self.q_norm(q).to(q.dtype), self.k_norm(k).to(k.dtype)
        q, k = self.apply_rotary_emb(q, k, freqs_cis)
        qkv = torch.stack([q, k, v], dim=0).permute(2, 0, 1, 3)

        qkv_dtype = qkv.dtype
        x = flash_attn_varlen_qkvpacked_func(
            qkv.half(),
            cu_seqlens,
            max_seqlen=max_seqlen,
        )

        x = x.reshape(-1, C).to(qkv_dtype)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        init_values: float | None = None,
        qk_norm: bool = False,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPE_Attention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), freqs_cis, cu_seq_lens, max_seqlen))
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Volt-MAE specific utilities
# ---------------------------------------------------------------------------
def _pack_indices(batch: torch.Tensor, ijk: torch.Tensor, shape_hash: int) -> torch.Tensor:
    """Pack (batch, i, j, k) into a single int64 hash per row."""
    b = batch.to(torch.int64)
    i = ijk[..., 0].to(torch.int64)
    j = ijk[..., 1].to(torch.int64)
    k = ijk[..., 2].to(torch.int64)
    return ((b * shape_hash + i) * shape_hash + j) * shape_hash + k


def sort_tokens_by_batch(
    token_features: torch.Tensor,
    token_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort sparse-conv tokens by (batch, i, j, k).

    spconv does not guarantee batch-contiguous output indices, while
    flash_attn_varlen_qkvpacked_func requires tokens for each sequence to be
    laid out contiguously according to cu_seqlens.
    """
    if token_indices.numel() == 0:
        return token_features, token_indices

    shape_hash = int(token_indices[:, 1:].max().item()) + 2
    sort_key = _pack_indices(token_indices[:, 0], token_indices[:, 1:], shape_hash)
    order = torch.argsort(sort_key)
    return token_features.index_select(0, order), token_indices.index_select(0, order)


def build_point_to_token(
    grid_coord: torch.Tensor,
    batch: torch.Tensor,
    token_indices: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Map each input point to its parent token id.

    Args:
        grid_coord: (N, 3) int, point voxel indices at the fine resolution.
        batch: (N,) int, batch index per point.
        token_indices: (T, 4) int, (batch, i, j, k) from x.indices after the tokenizer.
        stride: tokenizer stride (e.g. 5). With SparseConv3d(padding=0) the parent
            token index of a fine voxel `p` is `p // stride`.

    Returns:
        point_to_token: (N,) int64, index into `token_indices` for each point.
    """
    parent = grid_coord // stride  # (N, 3)
    shape_hash = int(max(
        parent.max().item() if parent.numel() else 0,
        token_indices[:, 1:].max().item() if token_indices.numel() else 0,
    )) + 2

    point_hash = _pack_indices(batch, parent, shape_hash)
    token_hash = _pack_indices(token_indices[:, 0], token_indices[:, 1:], shape_hash)

    sorted_token_hash, token_order = token_hash.sort()
    pos = torch.searchsorted(sorted_token_hash, point_hash)
    # Guard against out-of-range before gather
    pos_clamped = pos.clamp(max=sorted_token_hash.numel() - 1)
    matched_hash = sorted_token_hash[pos_clamped]
    ok = matched_hash == point_hash
    if not ok.all():
        missing = (~ok).sum().item()
        raise RuntimeError(
            f"build_point_to_token: {missing}/{point_hash.numel()} points could not "
            f"be matched to any token. This indicates the tokenizer did not produce "
            f"an output voxel for every input parent — check stride/padding."
        )
    return token_order[pos_clamped]


def build_targets(
    point_to_token: torch.Tensor,
    grid_coord: torch.Tensor,
    token_indices: torch.Tensor,
    energy: torch.Tensor,
    stride: int,
    num_tokens: int,
):
    """Build per-token dense sub-voxel energy + occupancy targets.

    After GridSample at the fine resolution, at most one point occupies each
    sub-voxel. Occupancy is tracked by a dedicated scatter of ones so we
    don't have to infer it from energy (log-transformed energies can be
    negative, so `energy > 0` is unreliable).

    Args:
        point_to_token: (N,) int64 token id per point.
        grid_coord: (N, 3) fine-resolution voxel indices.
        token_indices: (T, 4) post-tokenizer indices.
        energy: (N, 1) or (N,) energy per point (may already be log-transformed).
        stride: tokenizer stride.
        num_tokens: T.

    Returns:
        energy_target: (T, stride**3) float, summed energies per sub-voxel.
        occ_target: (T, stride**3) float in {0, 1}, occupancy mask.
    """
    s3 = stride ** 3
    parent_ijk = token_indices[point_to_token, 1:]  # (N, 3)
    sub = grid_coord - parent_ijk * stride  # (N, 3) in [0, stride)
    sub_idx = (sub[:, 0] * stride + sub[:, 1]) * stride + sub[:, 2]  # (N,)

    eng = energy.squeeze(-1) if energy.dim() == 2 else energy
    eng = eng.to(torch.float32)

    flat_idx = point_to_token * s3 + sub_idx.to(point_to_token.dtype)
    energy_target = torch.zeros(num_tokens * s3, dtype=torch.float32, device=eng.device)
    energy_target.scatter_add_(0, flat_idx, eng)

    occ_target = torch.zeros(num_tokens * s3, dtype=torch.float32, device=eng.device)
    occ_target.scatter_add_(0, flat_idx, torch.ones_like(eng))
    occ_target.clamp_(max=1.0)

    return energy_target.view(num_tokens, s3), occ_target.view(num_tokens, s3)


def build_pointset_targets(
    point_to_token: torch.Tensor,
    grid_coord: torch.Tensor,
    token_indices: torch.Tensor,
    charge: torch.Tensor,
    stride: int,
    num_tokens: int,
    max_points_per_token: int | None = None,
    overflow_policy: str = "first",
) -> dict[str, torch.Tensor]:
    """Build ragged per-token local point-set targets.

    Local coordinates are derived from fine-grid voxel centers inside each
    sparse-conv token patch:

        local_xyz = 2 * ((grid_coord - patch_origin + 0.5) / stride) - 1

    For stride=5 this maps sub-voxel centers to {-0.8, -0.4, 0, 0.4, 0.8}.
    Empty tokens are represented by equal consecutive offsets.

    Args:
        point_to_token: (N,) int64 token id per point.
        grid_coord: (N, 3) fine-resolution voxel indices.
        token_indices: (T, 4) post-tokenizer indices.
        charge: (N, 1) or (N,) charge/energy per point.
        stride: tokenizer stride.
        num_tokens: T.
        max_points_per_token: optional K cap. If None, all points are kept.
        overflow_policy: "first" keeps the deterministic sub-voxel order;
            "error" raises if any token has more than K points.

    Returns:
        Dictionary with flattened ragged targets and per-token metadata:
        `target_xyz_local`, `target_charge`, `target_patch_index`,
        `target_offsets`, `target_counts`, `original_counts`, and overflow
        diagnostics.
    """
    if overflow_policy not in {"first", "error"}:
        raise ValueError(f"Unsupported overflow_policy: {overflow_policy!r}")

    device = grid_coord.device
    point_to_token = point_to_token.to(torch.long)
    grid_coord = grid_coord.to(torch.long)
    token_indices = token_indices.to(torch.long)

    chg = charge.squeeze(-1) if charge.dim() == 2 and charge.shape[-1] == 1 else charge
    if chg.dim() != 1:
        raise ValueError(
            f"charge must have shape (N,) or (N, 1), got {tuple(charge.shape)}"
        )
    chg = chg.to(torch.float32)

    if point_to_token.numel() == 0:
        counts = torch.zeros(num_tokens, dtype=torch.long, device=device)
        offsets = torch.zeros(num_tokens + 1, dtype=torch.long, device=device)
        zero = chg.new_zeros(())
        return {
            "target_xyz_local": chg.new_zeros((0, 3)),
            "target_charge": chg.new_zeros((0, 1)),
            "target_patch_index": torch.empty(0, dtype=torch.long, device=device),
            "target_offsets": offsets,
            "target_counts": counts,
            "original_counts": counts,
            "overflow_counts": counts,
            "overflow_mask": torch.zeros(num_tokens, dtype=torch.bool, device=device),
            "overflow_patch_fraction": zero,
            "overflow_point_fraction": zero,
            "overflow_charge_fraction": zero,
            "num_target_points": torch.zeros((), dtype=torch.long, device=device),
            "num_original_points": torch.zeros((), dtype=torch.long, device=device),
        }

    parent_ijk = token_indices[point_to_token, 1:]
    sub = grid_coord - parent_ijk * stride
    if not bool(((sub >= 0) & (sub < stride)).all()):
        bad = (~((sub >= 0) & (sub < stride)).all(dim=1)).sum().item()
        raise RuntimeError(
            f"build_pointset_targets: {bad}/{sub.shape[0]} points have local "
            f"sub-voxel coordinates outside [0, {stride})."
        )

    local_xyz = 2.0 * ((sub.to(torch.float32) + 0.5) / float(stride)) - 1.0
    sub_idx = (sub[:, 0] * stride + sub[:, 1]) * stride + sub[:, 2]
    sort_key = point_to_token * (stride ** 3) + sub_idx.to(point_to_token.dtype)
    order = torch.argsort(sort_key, stable=True)

    token_sorted = point_to_token.index_select(0, order)
    xyz_sorted = local_xyz.index_select(0, order)
    charge_sorted = chg.index_select(0, order)

    original_counts = torch.bincount(point_to_token, minlength=num_tokens).to(torch.long)
    original_offsets = torch.zeros(num_tokens + 1, dtype=torch.long, device=device)
    original_offsets[1:] = torch.cumsum(original_counts, dim=0)

    if max_points_per_token is None:
        keep_mask = torch.ones_like(token_sorted, dtype=torch.bool)
    else:
        max_points_per_token = int(max_points_per_token)
        if max_points_per_token <= 0:
            raise ValueError("max_points_per_token must be positive when set")
        overflow_mask = original_counts > max_points_per_token
        if overflow_policy == "error" and bool(overflow_mask.any()):
            max_count = int(original_counts.max().item())
            raise RuntimeError(
                f"build_pointset_targets: token has {max_count} points, "
                f"exceeding max_points_per_token={max_points_per_token}"
            )
        position_in_token = (
            torch.arange(token_sorted.numel(), device=device, dtype=torch.long)
            - original_offsets.index_select(0, token_sorted)
        )
        keep_mask = position_in_token < max_points_per_token

    target_patch_index = token_sorted[keep_mask]
    target_xyz_local = xyz_sorted[keep_mask]
    target_charge = charge_sorted[keep_mask].unsqueeze(-1)

    target_counts = torch.bincount(target_patch_index, minlength=num_tokens).to(torch.long)
    target_offsets = torch.zeros(num_tokens + 1, dtype=torch.long, device=device)
    target_offsets[1:] = torch.cumsum(target_counts, dim=0)

    overflow_counts = original_counts - target_counts
    overflow_mask = overflow_counts > 0
    occupied_mask = original_counts > 0
    dropped_mask = ~keep_mask

    num_original = torch.tensor(
        int(point_to_token.numel()), dtype=torch.long, device=device
    )
    num_target = torch.tensor(
        int(target_patch_index.numel()), dtype=torch.long, device=device
    )
    denom_occ = occupied_mask.sum().clamp(min=1).to(torch.float32)
    denom_points = num_original.clamp(min=1).to(torch.float32)
    total_charge_abs = charge_sorted.abs().sum().clamp(min=1.0e-12)
    dropped_charge_abs = charge_sorted[dropped_mask].abs().sum()

    return {
        "target_xyz_local": target_xyz_local,
        "target_charge": target_charge,
        "target_patch_index": target_patch_index,
        "target_offsets": target_offsets,
        "target_counts": target_counts,
        "original_counts": original_counts,
        "overflow_counts": overflow_counts,
        "overflow_mask": overflow_mask,
        "overflow_patch_fraction": overflow_mask.sum().to(torch.float32) / denom_occ,
        "overflow_point_fraction": overflow_counts.sum().to(torch.float32) / denom_points,
        "overflow_charge_fraction": dropped_charge_abs / total_charge_abs,
        "num_target_points": num_target,
        "num_original_points": num_original,
    }


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 0.0,
    alpha: float | None = None,
    reduction: str = "none",
) -> torch.Tensor:
    """Numerically-stable focal BCE.

    gamma=0 recovers plain BCE. `alpha` (optional) reweights classes.
    Accepts soft targets in [0, 1] so future label smoothing works unchanged.
    """
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if gamma > 0.0:
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        ce = (1.0 - p_t).pow(gamma) * ce
    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        ce = alpha_t * ce
    if reduction == "mean":
        return ce.mean()
    if reduction == "sum":
        return ce.sum()
    return ce


def occ_supervision_mask(
    occ_target: torch.Tensor,
    stride: int,
    dilate: int = 0,
    empty_beta: float = 1.0,
    generator: torch.Generator | None = None,
):
    """Build the per-sub-voxel supervision mask for occupancy.

    Args:
        occ_target: (M, stride**3) {0, 1} — raw per-sub-voxel occupancy for masked tokens.
        stride: tokenizer stride (patch side length).
        dilate: radius of the positive dilation; 0 disables.
        empty_beta: fraction of empties to supervise (1.0 = keep all, 0.0 = drop all).
        generator: optional torch.Generator for reproducible negative sampling.

    Returns:
        sup_mask    (M, s**3) bool  — entries to include in the occupancy loss
        sup_targ    (M, s**3) float — 1.0 on positives+border, 0.0 on sampled empties
        pos_mask    (M, s**3) bool  — raw positives (occ_target == 1)
        border_mask (M, s**3) bool  — dilated shell (not counting the positives themselves)
        neg_mask    (M, s**3) bool  — sampled empties
    """
    M, s3 = occ_target.shape
    assert s3 == stride ** 3, f"occ_target shape {occ_target.shape} mismatched to stride {stride}"

    pos_mask = occ_target > 0  # (M, s³)

    if dilate > 0:
        k = 2 * dilate + 1
        occ_grid = pos_mask.view(M, 1, stride, stride, stride).float()
        dilated = F.max_pool3d(occ_grid, kernel_size=k, stride=1, padding=dilate) > 0
        dilated = dilated.view(M, s3)
        border_mask = dilated & ~pos_mask
    else:
        border_mask = torch.zeros_like(pos_mask)

    positive_region = pos_mask | border_mask  # (M, s³)

    if empty_beta >= 1.0:
        neg_mask = ~positive_region
    elif empty_beta <= 0.0:
        neg_mask = torch.zeros_like(pos_mask)
    else:
        rand = torch.rand(
            occ_target.shape,
            device=occ_target.device,
            generator=generator,
            dtype=torch.float32,
        )
        neg_mask = (~positive_region) & (rand < empty_beta)

    sup_mask = positive_region | neg_mask
    sup_targ = positive_region.float()
    return sup_mask, sup_targ, pos_mask, border_mask, neg_mask


@torch.no_grad()
def reconstruction_diagnostics(
    occ_logits: torch.Tensor,
    pos_mask: torch.Tensor,
    border_mask: torch.Tensor,
    thresholds: tuple[float, ...] = (0.7,),
    prefix: str = "recon",
) -> dict[str, torch.Tensor]:
    """Scalar diagnostics for masked-token occupancy reconstruction."""
    prob = torch.sigmoid(occ_logits.detach())
    true_target = pos_mask
    dilated_target = pos_mask | border_mask
    neg_mask = ~dilated_target

    zero = prob.new_zeros(())

    def masked_mean(mask):
        return prob[mask].mean() if mask.any() else zero

    def prf(pred, target):
        tp = (pred & target).sum().float()
        fp = (pred & ~target).sum().float()
        fn = (~pred & target).sum().float()
        precision = tp / (tp + fp + 1.0e-10)
        recall = tp / (tp + fn + 1.0e-10)
        f1 = 2.0 * precision * recall / (precision + recall + 1.0e-10)
        return precision, recall, f1

    def topk_recall():
        counts = true_target.sum(dim=1).long()
        valid = counts > 0
        if not valid.any():
            return zero
        order = prob.argsort(dim=1, descending=True)
        ranked_hits = true_target.gather(1, order).float()
        cumulative_hits = ranked_hits.cumsum(dim=1)
        gather_idx = (counts.clamp(min=1) - 1).unsqueeze(1)
        per_token = cumulative_hits.gather(1, gather_idx).squeeze(1)
        per_token = per_token / counts.clamp(min=1).float()
        return per_token[valid].mean()

    metrics = {
        f"{prefix}_prob_true": masked_mean(true_target),
        f"{prefix}_prob_border": masked_mean(border_mask),
        f"{prefix}_prob_negative": masked_mean(neg_mask),
        f"{prefix}_topk_true_occ_recall": topk_recall(),
    }

    for threshold in thresholds:
        tag = f"t{int(round(float(threshold) * 100)):02d}"
        pred = prob >= float(threshold)
        true_precision, true_recall, true_f1 = prf(pred, true_target)
        dil_precision, dil_recall, dil_f1 = prf(pred, dilated_target)
        metrics.update({
            f"{prefix}_true_precision_{tag}": true_precision,
            f"{prefix}_true_recall_{tag}": true_recall,
            f"{prefix}_true_f1_{tag}": true_f1,
            f"{prefix}_dilated_precision_{tag}": dil_precision,
            f"{prefix}_dilated_recall_{tag}": dil_recall,
            f"{prefix}_dilated_f1_{tag}": dil_f1,
            f"{prefix}_pred_per_masked_{tag}": pred.sum(dim=1).float().mean(),
        })

    return metrics


def random_token_mask(
    token_batch_ids: torch.Tensor,
    mask_ratio: float,
    generator: torch.Generator | None = None,
):
    """Per-batch random patch-level masking.

    For each event, randomly pick `mask_ratio` fraction of its tokens to mask.
    Returns index tensors plus cu_seqlens tensors (int32) sized for
    `flash_attn_varlen_qkvpacked_func`.
    """
    device = token_batch_ids.device
    T = token_batch_ids.numel()
    rand = torch.rand(T, device=device, generator=generator)

    # Sort by (batch_id, rand). Integer batch id dominates, fractional rand
    # shuffles within each batch.
    batch_counts = torch.bincount(token_batch_ids)
    B = batch_counts.numel()

    sort_key = token_batch_ids.to(torch.float64) + rand.to(torch.float64)
    order = torch.argsort(sort_key)
    # `order` lays out tokens grouped by batch, shuffled within each group.
    batch_offsets = torch.zeros(B + 1, dtype=torch.int64, device=device)
    batch_offsets[1:] = torch.cumsum(batch_counts, dim=0)

    keep_flags = torch.zeros(T, dtype=torch.bool, device=device)
    mask_flags = torch.zeros(T, dtype=torch.bool, device=device)
    kept_counts = torch.zeros(B, dtype=torch.int64, device=device)
    for b in range(B):
        start = batch_offsets[b].item()
        end = batch_offsets[b + 1].item()
        n = end - start
        n_mask = int(round(n * mask_ratio))
        n_mask = min(max(n_mask, 0), n)
        batch_order = order[start:end]
        mask_flags[batch_order[:n_mask]] = True
        keep_flags[batch_order[n_mask:]] = True
        kept_counts[b] = n - n_mask

    ids_kept = torch.nonzero(keep_flags, as_tuple=False).squeeze(1)
    ids_masked = torch.nonzero(mask_flags, as_tuple=False).squeeze(1)

    # Keep ids sorted by original token order so that scatter + flash_attn cu_seqlens
    # stay aligned with token_batch_ids ordering.
    ids_kept, _ = ids_kept.sort()
    ids_masked, _ = ids_masked.sort()

    # cu_seqlens: int32, shape (B+1,), prefix sum of per-batch counts
    cu_kept = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_kept[1:] = torch.cumsum(
        torch.bincount(token_batch_ids[ids_kept], minlength=B).to(torch.int32), dim=0
    )
    cu_full = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_full[1:] = torch.cumsum(batch_counts.to(torch.int32), dim=0)

    max_kept = int((cu_kept[1:] - cu_kept[:-1]).max().item()) if B else 0
    max_full = int(batch_counts.max().item()) if B else 0

    return ids_kept, ids_masked, cu_kept, max_kept, cu_full, max_full


@torch.no_grad()
def sample_empty_patch_candidates(
    occupied_token_indices: torch.Tensor,
    masked_occupied_ids: torch.Tensor,
    ratio: float = 0.25,
    dilation_radius: int = 1,
    neighborhood: str = "26",
    max_per_event: int | None = 512,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample near-event empty token coordinates from occupied-token dilation.

    Args:
        occupied_token_indices: (T_occ, 4) sorted occupied tokens as
            (batch, i, j, k).
        masked_occupied_ids: token ids into ``occupied_token_indices`` selected
            for masked occupied reconstruction.
        ratio: number of empties per event is ceil(ratio * masked count).
        dilation_radius: Chebyshev radius around occupied tokens.
        neighborhood: ``"26"`` uses all nonzero offsets in the local cube.
        max_per_event: cap sampled empties per batch element.
        generator: optional RNG for tests.

    Returns:
        (E, 4) sampled empty token coordinates. They are guaranteed unique and
        absent from the original occupied-token set.
    """
    device = occupied_token_indices.device
    occupied = occupied_token_indices.to(torch.long)
    if occupied.numel() == 0 or float(ratio) <= 0.0:
        return occupied.new_empty((0, 4))

    masked_occupied_ids = masked_occupied_ids.to(device=device, dtype=torch.long).flatten()
    if masked_occupied_ids.numel() == 0:
        return occupied.new_empty((0, 4))
    if bool(((masked_occupied_ids < 0) | (masked_occupied_ids >= occupied.shape[0])).any()):
        raise ValueError("masked_occupied_ids contains an out-of-range occupied token id")

    radius = int(dilation_radius)
    if radius <= 0:
        return occupied.new_empty((0, 4))
    if neighborhood != "26":
        raise ValueError(f"Unsupported empty-candidate neighborhood: {neighborhood!r}")

    axis = torch.arange(-radius, radius + 1, dtype=torch.long, device=device)
    offsets = torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    offsets = offsets[(offsets.abs().sum(dim=1) > 0)]

    neigh_ijk = occupied[:, None, 1:] + offsets[None, :, :]
    neigh_batch = occupied[:, None, 0].expand(-1, offsets.shape[0])
    candidates = torch.cat(
        [neigh_batch.reshape(-1, 1), neigh_ijk.reshape(-1, 3)], dim=1
    )
    in_bounds = (candidates[:, 1:] >= 0).all(dim=1)
    candidates = candidates[in_bounds]
    if candidates.numel() == 0:
        return occupied.new_empty((0, 4))

    def _unique_sorted(rows: torch.Tensor) -> torch.Tensor:
        if rows.numel() == 0:
            return rows.reshape(0, 4)
        shape_hash = int(rows[:, 1:].max().item()) + 2
        key = _pack_indices(rows[:, 0], rows[:, 1:], shape_hash)
        order = torch.argsort(key)
        rows = rows.index_select(0, order)
        key = key.index_select(0, order)
        keep = torch.ones(rows.shape[0], dtype=torch.bool, device=rows.device)
        keep[1:] = key[1:] != key[:-1]
        return rows[keep]

    candidates = _unique_sorted(candidates)
    shape_hash = int(max(candidates[:, 1:].max().item(), occupied[:, 1:].max().item())) + 2
    candidate_hash = _pack_indices(candidates[:, 0], candidates[:, 1:], shape_hash)
    occupied_hash = _pack_indices(occupied[:, 0], occupied[:, 1:], shape_hash)
    sorted_occ_hash, _ = torch.sort(occupied_hash)
    pos = torch.searchsorted(sorted_occ_hash, candidate_hash)
    pos_clamped = pos.clamp(max=max(sorted_occ_hash.numel() - 1, 0))
    is_occupied = (pos < sorted_occ_hash.numel()) & (
        sorted_occ_hash.index_select(0, pos_clamped) == candidate_hash
    )
    candidates = candidates[~is_occupied]
    if candidates.numel() == 0:
        return occupied.new_empty((0, 4))

    selected: list[torch.Tensor] = []
    batch_values = torch.unique(occupied[:, 0], sorted=True)
    masked_batches = occupied.index_select(0, masked_occupied_ids)[:, 0]
    for batch_id in batch_values:
        masked_count = int((masked_batches == batch_id).sum().item())
        if masked_count == 0:
            continue
        n_target = int(math.ceil(float(masked_count) * float(ratio)))
        if max_per_event is not None:
            n_target = min(n_target, int(max_per_event))
        if n_target <= 0:
            continue
        batch_candidates = candidates[candidates[:, 0] == batch_id]
        if batch_candidates.numel() == 0:
            continue
        n_take = min(n_target, batch_candidates.shape[0])
        perm = torch.randperm(
            batch_candidates.shape[0], device=device, generator=generator
        )[:n_take]
        selected.append(batch_candidates.index_select(0, perm))

    if not selected:
        return occupied.new_empty((0, 4))
    return torch.cat(selected, dim=0)


class PointSetPredictionHead(nn.Module):
    """Per-token fixed-capacity point-set head for Gate-2 VoltMAE."""

    def __init__(
        self,
        in_dim: int,
        num_points: int,
        charge_dim: int = 1,
        xyz_range: float = 1.1,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.charge_dim = int(charge_dim)
        self.xyz_range = float(xyz_range)

        out_dim = self.num_points * (3 + self.charge_dim)
        if hidden_dim is None:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        y = y.view(x.shape[0], self.num_points, 3 + self.charge_dim)
        xyz = self.xyz_range * torch.tanh(y[..., :3])
        return torch.cat([xyz, y[..., 3:]], dim=-1)


def pointset_chamfer_loss(
    pred: torch.Tensor,
    target_points: torch.Tensor,
    offsets: torch.Tensor,
    counts: torch.Tensor,
    charge_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """First-M point-set baseline loss for ragged per-token targets.

    Args:
        pred: (T, K, 4) predicted local xyz + charge for predicted tokens.
        target_points: (P, 4) flattened ragged targets grouped by token.
        offsets: (T + 1,) ragged offsets into target_points.
        counts: (T,) target count per predicted token.
        charge_weight: scalar weight for nearest-neighbor charge loss.
    """
    losses: list[torch.Tensor] = []
    xyz_losses: list[torch.Tensor] = []
    q_losses: list[torch.Tensor] = []

    T, K, _ = pred.shape
    for t in range(T):
        m = int(counts[t].item())
        if m == 0:
            continue
        m = min(m, K)

        start = int(offsets[t].item())
        end = start + m
        target_t = target_points[start:end]
        pred_t = pred[t, :m]

        pred_xyz = pred_t[:, :3].float()
        targ_xyz = target_t[:, :3].float()
        pred_q = pred_t[:, 3:4].float()
        targ_q = target_t[:, 3:4].float()

        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            dist = torch.cdist(pred_xyz, targ_xyz, p=1)

        pred_to_targ_dist, pred_to_targ_idx = dist.min(dim=1)
        targ_to_pred_dist, targ_to_pred_idx = dist.min(dim=0)

        xyz_loss = pred_to_targ_dist.mean() + targ_to_pred_dist.mean()
        q_loss_pred = F.l1_loss(
            pred_q,
            targ_q.index_select(0, pred_to_targ_idx),
            reduction="mean",
        )
        q_loss_targ = F.l1_loss(
            pred_q.index_select(0, targ_to_pred_idx),
            targ_q,
            reduction="mean",
        )
        q_loss = 0.5 * (q_loss_pred + q_loss_targ)

        loss_t = xyz_loss + float(charge_weight) * q_loss
        losses.append(loss_t)
        xyz_losses.append(xyz_loss.detach())
        q_losses.append(q_loss.detach())

    if not losses:
        zero = pred.sum() * 0.0
        zero_detached = zero.detach()
        return {
            "loss": zero,
            "loss_pointset_xyz": zero_detached,
            "loss_pointset_charge": zero_detached,
            "num_supervised_patches": pred.new_tensor(0.0),
        }

    return {
        "loss": torch.stack(losses).mean(),
        "loss_pointset_xyz": torch.stack(xyz_losses).mean(),
        "loss_pointset_charge": torch.stack(q_losses).mean(),
        "num_supervised_patches": pred.new_tensor(float(len(losses))),
    }


class SlotPointSetPredictionHead(nn.Module):
    """Unordered fixed-capacity slot head for local point-set prediction."""

    def __init__(
        self,
        in_dim: int,
        num_slots: int = 64,
        slot_dim: int | None = None,
        num_heads: int = 6,
        slot_blocks: int = 1,
        mlp_ratio: float = 2.0,
        xyz_range: float = 1.1,
        obj_init_prob: float = 0.10,
    ) -> None:
        super().__init__()
        slot_dim = int(slot_dim or in_dim)
        if slot_dim % int(num_heads) != 0:
            raise ValueError(
                f"slot_dim={slot_dim} must be divisible by num_heads={num_heads}"
            )
        if not 0.0 < float(obj_init_prob) < 1.0:
            raise ValueError("obj_init_prob must be in (0, 1)")

        self.num_slots = int(num_slots)
        self.slot_dim = slot_dim
        self.xyz_range = float(xyz_range)

        self.token_proj = nn.Linear(in_dim, slot_dim)
        self.slot_embed = nn.Parameter(torch.zeros(self.num_slots, slot_dim))
        self.slot_blocks = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=slot_dim,
                    nhead=int(num_heads),
                    dim_feedforward=int(float(mlp_ratio) * slot_dim),
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(slot_blocks))
            ]
        )
        self.norm = nn.LayerNorm(slot_dim)
        self.xyz_head = nn.Linear(slot_dim, 3)
        self.charge_head = nn.Linear(slot_dim, 1)
        self.obj_head = nn.Linear(slot_dim, 1)
        self.reset_parameters(float(obj_init_prob))

    def reset_parameters(self, obj_init_prob: float) -> None:
        nn.init.normal_(self.slot_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        obj_bias = math.log(obj_init_prob / (1.0 - obj_init_prob))
        nn.init.zeros_(self.obj_head.weight)
        nn.init.constant_(self.obj_head.bias, obj_bias)

    def forward(self, decoded_tokens: torch.Tensor):
        base = self.token_proj(decoded_tokens)[:, None, :]
        slots = base + self.slot_embed[None, :, :]
        slots = self.slot_blocks(slots)
        slots = self.norm(slots)

        xyz = self.xyz_range * torch.tanh(self.xyz_head(slots))
        charge = self.charge_head(slots)
        obj_logits = self.obj_head(slots).squeeze(-1)
        pred_points = torch.cat([xyz, charge], dim=-1)
        return pred_points, obj_logits


@torch.no_grad()
def greedy_one_to_one_match(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy injective target-to-slot assignment for a single patch."""
    K, M = cost.shape
    if M > K:
        raise ValueError(f"Cannot match M={M} targets to K={K} slots")
    if M == 0:
        empty = torch.empty(0, dtype=torch.long, device=cost.device)
        return empty, empty

    work = cost.detach().clone()
    pred_matches = []
    targ_matches = []
    inf = torch.tensor(float("inf"), dtype=work.dtype, device=work.device)
    for _ in range(M):
        flat = torch.argmin(work)
        k = torch.div(flat, M, rounding_mode="floor")
        m = flat % M
        pred_matches.append(k)
        targ_matches.append(m)
        work[k, :] = inf
        work[:, m] = inf
    return torch.stack(pred_matches), torch.stack(targ_matches)


def pointset_slot_loss(
    pred_points: torch.Tensor,
    obj_logits: torch.Tensor,
    target_points: torch.Tensor,
    offsets: torch.Tensor,
    counts: torch.Tensor | None = None,
    xyz_weight: float = 1.0,
    charge_weight: float = 0.10,
    objectness_weight: float = 0.05,
    count_weight: float = 0.10,
    negative_objectness_weight: float = 0.25,
    is_empty_candidate: torch.Tensor | None = None,
    empty_loss_weight: float = 1.0,
    group_by_empty_candidate: bool = False,
    soft_objectness_tau: float = 0.20,
    pred_to_target_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Chamfer-based slot loss for unordered variable-cardinality point sets.

    Geometry uses the repo's padded Chamfer implementation. Target-to-prediction
    Chamfer guarantees target coverage. The M closest predicted slots are marked
    as hard objectness positives for occupied patches; empty candidates receive
    all-zero objectness targets and count 0.
    """
    _ = soft_objectness_tau  # Kept for backward-compatible configs.
    T, K, C = pred_points.shape
    if T == 0:
        zero = pred_points.sum() * 0.0 + obj_logits.sum() * 0.0
        zero_detached = zero.detach()
        return {
            "loss": zero,
            "loss_pointset_xyz": zero_detached,
            "loss_pointset_charge": zero_detached,
            "loss_pointset_objectness": zero_detached,
            "loss_pointset_count": zero_detached,
            "loss_pointset_occupied": zero_detached,
            "loss_pointset_empty": zero_detached,
            "pointset_count_mae": zero_detached,
            "mean_predicted_count": zero_detached,
            "mean_target_count": zero_detached,
            "mean_objectness_prob": zero_detached,
            "mean_target_count_occupied": zero_detached,
            "mean_predicted_count_occupied": zero_detached,
            "mean_predicted_count_empty": zero_detached,
            "mean_objectness_prob_occupied": zero_detached,
            "mean_objectness_prob_empty": zero_detached,
            "empty_false_positive_rate_obj_0p5": zero_detached,
            "occupied_count_mae": zero_detached,
            "empty_count_mae": zero_detached,
            "mean_positive_slots_occupied": zero_detached,
            "mean_objectness_target_occupied": zero_detached,
            "num_supervised_patches": pred_points.new_tensor(0.0),
        }
    if C < 4:
        raise ValueError(f"pred_points must contain xyz+charge, got last dim {C}")

    device = pred_points.device
    offsets = offsets.to(device=device, dtype=torch.long).flatten()
    if counts is None:
        count_t = offsets[1:] - offsets[:-1]
    else:
        count_t = counts.to(device=device, dtype=torch.long).flatten()
    if count_t.numel() != T:
        raise ValueError(
            f"counts must have one entry per predicted patch; got {count_t.numel()} for T={T}"
        )

    max_count = int(count_t.max().item()) if count_t.numel() else 0
    if max_count > K:
        raise RuntimeError(
            f"pointset_slot_loss received up to {max_count} targets for K={K} slots; "
            "targets should be capped before loss computation"
        )

    if is_empty_candidate is not None:
        is_empty_candidate = is_empty_candidate.to(device=device, dtype=torch.bool).flatten()
        if is_empty_candidate.numel() != T:
            raise ValueError(
                "is_empty_candidate must have one entry per predicted patch; "
                f"got {is_empty_candidate.numel()} for T={T}"
            )
        empty_mask = is_empty_candidate
    else:
        empty_mask = count_t == 0
    occupied_mask = ~empty_mask
    nonempty_mask = count_t > 0

    logits_f = obj_logits.float()
    obj_prob = logits_f.sigmoid()
    pred_count_t = obj_prob.sum(dim=1)
    target_count_t = count_t.to(dtype=pred_count_t.dtype)

    xyz_loss_t = logits_f.new_zeros(T)
    q_loss_t = logits_f.new_zeros(T)
    obj_labels = logits_f.new_zeros(T, K)
    positive_slot_count_t = logits_f.new_zeros(T)

    if max_count > 0 and bool(nonempty_mask.any()):
        target_points = target_points.to(device=device, dtype=pred_points.dtype)
        target_padded = pred_points.new_zeros((T, max_count, C))
        for m in range(1, max_count + 1):
            group_idx = torch.nonzero(count_t == m, as_tuple=False).flatten()
            if group_idx.numel() == 0:
                continue
            starts = offsets.index_select(0, group_idx)
            point_idx = starts[:, None] + torch.arange(m, device=device)[None, :]
            target_g = target_points.index_select(0, point_idx.reshape(-1)).view(
                group_idx.numel(), m, target_points.shape[-1]
            )
            target_padded[group_idx, :m, : target_g.shape[-1]] = target_g

        occ_idx = torch.nonzero(nonempty_mask, as_tuple=False).flatten()
        occ_counts = count_t.index_select(0, occ_idx)
        pred_occ = pred_points.index_select(0, occ_idx)
        target_occ = target_padded.index_select(0, occ_idx)

        with torch.amp.autocast(device_type=pred_points.device.type, enabled=False):
            (target_to_pred, pred_to_target), _, (target_to_pred_idx, _) = chamfer_distance(
                target_occ[:, :, :3].float(),
                pred_occ[:, :, :3].float(),
                x_lengths=occ_counts,
                y_lengths=torch.full_like(occ_counts, K),
                batch_reduction=None,
                point_reduction=None,
                norm=1,
                single_directional=False,
            )

        valid_target = torch.arange(max_count, device=device)[None, :] < occ_counts[:, None]
        target_denom = occ_counts.to(dtype=target_to_pred.dtype).clamp_min(1)
        target_cover_loss = (target_to_pred * valid_target).sum(dim=1) / target_denom

        labels_occ = logits_f.new_zeros((occ_idx.numel(), K))
        slot_order = torch.argsort(pred_to_target.detach(), dim=1)[:, :max_count]
        valid_slot = torch.arange(max_count, device=device)[None, :] < occ_counts[:, None]
        row_idx = (
            torch.arange(occ_idx.numel(), device=device)[:, None]
            .expand(-1, max_count)[valid_slot]
        )
        labels_occ[row_idx, slot_order[valid_slot]] = 1.0
        obj_labels.index_copy_(0, occ_idx, labels_occ)
        positive_slot_count_t.index_copy_(0, occ_idx, labels_occ.sum(dim=1))

        pred_weights = labels_occ.detach()
        pred_push_loss = (
            (pred_to_target * pred_weights).sum(dim=1)
            / pred_weights.sum(dim=1).clamp_min(1.0)
        )
        xyz_loss_t.index_copy_(
            0,
            occ_idx,
            target_cover_loss + float(pred_to_target_weight) * pred_push_loss,
        )

        pred_charge_occ = pred_occ[:, :, 3:4].float()
        nn_charge = pred_charge_occ.gather(
            1, target_to_pred_idx[:, :, None].expand(-1, -1, 1)
        )
        charge_err = F.smooth_l1_loss(
            nn_charge,
            target_occ[:, :, 3:4].float(),
            reduction="none",
        ).squeeze(-1)
        q_loss_occ = (charge_err * valid_target).sum(dim=1) / target_denom
        q_loss_t.index_copy_(0, occ_idx, q_loss_occ)


    obj_raw = F.binary_cross_entropy_with_logits(
        logits_f,
        obj_labels,
        reduction="none",
    )
    obj_weights = obj_labels + float(negative_objectness_weight) * (1.0 - obj_labels)
    obj_loss_t = (obj_raw * obj_weights).sum(dim=1) / obj_weights.sum(dim=1).clamp_min(1.0)

    count_loss_t = F.smooth_l1_loss(
        pred_count_t,
        target_count_t,
        reduction="none",
    )
    count_mae_t = (pred_count_t - target_count_t).abs()
    obj_prob_mean_t = obj_prob.mean(dim=1)
    obj_fp_t = (obj_prob > 0.5).to(torch.float32).mean(dim=1)

    stacked_losses = (
        float(xyz_weight) * xyz_loss_t
        + float(charge_weight) * q_loss_t
        + float(objectness_weight) * obj_loss_t
        + float(count_weight) * count_loss_t
    )

    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if bool(mask.any()):
            return values[mask].mean()
        return values.new_zeros(())

    occupied_loss = _masked_mean(stacked_losses, occupied_mask)
    empty_loss = _masked_mean(stacked_losses, empty_mask)
    if group_by_empty_candidate:
        loss = occupied_loss + float(empty_loss_weight) * empty_loss
    else:
        loss = stacked_losses.mean()

    return {
        "loss": loss,
        "loss_pointset_xyz": _masked_mean(xyz_loss_t, nonempty_mask),
        "loss_pointset_charge": _masked_mean(q_loss_t, nonempty_mask),
        "loss_pointset_objectness": obj_loss_t.mean(),
        "loss_pointset_count": count_loss_t.mean(),
        "loss_pointset_occupied": occupied_loss.detach(),
        "loss_pointset_empty": empty_loss.detach(),
        "pointset_count_mae": count_mae_t.mean(),
        "mean_predicted_count": pred_count_t.mean(),
        "mean_target_count": target_count_t.mean(),
        "mean_objectness_prob": obj_prob_mean_t.mean(),
        "mean_target_count_occupied": _masked_mean(target_count_t, occupied_mask),
        "mean_predicted_count_occupied": _masked_mean(pred_count_t, occupied_mask),
        "mean_predicted_count_empty": _masked_mean(pred_count_t, empty_mask),
        "mean_objectness_prob_occupied": _masked_mean(obj_prob_mean_t, occupied_mask),
        "mean_objectness_prob_empty": _masked_mean(obj_prob_mean_t, empty_mask),
        "empty_false_positive_rate_obj_0p5": _masked_mean(obj_fp_t, empty_mask),
        "occupied_count_mae": _masked_mean(count_mae_t, occupied_mask),
        "empty_count_mae": _masked_mean(count_mae_t, empty_mask),
        "mean_positive_slots_occupied": _masked_mean(
            positive_slot_count_t, occupied_mask
        ),
        "mean_objectness_target_occupied": _masked_mean(
            obj_labels.mean(dim=1), occupied_mask
        ),
        "num_supervised_patches": pred_points.new_tensor(float(T)),
    }


class ReconHead(nn.Module):
    """Per-token dense sub-voxel head emitting (occ_logits, energy_pred).

    Single shared MLP with output width `num_targets * kernel**3`. Returns a
    tensor of shape `(..., num_targets, kernel**3)` so callers can index the
    occupancy and energy channels cleanly.
    """

    def __init__(
        self,
        dim: int,
        kernel: int = 5,
        hidden_mult: int = 2,
        num_targets: int = 2,
    ):
        super().__init__()
        self.num_targets = num_targets
        self.kernel = kernel
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * hidden_mult)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * hidden_mult, num_targets * kernel ** 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc2(self.act(self.fc1(self.norm(x))))
        return out.view(*out.shape[:-1], self.num_targets, self.kernel ** 3)
