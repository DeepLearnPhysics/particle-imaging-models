"""Transformer and point-to-token utilities for Volt-v1m1."""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import LayerScale

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except ImportError:
    flash_attn_varlen_qkvpacked_func = None


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


def _pack_indices(
    batch: torch.Tensor, ijk: torch.Tensor, shape_hash: int
) -> torch.Tensor:
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
    shape_hash = (
        int(
            max(
                parent.max().item() if parent.numel() else 0,
                token_indices[:, 1:].max().item() if token_indices.numel() else 0,
            )
        )
        + 2
    )

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
