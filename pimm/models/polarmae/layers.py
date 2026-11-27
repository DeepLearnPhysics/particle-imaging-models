"""
Self-contained building blocks for PoLAr-MAE.

Reimplements the layers from the external PoLAr-MAE library without any
dependency on it.  All modules operate on padded (B, N, C) tensors with
boolean or integer masks for variable-length batches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pimm.models.polarmae import ops


def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    if dtype in (torch.float, torch.double, torch.bfloat16):
        return 1e-13
    if dtype == torch.half:
        return 1e-4
    raise TypeError(f"Unsupported dtype {dtype}")


_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
_MASK_DTYPE = torch.bool if _TORCH_VERSION >= (2, 5) else torch.float32


class MaskedLayerNorm(nn.Module):
    """LayerNorm that respects a (B, T) padding mask."""

    def __init__(self, size: int, gamma0: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, 1, size) * gamma0)
        self.beta = nn.Parameter(torch.zeros(1, 1, size))
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        bm = mask.unsqueeze(-1)
        n = bm.sum() * self.size
        mu = (x * bm).sum() / n
        centred = (x - mu) * bm
        std = torch.sqrt((centred * centred).sum() / n + tiny_value_of_dtype(x.dtype))
        return (self.gamma * (x - mu) / (std + tiny_value_of_dtype(x.dtype)) + self.beta) * bm


class MaskedBatchNorm1d(nn.Module):
    """BatchNorm1d for (B, C, L) with a (B, 1, L) mask."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, C, L = x.size()
        if mask is None:
            mask = x.new_ones(B, 1, L)
        mask = mask.float()
        valid = mask.sum().clamp(min=1)

        if self.training:
            mean = (x * mask).sum(dim=(0, 2)) / valid
            x = x - mean.view(1, C, 1)
            var = ((x * mask) ** 2).sum(dim=(0, 2)) / valid
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var, self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var
            x = x - mean.view(1, C, 1)

        x = x / torch.sqrt(var + self.eps).view(1, C, 1) * mask
        return x * self.weight.view(1, C, 1) + self.bias.view(1, C, 1)


class MaskedDropPath(nn.Module):
    """Stochastic depth per token, respecting padding."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        rand = x.new_empty(x.shape[0], x.shape[1], 1).bernoulli_(keep).div_(keep)
        x = x * rand
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class _Identity(nn.Module):
    def forward(self, x, mask=None):
        return x


def prepare_attn_mask(
    q_mask: torch.Tensor,
    kv_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build (B, 1, Nq, Nkv) attention mask from (B, Nq) and (B, Nkv) masks."""
    if kv_mask is None:
        kv_mask = q_mask
    attn = q_mask.unsqueeze(1).unsqueeze(3) & kv_mask.unsqueeze(1).unsqueeze(2)
    if _MASK_DTYPE != torch.bool:
        attn = attn.float().masked_fill_(~attn, float("-inf"))
    return attn


class Attention(nn.Module):
    """Multi-head attention supporting both self- and cross-attention."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _q(self, x):
        return F.linear(x, self.qkv.weight[: self.qkv.weight.shape[0] // 3],
                        self.qkv.bias[: self.qkv.bias.shape[0] // 3] if self.qkv.bias is not None else None)

    def _k(self, x):
        d = self.qkv.weight.shape[0] // 3
        return F.linear(x, self.qkv.weight[d:2*d],
                        self.qkv.bias[d:2*d] if self.qkv.bias is not None else None)

    def _v(self, x):
        d = self.qkv.weight.shape[0] // 3
        return F.linear(x, self.qkv.weight[2*d:],
                        self.qkv.bias[2*d:] if self.qkv.bias is not None else None)

    def forward(self, q: torch.Tensor, attn_mask: torch.Tensor | None = None,
                kv: torch.Tensor | None = None) -> Tuple[torch.Tensor, None]:
        if kv is None:
            return self._self_attn(q, attn_mask)
        return self._cross_attn(q, kv, attn_mask)

    def _self_attn(self, x, mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=self.attn_drop.p if self.training else 0.0)
        x = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x)), None

    def _cross_attn(self, q_in, kv_in, mask):
        B, Nq, C = q_in.shape
        Nv = kv_in.shape[1]
        q = self._q(q_in).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self._k(kv_in).reshape(B, Nv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self._v(kv_in).reshape(B, Nv, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=self.attn_drop.p if self.training else 0.0)
        x = out.transpose(1, 2).reshape(B, Nq, C)
        return self.proj_drop(self.proj(x)), None


class _MLP(nn.Module):
    """MLP with named fc1/fc2 (matches original PoLAr-MAE checkpoint keys)."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Transformer block: Norm → Attn → Residual → Norm → MLP → Residual."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0,
                 attn_drop: float = 0.0, drop_path: float = 0.0,
                 use_kv: bool = False):
        super().__init__()
        self.norm1 = MaskedLayerNorm(dim)
        self.norm1_kv = MaskedLayerNorm(dim) if use_kv else None
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = MaskedDropPath(drop_path) if drop_path > 0.0 else _Identity()
        self.norm2 = MaskedLayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio))

    def forward(self, q, q_mask, attn_mask, kv=None, kv_mask=None):
        q_n = self.norm1(q, q_mask)
        kv_n = self.norm1_kv(kv, kv_mask) if self.norm1_kv is not None else None
        h, _ = self.attn(q_n, attn_mask, kv=kv_n)
        q = q + self.drop_path(h, q_mask)
        ffn = self.mlp(self.norm2(q, q_mask))
        if q_mask is not None:
            ffn = ffn * q_mask.unsqueeze(-1)
        q = q + self.drop_path(ffn, q_mask)
        return q


@dataclass
class TransformerOutput:
    last_hidden_state: torch.Tensor
    hidden_states: Optional[List[torch.Tensor]] = None


VIT_CONFIGS = {
    "vit_tiny":  dict(embed_dim=192,  depth=12, num_heads=6),
    "vit_small": dict(embed_dim=384,  depth=12, num_heads=6),
    "vit_base":  dict(embed_dim=768,  depth=12, num_heads=12),
}


class Transformer(nn.Module):
    """Stack of Blocks with optional postnorm."""

    def __init__(self, embed_dim: int = 384, depth: int = 12, num_heads: int = 6,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0, add_pos_at_every_layer: bool = False,
                 postnorm: bool = True, use_kv: bool = False, **_kw):
        super().__init__()
        self.embed_dim = embed_dim
        self.add_pos_at_every_layer = add_pos_at_every_layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate,
                  attn_drop_rate, dpr[i], use_kv=use_kv)
            for i in range(depth)
        ])
        self.norm = MaskedLayerNorm(embed_dim) if postnorm else _Identity()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, q, pos_q, q_mask, kv=None, pos_kv=None, kv_mask=None,
                return_hidden_states: bool = False, final_norm: bool = True) -> TransformerOutput:
        if kv is not None:
            kv = kv + pos_kv
        if not self.add_pos_at_every_layer:
            q = q + pos_q
        attn_mask = prepare_attn_mask(q_mask, kv_mask)
        hidden_states = [] if return_hidden_states else None
        for blk in self.blocks:
            if self.add_pos_at_every_layer:
                q = q + pos_q
            q = blk(q, q_mask, attn_mask, kv, kv_mask)
            if return_hidden_states:
                hidden_states.append(q)
        if final_norm:
            q = self.norm(q, q_mask)
        return TransformerOutput(q, hidden_states)


def make_transformer(arch: str, use_kv: bool = False, **kwargs) -> Transformer:
    cfg = dict(VIT_CONFIGS[arch])
    cfg.update(kwargs)
    return Transformer(use_kv=use_kv, **cfg)


@torch.no_grad()
def sample_farthest_points(
    points: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    random_start: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Farthest point sampling over xyz from a random (or the first) point.  Returns (sampled_points, indices)."""
    start = None
    if random_start:
        start = (torch.rand(points.shape[0], device=points.device) * lengths.float()).long()
    idx = ops.sample_farthest_points(points, lengths, K, start_idxs=start)
    return masked_gather(points, idx), idx


@torch.no_grad()
def fill_empty_indices(idx: torch.Tensor) -> torch.Tensor:
    """Replace -1 entries with the first valid index in each group."""
    mask = idx == -1
    first = idx[..., 0].unsqueeze(-1).expand_as(idx)
    out = idx.clone()
    out[mask] = first[mask]
    return out


def masked_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather with -1-padding support. points: (B,N,D) or (B,G,N,D), idx: (B,K) or (B,G,K)."""
    idx = idx.clone()
    idx[idx == -1] = 0
    if idx.ndim == 2:
        # (B, K) -> gather from (B, N, D)
        return points.gather(1, idx.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
    # idx: (B, G, K) -> gather from (B, N, D)
    # Gather along dim=1 with flattened indices to avoid expanding points to
    # (B, G, N, D) — that expand+gather pattern forces backward to allocate
    # the full (B, G, N, D) gradient tensor (can be 100+ GiB).
    B, G, K = idx.shape
    D = points.shape[-1]
    idx_flat = idx.reshape(B, G * K)                                   # (B, G*K)
    idx_flat_exp = idx_flat.unsqueeze(-1).expand(-1, -1, D)            # (B, G*K, D)
    gathered = points.gather(1, idx_flat_exp)                          # (B, G*K, D)
    return gathered.reshape(B, G, K, D)


@torch.no_grad()
def _select_topk_by_fps(points: torch.Tensor, idx: torch.Tensor, K: int) -> torch.Tensor:
    """Reduce groups from K_big to K via FPS."""
    B, G, K_big = idx.shape
    C = points.shape[-1]
    grouped = masked_gather(points, idx).view(B * G, K_big, C)
    lens = (~idx.eq(-1)).sum(2).view(B * G).to(torch.int64)
    _, fps_idx = sample_farthest_points(grouped, lens, K, random_start=True)
    fps_idx = fps_idx.view(B, G, K)
    invalid = fps_idx == -1
    fps_idx = fps_idx.clamp(min=0)
    result = torch.gather(idx, 2, fps_idx)
    result[invalid] = -1
    return result


class PointcloudGrouping(nn.Module):
    """FPS → CNMS → Ball-query → FPS-reduction → local-coord normalisation."""

    def __init__(self, num_groups: int, context_length: int, group_max_points: int,
                 group_radius: float, group_upscale_points: int,
                 overlap_factor: float, reduction_method: str = "fps",
                 rescale_by_group_radius: bool | float = True):
        super().__init__()
        self.num_groups = num_groups
        self.context_length = context_length
        self.group_max_points = group_max_points
        self.group_radius = group_radius
        self.group_upscale_points = group_upscale_points
        self.overlap_factor = overlap_factor
        self.reduction_method = reduction_method
        self.rescale_by_group_radius = group_radius if isinstance(rescale_by_group_radius, bool) else rescale_by_group_radius

    def forward(self, points: torch.Tensor, lengths: torch.Tensor) -> dict:
        # FPS seed selection
        _, fps_idx = sample_farthest_points(points[..., :3].float(), lengths, self.num_groups, random_start=True)
        possible_centers = masked_gather(points[..., :3], fps_idx)
        possible_lengths = fps_idx.ne(-1).sum(-1)

        # CNMS
        centers, lens1 = ops.cnms(
            possible_centers, overlap_factor=self.overlap_factor,
            radius=self.group_radius, K=self.num_groups, lengths=possible_lengths,
        )
        lens1 = lens1.clamp_max(self.context_length)
        centers = centers[:, :self.context_length]

        # Ball query
        bq_idx = ops.ball_query(
            centers[..., :3], points[..., :3].float(),
            K=self.group_upscale_points, radius=self.group_radius,
            lengths1=lens1, lengths2=lengths,
        )

        # Reduction
        if self.reduction_method == "fps":
            bq_idx = _select_topk_by_fps(points, bq_idx, self.group_max_points)

        # Build groups
        K = self.group_max_points
        B, G, _ = bq_idx.shape
        T = min(self.context_length, G)
        point_lengths = (~bq_idx.eq(-1)).sum(2)
        groups = masked_gather(points, fill_empty_indices(bq_idx))  # (B, G, K_big, C)

        # Trim to context length
        groups = groups[:, :T, :K]
        centers = centers[:, :T]
        point_mask = torch.arange(K, device=points.device).expand(B, T, -1) < point_lengths[:, :T].unsqueeze(-1)
        group_lengths = (~bq_idx.eq(-1)).any(2).sum(1)
        emb_mask = torch.arange(T, device=points.device).unsqueeze(0).expand(B, -1) < group_lengths.unsqueeze(1)

        # Local coords
        groups[..., :3] = groups[..., :3] - centers.unsqueeze(2)
        if self.rescale_by_group_radius:
            groups[..., :3] = groups[..., :3] / self.rescale_by_group_radius
        groups *= emb_mask.unsqueeze(-1).unsqueeze(-1)

        return dict(groups=groups, centers=centers, embedding_mask=emb_mask,
                    point_mask=point_mask, idx=bq_idx)


class VariablePointcloudMasking(nn.Module):
    """Random masking of a fraction of tokens, handling variable-length batches."""

    def __init__(self, ratio: float = 0.6):
        super().__init__()
        self.ratio = ratio

    def forward(self, lengths: torch.Tensor):
        B, = lengths.shape
        G = lengths.max().item()
        device = lengths.device

        valid = torch.arange(G, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = torch.rand(B, G, device=device)
        scores[~valid] = float("inf")
        _, sorted_idx = scores.sort(dim=1)

        num_mask = (self.ratio * lengths).long()
        max_mask = num_mask.max().item()
        masked_indices = sorted_idx[:, :max_mask].clone()
        masked_mask = torch.arange(max_mask, device=device).unsqueeze(0) < num_mask.unsqueeze(1)

        num_unmask = lengths - num_mask
        max_unmask = num_unmask.max().item()
        offset = num_mask.unsqueeze(1) + torch.arange(max_unmask, device=device).unsqueeze(0)
        offset = offset.clamp(max=G - 1)
        unmasked_indices = torch.gather(sorted_idx, 1, offset).clone()
        unmasked_mask = torch.arange(max_unmask, device=device).unsqueeze(0) < num_unmask.unsqueeze(1)

        return masked_indices, masked_mask, unmasked_indices, unmasked_mask


class _TimeEmbedding(nn.Module):
    """Parameter-free sinusoidal embedding of a point's order index.

    Matches DeepLearnPhysics/PoLAr-MAE's ``TimeEmbedding`` exactly so the
    ``original`` PointOrderEncoder reproduces that checkpoint's behavior.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.emb_dim = dim

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        half = self.emb_dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=ts.device).float() * -emb)
        emb = ts[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class PointOrderEncoder(nn.Module):
    """Sinusoidal positional encoding by point index (for equivariant MiniPointNet).

    ``style="mlp"`` (pimm default): inline sinusoidal embedding then a 2-layer
    MLP (``net``). ``style="original"``: a ``TimeEmbedding`` followed by a single
    ``Linear`` (``time_embed``), reproducing the original PoLAr-MAE module and its
    checkpoint keys.
    """

    def __init__(self, dim: int, style: str = "mlp"):
        super().__init__()
        self.dim = dim
        self.style = style
        if style == "mlp":
            self.net = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim),
            )
        elif style == "original":
            self.time_embed = nn.Sequential(_TimeEmbedding(dim), nn.Linear(dim, dim))
        else:
            raise ValueError(f"unknown PointOrderEncoder style: {style!r}")

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        N = points.shape[1]
        device = points.device
        if self.style == "original":
            inp = torch.arange(N, device=device)
            return self.time_embed(inp).unsqueeze(0)
        half = self.dim // 2
        omega = torch.exp(torch.arange(half, device=device).float() * -(math.log(10000) / (half - 1)))
        pos = torch.arange(N, device=device).float()
        emb = torch.cat([torch.sin(pos[:, None] * omega[None, :]),
                          torch.cos(pos[:, None] * omega[None, :])], dim=-1)
        return self.net(emb).unsqueeze(0)


class MaskedMiniPointNet(nn.Module):
    """Mini PointNet that respects point masks."""

    def __init__(self, channels: int, feature_dim: int,
                 hidden1: int = 128, hidden2: int = 256,
                 equivariant: bool = False, pos_enc_style: str = "mlp"):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(channels, hidden1, 1, bias=False), MaskedBatchNorm1d(hidden1),
            nn.ReLU(inplace=True), nn.Conv1d(hidden1, hidden2, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(hidden2 * 2, hidden2 * 2, 1, bias=False), MaskedBatchNorm1d(hidden2 * 2),
            nn.ReLU(inplace=True), nn.Conv1d(hidden2 * 2, feature_dim, 1),
        )
        self.equivariant = equivariant
        if equivariant:
            self.pos_enc = PointOrderEncoder(hidden2, style=pos_enc_style)

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # points: (B, N, C) or (B, G, N, C); mask: (B, 1, N) or (B, G, 1, N)
        reshape = points.ndim == 4
        if reshape:
            Bi, Gi, S, C = points.shape
            points = points.reshape(Bi * Gi, S, C)
            mask = mask.reshape(Bi * Gi, 1, S)

        x = points.transpose(2, 1)  # (B, C, N)
        for layer in self.first_conv:
            x = layer(x, mask) if isinstance(layer, MaskedBatchNorm1d) else layer(x)

        if self.equivariant:
            x = x + self.pos_enc(points).transpose(2, 1)

        glob = x.max(dim=2, keepdim=True).values
        x = torch.cat([glob.expand_as(x), x], dim=1)

        for layer in self.second_conv:
            x = layer(x, mask) if isinstance(layer, MaskedBatchNorm1d) else layer(x)

        out = x.max(dim=2).values
        if reshape:
            out = out.reshape(Bi, Gi, -1)
        return out


class LearnedPositionalEncoder(nn.Module):
    """MLP positional encoding from 3D group centers."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.pos_enc = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim))

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        return self.pos_enc(pos[..., :3])


def masked_layer_norm(
    input: torch.Tensor, normalized_shape: int,
    mask: torch.Tensor, gamma: float = 1.0, beta: float = 0.0,
) -> torch.Tensor:
    """Layer-norm over valid tokens only (batch-global statistics).

    Args:
        input: (B, T, C)
        normalized_shape: C
        mask: (B, T) bool
        gamma, beta: scale/shift
    """
    bm = mask.unsqueeze(-1)
    n = bm.sum() * normalized_shape
    mu = (input * bm).sum() / n
    centred = (input - mu) * bm
    std = torch.sqrt((centred * centred).sum() / n + tiny_value_of_dtype(input.dtype))
    return (gamma * (input - mu) / (std + tiny_value_of_dtype(input.dtype)) + beta) * bm


class PointNetFeatureUpsampling(nn.Module):
    """KNN-based upsampling from token centers to original points."""

    def __init__(self, in_channel: int, mlp: List[int], K: int = 5):
        super().__init__()
        self.K = K
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3  # concat coords
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(MaskedBatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2, point_lens, embedding_lens, point_mask):
        """
        Args:
            xyz1: (B, N_max, 3) original point positions
            xyz2: (B, T, 3) token center positions
            points1: (B, N_max, 3) original point coords (for concat)
            points2: (B, T, D) token features (downcast)
            point_lens: (B,) valid point counts
            embedding_lens: (B,) valid token counts
            point_mask: (B, N_max) bool
        Returns:
            new_points: (B, N_max, D'), idx
        """
        dists, idx = ops.knn_points(
            xyz1[..., :3], xyz2, lengths1=point_lens, lengths2=embedding_lens,
            K=self.K, return_sorted=False,
        )
        dist_recip = 1.0 / (dists + torch.finfo(dists.dtype).eps)
        weight = dist_recip / dist_recip.sum(dim=2, keepdim=True)

        interpolated = masked_gather(points2, idx)  # (B, N, K, D)
        interpolated = (interpolated * weight.unsqueeze(-1)).sum(dim=2)

        new_points = torch.cat([points1, interpolated], dim=-1)
        new_points = new_points.transpose(1, 2)  # (B, D', N)
        pmask = point_mask.unsqueeze(1).float()

        for i, conv in enumerate(self.mlp_convs):
            new_points = conv(new_points)
            new_points = self.mlp_bns[i](new_points, pmask)
            if i < len(self.mlp_convs) - 1:
                new_points = F.gelu(new_points)

        return new_points.transpose(1, 2), idx


class SegmentationHead(nn.Module):
    """3-layer Conv1d MLP for per-point classification."""

    def __init__(self, in_channels: int, seg_head_dim: int,
                 seg_head_dropout: float, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, seg_head_dim, 1, bias=False)
        self.bn1 = MaskedBatchNorm1d(seg_head_dim)
        self.conv2 = nn.Conv1d(seg_head_dim, seg_head_dim // 2, 1, bias=False)
        self.bn2 = MaskedBatchNorm1d(seg_head_dim // 2)
        self.conv3 = nn.Conv1d(seg_head_dim // 2, num_classes, 1)
        self.dropout = nn.Dropout(seg_head_dropout)

    def forward(self, x: torch.Tensor, point_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, C, N), point_mask: (B, N) → (B, num_classes, N)"""
        mask = point_mask.unsqueeze(1).float() if point_mask is not None else None
        x = self.dropout(F.relu(self.bn1(self.conv1(x), mask)))
        x = F.relu(self.bn2(self.conv2(x), mask))
        return self.conv3(x)
