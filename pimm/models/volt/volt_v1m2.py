"""Volt-v1m2: Sonata-compatible Volt encoder + transposed-conv decoder.

Differs from v1m1 by upsampling the encoder's per-token features back to
per-input-point features through a SparseInverseConv3d decoder (matching
original Volt's design at libs/Volt/pointcept/models/volt/decoder.py).
The output is a Point at input resolution — no `pooling_parent`/
`pooling_inverse` chain, no `head_indices`-based label propagation.

Why: with v1m1 the SSL matching loss runs at token resolution, so KNN
across views works fine; but per-point downstream tasks see *patch-uniform*
features (every input point in a patch shares its parent token). v1m2
gives every input point a unique upsampled feature, which (a) avoids
the patch-uniform pathology entirely and (b) gives ~10x more samples
per batch to a SIGReg-style loss.

Pipeline:
    stem -> mask injection -> strided tokenizer -> RoPE blocks -> final norm
        -> SparseInverseConv3d decoder -> per-input-point Point

Pair with Sonata using `up_cast_level=0` (no pooling chain) and
`head_in_channels=decoder_dim` (typically 256, matching Volt's default).
"""

from __future__ import annotations

from typing import Optional

import spconv.pytorch as spconv
import torch
import torch.nn as nn
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import LayerScale
from torch.nn.init import trunc_normal_

from pimm.models.builder import MODELS
from pimm.models.modules import PointModel, PointModule, PointSequential
from pimm.models.utils.structure import Point
from pimm.utils.logger import get_logger

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except ImportError:
    flash_attn_varlen_qkvpacked_func = None

logger = get_logger(__name__)


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

    @staticmethod
    def _precompute(freqs, max_pos):
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


class Block(PointModule):
    """RoPE+FlashAttn transformer block taking and returning a `Point`.

    On the first call within a forward pass, computes the attention
    bookkeeping (`cu_seqlens`, `max_seqlen`, `freqs_cis`) from the Point's
    `sparse_conv_feat.indices` and caches it on the Point itself; subsequent
    blocks reuse the cache. Mirrors PT-v3m2's pattern of caching attention
    metadata on the Point (e.g. `point.serialized_order`) so the encoder
    body doesn't have to plumb it through every call.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = None,
        qk_norm: bool = False,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        pos_enc: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # Shared RoPE module (registered on parent too); PyTorch handles
        # the shared sub-module fine and RoPE's caches are non-persistent
        # buffers so state_dict isn't polluted.
        self.pos_enc = pos_enc

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

    def _compute_seqlens_and_freqs(self, point: Point):
        """Fetch (cu_seqlens, max_seqlen, freqs_cis) from the Point, computing
        and caching them on the Point itself if absent."""
        if "cu_seqlens" not in point.keys():
            indices = point.sparse_conv_feat.indices
            counts = torch.bincount(indices[:, 0])
            cu = torch.zeros(
                counts.numel() + 1, dtype=torch.int32, device=indices.device
            )
            cu[1:] = torch.cumsum(counts.to(torch.int32), dim=0)
            point["cu_seqlens"] = cu
            point["max_seqlen"] = int(counts.max().item()) if counts.numel() else 0
            point["freqs_cis"] = self.pos_enc.compute_axial_cis_efficient(
                indices[:, 1:].long()
            )
        return point.cu_seqlens, point.max_seqlen, point.freqs_cis

    def forward(self, point: Point) -> Point:
        cu_seqlens, max_seqlen, freqs_cis = self._compute_seqlens_and_freqs(point)
        x = point.feat
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), freqs_cis, cu_seqlens, max_seqlen))
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        point.feat = x
        if "sparse_conv_feat" in point.keys():
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(x)
        return point


class Stem(PointModule):
    """Per-point input projection + optional learnable mask token.

    Mirrors PT-v3m2's `Embedding` (point_transformer_v3m2_sonata.py:585-619):
    a `PointSequential` of `Linear` (+ optional `norm_layer` / `act_layer`)
    followed by a learnable mask-token override on points where `point.mask`
    is True. `PointSequential` keeps `point.feat` and `point.sparse_conv_feat`
    in sync through the linear projection; we sync once more after the mask
    `where` so the downstream `Tokenizer` reads the mask-injected SCT.
    """

    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        norm_layer=None,
        act_layer=None,
        mask_token: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stem_channels = stem_channels

        self.stem = PointSequential(linear=nn.Linear(in_channels, stem_channels))
        if norm_layer is not None:
            self.stem.add(norm_layer(stem_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

        if mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, stem_channels))
        else:
            self.mask_token = None

    def forward(self, point: Point) -> Point:
        point = self.stem(point)
        if (
            self.mask_token is not None
            and "mask" in point.keys()
            and point.mask is not None
        ):
            point.feat = torch.where(
                point.mask.unsqueeze(-1),
                self.mask_token.to(point.feat.dtype),
                point.feat,
            )
            if "sparse_conv_feat" in point.keys():
                point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(
                    point.feat
                )
        return point


class Tokenizer(PointModule):
    """Strided sparse-conv tokenizer: input voxels -> 5x5x5 patch tokens."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kernel_size: int = 5,
        stride: int = 5,
        indice_key: str = "embedding",
    ):
        super().__init__()
        self.conv = spconv.SparseConv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
            indice_key=indice_key,
        )

    def forward(self, point: Point) -> Point:
        x = self.conv(point.sparse_conv_feat)
        point.sparse_conv_feat = x
        point.feat = x.features
        return point


class Decoder(PointModule):
    """Transposed-conv decoder upsampling 5x5x5 patches back to per-input-voxel features.

    Structurally mirrors libs/Volt/pointcept/models/volt/decoder.py (Linear
    projection `embed_dim -> out_channels` followed by SparseInverseConv3d
    using the cached `indice_key` from the parent tokenizer to restore the
    original input voxel positions). Differs from original Volt by using
    `LayerNorm` instead of `BatchNorm1d` — BN was fine for supervised
    fine-tuning but produces train/eval-mode-asymmetric features that
    break SSL online linear probing. LN is batch-independent.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        indice_key: str = "embedding",
    ):
        super().__init__()
        act_layer = nn.GELU
        self.up = spconv.SparseSequential(
            nn.LayerNorm(in_channels),
            act_layer(),
            nn.Linear(in_channels, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            act_layer(),
            spconv.SparseInverseConv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                indice_key=indice_key,
                bias=False,
            ),
            nn.LayerNorm(out_channels),
            act_layer(),
        )

    def forward(self, point: Point) -> Point:
        x = self.up(point.sparse_conv_feat)
        point.sparse_conv_feat = x
        point.feat = x.features
        return point


@MODELS.register_module("Volt-v1m2")
class VoltBackboneV2(PointModel):
    """Flat Volt encoder + transposed-conv decoder, returning per-input-point features."""

    def __init__(
        self,
        in_channels: int = 4,
        stem_channels: int = 32,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = None,
        qk_norm: bool = True,
        drop_path: float = 0.3,
        increase_drop_path: bool = True,
        stride: int = 5,
        kernel_size: int = 5,
        mask_token: bool = True,
        decoder_dim: int = 256,
        rope_max_grid_size: tuple = (1024, 1024, 1024),
        rope_freq_split: tuple = (11, 11, 10),
    ):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim

        self.stem = Stem(in_channels, stem_channels, mask_token=mask_token)

        self.tokenizer = Tokenizer(
            in_channels=stem_channels,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            indice_key="embedding",
        )

        h_dim = embed_dim // num_heads
        assert sum(rope_freq_split) * 2 == h_dim, (
            f"rope_freq_split {tuple(rope_freq_split)} sums to {sum(rope_freq_split)}, "
            f"but h_dim//2 = {h_dim // 2} for embed_dim={embed_dim}, num_heads={num_heads}. "
            f"Adjust rope_freq_split so that sum(..)*2 == embed_dim // num_heads."
        )
        self.pos_enc = RoPE(
            freq_split=tuple(rope_freq_split),
            max_grid_size=tuple(rope_max_grid_size),
        )

        if increase_drop_path:
            dp = torch.linspace(0, drop_path, depth).tolist()
        else:
            dp = [drop_path] * depth

        self.blocks = PointSequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    drop_path=dp[i],
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    pos_enc=self.pos_enc,
                )
                for i in range(depth)
            ]
        )

        self.decoder = Decoder(
            in_channels=embed_dim,
            out_channels=decoder_dim,
            kernel_size=kernel_size,
            indice_key="embedding",
        )

        for name, module in self.named_modules():
            self._init_weights(module, name=name)

        logger.info(
            f"Volt-v1m2: in_channels={in_channels}, stem_channels={stem_channels}, "
            f"embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}, "
            f"stride={stride}, kernel_size={kernel_size}, mask_token={mask_token}, "
            f"decoder_dim={decoder_dim}"
        )

    @staticmethod
    def _init_weights(module, name: str = ""):
        if isinstance(module, nn.Linear):
            # keep kaiming for stem and decoder
            if name.startswith("stem.") or name.startswith("decoder."):
                return
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, "init_weights"):
            module.init_weights()

    def forward(self, point):
        if not isinstance(point, Point):
            point = Point(point)
        point.sparsify(pad=96)

        point = self.stem(point)
        point = self.tokenizer(point)
        point = self.blocks(point)
        point = self.decoder(point)
        return point
