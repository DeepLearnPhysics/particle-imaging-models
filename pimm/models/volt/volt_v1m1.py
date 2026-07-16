"""Volt-v1m1: Sonata-compatible Volt encoder backbone.

A flat (non-hierarchical) sparse-conv-tokenized transformer that satisfies
Sonata's `(point) -> point` backbone contract by emitting a *coarse* Point
at token resolution (one row per 5x5x5 voxel patch) with a
`pooling_parent`/`pooling_inverse` chain — the same shape PT-v3m2's
`SerializedPooling` produces for its hierarchical stages.

    point in:  feat, coord, origin_coord, offset (-> batch), grid_size,
               optional `mask` (per-point bool)
    point out: token-resolution Point with
                 feat / coord / origin_coord / batch / grid_coord
                 pooling_parent (the input Point)
                 pooling_inverse (per-input-point parent-token id)

The forward pipeline is `stem -> mask injection -> strided tokenizer ->
RoPE transformer blocks -> per-token Point construction`. The stem
(1x1 SubMConv3d into `stem_channels`) provides a per-point post-projection
slot where a learnable mask token can replace masked-point features in
the same way PT-v3m2's `Embedding` module does — keeping mask injection
in learned embedding space rather than mixing it into the 4-dim
physical-units input.

Why coarse output: Sonata's `match_neighbour` runs KNN over the returned
Point. Per-input-point broadcast tokens make matched pairs *byte-identical
within a patch*, which collapses the contrastive loss to its trivial
fixed point. Returning at token resolution gives every matched pair
distinct features. The `PretrainEvaluator` walks the `pooling_parent`
chain back up to per-point features for linear probing.

Pair with Sonata using `up_cast_level=0` and `head_in_channels=embed_dim`.
"""

from __future__ import annotations

from typing import Optional

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch_scatter
from torch.nn.init import trunc_normal_

from pimm.models.builder import MODELS
from pimm.models.modules import PointModel
from pimm.models.utils.structure import Point
from pimm.models.volt.layers import (
    Block,
    RoPE,
    build_point_to_token,
    sort_tokens_by_batch,
)
from pimm.utils.logger import get_logger

logger = get_logger(__name__)


@MODELS.register_module("Volt-v1m1")
class VoltBackbone(PointModel):
    """Flat Volt encoder for Sonata pretraining."""

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
        final_norm: bool = True,
        rope_max_grid_size: tuple = (1024, 1024, 1024),
        rope_freq_split: tuple = (11, 11, 10),
    ):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim

        self.stem = spconv.SubMConv3d(
            in_channels,
            stem_channels,
            kernel_size=1,
            bias=True,
            indice_key="volt_stem",
        )

        if mask_token:
            self.point_mask_token = nn.Parameter(torch.zeros(stem_channels))
            trunc_normal_(self.point_mask_token, std=0.02)
        else:
            self.point_mask_token = None

        self.tokenizer = spconv.SparseConv3d(
            stem_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
            indice_key="embedding",
        )

        if increase_drop_path:
            dp = torch.linspace(0, drop_path, depth).tolist()
        else:
            dp = [drop_path] * depth

        self.blocks = nn.Sequential(
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
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim) if final_norm else nn.Identity()

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

        self.apply(self._init_weights)

        logger.info(
            f"Volt-v1m1: in_channels={in_channels}, stem_channels={stem_channels}, "
            f"embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}, "
            f"stride={stride}, kernel_size={kernel_size}, mask_token={mask_token}, "
            f"final_norm={final_norm}"
        )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, "init_weights"):
            module.init_weights()

    @staticmethod
    def _compute_cu_seqlens(batch_indices: torch.Tensor):
        counts = torch.bincount(batch_indices)
        cu = torch.zeros(
            counts.numel() + 1, dtype=torch.int32, device=batch_indices.device
        )
        cu[1:] = torch.cumsum(counts.to(torch.int32), dim=0)
        max_seqlen = int(counts.max().item()) if counts.numel() else 0
        return cu, max_seqlen

    def forward(self, point):
        if not isinstance(point, Point):
            point = Point(point)
        if "grid_coord" not in point.keys():
            assert {"coord", "grid_size"}.issubset(point.keys()), (
                "Volt-v1m1 requires either `grid_coord` or both `coord` and "
                "`grid_size` on the input Point."
            )
            point["grid_coord"] = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()

        grid_coord = point.grid_coord
        batch = point.batch

        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        indices = torch.cat(
            [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
        ).contiguous()
        x = spconv.SparseConvTensor(
            features=point.feat,
            indices=indices,
            spatial_shape=sparse_shape,
            batch_size=int(batch[-1].item()) + 1,
        )

        x = self.stem(x)

        if (
            self.point_mask_token is not None
            and "mask" in point.keys()
            and point.mask is not None
        ):
            mask = point.mask
            x = x.replace_feature(
                torch.where(
                    mask.unsqueeze(-1),
                    self.point_mask_token.to(x.features.dtype),
                    x.features,
                )
            )

        x = self.tokenizer(x)

        token_features, token_indices = sort_tokens_by_batch(
            x.features, x.indices.long()
        )
        cu, max_seq = self._compute_cu_seqlens(token_indices[:, 0])
        freqs = self.pos_enc.compute_axial_cis_efficient(token_indices[:, 1:])

        for blk in self.blocks:
            token_features = blk(token_features, freqs, cu, max_seq)
        token_features = self.norm(token_features)

        point_to_token = build_point_to_token(
            grid_coord.long(), batch.long(), token_indices, self.stride
        )

        # Build the coarse (token-resolution) Point. Mirrors the
        # PT-v3m2 SerializedPooling output shape: per-cluster feat/coord
        # plus a pooling_parent/pooling_inverse chain so the evaluator
        # can broadcast back to per-point features for linear probing.
        T = token_features.shape[0]
        coarse_dict = dict(
            feat=token_features,
            coord=torch_scatter.scatter_mean(
                point.coord, point_to_token, dim=0, dim_size=T
            ),
            grid_coord=token_indices[:, 1:].int(),
            batch=token_indices[:, 0].long(),
            grid_size=(
                point.grid_size * self.stride if "grid_size" in point.keys() else None
            ),
            pooling_parent=point,
            pooling_inverse=point_to_token,
        )
        if "origin_coord" in point.keys():
            coarse_dict["origin_coord"] = torch_scatter.scatter_mean(
                point.origin_coord, point_to_token, dim=0, dim_size=T
            )
        # Propagate per-input-point classification labels (e.g. for
        # OnlineLinearProbe / PretrainEvaluator) by picking the first
        # input point in each token — matches PT-v3m2's `head_indices`
        # semantics (point_transformer_v3m2_sonata.py:498,526).
        if "segment_motif" in point.keys() or "segment" in point.keys():
            input_idx = torch.arange(point.feat.shape[0], device=point.feat.device)
            head_indices, _ = torch_scatter.scatter_min(
                input_idx, point_to_token, dim=0, dim_size=T
            )
            for label_key in ("segment_motif", "segment"):
                if label_key in point.keys():
                    coarse_dict[label_key] = point[label_key][head_indices]
        # Drop None-valued keys so addict-style Point doesn't store them.
        coarse_dict = {k: v for k, v in coarse_dict.items() if v is not None}
        return Point(coarse_dict)
