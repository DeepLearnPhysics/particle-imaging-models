"""
PoLAr-MAE Semantic Segmentation model for pimm.

Encodes all tokens (no masking), combines multi-scale features from
intermediate transformer layers, upsamples to per-point features via KNN,
and classifies with a Conv1d segmentation head.
"""

from __future__ import annotations

from math import sqrt
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn

from pimm.models.builder import MODELS
from pimm.models.losses import build_criteria
from pimm.models.modules import PointModel
from pimm.models.polarmae.data import batched_to_packed, packed_to_batched
from pimm.models.polarmae.layers import (
    LearnedPositionalEncoder,
    MaskedMiniPointNet,
    PointcloudGrouping,
    PointNetFeatureUpsampling,
    SegmentationHead,
    VIT_CONFIGS,
    make_transformer,
    masked_layer_norm,
)
from pimm.utils.logger import get_logger

logger = get_logger(__name__)

_SCALE = 768 * sqrt(3) / 2  # ≈ 665.1076

_TOKENIZER_PRESETS = {
    ("vit_small", 5): dict(
        num_groups=2048, context_length=512, group_max_points=32,
        group_radius=5 / _SCALE, group_upscale_points=256, overlap_factor=0.72,
    ),
    ("vit_small", 2.5): dict(
        num_groups=2048, context_length=1024, group_max_points=24,
        group_radius=2.5 / _SCALE, group_upscale_points=64, overlap_factor=0.75,
    ),
    ("vit_tiny", 5): dict(
        num_groups=2048, context_length=512, group_max_points=32,
        group_radius=5 / _SCALE, group_upscale_points=256, overlap_factor=0.72,
    ),
    ("vit_base", 5): dict(
        num_groups=2048, context_length=512, group_max_points=32,
        group_radius=5 / _SCALE, group_upscale_points=256, overlap_factor=0.72,
    ),
}


@MODELS.register_module("PoLArMAE-SemSeg")
class PoLArMAESemSeg(PointModel):
    """PoLAr-MAE semantic segmentation for pimm's DefaultTrainer."""

    def __init__(
        self,
        num_classes: int = 5,
        arch: Literal["vit_tiny", "vit_small", "vit_base"] = "vit_small",
        voxel_size: float = 5,
        num_channels: int = 4,
        seg_head_fetch_layers: List[int] = [3, 7, 11],
        seg_head_combination_method: Literal["mean", "concat"] = "mean",
        seg_head_dim: int = 384,
        seg_head_dropout: float = 0.5,
        freeze_encoder: bool = False,
        apply_encoder_postnorm: bool = True,
        condition_global_features: bool = False,
        upsampling_k: int = 5,
        upsampling_dim: Optional[int] = None,
        center: List[float] = [384.0, 384.0, 384.0],
        scale: float = 1.0 / _SCALE,
        criteria=None,
        transformer_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        transformer_kwargs = dict(transformer_kwargs or {})
        embed_dim = VIT_CONFIGS[arch]["embed_dim"]
        # If upsampling_dim is None, use embed_dim directly (no downcast)
        up_dim = upsampling_dim if upsampling_dim is not None else embed_dim

        self.seg_head_fetch_layers = seg_head_fetch_layers
        self.seg_head_combination_method = seg_head_combination_method
        self.apply_encoder_postnorm = apply_encoder_postnorm
        self.freeze_encoder = freeze_encoder
        self.condition_global_features = condition_global_features
        self.register_buffer("center", torch.tensor(center))
        self.scale = scale

        # --- Tokenizer ---
        tok_cfg = dict(_TOKENIZER_PRESETS.get(
            (arch, voxel_size), _TOKENIZER_PRESETS[("vit_small", 5)]
        ))
        if tokenizer_kwargs:
            tok_cfg.update(tokenizer_kwargs)

        self.grouping = PointcloudGrouping(reduction_method="fps", **tok_cfg)
        self.embedding = MaskedMiniPointNet(num_channels, embed_dim)
        self.pos_embed = LearnedPositionalEncoder(embed_dim)

        # --- Encoder ---
        self.encoder = make_transformer(arch, use_kv=False, **transformer_kwargs)

        # --- Downstream head ---
        if up_dim != embed_dim:
            self.point_downcast = nn.Linear(embed_dim, up_dim)
        else:
            self.point_downcast = nn.Identity()

        self.upsampler = PointNetFeatureUpsampling(
            in_channel=up_dim, mlp=[up_dim, up_dim], K=upsampling_k,
        )

        # Seg head input: up_dim + 2*up_dim (if global features) else up_dim
        seg_in = up_dim * 3 if condition_global_features else up_dim
        self.seg_head = SegmentationHead(
            in_channels=seg_in,
            seg_head_dim=seg_head_dim,
            seg_head_dropout=seg_head_dropout,
            num_classes=num_classes,
        )

        # --- Loss ---
        self.criteria = build_criteria(criteria)

        # --- Freeze encoder if requested ---
        if freeze_encoder:
            for module in [self.grouping, self.embedding, self.pos_embed, self.encoder]:
                module.requires_grad_(False)

        logger.info(
            f"PoLArMAE-SemSeg: arch={arch}, classes={num_classes}, "
            f"freeze={freeze_encoder}, fetch_layers={seg_head_fetch_layers}, "
            f"upsampling_dim={up_dim}, condition_global={condition_global_features}"
        )

    def _combine_intermediate_layers(
        self,
        hidden_states: List[torch.Tensor],
        mask: torch.Tensor,
        layers: List[int],
    ) -> torch.Tensor:
        """Normalize each layer's output and average across selected layers."""
        normed = [
            masked_layer_norm(hidden_states[i], hidden_states[i].shape[-1], mask)
            for i in layers
        ]
        return torch.stack(normed, dim=0).mean(0)

    def forward(self, data_dict):
        feat = data_dict["feat"]      # (N_total, C)
        offset = data_dict["offset"]  # (B,)

        # 1. Packed → padded
        points, lengths = packed_to_batched(feat, offset)

        # 2. Normalize coordinates
        points[..., :3] = (points[..., :3] - self.center) * self.scale

        # 3. Grouping (no masking — encode all tokens for downstream)
        g = self.grouping(points, lengths)
        groups, centers = g["groups"], g["centers"]
        emb_mask, point_mask_g = g["embedding_mask"], g["point_mask"]

        # 4. Embed all tokens
        with torch.amp.autocast(device_type=feat.device.type, dtype=torch.float32):
            flat_tok = self.embedding(
                groups[emb_mask], point_mask_g[emb_mask].unsqueeze(1),
            )
            tokens = groups.new_zeros(groups.shape[0], groups.shape[1], flat_tok.shape[-1])
            tokens[emb_mask] = flat_tok

        # 5. Positional encoding
        pos = self.pos_embed(centers)

        # 6. Encoder with hidden states
        use_hidden = len(self.seg_head_fetch_layers) > 0
        enc_out = self.encoder(
            tokens, pos, emb_mask,
            return_hidden_states=use_hidden,
            final_norm=self.apply_encoder_postnorm,
        )

        # 7. Combine intermediate layers
        if use_hidden:
            token_features = self._combine_intermediate_layers(
                enc_out.hidden_states, emb_mask, self.seg_head_fetch_layers,
            )
        else:
            token_features = enc_out.last_hidden_state

        # 8. Downcast + upsample to per-point features
        downcast = self.point_downcast(token_features)

        point_mask = torch.arange(
            points.shape[1], device=points.device,
        ).unsqueeze(0) < lengths.unsqueeze(1)
        emb_lengths = emb_mask.sum(dim=1)

        upsampled, _ = self.upsampler(
            points[..., :3], centers[..., :3], points[..., :3],
            downcast, lengths, emb_lengths, point_mask,
        )  # (B, N_max, up_dim)

        # 9. Optionally condition on global features
        if self.condition_global_features:
            B, N, _ = upsampled.shape
            bm = emb_mask.unsqueeze(-1).float()
            valid_count = bm.sum(dim=1, keepdim=True).clamp(min=1)
            # Masked mean
            global_mean = (token_features * bm).sum(dim=1) / valid_count.squeeze(1)
            # Masked max
            tf_masked = token_features.clone()
            tf_masked[~emb_mask] = float("-inf")
            global_max = tf_masked.max(dim=1).values
            # Apply downcast and concat
            global_feat = torch.cat([
                self.point_downcast(global_max),
                self.point_downcast(global_mean),
            ], dim=-1)  # (B, 2*up_dim)
            upsampled = torch.cat([
                upsampled,
                global_feat.unsqueeze(1).expand(-1, N, -1),
            ], dim=-1)  # (B, N, 3*up_dim)

        # 10. Segmentation head
        logits = self.seg_head(
            upsampled.transpose(1, 2), point_mask,
        ).transpose(1, 2)  # (B, N_max, num_classes)

        # 11. Batched → packed
        seg_logits, _ = batched_to_packed(logits, lengths)

        # 12. Loss
        num_classes = seg_logits.shape[-1]
        result = dict(seg_logits=seg_logits)
        if "segment" in data_dict:
            segment = data_dict["segment"]
            # Remap out-of-range labels to ignore_index (-1)
            if segment.max() >= num_classes:
                segment = segment.clone()
                segment[segment >= num_classes] = -1
            loss = self.criteria(seg_logits, segment)
            result["loss"] = loss
        if not self.training:  # the points are the original points
            result["offset"] = offset
            if "segment" in data_dict:
                result["segment"] = data_dict["segment"]
        return result
