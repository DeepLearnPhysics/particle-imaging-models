"""
Self-contained PoLAr-MAE for pimm.

Reimplements the PoLAr-MAE self-supervised model (Chamfer reconstruction +
energy infilling) natively within pimm, with no dependency on the external
PoLAr-MAE library.  Transforms (centering, scaling, rotation) are expected
to be applied in the dataset pipeline rather than inside the model.
"""

from __future__ import annotations

from math import sqrt
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pimm.models.builder import MODELS
from pimm.models.modules import PointModel
from pimm.models.polarmae.data import packed_to_batched
from pimm.models.polarmae.ops import chamfer_distance
from pimm.models.polarmae.layers import (
    LearnedPositionalEncoder,
    MaskedMiniPointNet,
    PointcloudGrouping,
    TransformerOutput,
    VariablePointcloudMasking,
    VIT_CONFIGS,
    make_transformer,
)
from pimm.utils.logger import get_logger

logger = get_logger(__name__)

# Default tokenizer configs keyed by (arch, voxel_size)
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
    ("vit_small", 25): dict(
        num_groups=256, context_length=128, group_max_points=128,
        group_radius=25 / _SCALE, group_upscale_points=2048, overlap_factor=0.72,
    ),
    ("vit_tiny", 5): dict(
        num_groups=2048, context_length=512, group_max_points=32,
        group_radius=5 / _SCALE, group_upscale_points=256, overlap_factor=0.72,
    ),
    ("vit_tiny", 2.5): dict(
        num_groups=2048, context_length=1024, group_max_points=24,
        group_radius=2.5 / _SCALE, group_upscale_points=64, overlap_factor=0.75,
    ),
    ("vit_base", 5): dict(
        num_groups=2048, context_length=512, group_max_points=32,
        group_radius=5 / _SCALE, group_upscale_points=256, overlap_factor=0.72,
    ),
}


@MODELS.register_module("PoLAr-MAE")
class PoLArMAE(PointModel):
    """Self-contained PoLAr-MAE for pimm's DefaultTrainer."""

    def __init__(
        self,
        arch: Literal["vit_tiny", "vit_small", "vit_base"] = "vit_small",
        num_channels: int = 4,
        voxel_size: float = 5,
        masking_ratio: float = 0.6,
        masking_type: Literal["rand"] = "rand",
        decoder_arch: Literal["vit_tiny", "vit_small", "vit_base"] = "vit_small",
        decoder_kwargs: Optional[dict] = None,
        mae_prediction: Literal["full", "pos"] = "full",
        loss_weights: Optional[Dict[str, float]] = None,
        transformer_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        decoder_use_kv: bool = True,
        patch_encoder_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        loss_weights = loss_weights or {"chamfer": 1.0, "energy": 1.0}
        transformer_kwargs = dict(transformer_kwargs or {})
        decoder_kwargs = dict(decoder_kwargs or {})

        embed_dim = VIT_CONFIGS[arch]["embed_dim"]
        self.mae_channels = 4 if mae_prediction == "full" else 3
        self.loss_weights = loss_weights

        # --- Tokenizer (grouping + embedding + masking) ---
        tok_cfg = dict(_TOKENIZER_PRESETS.get((arch, voxel_size), _TOKENIZER_PRESETS[("vit_small", 5)]))
        if tokenizer_kwargs:
            tok_cfg.update(tokenizer_kwargs)
        group_max_points = tok_cfg["group_max_points"]

        self.grouping = PointcloudGrouping(reduction_method="fps", **tok_cfg)
        self.embedding = MaskedMiniPointNet(num_channels, embed_dim)
        self.masking = VariablePointcloudMasking(ratio=masking_ratio)
        self.pos_embed = LearnedPositionalEncoder(embed_dim)

        # --- Encoder & Decoder ---
        self.encoder = make_transformer(arch, use_kv=False, **transformer_kwargs)
        self.decoder_use_kv = decoder_use_kv
        self.decoder = make_transformer(
            decoder_arch, use_kv=decoder_use_kv, **decoder_kwargs.get("transformer_kwargs", {}),
        )

        # --- Loss heads ---
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02, a=-0.02, b=0.02)

        self.increase_dim = nn.Conv1d(embed_dim, self.mae_channels * group_max_points, 1)

        # Equivariant patch encoder for the energy-infilling head. Defaults match
        # pimm-native; `patch_encoder_kwargs` (feature_dim/hidden1/hidden2/
        # pos_enc_style) lets a config reproduce the original PoLAr-MAE checkpoint.
        pe_cfg = dict(channels=3, feature_dim=96, hidden1=64, hidden2=64,
                      equivariant=True, pos_enc_style="mlp")
        if patch_encoder_kwargs:
            pe_cfg.update(patch_encoder_kwargs)
        self.equivariant_patch_encoder = MaskedMiniPointNet(**pe_cfg)
        self.energy_decoder = nn.Conv1d(embed_dim + pe_cfg["feature_dim"], group_max_points, 1)

        logger.info(
            f"PoLAr-MAE-Native: arch={arch}, voxel={voxel_size}, "
            f"mask={masking_ratio}, pred={mae_prediction}, weights={loss_weights}"
        )

    # ---------------------------------------------------------------
    def forward(self, data_dict):
        feat = data_dict["feat"]      # (N_total, C)
        offset = data_dict["offset"]  # (B,)

        # 1. Packed → padded
        points, lengths = packed_to_batched(feat, offset)

        # 2. Grouping
        g = self.grouping(points, lengths)
        groups, centers = g["groups"], g["centers"]
        emb_mask, point_mask = g["embedding_mask"], g["point_mask"]

        # 3. Masking
        m_idx, m_mask, um_idx, um_mask = self.masking(emb_mask.sum(-1))

        _gather = lambda x, idx: torch.gather(
            x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        _lgather = lambda x, idx: torch.gather(
            x, 1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3]))

        # Unmasked tokens
        um_groups = _lgather(groups, um_idx)
        um_pmask = _gather(point_mask, um_idx)
        um_emb_mask = um_mask

        # Embed only valid unmasked tokens
        with torch.amp.autocast(device_type=feat.device.type, dtype=torch.float32):
            flat_tok = self.embedding(um_groups[um_emb_mask], um_pmask[um_emb_mask].unsqueeze(1))
            tokens = um_groups.new_zeros(um_groups.shape[0], um_groups.shape[1], flat_tok.shape[-1])
            tokens[um_emb_mask] = flat_tok

        um_centers = _gather(centers, um_idx)
        um_pos = self.pos_embed(um_centers)

        # Masked metadata
        m_centers = _gather(centers, m_idx)
        m_pos = self.pos_embed(m_centers)
        m_groups = _lgather(groups, m_idx)
        m_pmask = _gather(point_mask, m_idx) * m_mask.unsqueeze(-1)

        # 4. Encode unmasked tokens
        enc_out = self.encoder(tokens, um_pos, um_mask).last_hidden_state

        # 5. Decode masked tokens
        tok_m = self.mask_token.expand(m_mask.shape[0], m_mask.shape[1], -1)
        if self.decoder_use_kv:  # mask tokens cross-attend to the encoded tokens
            dec_out = self.decoder(tok_m, m_pos, m_mask, kv=enc_out, pos_kv=um_pos, kv_mask=um_mask).last_hidden_state
        else:  # as in PoLAr-MAE: encoded and mask tokens are decoded together with self-attention
            visible = enc_out.shape[1]
            dec_out = self.decoder(
                torch.cat([enc_out, tok_m], 1), torch.cat([um_pos, m_pos], 1), torch.cat([um_mask, m_mask], 1)
            ).last_hidden_state[:, visible:]

        # Flatten to valid masked tokens only
        masked_output = dec_out[m_mask]
        masked_groups = m_groups[m_mask]
        flat_pmask = m_pmask[m_mask]
        point_lengths = flat_pmask.sum(-1)

        # 6. Chamfer loss
        with torch.amp.autocast(device_type=feat.device.type, dtype=torch.float32):
            up = self.increase_dim(masked_output.unsqueeze(-1)).squeeze(-1)
            up = up.reshape(up.shape[0], -1, self.mae_channels)
            chamfer_loss = chamfer_distance(
                up.float(), masked_groups[..., :self.mae_channels].float(),
                x_lengths=point_lengths, y_lengths=point_lengths,
            )

        # 7. Energy loss
        pmask_1 = flat_pmask.unsqueeze(1)
        with torch.amp.autocast(device_type=feat.device.type, dtype=torch.float32):
            eq_enc = self.equivariant_patch_encoder(masked_groups[..., :3], pmask_1)
            dec_inp = torch.cat([eq_enc, masked_output], dim=1)
            energy_pred = self.energy_decoder(dec_inp.unsqueeze(-1)).squeeze(-1)
            energy_loss = F.mse_loss(
                energy_pred[flat_pmask].float(),
                masked_groups[flat_pmask][..., -1].float(),
            )

        loss = sum(self.loss_weights.get(k, 1.0) * v
                   for k, v in [("chamfer", chamfer_loss), ("energy", energy_loss)])

        return {
            "loss": loss,
            "chamfer_loss": chamfer_loss,
            "energy_loss": energy_loss,
            "mean_points": lengths.float().mean(),
            "mean_groups": emb_mask.sum(-1).float().mean(),
        }
