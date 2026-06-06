"""Volt-MAE v1m2: minimal masked point-set reconstruction baseline.

Gate-2 implementation for VoltMAE-PointSet. This keeps the v1m1 sparse-conv
Volt tokenizer, visible-token encoder, full-token MAE decoder, and
PretrainEvaluator-compatible ``encode`` path, but replaces dense sub-voxel
occupancy reconstruction with a fixed-capacity point-set head. Each masked
token predicts ``K`` local ``(xyz, charge)`` slots and the loss compares the
first ``M_t`` predicted slots against that token's ragged target point set.

Gate 3 adds an opt-in ``mask_mode="pre_tokenizer"`` path that removes masked
points before sparse CNN tokenization while preserving the same point-set head
and first-M Chamfer objective.
"""

from __future__ import annotations

from typing import Optional

import spconv.pytorch as spconv
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from pimm.models.builder import MODELS
from pimm.models.modules import PointModel
from pimm.models.utils.misc import offset2batch
from pimm.models.voltmae.layers import (
    Block,
    PointSetPredictionHead,
    RoPE,
    _pack_indices,
    build_point_to_token,
    build_pointset_targets,
    pointset_chamfer_loss,
    random_token_mask,
    sort_tokens_by_batch,
)
from pimm.utils.logger import get_logger

logger = get_logger(__name__)


@MODELS.register_module("Volt-MAE-v1m2")
class VoltMAEPointSet(PointModel):
    """Volt backbone + MAE pretext task on local point sets."""

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 384,
        enc_depth: int = 12,
        dec_depth: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = None,
        qk_norm: bool = True,
        drop_path: float = 0.3,
        stride: int = 5,
        kernel_size: int = 5,
        mask_ratio: float = 0.6,
        mask_mode: str = "post_tokenizer",
        remove_masked_points_before_sparse_tokenizer: Optional[bool] = None,
        increase_drop_path: bool = True,
        energy_key: str = "energy",
        rope_max_grid_size: tuple = (1024, 1024, 1024),
        rope_freq_split: tuple = (5, 5, 6),
        recon_target: str = "pointset",
        points_per_token: int = 64,
        pointset_num_points: Optional[int] = None,
        pointset_hidden_dim: Optional[int] = None,
        pointset_xyz_range: float = 1.1,
        pointset_charge_weight: float = 0.1,
        overflow_policy: str = "first",
    ):
        super().__init__()
        if recon_target != "pointset":
            raise ValueError(
                "Volt-MAE-v1m2 is the Gate-2 point-set baseline; "
                f"expected recon_target='pointset', got {recon_target!r}"
            )
        if remove_masked_points_before_sparse_tokenizer is not None:
            implied = "pre_tokenizer" if remove_masked_points_before_sparse_tokenizer else "post_tokenizer"
            if mask_mode != "post_tokenizer" and mask_mode != implied:
                raise ValueError(
                    "remove_masked_points_before_sparse_tokenizer conflicts with "
                    f"mask_mode={mask_mode!r}"
                )
            mask_mode = implied
        if mask_mode not in {"post_tokenizer", "pre_tokenizer"}:
            raise ValueError(
                "mask_mode must be 'post_tokenizer' or 'pre_tokenizer', "
                f"got {mask_mode!r}"
            )

        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

        if pointset_num_points is not None:
            points_per_token = pointset_num_points

        self.stride = stride
        self.kernel_size = kernel_size
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.energy_key = energy_key
        self.points_per_token = int(points_per_token)
        self.pointset_charge_weight = float(pointset_charge_weight)
        self.overflow_policy = overflow_policy

        self.tokenizer = spconv.SparseConv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
            indice_key="embedding",
        )

        total_depth = enc_depth
        if increase_drop_path:
            enc_dp = torch.linspace(0, drop_path, total_depth).tolist()
        else:
            enc_dp = [drop_path] * total_depth

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    drop_path=enc_dp[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for i in range(enc_depth)
            ]
        )

        self.decoder_blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    drop_path=0.0,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(dec_depth)
            ]
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
        self.recon_head = PointSetPredictionHead(
            in_dim=embed_dim,
            num_points=self.points_per_token,
            charge_dim=1,
            xyz_range=pointset_xyz_range,
            hidden_dim=embed_dim if pointset_hidden_dim is None else pointset_hidden_dim,
        )

        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        trunc_normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

        logger.info(
            f"Volt-MAE-v1m2 PointSet: in_channels={in_channels}, "
            f"embed_dim={embed_dim}, enc_depth={enc_depth}, dec_depth={dec_depth}, "
            f"stride={stride}, mask_ratio={mask_ratio}, mask_mode={mask_mode}, "
            f"K={self.points_per_token}, charge_weight={self.pointset_charge_weight}, "
            f"overflow={overflow_policy}"
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

    def _make_sparse_tensor(
        self,
        grid_coord: torch.Tensor,
        feat: torch.Tensor,
        batch: torch.Tensor,
    ):
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        indices = torch.cat(
            [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
        ).contiguous()
        return spconv.SparseConvTensor(
            features=feat,
            indices=indices,
            spatial_shape=sparse_shape,
            batch_size=int(batch[-1].item()) + 1,
        )

    def _occupied_token_indices(
        self,
        grid_coord: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        parent = grid_coord.long() // self.stride
        rows = torch.cat([batch.long().unsqueeze(1), parent], dim=1)
        if rows.numel() == 0:
            return rows
        shape_hash = int(parent.max().item()) + 2
        key = _pack_indices(rows[:, 0], rows[:, 1:], shape_hash)
        order = torch.argsort(key)
        rows = rows.index_select(0, order)
        key = key.index_select(0, order)
        keep = torch.ones(rows.shape[0], dtype=torch.bool, device=rows.device)
        keep[1:] = key[1:] != key[:-1]
        return rows[keep]

    @staticmethod
    def _map_token_indices_to_full(
        token_indices: torch.Tensor,
        full_token_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_indices.numel() == 0:
            ids = torch.empty(0, dtype=torch.long, device=token_indices.device)
            ok = torch.empty(0, dtype=torch.bool, device=token_indices.device)
            return ids, ok
        max_coord = max(
            token_indices[:, 1:].max().item() if token_indices.numel() else 0,
            full_token_indices[:, 1:].max().item() if full_token_indices.numel() else 0,
        )
        shape_hash = int(max_coord) + 2
        token_hash = _pack_indices(token_indices[:, 0], token_indices[:, 1:], shape_hash)
        full_hash = _pack_indices(
            full_token_indices[:, 0], full_token_indices[:, 1:], shape_hash
        )
        sorted_hash, sorted_ids = torch.sort(full_hash)
        pos = torch.searchsorted(sorted_hash, token_hash)
        pos_clamped = pos.clamp(max=max(sorted_hash.numel() - 1, 0))
        matched_hash = sorted_hash.index_select(0, pos_clamped)
        ok = (pos < sorted_hash.numel()) & (matched_hash == token_hash)
        ids = sorted_ids.index_select(0, pos_clamped)
        ids = torch.where(ok, ids, torch.full_like(ids, -1))
        return ids, ok

    @staticmethod
    def _resolve_mask(
        token_batch_ids: torch.Tensor,
        mask_ratio: float,
        fixed_masked_ids: Optional[torch.Tensor] = None,
    ):
        if fixed_masked_ids is None:
            return random_token_mask(token_batch_ids, mask_ratio)

        device = token_batch_ids.device
        fixed_masked_ids = fixed_masked_ids.to(device=device, dtype=torch.long).flatten()
        if fixed_masked_ids.numel() > 0:
            fixed_masked_ids = torch.unique(fixed_masked_ids, sorted=True)
        T = token_batch_ids.numel()
        if bool(((fixed_masked_ids < 0) | (fixed_masked_ids >= T)).any()):
            raise ValueError("fixed_masked_ids contains an out-of-range token id")

        mask = torch.zeros(T, dtype=torch.bool, device=device)
        mask.index_fill_(0, fixed_masked_ids, True)
        ids_masked = torch.nonzero(mask, as_tuple=False).squeeze(1)
        ids_kept = torch.nonzero(~mask, as_tuple=False).squeeze(1)
        batch_counts = torch.bincount(token_batch_ids)
        B = batch_counts.numel()
        cu_kept = torch.zeros(B + 1, dtype=torch.int32, device=device)
        if ids_kept.numel() > 0:
            cu_kept[1:] = torch.cumsum(
                torch.bincount(token_batch_ids[ids_kept], minlength=B).to(torch.int32),
                dim=0,
            )
        cu_full = torch.zeros(B + 1, dtype=torch.int32, device=device)
        cu_full[1:] = torch.cumsum(batch_counts.to(torch.int32), dim=0)
        max_kept = int((cu_kept[1:] - cu_kept[:-1]).max().item()) if B else 0
        max_full = int(batch_counts.max().item()) if B else 0
        return ids_kept, ids_masked, cu_kept, max_kept, cu_full, max_full

    @staticmethod
    def _select_masked_pointset_targets(
        targets: dict[str, torch.Tensor],
        ids_masked: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert full-token ragged targets to masked-token-local ragged targets."""
        num_masked = ids_masked.numel()
        full_counts = targets["target_counts"]
        device = full_counts.device

        masked_slot = torch.full(
            (full_counts.shape[0],), -1, dtype=torch.long, device=device
        )
        masked_slot.index_copy_(
            0,
            ids_masked,
            torch.arange(num_masked, dtype=torch.long, device=device),
        )

        target_patch = targets["target_patch_index"]
        if target_patch.numel() == 0:
            counts = torch.zeros(num_masked, dtype=torch.long, device=device)
            offsets = torch.zeros(num_masked + 1, dtype=torch.long, device=device)
            points = targets["target_xyz_local"].new_zeros((0, 4))
            return points, offsets, counts

        local_patch = masked_slot.index_select(0, target_patch)
        keep = local_patch >= 0
        points = torch.cat(
            [targets["target_xyz_local"], targets["target_charge"]], dim=-1
        ).index_select(0, torch.nonzero(keep, as_tuple=False).squeeze(1))
        local_patch = local_patch[keep]

        counts = torch.bincount(local_patch, minlength=num_masked).to(torch.long)
        offsets = torch.zeros(num_masked + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(counts, dim=0)
        return points, offsets, counts

    def _build_debug(
        self,
        ids_masked: torch.Tensor,
        encoder_input_patch_ids: torch.Tensor,
        encoder_visible_token_patch_ids: torch.Tensor,
        full_point_count: int,
    ) -> dict[str, torch.Tensor]:
        device = ids_masked.device
        return {
            "masked_patch_ids": ids_masked.detach(),
            "encoder_input_patch_ids": encoder_input_patch_ids.detach(),
            "encoder_visible_token_patch_ids": encoder_visible_token_patch_ids.detach(),
            "full_point_count": torch.tensor(float(full_point_count), device=device),
            "encoder_visible_point_count": torch.tensor(
                float(encoder_input_patch_ids.numel()), device=device
            ),
        }

    def _predict_pointset(self, decoded_tokens: torch.Tensor):
        return self.recon_head(decoded_tokens)

    def _prediction_tensor(self, pred_output):
        return pred_output

    def _compute_pointset_loss(
        self,
        pred_output,
        target_points: torch.Tensor,
        target_offsets: torch.Tensor,
        target_counts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return pointset_chamfer_loss(
            pred_output,
            target_points,
            target_offsets,
            target_counts,
            charge_weight=self.pointset_charge_weight,
        )

    def _add_prediction_outputs(self, output: dict, pred_output) -> None:
        output["pred_pointset"] = self._prediction_tensor(pred_output)

    def _forward_post_tokenizer(
        self,
        grid_coord: torch.Tensor,
        feat: torch.Tensor,
        batch: torch.Tensor,
        energy: torch.Tensor,
        fixed_masked_ids: Optional[torch.Tensor],
    ):
        x = self._make_sparse_tensor(grid_coord, feat, batch)
        x = self.tokenizer(x)
        token_features, full_token_indices = sort_tokens_by_batch(
            x.features, x.indices.long()
        )
        T = token_features.shape[0]
        point_to_token = build_point_to_token(
            grid_coord.long(), batch.long(), full_token_indices, self.stride
        )
        token_batch_ids = full_token_indices[:, 0].to(torch.int64)
        (
            ids_kept,
            ids_masked,
            cu_kept,
            max_kept,
            cu_full,
            max_full,
        ) = self._resolve_mask(token_batch_ids, self.mask_ratio, fixed_masked_ids)

        encoder_input_patch_ids = point_to_token
        encoder_visible_token_patch_ids = ids_kept
        enc_tokens = token_features.index_select(0, ids_kept)
        enc_full_ids = ids_kept
        return (
            full_token_indices,
            token_features,
            point_to_token,
            ids_kept,
            ids_masked,
            enc_tokens,
            enc_full_ids,
            cu_kept,
            max_kept,
            cu_full,
            max_full,
            encoder_input_patch_ids,
            encoder_visible_token_patch_ids,
        )

    def _forward_pre_tokenizer(
        self,
        grid_coord: torch.Tensor,
        feat: torch.Tensor,
        batch: torch.Tensor,
        energy: torch.Tensor,
        fixed_masked_ids: Optional[torch.Tensor],
    ):
        full_token_indices = self._occupied_token_indices(grid_coord, batch)
        T = full_token_indices.shape[0]
        point_to_token = build_point_to_token(
            grid_coord.long(), batch.long(), full_token_indices, self.stride
        )
        token_batch_ids = full_token_indices[:, 0].to(torch.int64)
        (
            ids_kept,
            ids_masked,
            _cu_kept_unused,
            _max_kept_unused,
            cu_full,
            max_full,
        ) = self._resolve_mask(token_batch_ids, self.mask_ratio, fixed_masked_ids)

        token_is_masked = torch.zeros(T, dtype=torch.bool, device=grid_coord.device)
        token_is_masked.index_fill_(0, ids_masked, True)
        point_is_masked = token_is_masked.index_select(0, point_to_token)
        visible_point_mask = ~point_is_masked
        if not bool(visible_point_mask.any()):
            raise RuntimeError(
                "mask_mode='pre_tokenizer' removed every point before the tokenizer; "
                "use a lower mask ratio or larger batch for this smoke."
            )

        grid_visible = grid_coord.index_select(0, torch.nonzero(visible_point_mask, as_tuple=False).squeeze(1))
        feat_visible = feat.index_select(0, torch.nonzero(visible_point_mask, as_tuple=False).squeeze(1))
        batch_visible = batch.index_select(0, torch.nonzero(visible_point_mask, as_tuple=False).squeeze(1))
        encoder_input_patch_ids = point_to_token.index_select(
            0, torch.nonzero(visible_point_mask, as_tuple=False).squeeze(1)
        )

        x = self._make_sparse_tensor(grid_visible, feat_visible, batch_visible)
        x = self.tokenizer(x)
        token_features, token_indices_visible = sort_tokens_by_batch(
            x.features, x.indices.long()
        )
        visible_full_ids, matched_full = self._map_token_indices_to_full(
            token_indices_visible,
            full_token_indices,
        )
        keep_token = matched_full & (~token_is_masked.index_select(0, visible_full_ids.clamp(min=0)))
        keep_idx = torch.nonzero(keep_token, as_tuple=False).squeeze(1)
        token_features = token_features.index_select(0, keep_idx)
        token_indices_visible = token_indices_visible.index_select(0, keep_idx)
        visible_full_ids = visible_full_ids.index_select(0, keep_idx)
        encoder_visible_token_patch_ids = visible_full_ids

        cu_kept, max_kept = self._compute_cu_seqlens(token_indices_visible[:, 0].to(torch.long))
        return (
            full_token_indices,
            token_features,
            point_to_token,
            ids_kept,
            ids_masked,
            token_features,
            visible_full_ids,
            cu_kept,
            max_kept,
            cu_full,
            max_full,
            encoder_input_patch_ids,
            encoder_visible_token_patch_ids,
        )

    def forward(
        self,
        data_dict,
        fixed_masked_ids: Optional[torch.Tensor] = None,
        fixed_masked_patch_ids: Optional[torch.Tensor] = None,
        return_debug: bool = False,
        return_preds: bool = False,
    ):
        grid_coord = data_dict["grid_coord"]
        feat = data_dict["feat"]
        if "batch" in data_dict:
            batch = data_dict["batch"]
        else:
            batch = offset2batch(data_dict["offset"])
        energy = data_dict[self.energy_key]

        if fixed_masked_ids is None:
            fixed_masked_ids = fixed_masked_patch_ids
        if fixed_masked_ids is None and "fixed_masked_ids" in data_dict:
            fixed_masked_ids = data_dict["fixed_masked_ids"]
        if fixed_masked_ids is None and "fixed_masked_patch_ids" in data_dict:
            fixed_masked_ids = data_dict["fixed_masked_patch_ids"]

        if self.mask_mode == "post_tokenizer":
            pack = self._forward_post_tokenizer(
                grid_coord, feat, batch, energy, fixed_masked_ids
            )
        else:
            pack = self._forward_pre_tokenizer(
                grid_coord, feat, batch, energy, fixed_masked_ids
            )

        (
            full_token_indices,
            token_features_for_zero,
            point_to_token,
            ids_kept,
            ids_masked,
            enc_tokens,
            enc_full_ids,
            cu_kept,
            max_kept,
            cu_full,
            max_full,
            encoder_input_patch_ids,
            encoder_visible_token_patch_ids,
        ) = pack
        T = full_token_indices.shape[0]

        pointset_targets = build_pointset_targets(
            point_to_token,
            grid_coord.long(),
            full_token_indices,
            energy,
            self.stride,
            T,
            max_points_per_token=self.points_per_token,
            overflow_policy=self.overflow_policy,
        )
        freqs_cis_full = self.pos_enc.compute_axial_cis_efficient(full_token_indices[:, 1:])

        if ids_masked.numel() == 0:
            loss = token_features_for_zero.sum() * 0.0
            zero = loss.detach()
            B = int(batch[-1].item()) + 1
            output = {
                "loss": loss,
                "loss_pointset_xyz": zero,
                "loss_pointset_charge": zero,
                "chamfer_loss": zero,
                "charge_loss": zero,
                "num_supervised_patches": token_features_for_zero.new_tensor(0.0),
                "full_point_count": token_features_for_zero.new_tensor(float(grid_coord.shape[0])),
                "encoder_visible_point_count": token_features_for_zero.new_tensor(float(encoder_input_patch_ids.numel())),
                "masked_target_point_count": token_features_for_zero.new_tensor(0.0),
                "masked_point_count": token_features_for_zero.new_tensor(0.0),
                "masked_patch_count": token_features_for_zero.new_tensor(0.0),
                "encoder_visible_token_count": token_features_for_zero.new_tensor(float(enc_full_ids.numel())),
                "mean_tokens": torch.tensor(float(T) / max(1, B), device=feat.device),
                "mean_masked": torch.tensor(0.0, device=feat.device),
                "mean_points_per_masked": torch.tensor(0.0, device=feat.device),
                "mean_target_points_per_patch": torch.tensor(0.0, device=feat.device),
                "max_target_points_per_patch": torch.tensor(0.0, device=feat.device),
                "overflow_patch_fraction": pointset_targets["overflow_patch_fraction"].detach(),
                "overflow_point_fraction": pointset_targets["overflow_point_fraction"].detach(),
                "overflow_charge_fraction": pointset_targets["overflow_charge_fraction"].detach(),
            }
            if return_debug:
                output["debug"] = self._build_debug(
                    ids_masked,
                    encoder_input_patch_ids,
                    encoder_visible_token_patch_ids,
                    grid_coord.shape[0],
                )
            return output

        enc_freqs = freqs_cis_full[:, enc_full_ids]
        for blk in self.blocks:
            enc_tokens = blk(enc_tokens, enc_freqs, cu_kept, max_kept)

        full_tokens = torch.zeros(
            T, enc_tokens.shape[1], dtype=enc_tokens.dtype, device=enc_tokens.device
        )
        full_tokens.index_copy_(0, enc_full_ids, enc_tokens)
        mask_tok = self.mask_token.to(full_tokens.dtype).expand(ids_masked.numel(), -1)
        full_tokens.index_copy_(0, ids_masked, mask_tok)

        dec_tokens = full_tokens
        for blk in self.decoder_blocks:
            dec_tokens = blk(dec_tokens, freqs_cis_full, cu_full, max_full)

        pred_output = self._predict_pointset(dec_tokens.index_select(0, ids_masked))
        pred_m = self._prediction_tensor(pred_output)
        target_points_m, target_offsets_m, target_counts_m = self._select_masked_pointset_targets(
            pointset_targets,
            ids_masked,
        )
        loss_dict = self._compute_pointset_loss(
            pred_output,
            target_points_m,
            target_offsets_m,
            target_counts_m,
        )
        loss = loss_dict["loss"]

        B = int(batch[-1].item()) + 1
        M = max(1, ids_masked.numel())
        mean_target_points = target_counts_m.float().mean() if target_counts_m.numel() else pred_m.new_zeros(())
        max_target_points = target_counts_m.max().float() if target_counts_m.numel() else pred_m.new_zeros(())

        output = {
            "loss": loss,
            "loss_pointset_xyz": loss_dict["loss_pointset_xyz"],
            "loss_pointset_charge": loss_dict["loss_pointset_charge"],
            "chamfer_loss": loss_dict["loss_pointset_xyz"],
            "charge_loss": loss_dict["loss_pointset_charge"],
            "num_supervised_patches": loss_dict["num_supervised_patches"],
            "full_point_count": pred_m.new_tensor(float(grid_coord.shape[0])),
            "encoder_visible_point_count": pred_m.new_tensor(float(encoder_input_patch_ids.numel())),
            "masked_target_point_count": target_counts_m.sum().to(torch.float32).detach(),
            "masked_point_count": target_counts_m.sum().to(torch.float32).detach(),
            "masked_patch_count": pred_m.new_tensor(float(ids_masked.numel())),
            "encoder_visible_token_count": pred_m.new_tensor(float(enc_full_ids.numel())),
            "mean_tokens": torch.tensor(float(T) / max(1, B), device=feat.device),
            "mean_masked": torch.tensor(float(ids_masked.numel()) / max(1, B), device=feat.device),
            "mean_points_per_masked": (target_counts_m.sum().float() / M).detach(),
            "mean_target_points_per_patch": mean_target_points.detach(),
            "max_target_points_per_patch": max_target_points.detach(),
            "overflow_patch_fraction": pointset_targets["overflow_patch_fraction"].detach(),
            "overflow_point_fraction": pointset_targets["overflow_point_fraction"].detach(),
            "overflow_charge_fraction": pointset_targets["overflow_charge_fraction"].detach(),
        }
        if return_debug:
            output["debug"] = self._build_debug(
                ids_masked,
                encoder_input_patch_ids,
                encoder_visible_token_patch_ids,
                grid_coord.shape[0],
            )
        for key, value in loss_dict.items():
            if key != "loss" and key not in output:
                output[key] = value
        if return_preds:
            self._add_prediction_outputs(output, pred_output)
        return output

    @torch.no_grad()
    def encode(self, data_dict):
        """Per-point encoder features for downstream linear probing."""
        grid_coord = data_dict["grid_coord"]
        feat = data_dict["feat"]
        if "batch" in data_dict:
            batch = data_dict["batch"]
        else:
            batch = offset2batch(data_dict["offset"])

        x = self._make_sparse_tensor(grid_coord, feat, batch)
        x = self.tokenizer(x)
        tokens, token_indices = sort_tokens_by_batch(
            x.features, x.indices.long()
        )

        cu_full, max_full = self._compute_cu_seqlens(token_indices[:, 0])
        freqs = self.pos_enc.compute_axial_cis_efficient(token_indices[:, 1:])
        for blk in self.blocks:
            tokens = blk(tokens, freqs, cu_full, max_full)

        p2t = build_point_to_token(
            grid_coord.long(), batch.long(), token_indices, self.stride
        )
        return tokens.index_select(0, p2t)


# Backward-compatible import name for direct module imports.
VoltMAE = VoltMAEPointSet
