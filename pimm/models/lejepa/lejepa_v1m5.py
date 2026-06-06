"""
LeJEPA v1m5 — LeJEPA + masked prediction for 3D point clouds.

Adapted from: Balestriero & LeCun, "LeJEPA: Provable and Scalable
Self-Supervised Learning Without the Heuristics", arXiv:2511.08544 (2025).

This is basically Panda/Sonata, except:
- we remove EMA / stop-grad, using SIGReg on projected (dim=32) featuresinstead to prevent collapse.
- we perform MSE between projected features instead of x-entropy on prototypes.
"""

from itertools import chain

import torch
import torch.nn as nn
import torch.distributed as dist
import torch_scatter
import pointops

from pimm.models.utils.structure import Point
from pimm.models.builder import MODELS, build_model
from pimm.models.modules import PointModel
from pimm.models.utils import offset2batch, offset2bincount, batch2offset
from pimm.utils.comm import get_world_size
from pimm.utils.scheduler import CosineScheduler
from pimm.models.utils.stats import compute_enc_feat_stats


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization."""

    def __init__(self, knots=17, num_slices=256):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj, batch):
        # proj (N_total, proj_dim), batch (N_total,)
        A = torch.randn(proj.size(-1), self.num_slices, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))

        x_t = (proj @ A).unsqueeze(-1) * self.t  # (N, slices, knots)
        cos_mean = torch_scatter.scatter_mean(
            x_t.cos(), batch, dim=0
        )  # (V, slices, knots)
        sin_mean = torch_scatter.scatter_mean(
            x_t.sin(), batch, dim=0
        )  # (V, slices, knots)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        counts = torch_scatter.scatter_add(
            torch.ones(proj.shape[0], device=proj.device), batch
        )  # (V,)
        statistic = (err @ self.weights) * counts.unsqueeze(-1)
        return statistic.mean()


@MODELS.register_module("LeJEPA-v1m5")
class LeJEPAv5(PointModel):
    def __init__(
        self,
        backbone,
        head_in_channels=1088,
        proj_hidden_channels=(2048, 2048),
        proj_dim=16,
        lamb=0.02,
        mask_weight=2 / 8,
        roll_mask_weight=2 / 8,
        unmask_weight=4 / 8,
        num_global_view=2,
        num_local_view=6,
        up_cast_level=2,
        sigreg_knots=17,
        sigreg_num_slices=256,
        # Masking schedule (set start==base to disable warmup)
        mask_ratio_start=0.3,
        mask_ratio_base=0.7,
        mask_ratio_warmup_ratio=0.05,
        mask_size_start=0.01,
        mask_size_base=0.075,
        mask_size_warmup_ratio=0.05,
        mask_jitter=None,
        mask_jitter_start=0.0,
        mask_jitter_base=0.0005,
        mask_jitter_warmup_ratio=0.05,
        match_max_r=0.002,
        tie_heads=False,
    ):
        super().__init__()
        self.lamb = lamb
        self.mask_weight = mask_weight
        self.roll_mask_weight = roll_mask_weight
        self.unmask_weight = unmask_weight
        self.num_global_view = num_global_view
        self.num_local_view = num_local_view
        self.up_cast_level = up_cast_level
        self.match_max_r = match_max_r
        self.tie_heads = tie_heads

        # Masking state (updated by schedulers)
        self.mask_ratio = mask_ratio_start
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_base = mask_ratio_base
        self.mask_ratio_warmup_ratio = mask_ratio_warmup_ratio

        self.mask_size = mask_size_start
        self.mask_size_start = mask_size_start
        self.mask_size_base = mask_size_base
        self.mask_size_warmup_ratio = mask_size_warmup_ratio

        self.mask_jitter = mask_jitter
        if mask_jitter is None:
            self.mask_jitter_start = mask_jitter_start
            self.mask_jitter_base = mask_jitter_base
            self.mask_jitter_warmup_ratio = mask_jitter_warmup_ratio

        self.backbone = build_model(backbone)

        def _build_proj():
            layers = []
            in_dim = head_in_channels
            for h in proj_hidden_channels:
                layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
                in_dim = h
            layers.append(nn.Linear(in_dim, proj_dim))
            return nn.Sequential(*layers)

        if tie_heads:
            self.mask_head = self.unmask_head = _build_proj()
        else:
            self.mask_head = _build_proj()
            self.unmask_head = _build_proj()

        self.sigreg = SIGReg(knots=sigreg_knots, num_slices=sigreg_num_slices)

    # ------------------------------------------------------------------
    # Lifecycle hooks (masking schedule)
    # ------------------------------------------------------------------

    def before_train(self):
        total_steps = self.trainer.cfg.scheduler.total_steps
        curr_step = getattr(self.trainer, "global_step", 0) or (
            self.trainer.start_epoch * len(self.trainer.train_loader)
        )

        self.mask_ratio_scheduler = CosineScheduler(
            start_value=self.mask_ratio_start,
            base_value=self.mask_ratio_base,
            final_value=self.mask_ratio_base,
            warmup_iters=int(total_steps * self.mask_ratio_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_ratio_scheduler.iter = curr_step

        self.mask_size_scheduler = CosineScheduler(
            start_value=self.mask_size_start,
            base_value=self.mask_size_base,
            final_value=self.mask_size_base,
            warmup_iters=int(total_steps * self.mask_size_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_size_scheduler.iter = curr_step

        if self.mask_jitter is None:
            self.mask_jitter_scheduler = CosineScheduler(
                start_value=self.mask_jitter_start,
                base_value=self.mask_jitter_start,
                final_value=self.mask_jitter_base,
                warmup_iters=int(
                    total_steps * self.mask_jitter_warmup_ratio
                ),
                total_iters=total_steps,
            )
            self.mask_jitter_scheduler.iter = curr_step

    def before_step(self):
        self.mask_ratio = self.mask_ratio_scheduler.step()
        self.mask_size = self.mask_size_scheduler.step()
        if hasattr(self, "mask_jitter_scheduler"):
            self.mask_jitter = self.mask_jitter_scheduler.step()

        if self.trainer.writer is not None:
            step = self.mask_ratio_scheduler.iter
            self.trainer.writer.add_scalar(
                "params/mask_ratio", self.mask_ratio, step
            )
            self.trainer.writer.add_scalar(
                "params/mask_size", self.mask_size, step
            )
            if hasattr(self, "mask_jitter_scheduler"):
                self.trainer.writer.add_scalar(
                    "params/mask_jitter", self.mask_jitter, step
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def up_cast(self, point):
        for _ in range(self.up_cast_level):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat(
                [parent.feat, point.feat[inverse]], dim=-1
            )
            point = parent
        return point

    @torch.no_grad()
    def match_neighbour(
        self, view1_coord, view1_offset, view2_coord, view2_offset
    ):
        index2, distance = pointops.knn_query(
            1,
            view2_coord.float(),
            view2_offset.int(),
            view1_coord.float(),
            view1_offset.int(),
        )
        index1 = torch.arange(
            index2.shape[0], device=index2.device, dtype=torch.long
        ).unsqueeze(-1)
        index = torch.cat([index1, index2], dim=-1)[
            distance.squeeze(-1) < self.match_max_r
        ]
        return index

    @torch.no_grad()
    def roll_point(self, point):
        """Swap view order within each sample (only supports num_global_view==2)."""
        n = self.num_global_view
        bs = len(point.offset) // n
        data_dict = {}
        for key in point.keys():
            if key in ["feat", "coord", "origin_coord", "batch"]:
                value = point[key].split(
                    offset2bincount(point.offset).tolist()
                )
                value = chain(
                    *[value[n * b : n * (b + 1)][::-1] for b in range(bs)]
                )
                if key == "batch":
                    value = [
                        torch.ones_like(v) * i for i, v in enumerate(value)
                    ]
                data_dict[key] = torch.cat(list(value), dim=0)
        return Point(data_dict)

    @torch.no_grad()
    def _roll_index(self, point):
        """Return index tensor that reorders packed points as roll_point does."""
        n = self.num_global_view
        bs = len(point.offset) // n
        bincounts = offset2bincount(point.offset).tolist()
        roll_indices = []
        offset = 0
        for b in range(bs):
            views = []
            for v in range(n):
                views.append(
                    torch.arange(
                        offset,
                        offset + bincounts[n * b + v],
                        device=point.offset.device,
                    )
                )
                offset += bincounts[n * b + v]
            roll_indices.extend(reversed(views))
        return torch.cat(roll_indices)

    def generate_mask(self, coord, offset):
        batch = offset2batch(offset)
        min_coord = torch_scatter.segment_coo(coord, batch, reduce="min")
        grid_coord = ((coord - min_coord[batch]) // self.mask_size).int()
        grid_coord = torch.cat([batch.unsqueeze(-1), grid_coord], dim=-1)
        unique, point_cluster, counts = torch.unique(
            grid_coord, dim=0, sorted=True, return_inverse=True,
            return_counts=True,
        )
        patch_num = unique.shape[0]
        mask_patch_num = int(patch_num * self.mask_ratio)
        patch_index = torch.randperm(patch_num, device=coord.device)
        mask_patch_index = patch_index[:mask_patch_num]
        point_mask = torch.isin(point_cluster, mask_patch_index)
        return point_mask, point_cluster

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data_dict):
        grid_size = data_dict["grid_size"][0]

        # --- 1. Build Point objects + masking ---
        has_labels = "global_segment_motif" in data_dict

        with torch.no_grad():
            global_point = Point(
                feat=data_dict["global_feat"],
                coord=data_dict["global_coord"],
                origin_coord=data_dict["global_origin_coord"],
                offset=data_dict["global_offset"],
                grid_size=grid_size,
            )
            if has_labels:
                global_point.segment_motif = data_dict["global_segment_motif"]

            global_mask, _ = self.generate_mask(
                global_point.coord, global_point.offset
            )
            mask_global_coord = global_point.coord.clone().detach()
            if self.mask_jitter is not None and self.mask_jitter > 0:
                mask_global_coord[global_mask] += torch.clip(
                    torch.randn_like(mask_global_coord[global_mask]).mul(
                        self.mask_jitter
                    ),
                    max=self.mask_jitter * 2,
                )

            mask_global_point = Point(
                feat=data_dict["global_feat"],
                coord=mask_global_coord,
                origin_coord=data_dict["global_origin_coord"],
                mask=global_mask,
                offset=data_dict["global_offset"],
                grid_size=grid_size,
            )

            local_point = Point(
                feat=data_dict["local_feat"],
                coord=data_dict["local_coord"],
                origin_coord=data_dict["local_origin_coord"],
                offset=data_dict["local_offset"],
                grid_size=grid_size,
            )

        # --- 2. Three backbone passes ---
        # 2a. Global (unmasked) — target for mask losses
        global_point_ = self.backbone(global_point)
        global_point_ = self.up_cast(global_point_)

        # 2b. Masked global — student for mask losses
        mask_global_point_ = self.backbone(mask_global_point)
        mask_global_point_ = self.up_cast(mask_global_point_)

        # 2c. Local — for local-global loss
        local_point_ = self.backbone(local_point)
        local_point_ = self.up_cast(local_point_)

        # --- Encoder feature stats (on global views) ---
        enc_feat_stats = compute_enc_feat_stats(global_point_)

        # --- 3. Project per-point features ---
        global_proj = self.mask_head(global_point_.feat)
        mask_global_proj = self.mask_head(mask_global_point_.feat)
        global_unmask_proj = self.unmask_head(global_point_.feat)
        local_proj = self.unmask_head(local_point_.feat)

        # --- 4. SIGReg (per-view, matching original LeJEPA paper) ---
        V_g = len(global_point_.offset)
        V_m = len(mask_global_point_.offset)
        V_l = len(local_point_.offset)
        sigreg_loss = (
            self.sigreg(global_proj, global_point_.batch) * V_g
            + self.sigreg(mask_global_proj, mask_global_point_.batch) * V_m
            + self.sigreg(local_proj, local_point_.batch) * V_l
        ) / (V_g + V_m + V_l)

        # --- 5. Mask loss: masked-global vs global ---
        result_dict = dict(loss=[])

        if self.mask_weight > 0:
            match_mask = self.match_neighbour(
                mask_global_point_.origin_coord,
                mask_global_point_.offset,
                global_point_.origin_coord,
                global_point_.offset,
            )
            if match_mask.shape[0] > 0:
                mask_mse = (
                    mask_global_proj[match_mask[:, 0]]
                    - global_proj[match_mask[:, 1]]
                ).square().mean(-1)
                mask_loss = torch_scatter.segment_coo(
                    mask_mse,
                    index=mask_global_point_.batch[match_mask[:, 0]],
                    reduce="mean",
                ).mean()
            else:
                mask_loss = mask_global_proj.new_tensor(0.0)
            result_dict["mask_loss"] = mask_loss
            result_dict["loss"].append(mask_loss * self.mask_weight)

        # --- 6. Roll mask loss: masked-global vs rolled-global ---
        if self.roll_mask_weight > 0:
            roll_global_point_ = self.roll_point(global_point_)
            roll_idx = self._roll_index(global_point_)
            roll_global_proj = global_proj[roll_idx]

            match_roll = self.match_neighbour(
                mask_global_point_.origin_coord,
                mask_global_point_.offset,
                roll_global_point_.origin_coord,
                roll_global_point_.offset,
            )
            if match_roll.shape[0] > 0:
                roll_diff = (
                    mask_global_proj[match_roll[:, 0]]
                    - roll_global_proj[match_roll[:, 1]]
                ).square().mean(-1)
                roll_mask_loss = torch_scatter.segment_coo(
                    roll_diff,
                    index=mask_global_point_.batch[match_roll[:, 0]],
                    reduce="mean",
                ).mean()
            else:
                roll_mask_loss = mask_global_proj.new_tensor(0.0)
            result_dict["roll_mask_loss"] = roll_mask_loss
            result_dict["loss"].append(roll_mask_loss * self.roll_mask_weight)

        # --- 7. Local-global loss ---
        if self.unmask_weight > 0:
            principal_mask = global_point_.batch % self.num_global_view == 0
            principal_batch = (
                global_point_.batch[principal_mask] // self.num_global_view
            )

            match_lg = self.match_neighbour(
                local_point_.origin_coord,
                local_point_.offset[
                    self.num_local_view - 1 :: self.num_local_view
                ],
                global_point_.origin_coord[principal_mask],
                batch2offset(principal_batch),
            )
            if match_lg.shape[0] > 0:
                lg_diff = (
                    local_proj[match_lg[:, 0]]
                    - global_unmask_proj[principal_mask][match_lg[:, 1]]
                ).square().mean(-1)
                unmask_loss = torch_scatter.segment_coo(
                    lg_diff,
                    index=local_point_.batch[match_lg[:, 0]],
                    reduce="mean",
                ).mean()
            else:
                unmask_loss = local_proj.new_tensor(0.0)
            result_dict["unmask_loss"] = unmask_loss
            result_dict["loss"].append(unmask_loss * self.unmask_weight)

        # --- 8. Combined loss ---
        inv_loss = sum(result_dict["loss"])
        loss = sigreg_loss * self.lamb + inv_loss * (1 - self.lamb)

        result_dict["loss"] = loss
        result_dict["total_loss"] = loss.detach().clone()
        result_dict["sigreg_loss"] = sigreg_loss
        result_dict["inv_loss"] = inv_loss
        result_dict.update(enc_feat_stats)

        # Online probe data — stored on model, not in result_dict
        # (result_dict keys get iterated by InformationWriter/storage)
        if has_labels and "segment_motif" in global_point_:
            self._probe_feat = global_point_.feat.detach()
            self._probe_segment_motif = global_point_.segment_motif

        if (ws := get_world_size()) > 1:
            for key in list(result_dict.keys()):
                if key == "loss":
                    continue
                val = result_dict[key]
                if not isinstance(val, torch.Tensor) or not val.is_floating_point():
                    continue
                synced = val.detach()
                dist.all_reduce(synced, op=dist.ReduceOp.SUM)
                synced.div_(ws)
                result_dict[key] = synced

        return result_dict
