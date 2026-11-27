"""Supervised event classification from point features.

Mean, learned-query, set-transformer and efficient-probing aggregators map point
features to event logits. Spatial pooling can retain the tokens' geometry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pimm.models.losses import build_criteria
from pimm.models.utils import offset2batch
from pimm.models.utils.misc import scatter_mean
from pimm.models.utils.structure import Point

from pimm.models.builder import MODELS, build_model


@MODELS.register_module("EventClassifier")
class EventClassifier(nn.Module):
    """Aggregate the backbone's features at the input points per event, then linear-classify."""

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        label_key="event_label",
        hidden_channels=None,
        dropout=0.0,
        collapse_reg_weight=0.0,
        collapse_reg_target=1.0,
        supcon_weight=0.0,
        supcon_temp=0.1,
        pool="mean",
        pool_heads=8,
        pool_layers=2,
        spatial_grid=16,
        spatial_layers=3,
        spatial_unroll=False,
        spatial_ndim=2,
        spatial_raw_channels=0,
        sensor_embed_num=0,
        sensor_embed_dim=32,
        sensor_key="sensor_idx",
        spatial_R=16.9,
        spatial_Z=18.1,
        spatial_gap=0.06,
        spatial_coord_scale=18.1,
        input_channels=None,
        freeze_backbone=False,
    ):
        super().__init__()
        # backbone=None -> identity control: raw per-point features are linearly
        # projected to backbone_out_channels and fed straight to the aggregator.
        # Comparing this against a PTv3 backbone under the IDENTICAL readout
        # isolates whether PTv3 features help, hurt, or do nothing.
        if backbone is None:
            assert input_channels, "identity backbone needs input_channels"
            self.backbone = None
            self.input_proj = nn.Linear(int(input_channels), backbone_out_channels)
        else:
            self.backbone = build_model(backbone)
            self.input_proj = None
        # freeze_backbone -> train ONLY the readout/head on top of fixed SSL
        # features. Full end-to-end CE lets a trainable backbone collapse the
        # SSL features back to a constant (loss -> ln2); freezing prevents that,
        # so the head reads stable pretrained features (a non-collapsing probe).
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone and self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.label_key = label_key
        if pool not in ("mean", "attn", "set_transformer", "spatial_conv", "ep"):
            raise ValueError(f"unknown pool {pool!r}")
        self.pool = pool
        # When the backbone runs in native 3D, unroll token positions to (u,v)
        # only at the head so the spatial conv reads the 2D ring plane.
        self.spatial_unroll = bool(spatial_unroll)
        self.spatial_ndim = int(spatial_ndim)
        self.spatial_R = float(spatial_R)
        self.spatial_Z = float(spatial_Z)
        self.spatial_gap = float(spatial_gap)
        self.spatial_coord_scale = float(spatial_coord_scale)
        # Fixed-detector prior: each sensor (PMT) gets a learned identity
        # embedding appended to its per-point features. The detector geometry
        # is FIXED across events, so a stable per-sensor chart (like ViT/BERT
        # position tokens) replaces the permutation-invariance PTv3 assumes.
        self.sensor_key = sensor_key
        if sensor_embed_num:
            self.sensor_embed = nn.Embedding(
                int(sensor_embed_num), int(sensor_embed_dim)
            )
            nn.init.trunc_normal_(self.sensor_embed.weight, std=0.02)
        else:
            self.sensor_embed = None
        C = backbone_out_channels

        if pool in ("attn", "set_transformer"):
            self.pool_query = nn.Parameter(torch.zeros(1, 1, C))
            nn.init.trunc_normal_(self.pool_query, std=0.02)
            self.pool_attn = nn.MultiheadAttention(
                C, num_heads=pool_heads, batch_first=True
            )
        if pool == "ep":
            self.ep_query = nn.Parameter(torch.zeros(pool_heads, C))
            nn.init.trunc_normal_(self.ep_query, std=0.02)
            self.ep_value = nn.Linear(C, (C // pool_heads) * pool_heads, bias=False)
        if pool == "set_transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=C,
                nhead=pool_heads,
                dim_feedforward=C * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            self.pool_blocks = nn.TransformerEncoder(layer, num_layers=pool_layers)

        if pool == "spatial_conv":
            # Scatter encoder tokens back onto a coarse grid and run a small conv
            # head, so the classifier sees WHERE features are (ring count = spatial
            # arrangement) instead of a permutation-invariant pool that is identical
            # for one ring vs two. ndim=2 uses a 2D grid (optionally unrolled);
            # ndim=3 uses a native (x,y,z) voxel grid + 3D conv -- NO geometry
            # assumptions, no unrolling at all.
            # spatial_raw_channels > 0 adds a RAW-FEATURE BYPASS: the input features
            # are scattered onto the same grid and channel-concatenated with the
            # token grid (fused by a 1x1 conv) so the head is guaranteed to see at
            # least the raw charge image even if the backbone obscures it.
            self.spatial_grid = int(spatial_grid)
            self.spatial_raw_channels = int(spatial_raw_channels)
            if self.spatial_ndim == 3:
                conv, bn, pool_l = nn.Conv3d, nn.BatchNorm3d, nn.AdaptiveAvgPool3d
            else:
                conv, bn, pool_l = nn.Conv2d, nn.BatchNorm2d, nn.AdaptiveAvgPool2d
            if self.spatial_raw_channels:
                # GATED two-stream: raw grid is the PRIMARY signal (projected to
                # C), PTv3 token grid enters as a residual through a gate init to 0
                # -> at init the head sees ONLY the raw charge image (the proven
                # E3 path), so it lifts off; PTv3 can only contribute if learned
                # to help. Avoids PTv3's degenerate optimum poisoning the run.
                self.spatial_raw_proj = conv(
                    self.spatial_raw_channels, C, kernel_size=1, bias=False
                )
                self.spatial_token_gate = nn.Parameter(torch.zeros(1))
            else:
                self.spatial_raw_proj = None
            convs = []
            for _ in range(int(spatial_layers)):
                convs += [
                    conv(C, C, kernel_size=3, padding=1, bias=False),
                    bn(C),
                    nn.GELU(),
                ]
            self.spatial_head = nn.Sequential(*convs)
            self.spatial_pool = pool_l(1)

        if pool == "ep":
            C = (C // pool_heads) * pool_heads  # the concatenated query outputs
        if hidden_channels:
            self.cls_head = nn.Sequential(
                nn.Linear(C, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_classes),
            )
        else:
            self.cls_head = nn.Linear(C, num_classes)
        self.criteria = build_criteria(criteria)
        # Anti-collapse: supervised CE through a pool sits on a plateau where the
        # backbone maps every event to the SAME pooled vector (pooled_std -> 0,
        # loss -> ln2). A VICReg-style variance hinge keeps each pooled dimension
        # spread across the batch (>= target std), pushing the optimizer OFF that
        # degenerate plateau so CE can find the generalizing ring feature.
        self.collapse_reg_weight = float(collapse_reg_weight)
        self.collapse_reg_target = float(collapse_reg_target)
        # Supervised contrastive (Khosla 2020) on the pooled vector: pull same-class
        # events together, push classes apart. Uses the LABELS to shape features ->
        # O(batch^2) pairwise gradient that CE's weak per-sample signal lacks, and
        # it cannot be satisfied by collapse OR by class-orthogonal variance (the
        # failure mode of the plain variance hinge). Vanilla representation loss.
        self.supcon_weight = float(supcon_weight)
        self.supcon_temp = float(supcon_temp)

    def train(self, mode=True):
        super().train(mode)
        # keep a frozen backbone in eval mode (stable norm/dropout statistics)
        if getattr(self, "freeze_backbone", False) and self.backbone is not None:
            self.backbone.eval()
        return self

    def _supcon(self, pooled, labels):
        """Supervised contrastive loss on L2-normalized pooled features."""
        z = F.normalize(pooled.float(), dim=1)
        sim = (z @ z.t()) / self.supcon_temp  # (B, B)
        n = z.shape[0]
        eye = torch.eye(n, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, float("-inf"))  # drop self-similarity
        logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        same = labels.view(-1, 1).eq(labels.view(1, -1)) & ~eye  # positives
        pos = same.sum(1)
        valid = pos > 0
        if not valid.any():
            return pooled.new_zeros(())
        mean_logprob_pos = (logprob.masked_fill(~same, 0.0).sum(1)[valid]) / pos[
            valid
        ].clamp(min=1)
        return -mean_logprob_pos.mean()

    @staticmethod
    def _to_dense(feat, batch, n_events):
        """Pack (N, C) event-grouped tokens into (B, Tmax, C) + validity mask."""
        counts = feat.new_zeros(n_events, dtype=torch.long)
        counts.scatter_add_(0, batch, torch.ones_like(batch))
        t_max = int(counts.max().item())
        starts = feat.new_zeros(n_events, dtype=torch.long)
        starts[1:] = torch.cumsum(counts, 0)[:-1]
        pos = torch.arange(feat.shape[0], device=feat.device) - starts[batch]
        dense = feat.new_zeros(n_events, t_max, feat.shape[1])
        dense[batch, pos] = feat
        valid = torch.arange(t_max, device=feat.device)[None, :] < counts[:, None]
        return dense, valid

    def _unroll_uv(self, coord):
        """Map (normalized native-3D) token coords to normalized unrolled (u,v).

        Mirrors UnrollCylinderCoord but in torch on the pooled token coords:
        un-normalize -> SK-style cylinder unroll -> renormalize to ~[-1,1].
        """
        p = coord * self.spatial_coord_scale  # back to physical meters
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        R, Z, gap = self.spatial_R, self.spatial_Z, self.spatial_gap
        r = torch.hypot(x, y)
        barrel = r > 0.9 * R
        phi = torch.atan2(y, x)
        off = Z + R + gap * Z
        u = torch.where(barrel, R * phi, x)
        v_cap = torch.where(z > 0, off + y, -off + y)
        v = torch.where(barrel, z, v_cap)
        u_n = u / (R * 3.141592653589793)
        v_n = v / (Z + 2.0 * R + gap * Z)
        return u_n, v_n

    @staticmethod
    def _bin(c, g):
        return (((c.clamp(-1, 1) + 1.0) * 0.5) * g).long().clamp(0, g - 1)

    def _grid(self, feat, coord, batch, n_events):
        """Scatter (N, C) features to a dense (B, C, [g,]g, g) grid by position."""
        import torch_scatter

        g = self.spatial_grid
        C = feat.shape[1]
        if self.spatial_ndim == 3:
            ix = self._bin(coord[:, 0], g)
            iy = self._bin(coord[:, 1], g)
            iz = self._bin(coord[:, 2], g)
            flat = batch * (g * g * g) + (iz * g + iy) * g + ix
            dense = torch_scatter.scatter(
                feat.float(), flat, dim=0, dim_size=n_events * g * g * g, reduce="mean"
            )
            return dense.view(n_events, g, g, g, C).permute(0, 4, 1, 2, 3).contiguous()
        if self.spatial_unroll:
            cu, cv = self._unroll_uv(coord)
        else:
            cu, cv = coord[:, 0], coord[:, 1]
        u = self._bin(cu, g)
        v = self._bin(cv, g)
        flat = batch * (g * g) + v * g + u
        dense = torch_scatter.scatter(
            feat.float(), flat, dim=0, dim_size=n_events * g * g, reduce="mean"
        )
        return dense.view(n_events, g, g, C).permute(0, 3, 1, 2).contiguous()

    def _spatial(self, feat, coord, batch, n_events, raw=None):
        token_dense = self._grid(feat, coord, batch, n_events)
        if self.spatial_raw_proj is not None and raw is not None:
            raw_feat, raw_coord, raw_batch = raw
            raw_dense = self._grid(raw_feat.float(), raw_coord, raw_batch, n_events)
            # primary = raw image; PTv3 enters as a gated residual (gate init 0)
            dense = (
                self.spatial_raw_proj(raw_dense) + self.spatial_token_gate * token_dense
            )
        else:
            dense = token_dense
        x = self.spatial_head(dense)
        return self.spatial_pool(x).flatten(1)

    def _aggregate(self, feat, coord, batch, n_events, raw=None):
        if self.pool == "mean":
            return scatter_mean(feat.float(), batch, n_events)
        if self.pool == "spatial_conv":
            return self._spatial(feat.float(), coord, batch, n_events, raw=raw)
        if (
            self.pool == "ep"
        ):  # each query's softmax over its event's points weights that query's values
            import torch_scatter

            with torch.autocast(
                feat.device.type, enabled=False
            ):  # fp32: an fp16 softmax sum stalls at 2048 points
                feat = feat.float()
                a = torch_scatter.composite.scatter_softmax(
                    feat @ self.ep_query.t(), batch, dim=0
                )  # (N, M)
                v = self.ep_value(feat).view(len(feat), a.shape[1], -1)  # (N, M, C / M)
                y = torch_scatter.scatter(
                    a[..., None] * v, batch, dim=0, dim_size=n_events, reduce="sum"
                )
            return y.flatten(1)
        dense, valid = self._to_dense(feat.float(), batch, n_events)
        key_padding = ~valid  # True positions are ignored by attention
        if self.pool == "set_transformer":
            dense = self.pool_blocks(dense, src_key_padding_mask=key_padding)
        q = self.pool_query.expand(n_events, -1, -1)
        pooled, _ = self.pool_attn(q, dense, dense, key_padding_mask=key_padding)
        return pooled.squeeze(1)

    def _event_labels(self, input_dict, n_events):
        labels = input_dict[self.label_key]
        labels = labels.detach().long().view(-1)
        if labels.numel() == n_events:
            return labels
        # per-point labels -> one per event via offsets
        offsets = [0] + input_dict["offset"].detach().cpu().tolist()
        return torch.stack(
            [labels[start] for start, _ in zip(offsets[:-1], offsets[1:])]
        ).to(labels.device)

    def forward(self, input_dict, return_point=False):
        if self.sensor_embed is not None:
            emb = self.sensor_embed(input_dict[self.sensor_key].long().view(-1))
            input_dict = dict(input_dict)
            input_dict["feat"] = torch.cat(
                [input_dict["feat"], emb.to(input_dict["feat"].dtype)], dim=-1
            )
        raw = None
        if getattr(self, "spatial_raw_channels", 0):
            raw = (
                input_dict["feat"],
                input_dict["coord"],
                offset2batch(input_dict["offset"]),
            )
        if self.backbone is None:
            feat = self.input_proj(input_dict["feat"])
            coord = input_dict["coord"]
            batch = offset2batch(input_dict["offset"])
            point = None
        else:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(Point(input_dict)).upcast()
            else:
                point = self.backbone(Point(input_dict)).upcast()
            feat = point.feat  # (N, C): features at the input points
            coord = point.coord
            if "batch" in point.keys():
                batch = point.batch
            else:
                batch = offset2batch(point.offset)
        n_events = int(batch.max().item()) + 1
        pooled = self._aggregate(feat, coord, batch, n_events, raw=raw)  # (n_events, C)
        logits = self.cls_head(pooled)  # (n_events, num_classes)

        result = dict()
        if return_point:
            result["point"] = point

        if self.label_key in input_dict.keys():
            labels = self._event_labels(input_dict, n_events)
            loss = self.criteria(logits, labels)
            if self.collapse_reg_weight > 0 and n_events > 1:
                std = pooled.float().std(0)  # per-dim std across events
                var_hinge = torch.relu(self.collapse_reg_target - std).mean()
                loss = loss + self.collapse_reg_weight * var_hinge
            if self.supcon_weight > 0 and n_events > 1:
                loss = loss + self.supcon_weight * self._supcon(pooled, labels)
            result["loss"] = loss
            if not self.training:
                result["cls_logits"] = logits
        else:
            result["cls_logits"] = logits
        return result


class MomentReadout(nn.Module):
    """Whitened moment readout with learned weights: each site's scalar s = g(f), and the event's
    C^-1 [mean(q s) + c mean(q |q|^2)] (3,), q the sites' positions from their centroid, C = mean(q q^T).
    """

    def __init__(self, g):
        super().__init__()
        self.g, self.c = g, nn.Parameter(torch.zeros(1))

    def forward(self, feat, q, batch, n_events, cov, cubic):
        import torch_scatter

        m = torch_scatter.scatter(
            q * self.g(feat), batch, dim=0, dim_size=n_events, reduce="mean"
        )
        return torch.linalg.solve(cov, m + self.c * cubic)


@MODELS.register_module("WANDEventProbes")
class WANDEventProbes(nn.Module):
    """A frozen backbone with one linear and one MLP probe per WAND single-particle task,
    trained side by side on the same batches. Each head's loss is the mean over the batch's events of its task
    (``task_mask``; zero without any), and the losses are summed: the heads share no parameters and the backbone gets
    no gradient, so each trains as its own run would. Heads (``<probe>_<task>``, probe ``linear`` or ``mlp``; ``g`` an
    ``nn.Linear`` or DefaultSegmentorV2's MLP head, ``mlp_hidden_channels`` wide):

    - ``pid``: the signal's logit, g of the event's mean feature (binary cross-entropy);
    - ``momentum``: log(|p| / 1 GeV/c), g of the mean feature (squared error);
    - ``vertex`` / ``direction``: MomentReadout of g over the sites, positions in metres (vertex: plus the sites'
      centroid; squared error summed over the axes against the vertex in metres / the unit direction).

    Heads run in fp32. In evaluation the output holds ``probe_pred`` (head -> per-event prediction).
    """

    def __init__(
        self,
        backbone,
        backbone_out_channels,
        tasks,
        mlp_hidden_channels=256,
        coord_scale=18.1,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.tasks, self.coord_scale = [dict(t) for t in tasks], float(coord_scale)
        c = backbone_out_channels
        hidden = mlp_hidden_channels
        probes = dict(
            linear=lambda: nn.Linear(c, 1),
            mlp=lambda: nn.Sequential(
                nn.Linear(c, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(hidden, 1),
            ),
        )
        self.heads = nn.ModuleDict(
            {
                f"{probe}_{task['name']}": (
                    MomentReadout(make())
                    if task["kind"] in ("vertex", "direction")
                    else make()
                )
                for probe, make in probes.items()
                for task in self.tasks
            }
        )
        self.head_task = {
            f"{probe}_{task['name']}": t
            for probe in probes
            for t, task in enumerate(self.tasks)
        }

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # frozen: fixed attention orders, no dropout
        return self

    def loss(self, kind, pred, input_dict, t, mask):
        if kind == "pid":
            return F.binary_cross_entropy_with_logits(
                pred[mask], input_dict["probe_label"][mask, t]
            )
        if kind == "momentum":
            return F.mse_loss(
                pred[mask], torch.log(input_dict["momentum"].view(-1)[mask] / 1000.0)
            )
        return ((pred[mask] - input_dict[kind][mask]) ** 2).sum(dim=1).mean()

    def forward(self, input_dict):
        import torch_scatter

        with torch.no_grad():
            point = self.backbone(Point(input_dict)).upcast()
        with torch.autocast(point.feat.device.type, enabled=False):
            feat = point.feat.float()
            batch = offset2batch(point.offset)
            n = len(point.offset)
            mean = scatter_mean(feat, batch, n)
            q = point.coord.float() * self.coord_scale
            centroid = torch_scatter.scatter(q, batch, dim=0, dim_size=n, reduce="mean")
            q = q - centroid[batch]
            cov = torch_scatter.scatter(
                (q[:, :, None] * q[:, None, :]).flatten(1),
                batch,
                dim=0,
                dim_size=n,
                reduce="mean",
            ).view(n, 3, 3)
            cov = cov + 1e-9 * cov.diagonal(dim1=1, dim2=2).sum(1)[
                :, None, None
            ] * torch.eye(3, device=cov.device)
            cubic = torch_scatter.scatter(
                q * (q * q).sum(1, keepdim=True),
                batch,
                dim=0,
                dim_size=n,
                reduce="mean",
            )
            pred, loss = {}, feat.new_zeros(())
            mask = input_dict["task_mask"].bool() if "task_mask" in input_dict else None
            for name, head in self.heads.items():
                t = self.head_task[name]
                task = self.tasks[t]
                if task["kind"] in ("vertex", "direction"):
                    pred[name] = head(feat, q, batch, n, cov, cubic) + (
                        centroid if task["kind"] == "vertex" else 0
                    )
                else:
                    pred[name] = head(mean)[:, 0]
                if mask is not None:
                    m = mask[:, t]
                    loss = loss + (
                        self.loss(task["kind"], pred[name], input_dict, t, m)
                        if m.any()
                        else 0.0 * pred[name].sum()
                    )
        out = dict() if self.training else dict(probe_pred=pred)
        if mask is not None:
            out["loss"] = loss
        return out
