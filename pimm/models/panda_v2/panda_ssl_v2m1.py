"""Panda V2 self-distillation with observable points and soft maps.

An EMA teacher and a student encode multiple masked views. With ``observable``, masked points are removed from the
student's views; otherwise the student keeps every point, and masked points get the backbone's mask token and
jittered coordinates. ``softmap`` normalizes targets and predictions over points per prototype. The losses are
computed ``up_cast_level`` upcasts above the backbone's coarsest stage: 3 reaches the input points, 2 the second stage.

Training batches must be prepared by the ``SonataPreparation`` transform, which draws the masks in the data workers
and finds the nearest-neighbour matches at the level of the losses. The features and coordinates of a view's rows
that share a backbone voxel are averaged into one site before encoding.

The heads return normalized embeddings. ``pimm.ops.sinkhorn_distillation_loss`` balances matched teacher rows and
computes prototype cross-entropy. ``fused_sinkhorn`` uses tiled BF16 products and FP32 reductions without storing the
full score table; the alternative uses dense FP32 scores.

Base author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com).
Adapted and extended: Samuel Young (youngsam@stanford.edu).
"""

from packaging import version
from functools import partial
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.init import trunc_normal_

from pimm.models.utils.structure import Point
from pimm.models.builder import MODELS, build_model
from pimm.models.modules import PointModel
from pimm.models.panda_v2.panda_encoder_v2m1 import PandaV2Encoder
from pimm.models.utils.checkpoint import recompute_activation
from pimm.utils.comm import get_world_size
from pimm.utils.config import resolve_arguments
from pimm.utils.scheduler import CosineScheduler, resolve_iters
from pimm.ops import sinkhorn_distillation_loss

class OnlineCluster(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=4096,
        embed_channels=512,
        num_prototypes=4096,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, embed_channels),
        )
        self.apply(self._init_weights)
        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            self.prototype = torch.nn.utils.parametrizations.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.parametrizations.weight.original0.data.fill_(1)
            self.prototype.parametrizations.weight.original0.requires_grad = False

        else:
            self.prototype = torch.nn.utils.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.weight_g.data.fill_(1)
            self.prototype.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, feat, return_embedding=False):
        feat = recompute_activation(
            self.mlp[1], self.mlp[0](feat), self.mlp[2]
        )  # backward recomputes the GELU
        eps = 1e-6 if feat.dtype == torch.float16 else 1e-12
        feat = nn.functional.normalize(feat, dim=-1, p=2, eps=eps)
        if return_embedding:
            return feat
        similarity = self.prototype(feat)
        return similarity


def masked_match_radius(match_max_r, observable_match_max_r):
    """The radius of the mask and roll-mask matches: observable_match_max_r, or match_max_r without it."""
    return match_max_r if observable_match_max_r is None else observable_match_max_r


def self_distillation_settings(config):
    """The arguments of a panda-ssl-v2m1 config, with the class's defaults for those it omits; the backbone's
    arguments are resolved the same way, and observable_match_max_r as the model resolves it."""
    settings = resolve_arguments(PandaV2SelfDistillation, config)
    settings["backbone"] = resolve_arguments(PandaV2Encoder, config["backbone"])
    settings["observable_match_max_r"] = masked_match_radius(
        settings["match_max_r"], settings["observable_match_max_r"]
    )
    return settings


@MODELS.register_module("panda-ssl-v2m1")
class PandaV2SelfDistillation(PointModel):
    def __init__(
        self,
        backbone,
        head_in_channels,
        head_hidden_channels=4096,
        head_embed_channels=512,
        head_num_prototypes=4096,
        teacher_custom=None,
        num_global_view=2,
        num_local_view=4,
        mask_size=0.4,
        mask_ratio=0.7,
        mask_jitter=0.01,
        teacher_temp_start=0.04,
        teacher_temp_base=0.07,
        teacher_temp_warmup_ratio=0.05,
        student_temp=0.1,
        mask_loss_weight=2 / 8,
        roll_mask_loss_weight=2 / 8,
        unmask_loss_weight=4 / 8,
        momentum_base=0.996,
        momentum_final=1,
        ema_power=1,  # teacher momentum = the cosine schedule's value ** ema_power (k for a k-times batch)
        match_max_r=0.08,
        up_cast_level=2,
        # --- DOS (arXiv:2512.11465); all-off == Sonata ---
        observable=False,
        observable_match_max_r=None,
        softmap=False,
        softmap_scope="sample",
        sinkhorn_iters=3,
        split_backward=False,  # loss as one part per student pass, each backpropagated before the next is computed
        fused_sinkhorn=False,  # the Triton kernel of pimm.ops.sinkhorn_distillation_loss
        sync_sinkhorn=True,  # balance Sinkhorn over all ranks, like sync_bn; False balances each rank's rows
    ):
        super().__init__()
        self.split_backward = split_backward
        self.fused_sinkhorn = fused_sinkhorn
        self.sync_sinkhorn = sync_sinkhorn
        self.mask_loss_weight = mask_loss_weight
        self.roll_mask_loss_weight = roll_mask_loss_weight
        self.unmask_loss_weight = unmask_loss_weight
        self.num_global_view = num_global_view
        self.num_local_view = num_local_view
        self.mask_size, self.mask_ratio, self.mask_jitter = mask_size, mask_ratio, mask_jitter

        # the teacher temperature warms up linearly, then stays at teacher_temp_base
        self.teacher_temp = teacher_temp_start
        self.teacher_temp_start = teacher_temp_start
        self.teacher_temp_base = teacher_temp_base
        self.teacher_temp_warmup_ratio = teacher_temp_warmup_ratio
        self.student_temp = student_temp

        # momentum and scheduler
        self.momentum = momentum_base
        self.momentum_base = momentum_base
        self.momentum_final = momentum_final
        self.momentum_scheduler = None
        self.ema_power = ema_power

        self.match_max_r = match_max_r
        if not 0 <= up_cast_level <= 3:
            raise ValueError("up_cast_level must be between 0 and 3")
        self.up_cast_level = up_cast_level

        # --- DOS knobs ---
        self.observable = observable
        # without the mask token, coarse cells of the student and teacher views need not coincide, so matching
        # in the masked views may use a looser radius
        self.observable_match_max_r = masked_match_radius(match_max_r, observable_match_max_r)
        self.softmap = softmap
        assert softmap_scope in ("sample", "batch")
        # "sample": normalize softmaps within each event (DOS default).
        # "batch": normalize over all matched points on the rank -- robust when
        # per-event information content varies widely (e.g. 1 muon vs several
        # interactions); simple events then leave most prototypes inactive
        # instead of being forced to spread them across the event.
        self.softmap_scope = softmap_scope
        self.sinkhorn_iters = sinkhorn_iters

        # one of unmask, mask, roll mask loss enable
        assert unmask_loss_weight + mask_loss_weight + roll_mask_loss_weight > 0
        # roll mask loss need more than one global view
        assert num_global_view > 1 or roll_mask_loss_weight == 0
        # current roll mask only support two global views
        assert num_global_view == 1 or num_global_view == 2

        student_model_dict = dict()
        teacher_model_dict = dict()
        if teacher_custom is None:
            teacher_custom = {}
        student_backbone = build_model(backbone)
        if not observable and getattr(student_backbone, "mask_token", None) is None:
            raise ValueError("observable=False needs a backbone with a mask token (mask_token=True)")
        # turn off parameters like drop path for teacher model
        backbone.update(teacher_custom)

        teacher_backbone = build_model(backbone)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        head = partial(
            OnlineCluster,
            in_channels=head_in_channels,
            hidden_channels=head_hidden_channels,
            embed_channels=head_embed_channels,
            num_prototypes=head_num_prototypes,
        )
        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            student_model_dict["mask_head"] = head()
            teacher_model_dict["mask_head"] = head()
        if self.unmask_loss_weight > 0:
            student_model_dict["unmask_head"] = head()
            teacher_model_dict["unmask_head"] = head()

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

    def before_train(self):
        # make ModelHook after CheckPointLoader
        total_steps = self.trainer.cfg.scheduler.total_steps
        self.step_count = getattr(self.trainer, "global_step", 0) or (
            self.trainer.start_epoch * len(self.trainer.train_loader)
        )
        self.teacher_temp_warmup_steps = resolve_iters(
            self.teacher_temp_warmup_ratio, total_steps
        )
        self.momentum_scheduler = CosineScheduler(
            base_value=self.momentum_base,
            final_value=self.momentum_final,
            total_iters=total_steps,
        )
        self.momentum_scheduler.iter = self.step_count

    def before_step(self):
        warmup, start, base = (
            self.teacher_temp_warmup_steps,
            self.teacher_temp_start,
            self.teacher_temp_base,
        )
        # numpy.linspace(start, base, warmup)'s values, step for step
        self.teacher_temp = (
            base
            if self.step_count >= warmup - 1
            else start + self.step_count * ((base - start) / (warmup - 1))
        )
        self.step_count += 1
        self.momentum = self.momentum_scheduler.step() ** self.ema_power
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                "params/teacher_temp", self.teacher_temp, self.step_count
            )
            self.trainer.writer.add_scalar(
                "params/momentum", self.momentum, self.momentum_scheduler.iter
            )

    def after_step(self):
        # EMA update teacher
        with torch.no_grad():
            m = self.momentum
            student_param_list = list(self.student.parameters())
            teacher_param_list = list(self.teacher.parameters())
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    @staticmethod
    @torch.no_grad()
    def merge_view_sites(point, sites):
        """The rows of one view in one backbone voxel become one site: the mean of their feat, coord and
        origin_coord. ``sites`` is prepared by SonataPreparation; it holds each row's site and the sites' voxels, so
        no sparse convolution sees two rows at one index. A site is masked if any of its rows is."""
        import torch_scatter

        rows, n = sites["index"], len(sites["batch"])
        order = torch.argsort(rows, stable=True)  # sums in row order, so the means are reproducible
        count = torch.bincount(rows, minlength=n)
        ptr = torch.cat([count.new_zeros(1), torch.cumsum(count, 0)])
        mean = {
            key: torch_scatter.segment_csr(point[key].index_select(0, order), ptr, reduce="sum")
            / count[:, None].to(point[key].dtype)
            for key in ("feat", "coord", "origin_coord")
        }
        merged = Point(
            grid_size=point.grid_size,
            grid_coord=sites["grid_coord"],
            batch=sites["batch"],
            offset=sites["offset"],
            **mean,
        )
        if "mask" in point.keys():
            merged.mask = rows.new_zeros(n).index_add_(0, rows, point.mask.long()) > 0
        return merged

    def check_plan(self, settings):
        """Raise unless a prepared batch's settings match this model and its backbones."""
        mismatched = [
            key
            for key in (
                "observable",
                "up_cast_level",
                "mask_size",
                "mask_ratio",
                "mask_jitter",
                "match_max_r",
                "observable_match_max_r",
                "num_global_view",
                "num_local_view",
            )
            if settings[key] != getattr(self, key)
        ]
        if any(
            getattr(model.backbone, "strides", None) != tuple(settings["strides"])  # pinning makes lists
            for model in (self.student, self.teacher)
        ):
            mismatched.append("strides (the backbone's)")
        if mismatched:
            raise ValueError(
                f"SonataPreparation settings do not match panda-ssl-v2m1: {', '.join(mismatched)}. "
                "Give the transform the model's config (model=...)."
            )

    def encode_view(self, model_dict, point, result_dict, plan, name):
        """Backbone and up cast of one set of views (``name``: teacher, student or local).

        The backbone sees the views' sites; if the up cast reaches the input level, each row gets its site's
        features, coord and origin_coord, so the output keeps one row per input row.
        """
        sites = plan["sites"][name]
        point_ = self.merge_view_sites(point, sites)
        result_dict[f"sites/merged_frac_{name}"] = torch.tensor(
            1 - len(point_.feat) / len(point.feat), device=point.feat.device
        )
        point_ = model_dict.backbone(point_).upcast(self.up_cast_level)
        if "pooling_parent" in point_.keys():
            return point_
        rows = sites["index"]
        return Point(
            feat=point_.feat[rows],
            coord=point_.coord[rows],
            origin_coord=point_.origin_coord[rows],
            offset=point.offset,
            batch=point.batch,
        )

    def _distill_loss(self, student_features, teacher_features, batch_idx, head):
        """Matched-row distillation, reduced equally over events."""
        import torch_scatter

        if not self.softmap:
            loss = sinkhorn_distillation_loss(
                student_features,
                self.student[head].prototype.weight,
                teacher_features,
                self.teacher[head].prototype.weight,
                self.student_temp,
                self.teacher_temp,
                iterations=self.sinkhorn_iters,
                fused=self.fused_sinkhorn,
                distributed=self.sync_sinkhorn,
            )
            return torch_scatter.segment_coo(
                loss, index=batch_idx, reduce="mean"
            ).mean()

        # --- softmap: per-prototype distribution over points. Normalization set
        # is the event ("sample") or the whole rank batch ("batch"); see the
        # softmap_scope note in __init__. ---
        group = (
            batch_idx if self.softmap_scope == "sample" else torch.zeros_like(batch_idx)
        )
        with torch.no_grad():
            # no Sinkhorn: softmax over the points of each group (softmaps are stable without centering)
            target = torch_scatter.scatter_softmax(
                teacher_features / self.teacher_temp, group, dim=0
            )
        log_p = torch_scatter.scatter_log_softmax(
            student_features / self.student_temp, group, dim=0
        )
        ce = -(target * log_p)  # (M, K)
        # CE over points within each (group, prototype), then mean over both
        # == (1/K) sum_k CE_k averaged over groups (paper's L_sigma).
        ce = torch_scatter.scatter(ce, group, dim=0, reduce="sum")  # (G, K)
        return ce.mean()

    def forward(self, data_dict, return_point=False):
        if return_point:
            return dict(point=self.student.backbone(data_dict).upcast())

        result_dict = {}
        parts = self._loss_parts(data_dict, result_dict)
        result_dict["loss"] = parts if self.split_backward else sum(parts)
        return result_dict

    def _loss_parts(self, data_dict, result_dict):
        """Yield the loss in parts that can be backpropagated one at a time: the student on the masked global views
        (mask and roll-mask losses), then on the local views (unmask loss). Loss terms go into ``result_dict``; the
        logged ones are averaged over ranks after the last part."""
        plan = data_dict.get("sonata_plan")
        if plan is None:
            raise ValueError(
                "panda-ssl-v2m1 trains on batches prepared by the SonataPreparation transform; "
                "append it to the training transforms after Collect"
            )
        self.check_plan(plan["settings"])
        sites, grid_size = plan["sites"], data_dict["grid_size"][0]
        with torch.no_grad():
            global_point = Point(
                feat=data_dict["global_feat"],
                coord=data_dict["global_coord"],
                origin_coord=data_dict["global_origin_coord"],
                offset=data_dict["global_offset"],
                batch=sites["teacher"]["row_batch"],
                grid_size=grid_size,
            )
            if self.observable:
                # DOS: the student encodes ONLY the visible points -- masked
                # points are dropped entirely (no mask token, no coordinate),
                # so there is no positional handle and no loss at masked coords.
                visible = plan["visible"]
                mask_global_point = Point(
                    feat=global_point.feat[visible],
                    coord=global_point.coord[visible],
                    origin_coord=global_point.origin_coord[visible],
                    offset=sites["student"]["row_offset"],
                    batch=sites["student"]["row_batch"],
                    grid_size=grid_size,
                )
            else:
                # Sonata: masked points keep their place with the mask token and jittered coordinates.
                mask_global_point = Point(
                    feat=global_point.feat,
                    coord=plan["coord"],
                    origin_coord=global_point.origin_coord,
                    mask=plan["mask"],
                    offset=global_point.offset,
                    batch=global_point.batch,
                    grid_size=grid_size,
                )
            local_point = Point(
                feat=data_dict["local_feat"],
                coord=data_dict["local_coord"],
                origin_coord=data_dict["local_origin_coord"],
                offset=data_dict["local_offset"],
                batch=sites["local"]["row_batch"],
                grid_size=grid_size,
            )
            # teacher backbone forward (shared with mask and unmask)
            global_point_ = self.encode_view(
                self.teacher, global_point, result_dict, plan, "teacher"
            )

        # one method per student pass, so each pass's tensors are freed before the next part runs
        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            yield self._masked_views_loss(
                global_point_, mask_global_point, result_dict, plan
            )
        if self.unmask_loss_weight > 0:
            yield self._local_views_loss(global_point_, local_point, result_dict, plan)

        # sync component losses for logging
        if (ws := get_world_size()) > 1:
            synced_keys = [key for key in result_dict if key != "loss"]
            synced_losses = torch.stack(
                [result_dict[key].detach() for key in synced_keys]
            )
            dist.all_reduce(synced_losses, op=dist.ReduceOp.SUM)
            synced_losses.div_(ws)
            for key, synced_loss in zip(synced_keys, synced_losses):
                result_dict[key] = synced_loss

    def _masked_views_loss(self, global_point_, mask_global_point, result_dict, plan):
        """The student on the masked global views: mask and roll-mask losses."""
        # teacher head forward, on a copy of the teacher's points
        with torch.no_grad():
            global_point_ = Point(
                feat=self.teacher.mask_head(
                    global_point_.feat, return_embedding=not self.softmap
                ),
                coord=global_point_.coord,
                origin_coord=global_point_.origin_coord,
                offset=global_point_.offset,
                batch=global_point_.batch,
            )
        # student forward
        mask_global_point_ = self.encode_view(
            self.student, mask_global_point, result_dict, plan, "student"
        )
        mask_pred_sim = self.student.mask_head(
            mask_global_point_.feat, return_embedding=not self.softmap
        )

        loss = 0
        # matches pair student rows with teacher rows of the same view (mask) or the other global view (roll_mask)
        for name, weight in (("mask", self.mask_loss_weight), ("roll_mask", self.roll_mask_loss_weight)):
            if weight > 0:
                match_index = plan[f"{name}_match"]
                result_dict[f"{name}_loss"] = self._distill_loss(
                    mask_pred_sim[match_index[:, 0]],
                    global_point_.feat[match_index[:, 1]],
                    mask_global_point_.batch[match_index[:, 0]],
                    head="mask_head",
                )
                loss = loss + result_dict[f"{name}_loss"] * weight
        result_dict["total_loss"] = result_dict.get("total_loss", 0) + loss.detach()
        return loss

    def _local_views_loss(self, global_point_, local_point, result_dict, plan):
        """The student on the local views: unmask loss."""
        # student forward
        local_point_ = self.encode_view(
            self.student, local_point, result_dict, plan, "local"
        )
        match_index = plan["local_match"]
        # matched embeddings are gathered first; the loss projects them onto the prototypes
        result_dict["unmask_loss"] = self._distill_loss(
            self.student.unmask_head(
                local_point_.feat, return_embedding=not self.softmap
            )[match_index[:, 0]],
            self.teacher.unmask_head(
                global_point_.feat, return_embedding=not self.softmap
            )[match_index[:, 1]],
            local_point_.batch[match_index[:, 0]],
            head="unmask_head",
        )
        loss = result_dict["unmask_loss"] * self.unmask_loss_weight
        result_dict["total_loss"] = result_dict.get("total_loss", 0) + loss.detach()
        return loss
