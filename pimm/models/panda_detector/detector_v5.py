"""A much simplified version of detector_v4.

This has the same functionality as detector_v4, but is much
simpler and easier to understand.

A breaking change is the keyword argument format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_
from torch import nn

from pimm.models.builder import MODELS, build_model
from pimm.models.losses import build_criteria
from pimm.models.modules import PointModel
from pimm.models.utils.misc import offset2bincount
from pimm.models.utils.structure import Point
from pimm.utils.comm import reduce_scalar_outputs_for_logging

from .layers import Block, MLP
from .postprocess import postprocess_batch, postprocess_overlap_batch

_PREDICTIONS = ("seg_logits", "pred_logits", "pred_masks")


@dataclass
class QueryHeadConfig:
    """Configuration for one extra prediction head on each query."""

    name: str
    kind: Literal["continuous", "categorical"] = "continuous"
    dim: int = 1
    num_classes: int | None = None
    pred_key: str | None = None
    target_key: str | None = None
    loss_weight: float = 1.0
    use_class_logits: bool = False
    detach_class_logits: bool = False
    required: bool = True
    head_mlp: bool = True
    aggregation: str | None = None
    squeeze: bool | None = None
    criterion: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.pred_key is None:
            self.pred_key = f"pred_{self.name}"


@dataclass
class LabelConfig:
    """Complete configuration for one named detector task."""

    num_queries: int
    num_classes: int
    instance_key: str
    segment_key: str
    criterion: dict[str, Any]
    stuff_classes: list[int] = field(default_factory=list)
    use_stuff_head: bool = True
    loss_weight: float = 1.0
    overlap: bool = False
    query_heads: list[QueryHeadConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Config files contain dictionaries; the detector uses typed objects.
        self.query_heads = [QueryHeadConfig(**head) for head in self.query_heads]


@dataclass
class PostprocessConfig:
    """Thresholds and NMS settings used by :meth:`UnifiedDetector.postprocess`."""

    stuff_threshold: float = 0.5
    mask_threshold: float = 0.5
    conf_threshold: float = 0.5
    nms_kernel: Literal["gaussian", "linear"] = "gaussian"
    nms_sigma: float = 2.0
    nms_pre: int = -1
    nms_max: int = -1
    min_points: int = 20
    fill_uncovered: bool = True
    background_class_label: int = -1
    dedup_iou: float | None = None
    dedup_same_pid: bool | None = None


@dataclass
class PointFilterConfig:
    """Optional point selection applied before decoder cross-attention."""

    type: Literal["stuff", "energy_threshold"] = "stuff"
    threshold: float = 0.5
    drop_classes: list[int] = field(default_factory=list)
    train_filter_use_gt: bool = False
    energy_channel: int = 0
    energy_threshold: float = 0.0


def _config_kwargs(config) -> dict[str, Any]:
    """Return dataclass fields that have concrete values."""
    return {name: value for name, value in asdict(config).items() if value is not None}


class PointFilter(nn.Module):
    """Select points before decoder cross-attention."""

    def __init__(
        self,
        type="stuff",
        full_in_channels=0,
        hidden_channels=256,
        threshold=0.5,
        drop_classes=None,
        train_filter_use_gt=False,
        energy_channel=0,
        energy_threshold=0.0,
    ):
        super().__init__()
        self.type = type
        self.threshold = float(threshold)
        self.drop_classes = list(drop_classes or [])
        self.train_filter_use_gt = bool(train_filter_use_gt)
        self.energy_channel = int(energy_channel)
        self.energy_threshold = float(energy_threshold)
        if type == "stuff":
            self.head = nn.Sequential(
                nn.Linear(full_in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
            )
        elif type == "energy_threshold":
            self.head = None
        else:
            raise ValueError(f"Unknown PointFilter type: {type!r}")

    @staticmethod
    def _ensure_nonempty(keep):
        if not keep.any():
            keep = keep.clone()
            keep[0] = True
        return keep

    def forward(self, feat, gt_segment=None, training=False):
        """Return the kept-point mask and optional learned-filter logits."""
        logits = None
        if self.type == "stuff":
            logits = self.head(feat).squeeze(-1)
            if (
                training
                and self.train_filter_use_gt
                and gt_segment is not None
                and self.drop_classes
            ):
                segment = gt_segment
                if segment.dim() == 2 and segment.shape[1] == 1:
                    segment = segment.squeeze(1)
                drop = torch.zeros_like(segment, dtype=torch.bool)
                for class_id in self.drop_classes:
                    drop |= segment == int(class_id)
                keep = ~drop
            else:
                keep = logits.sigmoid() < self.threshold
        else:
            keep = feat[:, self.energy_channel] > self.energy_threshold
        return self._ensure_nonempty(keep), logits


class GenericMultiTaskDecoder(nn.Module):
    """Decode one learned-query slice per configured label.

    All labels share the transformer blocks. Class, stuff, and extra query heads
    remain label-specific because their output sizes can differ.
    """

    def __init__(
        self,
        label_specs: dict[str, LabelConfig],
        full_in_channels: int,
        hidden_channels: int,
        num_heads: int,
        depth=3,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        pos_emb=True,
        mlp_point_proj=False,
        supervise_attn_mask=True,
        point_filter_cfg: PointFilterConfig | None = None,
    ) -> None:
        super().__init__()
        self.labels = tuple(label_specs)
        self.label_specs = label_specs
        self.query_slices = {}
        start = 0
        max_classes = 1
        for label, spec in label_specs.items():
            end = start + spec.num_queries
            self.query_slices[label] = (start, end)
            start = end
            max_classes = max(max_classes, spec.num_classes)
        self.total_queries = start

        self.full_in_channels = full_in_channels
        self.mask_channels = hidden_channels
        self.num_classes = max_classes
        self.enc_mode = False
        self.num_queries = self.total_queries
        self.query_type = "learned"
        self.query_feat = nn.Embedding(start, hidden_channels)
        self.query_embed = nn.Embedding(start, hidden_channels)
        self.pos_emb = (
            nn.Sequential(
                nn.Linear(3, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            if pos_emb
            else None
        )
        self.attn_mask_anneal = False
        self.attn_mask_anneal_steps = 10000
        self.attn_mask_warmup_steps = 0
        self.attn_mask_progressive = False
        self.attn_mask_progressive_delay = 0
        self.supervise_attn_mask = supervise_attn_mask
        self.blocks = nn.ModuleList(
            [
                Block(
                    channels=hidden_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=drop_path,
                    layer_scale=layer_scale,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    pre_norm=pre_norm,
                    enable_flash=enable_flash,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                    use_attn_mask=True,
                    attn_mask_eps=1e-6,
                    attn_mask_anneal=self.attn_mask_anneal,
                    attn_mask_anneal_steps=self.attn_mask_anneal_steps,
                    attn_mask_warmup_steps=self.attn_mask_warmup_steps,
                    supervise_attn_mask=self.supervise_attn_mask,
                    is_last_block=index == depth - 1,
                )
                for index in range(depth)
            ]
        )
        self.final_norm = norm_layer(hidden_channels)
        self.full_point_proj = (
            MLP(full_in_channels, hidden_channels, hidden_channels)
            if mlp_point_proj
            else nn.Linear(full_in_channels, hidden_channels)
        )
        self.cls_pred = nn.Identity()
        self.cls_pred_by_label = nn.ModuleDict()
        self.stuff_head_by_label = nn.ModuleDict()
        self.head_by_key = nn.ModuleDict()
        self.query_heads_by_label = {}
        for label, spec in label_specs.items():
            classes = spec.num_classes + 1
            self.cls_pred_by_label[label] = (
                MLP(hidden_channels, hidden_channels, classes)
                if mlp_point_proj
                else nn.Linear(hidden_channels, classes)
            )
            if spec.use_stuff_head:
                self.stuff_head_by_label[label] = nn.Sequential(
                    nn.Linear(full_in_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, 1),
                )
            heads = spec.query_heads
            self.query_heads_by_label[label] = heads
            for head in heads:
                output_dim = {
                    "continuous": head.dim,
                    "categorical": head.num_classes,
                }[head.kind]
                input_dim = hidden_channels
                if head.use_class_logits:
                    input_dim += spec.num_classes
                key = self._head_key(label, head.name)
                self.head_by_key[key] = (
                    MLP(input_dim, hidden_channels, int(output_dim))
                    if head.head_mlp
                    else nn.Linear(input_dim, int(output_dim))
                )
        self.point_filter = None
        if point_filter_cfg is not None:
            self.point_filter = PointFilter(
                full_in_channels=full_in_channels,
                hidden_channels=hidden_channels,
                **_config_kwargs(point_filter_cfg),
            )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def update_anneal_step(self, step):
        for block in self.blocks:
            block.set_anneal_step(step)

    def _get_queries(self, point):
        """Repeat the learned query bank once per event."""
        batch_size = point.offset.shape[0]
        queries = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        counts = torch.full(
            (batch_size,),
            self.num_queries,
            dtype=torch.int32,
            device=point.feat.device,
        )
        valid = torch.ones(
            batch_size,
            self.num_queries,
            dtype=torch.bool,
            device=point.feat.device,
        )
        return queries, query_pos, counts, valid

    def _forward_decoder(self, point, return_aux=False):
        """Run the decoder blocks on packed points and event-major queries."""
        point_pos = self.pos_emb(point.coord) if self.pos_emb else None
        cu_seqlens_kv = torch.cat([point.offset.new_zeros(1), point.offset]).int()
        max_seqlen_kv = cu_seqlens_kv.diff().max()

        queries, query_pos, query_counts, query_valid = self._get_queries(point)
        query_lengths = torch.cat(
            [query_counts.new_zeros(1), query_counts.cumsum(0)]
        ).int()
        max_queries = int(query_counts.max().item()) if query_counts.numel() > 0 else 0
        queries = queries.reshape(-1, self.mask_channels)
        query_pos = query_pos.reshape(-1, self.mask_channels)
        query_valid = query_valid.reshape(-1, 1)
        query_valid_f = query_valid.to(queries.dtype)

        auxiliary_queries = []
        auxiliary_masks = []
        final_masks = None
        for block in self.blocks:
            queries, masks, _, _ = block(
                queries,
                point.feat,
                query_lengths,
                cu_seqlens_kv,
                max_queries,
                max_seqlen_kv,
                query_pos,
                point_pos,
            )
            if masks is not None:
                masks = masks * query_valid_f
                final_masks = masks
            queries = queries * query_valid_f
            if return_aux:
                auxiliary_queries.append(self.final_norm(queries))
                auxiliary_masks.append(masks if self.supervise_attn_mask else None)

        output = {
            "out_q": self.final_norm(queries),
            "final_mask_logits": final_masks,
            "query_counts": query_counts.to(torch.long),
            "query_valid": query_valid.squeeze(-1).bool(),
        }
        if return_aux:
            output["aux_q_list"] = auxiliary_queries[:-1]
            output["aux_mask_logits_list"] = auxiliary_masks[:-1]
        return output

    def up_cast(self, point):
        if not self.enc_mode:
            return point
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    @staticmethod
    def _head_key(label, name):
        return f"{label}__{name}".replace(".", "_").replace("/", "_")

    def _compute_head_embeddings(self, label, queries, classes):
        """Apply a label's extra heads to the same flat query tensor."""
        embeddings = {}
        for head in self.query_heads_by_label[label]:
            values = queries
            if head.use_class_logits:
                class_values = classes[:, : self.label_specs[label].num_classes]
                if head.detach_class_logits:
                    class_values = class_values.detach()
                values = torch.cat([values, class_values], dim=-1)
            values = self.head_by_key[self._head_key(label, head.name)](values)
            squeeze = (
                head.kind == "continuous" and head.dim == 1
                if head.squeeze is None
                else head.squeeze
            )
            if squeeze and values.shape[-1] == 1:
                values = values.squeeze(-1)
            embeddings[head.name] = (head, values)
        return embeddings

    def _compute_label_predictions(self, label, queries, masks, point, query_valid):
        """Split flat decoder tensors into one list entry per event."""
        classes = self.cls_pred_by_label[label](queries)
        head_embeddings = self._compute_head_embeddings(label, queries, classes)
        event_masks, event_classes, semantic = [], [], []
        event_heads = {name: [] for name in head_embeddings}
        num_classes = self.label_specs[label].num_classes
        first, last = self.query_slices[label]
        counts = offset2bincount(point.offset).to(torch.long)
        offsets = torch.cat([counts.new_zeros(1), point.offset])

        for batch in range(point.offset.shape[0]):
            # Select this label's query rows and this event's point columns.
            query_start = batch * self.total_queries + first
            query_end = batch * self.total_queries + last
            mask = masks[query_start:query_end, offsets[batch] : offsets[batch + 1]]
            class_logits = classes[query_start:query_end]
            valid = query_valid[query_start:query_end]
            mask = mask[valid]
            class_logits = class_logits[valid]
            event_masks.append(mask)
            event_classes.append(class_logits)
            for name, (_, values) in head_embeddings.items():
                event_heads[name].append(values[query_start:query_end][valid])
            if mask.shape[0] > 0:
                logits = torch.logsumexp(
                    mask.transpose(0, 1).unsqueeze(-1)
                    + class_logits[:, :num_classes].unsqueeze(0),
                    dim=1,
                )
            else:
                logits = masks.new_zeros((counts[batch].item(), num_classes))
            semantic.append(logits)
        semantic = (
            torch.cat(semantic, dim=0)
            if semantic
            else masks.new_zeros((0, num_classes))
        )
        output = {
            "pred_masks": event_masks,
            "pred_logits": event_classes,
            "seg_logits": semantic,
        }
        if event_heads:
            output["pred_regression"] = event_heads
            for name, values in event_heads.items():
                head = head_embeddings[name][0]
                output[head.pred_key] = values
        return output

    @staticmethod
    def _scatter_mask_logits(masks, keep, num_points):
        if masks is None:
            return None
        full = masks.new_full((masks.shape[0], num_points), -1e4)
        full[:, keep] = masks
        return full

    def forward(self, point):
        """Decode a full-resolution point batch for every configured label."""
        point_full = self.up_cast(point)
        num_points = point_full.feat.shape[0]
        keep = None
        filter_logits = None
        # filter out points that we don't want to learn masks for (stuff like noise)
        if self.point_filter is not None:
            keep, filter_logits = self.point_filter(
                point_full.feat,
                gt_segment=getattr(point_full, "_filter_gt_segment", None),
                training=self.training,
            )
            decoder_point = point_full[keep]
        else:
            decoder_point = point_full.copy()
        # run an MLP through the backbone features that projects down
        # to the hidden dimension of the decoder (and mixes per-depth information)
        decoder_point.feat = self.full_point_proj(decoder_point.feat)

        # decode queries at each layer of the decoder
        decoder = self._forward_decoder(decoder_point, return_aux=self.training)
        queries = decoder["out_q"]
        query_valid = decoder["query_valid"]
        masks = decoder["final_mask_logits"]
        auxiliary_queries = decoder.get("aux_q_list", [])
        auxiliary_masks = decoder.get("aux_mask_logits_list", [])
        if keep is not None:
            # Losses and postprocessing use original point indexing, so filtered
            # mask columns are restored with an effectively-zero mask logit.
            masks = self._scatter_mask_logits(masks, keep, num_points)
            auxiliary_masks = [
                self._scatter_mask_logits(aux_masks, keep, num_points)
                for aux_masks in auxiliary_masks
            ]

        outputs = {}
        for label in self.labels:
            prediction = self._compute_label_predictions(
                label, queries, masks, point_full, query_valid
            )
            if self.training:
                aux_outputs = []
                for aux_queries, aux_masks in zip(auxiliary_queries, auxiliary_masks):
                    if aux_masks is None:
                        continue
                    aux_outputs.append(
                        self._compute_label_predictions(
                            label,
                            aux_queries,
                            aux_masks,
                            point_full,
                            query_valid,
                        )
                    )
                if aux_outputs:
                    prediction["aux_outputs"] = aux_outputs
            if label in self.stuff_head_by_label:
                logits = self.stuff_head_by_label[label](point_full.feat).squeeze(-1)
                prediction["stuff_logits"] = logits
                prediction["stuff_probs"] = logits.sigmoid()
            outputs[label] = prediction
        if filter_logits is not None:
            outputs["__filter_logits__"] = filter_logits
        return outputs


@MODELS.register_module("detector-v5")
class UnifiedDetector(PointModel):
    """Backbone, shared query decoder, losses, and panoptic postprocessing.

    ``label_configs`` insertion order defines the label and query order. Each
    label requires query/class counts, target keys, and a criterion. Stuff,
    overlap, loss-weight, and query-head fields have defaults. A query head only
    requires a name; continuous heads default to one output, while categorical
    heads also specify ``kind="categorical"`` and ``num_classes``.

    The last configured label is the default evaluation label unless
    ``eval_label`` names another one. Evaluation also returns every label under
    ``outputs_by_label`` and the ``*_by_label`` aliases.
    """

    def __init__(
        self,
        backbone: dict[str, Any],
        label_configs: dict[str, dict[str, Any]],
        postprocess: dict[str, Any],
        full_in_channels: int,
        hidden_channels: int,
        num_heads: int,
        eval_label: str | None = None,
        point_filter: dict[str, Any] | None = None,
        filter_loss_weight: float = 1.0,
        depth: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: float | None = None,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        pre_norm: bool = True,
        enable_flash: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        pos_emb: bool = True,
        supervise_attn_mask: bool = True,
        mlp_point_proj: bool = False,
    ) -> None:
        """Build the configured backbone, losses, and shared query decoder.

        Args:
            backbone: Registry configuration for the point backbone.
            label_configs: Ordered mapping of label names to complete task
                configurations described in the class docstring.
            postprocess: Keyword arguments forwarded to the panoptic
                postprocessor. Calls to :meth:`postprocess` may override them.
            full_in_channels: Width of the up-cast backbone features.
            hidden_channels: Decoder and query embedding width.
            num_heads: Attention heads in every decoder block.
            eval_label: Label used for unprefixed evaluation outputs.
            point_filter: Optional ``PointFilter`` constructor arguments.
            filter_loss_weight: Weight for learned point-filter supervision.
            depth: Number of shared transformer decoder blocks.
            supervise_attn_mask: Include intermediate mask predictions in the
                auxiliary losses.
            mlp_point_proj: Use MLPs instead of linear point and class heads.

        The remaining attention, normalization, and dropout arguments are
        forwarded unchanged to every decoder block.
        """
        super().__init__()
        self.label_specs = {
            label: LabelConfig(**config) for label, config in label_configs.items()
        }
        self.labels = tuple(self.label_specs)
        self.eval_label = self.labels[-1] if eval_label is None else eval_label
        self.postprocess_cfg = PostprocessConfig(**postprocess)
        self.point_filter_cfg = (
            None if point_filter is None else PointFilterConfig(**point_filter)
        )
        self.filter_loss_weight = float(filter_loss_weight)
        self.backbone = build_model(backbone)

        self.criteria_by_label = {}
        for label, spec in self.label_specs.items():
            criterion = {
                **spec.criterion,
                "query_heads": [_config_kwargs(head) for head in spec.query_heads],
                "overlap": spec.overlap,
                "truth_label": spec.instance_key,
                "segment_key": spec.segment_key,
            }
            self.criteria_by_label[label] = build_criteria([criterion])

        self.decoder = GenericMultiTaskDecoder(
            self.label_specs,
            full_in_channels,
            hidden_channels,
            num_heads,
            depth,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
            drop_path,
            layer_scale,
            norm_layer,
            act_layer,
            pre_norm,
            enable_flash,
            upcast_attention,
            upcast_softmax,
            pos_emb,
            mlp_point_proj,
            supervise_attn_mask,
            self.point_filter_cfg,
        )

    def up_cast(self, point):
        """Concatenate pooled-scale features back to input point resolution."""
        if not getattr(self.backbone, "enc_mode", True):
            return point
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    def update_anneal_step(self, step):
        self.decoder.update_anneal_step(step)

    def _compute_stuff_loss(self, label, stuff_logits, input_dict):
        segment = input_dict[self.label_specs[label].segment_key]
        if isinstance(segment, torch.Tensor) and segment.dim() == 2:
            if segment.shape[1] == 1:
                segment = segment.squeeze(1)
        target = torch.zeros_like(stuff_logits)
        for class_id in self.label_specs[label].stuff_classes:
            target[segment == int(class_id)] = 1.0
        return F.binary_cross_entropy_with_logits(
            stuff_logits, target, reduction="mean"
        )

    def _compute_filter_loss(self, filter_logits, input_dict):
        classes = self.point_filter_cfg.drop_classes
        if not classes:
            return None
        spec = self.label_specs[self.eval_label]
        segment_key = spec.segment_key
        if segment_key not in input_dict:
            return None
        segment = input_dict[segment_key]
        if segment.dim() == 2 and segment.shape[1] == 1:
            segment = segment.squeeze(1)
        target = torch.zeros_like(filter_logits)
        for class_id in classes:
            target[segment == int(class_id)] = 1.0
        return self.filter_loss_weight * F.binary_cross_entropy_with_logits(
            filter_logits, target
        )

    def forward(self, input_dict, return_point=False):
        """Run the backbone, query decoder, and configured task losses."""

        # run fwd and upcast backbone features
        point = Point(input_dict)
        point = self.backbone(point)
        point = self.up_cast(point)

        # find what we need to filter out
        if self.point_filter_cfg is not None and self.training:
            spec = self.label_specs[self.eval_label]
            segment_key = spec.segment_key
            if segment_key in input_dict:
                point._filter_gt_segment = input_dict[segment_key]

        # run query decoder on a copy of the point
        decoder_point = Point(point.copy())
        if hasattr(point, "_filter_gt_segment"):
            decoder_point._filter_gt_segment = point._filter_gt_segment
        outputs_by_label = self.decoder(decoder_point)
        filter_logits = outputs_by_label.pop("__filter_logits__", None)

        result = {}
        total_loss = None
        emit_predictions = not self.training  # return preds if in eval mode
        # loop over each type of query (particles, interactions, etc.)
        # and compute the loss for each, adding them up to get the total loss
        for label in self.labels:
            # set up targets
            spec = self.label_specs[label]
            segment_key = spec.segment_key
            has_targets = spec.instance_key in input_dict and segment_key in input_dict
            prediction = outputs_by_label[label]

            if emit_predictions:
                result[f"{label}_seg_logits"] = prediction["seg_logits"]
                result[f"{label}_pred_logits"] = prediction.get("pred_logits")
                result[f"{label}_pred_masks"] = prediction.get("pred_masks")
                for head in spec.query_heads:
                    result[f"{label}_{head.pred_key}"] = prediction[head.pred_key]
                if "stuff_probs" in prediction:
                    result[f"{label}_stuff_probs"] = prediction["stuff_probs"]

            if has_targets or self.training:
                loss_result = self.criteria_by_label[label](prediction, input_dict)
                if isinstance(loss_result, tuple):
                    label_loss, details = loss_result
                else:
                    label_loss, details = loss_result, {}
                label_loss = label_loss * spec.loss_weight
                total_loss = (
                    label_loss if total_loss is None else total_loss + label_loss
                )
                result[f"{label}_loss"] = label_loss
                result.update(
                    {f"{label}_{key}": value for key, value in details.items()}
                )
                if spec.use_stuff_head and "stuff_logits" in prediction:
                    stuff_loss = self._compute_stuff_loss(
                        label, prediction["stuff_logits"], input_dict
                    )
                    total_loss = total_loss + stuff_loss
                    result[f"{label}_stuff_loss"] = stuff_loss

        if filter_logits is not None and self.training:
            filter_loss = self._compute_filter_loss(filter_logits, input_dict)
            if filter_loss is not None:
                total_loss = (
                    filter_loss if total_loss is None else total_loss + filter_loss
                )
                result["filter_loss"] = filter_loss

        if emit_predictions or return_point:
            primary = outputs_by_label[self.eval_label]
            spec = self.label_specs[self.eval_label]
            point.outputs = primary
            point.outputs_by_label = outputs_by_label
            point.pred_cls = primary["pred_logits"]
            point.pred_masks = primary["pred_masks"]
            point.pred_logits = primary["seg_logits"]
            point.pred_regression = primary.get("pred_regression")
            if spec.instance_key in input_dict:
                point.instance = input_dict[spec.instance_key]
            segment_key = spec.segment_key
            if segment_key in input_dict:
                point.segment = input_dict[segment_key]

            if emit_predictions:
                result["outputs_by_label"] = outputs_by_label
                for key in _PREDICTIONS:
                    result[f"{key}_by_label"] = {
                        label: output[key] for label, output in outputs_by_label.items()
                    }
                result["pred_regression_by_label"] = {
                    label: output.get("pred_regression", {})
                    for label, output in outputs_by_label.items()
                }
                result["seg_logits"] = primary["seg_logits"]
                result["pred_logits"] = primary.get("pred_logits")
                result["pred_masks"] = primary.get("pred_masks")
                result["pred_regression"] = primary.get("pred_regression")
                for head in spec.query_heads:
                    result[head.pred_key] = primary[head.pred_key]
                result["stuff_probs"] = primary.get("stuff_probs")
                result["point_counts"] = offset2bincount(point.offset)
        if total_loss is not None:
            result["loss"] = total_loss
        if return_point:
            result["point"] = point
        return reduce_scalar_outputs_for_logging(result)

    def postprocess(self, forward_output, label=None, **overrides):
        """Fuse one label's query masks and classes into point assignments.

        ``forward_output`` is the dictionary returned by :meth:`forward` in
        evaluation mode. Keyword overrides replace values from the constructor's
        ``postprocess`` mapping for this call only.
        """
        label = self.eval_label if label is None else label
        cfg = {**_config_kwargs(self.postprocess_cfg), **overrides}
        prediction = forward_output["outputs_by_label"][label]
        point_counts = forward_output["point_counts"]
        spec = self.label_specs[label]
        if spec.overlap:
            return postprocess_overlap_batch(
                prediction["pred_masks"],
                prediction["pred_logits"],
                spec.num_classes,
                prediction.get("pred_regression"),
                **cfg,
            )
        return postprocess_batch(
            prediction["pred_masks"],
            prediction["pred_logits"],
            prediction.get("stuff_probs"),
            pred_regression=prediction.get("pred_regression"),
            point_counts=point_counts,
            stuff_classes=spec.stuff_classes,
            **cfg,
        )
