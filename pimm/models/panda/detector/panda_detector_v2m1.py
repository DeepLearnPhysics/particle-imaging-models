"""Panda V2 query decoder for exclusive and overlapping instances."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_

from pimm.models.builder import MODELS, build_model
from pimm.models.losses import build_criteria
from pimm.models.modules import PointModel
from pimm.models.utils.misc import offset2bincount
from pimm.models.utils.structure import Point
from pimm.utils.comm import reduce_scalar_outputs_for_logging

from .layers import MLP, Block
from .postprocess import postprocess_batch, postprocess_overlap_batch

PLACEMENTS = [
    (x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)
]  # voxel-grid offsets for average_placements

# Output keys the decoder/forward populate itself -- user head pred_keys may not
# collide with these.
_RESERVED_OUTPUT_KEYS = {
    "pred_masks",
    "pred_logits",
    "seg_logits",
    "aux_outputs",
    "stuff_logits",
    "stuff_probs",
    "pred_regression",
}

_DECODING_DEFAULTS = {
    "panda_v1": dict(
        conf_threshold=0.5, rank_by_class_score=False, argmax_fill_min_prob=None
    ),
    "panda_v2": dict(
        conf_threshold=0.3, rank_by_class_score=True, argmax_fill_min_prob=0.1
    ),
    "wc": dict(
        conf_threshold=0.65, rank_by_class_score=True, argmax_fill_min_prob=None
    ),
}

_DEFAULT_CONTINUOUS_CRITERION = dict(
    type="SmoothL1RegressionLoss", beta=1.0, reduction="mean"
)
_DEFAULT_CATEGORICAL_CRITERION = dict(type="CrossEntropyHeadLoss", reduction="mean")


def normalize_query_heads(heads):
    """Normalize a ``query_heads`` config into a list of fully-specified dicts.

    Each head is one of two kinds:

    * ``kind="continuous"`` -- a regression head. Keys: ``dim`` (default 1),
      ``aggregation`` ('mean'|'first', default 'mean'), ``squeeze``
      (default ``dim == 1``), ``criterion`` (default SmoothL1).
    * ``kind="categorical"`` -- an extra classification head (in addition to the
      primary PID head). Keys: ``num_classes`` (required), ``aggregation``
      ('mode'|'first', default 'mode'), ``criterion`` (default CE).

    Common keys: ``name`` (required, unique), ``pred_key`` (default
    ``pred_<name>``), ``target_key`` (default ``name``), ``loss_weight``
    (default 1.0), and ``required`` (default True).
    """
    targets = []
    seen = set()
    for cfg in heads or []:
        cfg = deepcopy(cfg)
        name = cfg.get("name")
        if not name:
            raise ValueError("Each query head needs a non-empty 'name'")
        if name in seen:
            raise ValueError(f"Duplicate query head name: {name!r}")
        seen.add(name)

        kind = cfg.setdefault("kind", "continuous")
        if kind not in ("continuous", "categorical"):
            raise ValueError(
                f"query head {name!r}: kind must be 'continuous' or 'categorical', "
                f"got {kind!r}"
            )

        cfg.setdefault("pred_key", f"pred_{name}")
        if cfg["pred_key"] in _RESERVED_OUTPUT_KEYS:
            raise ValueError(
                f"query head {name!r} uses reserved pred_key {cfg['pred_key']!r}"
            )
        cfg.setdefault("target_key", name)
        cfg.setdefault("loss_weight", 1.0)
        cfg.setdefault("required", True)

        if kind == "continuous":
            cfg["dim"] = int(cfg.get("dim", 1))
            cfg.setdefault("aggregation", "mean")
            cfg.setdefault("squeeze", cfg["dim"] == 1)
            cfg.setdefault("criterion", deepcopy(_DEFAULT_CONTINUOUS_CRITERION))
            if cfg["aggregation"] not in ("mean", "first"):
                raise ValueError(
                    f"continuous head {name!r}: aggregation must be 'mean'|'first'"
                )
        else:  # categorical
            if "num_classes" not in cfg:
                raise ValueError(f"categorical head {name!r} requires 'num_classes'")
            cfg["num_classes"] = int(cfg["num_classes"])
            cfg["dim"] = cfg["num_classes"]
            cfg.setdefault("aggregation", "mode")
            cfg.setdefault("squeeze", False)
            cfg.setdefault("criterion", deepcopy(_DEFAULT_CATEGORICAL_CRITERION))
            if cfg["aggregation"] not in ("mode", "first"):
                raise ValueError(
                    f"categorical head {name!r}: aggregation must be 'mode'|'first'"
                )
        targets.append(cfg)
    return targets


class GenericMultiTaskDecoder(nn.Module):
    """Masked-cross-attention query decoder (loosely Mask2Former / OneFormer3D),
    with per-label class and regression heads.
    """

    def __init__(
        self,
        label_specs,
        full_in_channels,
        hidden_channels,
        num_heads,
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
        upcast_attention=False,
        pos_emb=True,
        center_pos_k=False,
        mlp_point_proj=False,
    ):
        nn.Module.__init__(self)
        self.labels = tuple(label_specs.keys())
        self.label_specs = label_specs
        query_slices = {}
        query_start = 0
        for label, spec in label_specs.items():
            query_end = query_start + int(spec["num_queries"])
            query_slices[label] = (query_start, query_end)
            query_start = query_end
        self.query_slices = query_slices
        self.total_queries = query_start

        # ---- shared decoder modules ------------------------------------- #
        self.mask_channels = hidden_channels
        self.num_queries = self.total_queries
        self.query_feat = nn.Embedding(self.total_queries, hidden_channels)
        self.query_embed = nn.Embedding(self.total_queries, hidden_channels)
        self.pos_emb = (
            nn.Sequential(
                nn.Linear(3, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            if pos_emb
            else None
        )
        # Removing the event-wide offset keeps positional keys stable in half precision.
        self.center_pos_k = center_pos_k

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
                    upcast_attention=upcast_attention,
                    attn_mask_eps=1e-6,
                )
                for i in range(depth)
            ]
        )
        self.final_norm = norm_layer(hidden_channels)
        self.full_point_proj = (
            MLP(full_in_channels, hidden_channels, hidden_channels)
            if mlp_point_proj
            else nn.Linear(full_in_channels, hidden_channels)
        )

        # ---- per-label / per-query heads -------------------------------- #
        self.cls_pred_by_label = nn.ModuleDict()  # primary PID head (matching)
        self.stuff_head_by_label = nn.ModuleDict()  # auxiliary per-label stuff loss
        self.head_by_key = nn.ModuleDict()  # extra categorical/continuous
        self.query_heads_by_label = {}

        for label, spec in label_specs.items():
            out_classes = int(spec["num_classes"]) + 1
            self.cls_pred_by_label[label] = (
                MLP(hidden_channels, hidden_channels, out_classes)
                if mlp_point_proj
                else nn.Linear(hidden_channels, out_classes)
            )
            if spec["use_stuff_head"]:
                self.stuff_head_by_label[label] = nn.Sequential(
                    nn.Linear(full_in_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, 1),
                )

            heads = normalize_query_heads(spec.get("query_heads"))
            self.query_heads_by_label[label] = heads
            for head in heads:
                head_key = self._head_key(label, head["name"])
                self.head_by_key[head_key] = MLP(
                    hidden_channels, hidden_channels, int(head["dim"])
                )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward_decoder(self, point: Point, return_aux: bool = False):
        """Decode packed events, retaining intermediate predictions for training."""
        point_proj = point.feat  # projection done in caller
        pos_k = (
            self.pos_emb(point.coord)
            if self.pos_emb and not self.center_pos_k
            else None
        )
        if self.pos_emb and self.center_pos_k:
            with torch.autocast(point.coord.device.type, enabled=False):
                pos_k = self.pos_emb(point.coord.float())
            counts = offset2bincount(point.offset).to(torch.long)
            event = torch.repeat_interleave(
                torch.arange(len(counts), device=pos_k.device), counts
            )
            mean = (
                pos_k.new_zeros(len(counts), pos_k.shape[1]).index_add_(0, event, pos_k)
                / counts[:, None]
            )
            pos_k = pos_k - mean[event]
        cu_seqlens_kv = torch.cat(
            [point.offset.new_zeros(1), point.offset]
        ).int()  # [B + 1]
        max_seqlen_kv = cu_seqlens_kv.diff().max()

        batch_size = len(point.offset)
        q = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        pos_q = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_counts = torch.full(
            (batch_size,), self.total_queries, device=q.device, dtype=torch.long
        )
        cu_seqlens_q = torch.cat(
            [query_counts.new_zeros(1), query_counts.cumsum(dim=0)]
        ).int()
        max_seqlen_q = int(query_counts.max().item()) if query_counts.numel() > 0 else 0

        q = q.reshape(-1, self.mask_channels)  # [B * Q, C]
        pos_q = pos_q.reshape(-1, self.mask_channels) if pos_q is not None else None

        aux_outputs = []
        aux_p_hat_list = []
        query_dtype = q.dtype
        for blk in self.blocks:
            q, mask_logits, mask_embed, _ = blk(
                q,
                point_proj,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                pos_q,
                pos_k,
            )
            # Query embeddings determine residual and mask precision under autocast.
            q = q.to(query_dtype)
            mask_logits = mask_logits.to(query_dtype)
            if return_aux:
                aux_outputs.append(self.final_norm(q))
                aux_p_hat_list.append(mask_logits)

        outputs = {
            "out_q": self.final_norm(q),
            "mask_embed": mask_embed,
            "final_mask_logits": mask_logits,
        }
        if return_aux:
            outputs["aux_q_list"] = aux_outputs[:-1]
            outputs["aux_mask_logits_list"] = aux_p_hat_list[:-1]
        return outputs

    @staticmethod
    def _head_key(label, name):
        return f"{label}__{name}".replace(".", "_").replace("/", "_")

    def _compute_head_embeddings(self, label, q_features):
        embeddings = {}
        for head in self.query_heads_by_label.get(label, []):
            pred = self.head_by_key[self._head_key(label, head["name"])](q_features)
            if head.get("squeeze") and pred.dim() == 2 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            embeddings[head["name"]] = (head, pred)
        return embeddings

    def _compute_label_predictions(self, label, q_features, mask_logits, point):
        class_embed = self.cls_pred_by_label[label](q_features)
        head_embeddings = self._compute_head_embeddings(label, q_features)

        pred_masks = []
        pred_cls = []
        pred_logits = []
        pred_heads = {name: [] for name in head_embeddings}

        C = int(self.label_specs[label]["num_classes"])
        label_q_start, label_q_end = self.query_slices[label]
        total_q = self.total_queries

        B = point.offset.shape[0]
        counts = offset2bincount(point.offset).to(torch.long)
        point_offsets = torch.cat([counts.new_zeros(1), point.offset])

        for b in range(B):
            P_b = counts[b].item()
            q_start = b * total_q + label_q_start
            q_end = b * total_q + label_q_end
            p_start, p_end = point_offsets[b], point_offsets[b + 1]

            mask_logits_b = mask_logits[q_start:q_end, p_start:p_end]
            cls_b = class_embed[q_start:q_end]

            mask_logits_b = mask_logits_b.contiguous()
            cls_b = cls_b.contiguous()

            pred_masks.append(mask_logits_b)
            pred_cls.append(cls_b)
            for name, (_, values) in head_embeddings.items():
                pred_heads[name].append(values[q_start:q_end].contiguous())

            if mask_logits_b.shape[0] > 0:
                s = mask_logits_b.transpose(0, 1).unsqueeze(-1)
                c = cls_b[:, :C].unsqueeze(0)
                logits_b = torch.logsumexp(s + c, dim=1)
            else:
                logits_b = mask_logits.new_zeros((P_b, C))
            pred_logits.append(logits_b)

        pred_logits = (
            torch.cat(pred_logits, dim=0)
            if pred_logits
            else mask_logits.new_zeros((0, C))
        )

        output = {
            "pred_masks": pred_masks,
            "pred_logits": pred_cls,
            "seg_logits": pred_logits,
        }
        if pred_heads:
            output["pred_regression"] = pred_heads  # back-compat alias for postprocess
            for name, values in pred_heads.items():
                head = head_embeddings[name][0]
                output[head["pred_key"]] = values
        return output

    def forward(self, point: Point, return_geometry=False):
        point_full = point

        decoder_point = Point(point_full.copy())
        decoder_point.feat = self.full_point_proj(decoder_point.feat)

        return_aux = self.training
        decoder_outputs = self._forward_decoder(decoder_point, return_aux=return_aux)
        out_q = decoder_outputs["out_q"]

        final_mask_logits = decoder_outputs["final_mask_logits"]
        aux_q_list = decoder_outputs.get("aux_q_list", [])
        aux_mask_logits_list = decoder_outputs.get("aux_mask_logits_list", [])
        outputs_by_label = {}
        for label in self.labels:
            predictions = self._compute_label_predictions(
                label, out_q, final_mask_logits, point_full
            )
            if return_aux:
                aux_outputs = []
                for aux_q, aux_mask_logits in zip(aux_q_list, aux_mask_logits_list):
                    aux_outputs.append(
                        self._compute_label_predictions(
                            label, aux_q, aux_mask_logits, point_full
                        )
                    )
                if aux_outputs:
                    predictions["aux_outputs"] = aux_outputs

            if label in self.stuff_head_by_label:
                stuff_logits = self.stuff_head_by_label[label](point_full.feat).squeeze(
                    -1
                )
                predictions["stuff_logits"] = stuff_logits
                predictions["stuff_probs"] = stuff_logits.sigmoid()
            outputs_by_label[label] = predictions

        if return_geometry:
            return outputs_by_label, decoder_outputs["mask_embed"], decoder_point.feat
        return outputs_by_label


@MODELS.register_module("panda-detector-v2m1")
class PandaV2Detector(PointModel):
    """Panda instance detector; ``backbone=None`` trains the decoder on input features. Attention runs with the
    process's attention backend (``pimm.ops.set_attention_backend``)."""

    @staticmethod
    def _select_value(value, labels, label, idx, default=None):
        if value is None:
            return default
        if isinstance(value, dict):
            return value.get(label, default)
        if isinstance(value, (list, tuple)) and len(value) == len(labels):
            return value[idx]
        return value

    @classmethod
    def _select_stuff_classes(cls, value, labels, label, idx, default=None):
        if value is None:
            return default
        if isinstance(value, dict):
            return value.get(label, default)
        if isinstance(value, (list, tuple)) and len(value) == len(labels):
            if all(isinstance(v, (list, tuple, set)) for v in value):
                return value[idx]
        return value

    def up_cast(self, point: Point) -> Point:
        enc_mode = getattr(self.backbone, "enc_mode", True)
        return point.upcast() if enc_mode else point

    def _task_input(self, input_dict: Dict, label: str) -> Dict:
        spec = self.label_specs[label]
        instance_key = spec["instance_key"]
        segment_key = spec["segment_key"]
        if instance_key not in input_dict:
            raise KeyError(f"Missing instance labels for {label!r}: {instance_key}")
        if segment_key not in input_dict:
            raise KeyError(f"Missing semantic labels for {label!r}: {segment_key}")
        task_input = dict(input_dict)
        task_input["instance"] = input_dict[instance_key]
        task_input["segment"] = input_dict[segment_key]
        return task_input

    @staticmethod
    def _compute_stuff_loss(decoder, label, stuff_logits, task_input):
        segment = task_input["segment"]
        if (
            isinstance(segment, torch.Tensor)
            and segment.dim() == 2
            and segment.shape[1] == 1
        ):
            segment = segment.squeeze(1)
        stuff_target = torch.zeros_like(stuff_logits)
        for stuff_class in decoder.label_specs[label]["stuff_classes"]:
            stuff_target[segment == int(stuff_class)] = 1.0
        return F.binary_cross_entropy_with_logits(
            stuff_logits, stuff_target, reduction="mean"
        )

    def _sync_scalar_outputs(self, return_dict: Dict) -> Dict:
        return reduce_scalar_outputs_for_logging(return_dict)

    def __init__(
        self,
        full_in_channels,
        hidden_channels,
        num_heads,
        labels=("particle",),
        num_queries=None,
        num_classes=None,
        label_configs=None,
        overlap=None,  # None -> per-label `label_configs[...]["overlap"]` (default False);
        # a bool here forces it for ALL labels (see _select_value)
        loss_weights=None,
        eval_label=None,
        backbone=None,
        criteria=None,
        criteria_by_label=None,
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
        upcast_attention=False,
        pos_emb=True,
        center_pos_k=False,
        use_stuff_head=False,
        stuff_classes=None,
        mlp_point_proj=False,
        stuff_threshold=0.5,
        mask_threshold=0.5,
        conf_threshold=None,
        nms_kernel="gaussian",
        nms_sigma=2.0,
        nms_pre=-1,
        nms_max=-1,
        min_points=2,
        fill_uncovered=False,
        average_placements=False,  # inference only: mean encoder features over the 8 voxel-grid placements
        zero_init_positions=False,  # zero query_embed and pos_emb's output layer, for a synthetically pretrained core
        decoding="panda_v2",
    ):
        PointModel.__init__(self)
        if isinstance(labels, str):
            labels = (labels,)
        self.labels = tuple(labels)
        self.average_placements = bool(average_placements)
        if len(self.labels) == 0:
            raise ValueError("labels must contain at least one label name")

        self.label_specs = self._build_label_specs(
            self.labels,
            label_configs=label_configs,
            num_queries=num_queries,
            num_classes=num_classes,
            stuff_classes=stuff_classes,
            use_stuff_head=use_stuff_head,
            loss_weights=loss_weights,
            overlap=overlap,
        )
        self.eval_label = eval_label or self.labels[-1]
        if self.eval_label not in self.label_specs:
            raise ValueError(
                f"eval_label={self.eval_label!r} must be one of {self.labels}"
            )

        self.backbone = (
            build_model(backbone) if backbone is not None else nn.Identity()
        )
        self.criteria_by_label = {
            label: build_criteria(
                self._criteria_cfg(label, criteria, criteria_by_label)
            )
            for label in self.labels
        }

        self.decoder = GenericMultiTaskDecoder(
            label_specs=self.label_specs,
            full_in_channels=full_in_channels,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            layer_scale=layer_scale,
            norm_layer=norm_layer,
            act_layer=act_layer,
            upcast_attention=upcast_attention,
            pos_emb=pos_emb,
            center_pos_k=center_pos_k,
            mlp_point_proj=mlp_point_proj,
        )
        if zero_init_positions:  # the core's attention had no query or point positional terms, so both start as no-ops
            nn.init.zeros_(self.decoder.query_embed.weight)
            if self.decoder.pos_emb is not None:
                nn.init.zeros_(self.decoder.pos_emb[-1].weight)
                nn.init.zeros_(self.decoder.pos_emb[-1].bias)

        if decoding not in _DECODING_DEFAULTS:
            raise ValueError(f"Unknown decoding rule: {decoding!r}")
        self.decoding = decoding
        decoding_defaults = _DECODING_DEFAULTS[decoding]
        self.postprocess_cfg = dict(
            stuff_threshold=stuff_threshold,
            mask_threshold=mask_threshold,
            conf_threshold=(
                decoding_defaults["conf_threshold"]
                if conf_threshold is None
                else conf_threshold
            ),
            nms_kernel=nms_kernel,
            nms_sigma=nms_sigma,
            nms_pre=nms_pre,
            nms_max=nms_max,
            min_points=min_points,
            fill_uncovered=fill_uncovered,
            rank_by_class_score=decoding_defaults["rank_by_class_score"],
            argmax_fill_min_prob=decoding_defaults["argmax_fill_min_prob"],
        )

    def _criteria_cfg(self, label, criteria, criteria_by_label):
        """Auto-inject the per-label head specs + overlap flag into the loss cfg
        so the user does not have to duplicate them."""
        base = (
            criteria_by_label[label]
            if criteria_by_label is not None and label in criteria_by_label
            else criteria
        )
        spec = self.label_specs[label]

        def _inject(cfg):
            cfg = deepcopy(cfg)
            if isinstance(cfg, dict):
                if cfg.get("type") in (
                    "FastUnifiedInstanceLoss",
                    "ConsistentMatchingInstanceLoss",
                ):
                    cfg.setdefault("panda_v2", True)
                    cfg.setdefault("query_heads", spec.get("query_heads"))
                    cfg.setdefault("overlap", spec.get("overlap", False))
                return cfg
            if isinstance(cfg, (list, tuple)):
                return type(cfg)(_inject(c) for c in cfg)
            return cfg

        return _inject(base)

    @classmethod
    def _build_label_specs(
        cls,
        labels,
        label_configs=None,
        num_queries=None,
        num_classes=None,
        stuff_classes=None,
        use_stuff_head=True,
        loss_weights=None,
        overlap=None,
    ):
        label_configs = label_configs or {}
        specs = {}
        for idx, label in enumerate(labels):
            spec = deepcopy(label_configs.get(label, {}))

            # precedence: explicit top-level num_queries > label_configs > 32
            queries = cls._select_value(
                num_queries, labels, label, idx, spec.get("num_queries", 32)
            )
            classes = cls._select_value(
                num_classes, labels, label, idx, spec.get("num_classes")
            )
            stuff = cls._select_stuff_classes(
                stuff_classes, labels, label, idx, spec.get("stuff_classes", [])
            )
            task_use_stuff = cls._select_value(use_stuff_head, labels, label, idx, True)
            loss_weight = cls._select_value(loss_weights, labels, label, idx, 1.0)
            label_overlap = cls._select_value(
                overlap, labels, label, idx, spec.get("overlap", False)
            )

            if queries is None:
                raise ValueError(f"num_queries is required for label {label!r}")
            if classes is None:
                raise ValueError(
                    f"num_classes is required for label {label!r}; provide "
                    "num_classes or label_configs"
                )
            if "instance_key" not in spec:
                spec["instance_key"] = f"instance_{label}"
            if "segment_key" not in spec:
                spec["segment_key"] = f"segment_{label}"

            spec["num_queries"] = int(queries)
            spec["num_classes"] = int(classes)
            spec["stuff_classes"] = list(stuff or [])
            spec["use_stuff_head"] = bool(task_use_stuff)
            spec["loss_weight"] = float(loss_weight)
            spec["overlap"] = bool(label_overlap)
            spec["query_heads"] = normalize_query_heads(spec.get("query_heads"))
            specs[label] = spec
        return specs

    def _prediction_keys_for_label(self, label):
        return [
            head["pred_key"] for head in self.label_specs[label].get("query_heads", [])
        ]

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        point_full = self.up_cast(point)
        if self.average_placements and not self.training:
            # Grid shifts change pooling boundaries while keeping the original point order.
            grid = input_dict["grid_coord"]
            for offset in PLACEMENTS[1:]:
                shifted = dict(input_dict, grid_coord=grid + grid.new_tensor(offset))
                point_full.feat = (
                    point_full.feat + self.up_cast(self.backbone(Point(shifted))).feat
                )
            point_full.feat = point_full.feat / len(PLACEMENTS)

        decoder_point = Point(point_full.copy())
        outputs_by_label = self.decoder(decoder_point)

        loss = None
        return_dict = {}
        emit_predictions = not self.training

        for label in self.labels:
            task_input = None
            spec = self.label_specs[label]
            segment_key = spec["segment_key"]
            has_targets = (
                spec["instance_key"] in input_dict and segment_key in input_dict
            )
            if has_targets or self.training:
                task_input = self._task_input(input_dict, label)
            outputs = outputs_by_label[label]

            if emit_predictions:
                return_dict[f"{label}_seg_logits"] = outputs["seg_logits"]
                return_dict[f"{label}_pred_logits"] = outputs.get("pred_logits")
                return_dict[f"{label}_pred_masks"] = outputs.get("pred_masks")
                for pred_key in self._prediction_keys_for_label(label):
                    if pred_key in outputs:
                        return_dict[f"{label}_{pred_key}"] = outputs[pred_key]
                if "stuff_probs" in outputs:
                    return_dict[f"{label}_stuff_probs"] = outputs["stuff_probs"]

            if task_input is not None:
                loss_result = self.criteria_by_label[label](outputs, task_input)
                if isinstance(loss_result, tuple):
                    task_loss, components = loss_result
                else:
                    task_loss, components = loss_result, {}
                task_loss = task_loss * spec["loss_weight"]
                loss = task_loss if loss is None else loss + task_loss
                return_dict[f"{label}_loss"] = task_loss
                for key, value in components.items():
                    return_dict[f"{label}_{key}"] = value

                if spec["use_stuff_head"] and "stuff_logits" in outputs:
                    stuff_loss = self._compute_stuff_loss(
                        self.decoder, label, outputs["stuff_logits"], task_input
                    )
                    loss = loss + stuff_loss
                    return_dict[f"{label}_stuff_loss"] = stuff_loss

        if emit_predictions or return_point:
            primary_outputs = outputs_by_label[self.eval_label]
            primary_spec = self.label_specs[self.eval_label]
            point_full.outputs = primary_outputs
            point_full.outputs_by_label = outputs_by_label
            point_full.pred_cls = primary_outputs["pred_logits"]
            point_full.pred_masks = primary_outputs["pred_masks"]
            point_full.pred_logits = primary_outputs["seg_logits"]
            point_full.pred_regression = primary_outputs.get("pred_regression")
            if primary_spec["instance_key"] in input_dict:
                point_full.instance = input_dict[primary_spec["instance_key"]]
            primary_segment_key = primary_spec["segment_key"]
            if primary_segment_key in input_dict:
                point_full.segment = input_dict[primary_segment_key]

            if emit_predictions:
                return_dict["outputs_by_label"] = outputs_by_label
                return_dict["seg_logits_by_label"] = {
                    label: o["seg_logits"] for label, o in outputs_by_label.items()
                }
                return_dict["pred_logits_by_label"] = {
                    label: o["pred_logits"] for label, o in outputs_by_label.items()
                }
                return_dict["pred_masks_by_label"] = {
                    label: o["pred_masks"] for label, o in outputs_by_label.items()
                }
                return_dict["pred_regression_by_label"] = {
                    label: o.get("pred_regression", {})
                    for label, o in outputs_by_label.items()
                }
                return_dict["seg_logits"] = primary_outputs["seg_logits"]
                return_dict["pred_logits"] = primary_outputs.get("pred_logits")
                return_dict["pred_masks"] = primary_outputs.get("pred_masks")
                return_dict["pred_regression"] = primary_outputs.get("pred_regression")
                for pred_key in self._prediction_keys_for_label(self.eval_label):
                    if pred_key in primary_outputs:
                        return_dict[pred_key] = primary_outputs[pred_key]
                return_dict["stuff_probs"] = primary_outputs.get("stuff_probs")
                return_dict["point_counts"] = offset2bincount(point_full.offset)

        if loss is not None:
            return_dict["loss"] = loss
        if return_point:
            return_dict["point"] = point_full
        return self._sync_scalar_outputs(return_dict)

    @torch.inference_mode()
    def predict(self, input_dict):
        """Return masks, class probabilities and spherical-cap geometry per event."""
        point = self.up_cast(self.backbone(Point(input_dict)))
        outputs, mask_embed, features = self.decoder(point, return_geometry=True)
        norm = self.decoder.blocks[-1].norm_kv
        axis = mask_embed.float() * norm.weight.float()
        axis = axis - axis.mean(dim=1, keepdim=True)
        magnitude = axis.norm(dim=1).clamp_min(1e-12)
        sharpness = magnitude * mask_embed.shape[1] ** 0.5
        threshold = -(mask_embed.float() @ norm.bias.float()) / sharpness
        axis = axis / magnitude[:, None]
        from pimm.datasets.sphere_prior.geometry import to_sphere

        counts = offset2bincount(point.offset).tolist()
        spheres = to_sphere(features.float()).split(counts)
        start, end = self.decoder.query_slices[self.eval_label]
        total = self.decoder.total_queries
        output = outputs[self.eval_label]
        return [
            dict(
                mask_logits=masks.float(),
                class_probs=logits.float().softmax(-1),
                sphere_points=sphere,
                cap_axis=axis[b * total + start : b * total + end],
                cap_threshold=threshold[b * total + start : b * total + end],
                cap_sharpness=sharpness[b * total + start : b * total + end],
            )
            for b, (masks, logits, sphere) in enumerate(
                zip(output["pred_masks"], output["pred_logits"], spheres)
            )
        ]

    def postprocess(
        self,
        forward_output: dict,
        label: str = None,
        stuff_threshold: float = None,
        mask_threshold: float = None,
        conf_threshold: float = None,
        nms_kernel: str = None,
        nms_sigma: float = None,
        nms_pre: int = None,
        nms_max: int = None,
        min_points: int = None,
        background_class_label: int = None,
        fill_uncovered: bool = None,
        rank_by_class_score: bool = None,
        dedup_iou: float = None,
        decoding: str = None,
    ):
        label = label or self.eval_label
        cfg = self.postprocess_cfg.copy()
        decoding = self.decoding if decoding is None else decoding
        if decoding not in _DECODING_DEFAULTS:
            raise ValueError(f"Unknown decoding rule: {decoding!r}")
        if decoding != self.decoding:
            cfg.update(_DECODING_DEFAULTS[decoding])
        cfg["stuff_class_label"] = (
            self.label_specs[label].get("stuff_class_label")
            if decoding == "panda_v2"
            else None
        )
        overrides = {
            "stuff_threshold": stuff_threshold,
            "mask_threshold": mask_threshold,
            "conf_threshold": conf_threshold,
            "nms_kernel": nms_kernel,
            "nms_sigma": nms_sigma,
            "nms_pre": nms_pre,
            "nms_max": nms_max,
            "min_points": min_points,
            "background_class_label": background_class_label,
            "fill_uncovered": fill_uncovered,
            "rank_by_class_score": rank_by_class_score,
        }
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value

        if "outputs_by_label" in forward_output:
            task_output = forward_output["outputs_by_label"][label]
            pred_masks = task_output["pred_masks"]
            pred_logits = task_output["pred_logits"]
            stuff_probs = task_output.get("stuff_probs")
            pred_regression = task_output.get("pred_regression")
            point_counts = forward_output.get("point_counts")
        else:
            pred_masks = forward_output["pred_masks"]
            pred_logits = forward_output["pred_logits"]
            stuff_probs = forward_output.get("stuff_probs")
            pred_regression = forward_output.get("pred_regression")
            point_counts = forward_output.get("point_counts")

        if point_counts is None:
            point_counts = forward_output.get(f"{label}_point_counts")

        if self.label_specs[label].get("overlap", False):
            if decoding == "wc":
                cfg.update(dedup_iou=1.0, min_eg_points=50)
            if (
                dedup_iou is not None
            ):  # near-duplicate removal of overlapping rings (1.0 turns it off)
                cfg["dedup_iou"] = dedup_iou
            return postprocess_overlap_batch(
                pred_masks=pred_masks,
                pred_logits=pred_logits,
                num_classes=int(self.label_specs[label]["num_classes"]),
                pred_regression=pred_regression,
                **cfg,
            )
        return postprocess_batch(
            pred_masks=pred_masks,
            pred_logits=pred_logits,
            stuff_probs=stuff_probs,
            pred_regression=pred_regression,
            point_counts=point_counts,
            stuff_classes=self.label_specs[label]["stuff_classes"],
            **cfg,
        )
