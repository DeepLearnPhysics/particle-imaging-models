"""Unified, fully-configurable Panda detector (detector-v4).

A single query-based (DETR / Mask2Former-style) instance/panoptic decoder that
subsumes the whole panda detector family:

* ``detector-v1m1``  -- mask + PID.
* ``detector-v1m2``  -- + momentum / IoU heads.
* ``detector-v3m1/2``-- multi-label (shared decoder, per-label query slices).
* ``detector-v3m3``  -- generic per-query continuous regression heads.
* ``ring-panoptic-detector`` -- overlap-aware (per-query sigmoid masks).

It exposes four axes as pure config:

1. **Pre-cross-attention point removal** (``point_filter``): an optional,
   pluggable module that drops points (e.g. "stuff" / low-energy) *before* the
   decoder cross-attention; mask logits are scattered back to the full cloud.
2. **Arbitrary per-query heads** (``query_heads``): any number of categorical
   *and* continuous heads per label, on top of the primary categorical (PID)
   head that drives matching.
3. **Configurable query count** per label (inherited from the v3 machinery).
4. **Overlapping queries** (``overlap`` per label): switches ground-truth
   construction + fusion between mutually-exclusive (one-hot, greedy NMS) and
   overlap-aware (multi-hot membership, independent sigmoid).

The decoder (``GenericMultiTaskDecoder``) lives in this file and composes the
low-level transformer layers from ``layers.py`` (SDPA masked cross-attention with
a bounded ``log(sigmoid+eps)`` mask). It is **not** forced to fp32; precision is
controlled by ``upcast_softmax`` (cheap, softmax-only) and an optional
``decoder_fp32`` escape hatch.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from timm.layers import trunc_normal_

from pimm.models.builder import MODELS, build_model
from pimm.models.losses import build_criteria
from pimm.models.modules import PointModel
from pimm.models.utils.misc import offset2bincount
from pimm.models.utils.structure import Point
from pimm.utils.comm import get_world_size, reduce_scalar_outputs_for_logging

from .layers import MLP, Block  # low-level transformer layers; the decoder lives here
from .postprocess import postprocess_batch, postprocess_overlap_batch


# Default per-label specs (instance/segment keys, class counts, stuff classes).
_DEFAULT_LABEL_SPECS = {
    "particle": dict(
        instance_key="instance_particle",
        segment_key="segment_particle",
        segment_fallback_key="segment_pid",
        num_classes=6,
        stuff_classes=[5],
    ),
    "interaction": dict(
        instance_key="instance_interaction",
        segment_key="segment_interaction",
        num_classes=2,
        stuff_classes=[0],
    ),
}

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
    "filter_logits",
}


def _as_list(value):
    """Accept a list, a single dict (with ``name``), or a ``{name: cfg}`` dict."""
    if value is None:
        return []
    if isinstance(value, dict):
        if "name" in value:
            return [value]
        return [dict(cfg, name=name) for name, cfg in value.items()]
    return list(value)


# --------------------------------------------------------------------------- #
# Head-spec normalization                                                      #
# --------------------------------------------------------------------------- #
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
    (default 1.0), ``use_class_logits`` / ``detach_class_logits`` (default
    False -- condition the head input on the primary class logits), ``required``
    (default True), ``head_mlp`` (default True -- MLP vs Linear).
    """
    targets = []
    seen = set()
    for cfg in _as_list(heads):
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
        cfg.setdefault("use_class_logits", False)
        cfg.setdefault("detach_class_logits", False)
        cfg.setdefault("required", True)
        cfg.setdefault("head_mlp", True)

        if kind == "continuous":
            cfg["dim"] = int(cfg.get("dim", cfg.get("out_dim", 1)))
            cfg.setdefault("aggregation", "mean")
            cfg.setdefault("squeeze", cfg["dim"] == 1)
            cfg.setdefault("criterion", deepcopy(_DEFAULT_CONTINUOUS_CRITERION))
            if cfg["aggregation"] not in ("mean", "first"):
                raise ValueError(
                    f"continuous head {name!r}: aggregation must be 'mean'|'first'"
                )
        else:  # categorical
            if "num_classes" not in cfg:
                raise ValueError(
                    f"categorical head {name!r} requires 'num_classes'"
                )
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


def _with_query_heads(label_configs, labels, query_heads):
    """Fold a top-level ``query_heads`` (dict-by-label or list for single label)
    into per-label ``label_configs``."""
    label_configs = deepcopy(label_configs or {})
    if query_heads is None:
        return label_configs
    if isinstance(query_heads, dict) and not (
        "name" in query_heads or "kind" in query_heads
    ):
        for label, heads in query_heads.items():
            label_configs.setdefault(label, {})["query_heads"] = heads
        return label_configs
    if len(labels) != 1:
        raise ValueError(
            "Top-level query_heads must be a dict keyed by label when training "
            "multiple labels"
        )
    label_configs.setdefault(labels[0], {})["query_heads"] = query_heads
    return label_configs


# --------------------------------------------------------------------------- #
# Pre-cross-attention point filter                                             #
# --------------------------------------------------------------------------- #
class PointFilter(nn.Module):
    """Optional, pluggable removal of points before the decoder cross-attention.

    Types:

    * ``"stuff"`` -- a learned per-point binary classifier; keeps points with
      ``sigmoid(logit) < threshold``. During training, if ``train_filter_use_gt``
      and a GT segment is available, keeps points whose class is not in
      ``drop_classes``. Emits ``filter_logits`` for an auxiliary BCE.
    * ``"energy_threshold"`` -- fixed rule: keep points whose feature channel
      ``energy_channel`` exceeds ``energy_threshold`` (operates on the *raw*
      backbone feature; no learnable params).

    Returns ``(keep_mask, filter_logits_or_None)``. Always keeps at least one
    point per non-empty cloud so the decoder still runs.
    """

    def __init__(
        self,
        type: str = "stuff",
        full_in_channels: int = 0,
        hidden_channels: int = 256,
        threshold: float = 0.5,
        drop_classes=None,
        train_filter_use_gt: bool = False,
        energy_channel: int = 0,
        energy_threshold: float = 0.0,
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
    def _ensure_nonempty(keep_mask):
        if not keep_mask.any():
            # Keep a single point so the decoder doesn't crash on an empty cloud.
            keep_mask = keep_mask.clone()
            keep_mask[0] = True
        return keep_mask

    def forward(self, feat, gt_segment=None, training=False):
        filter_logits = None
        if self.type == "stuff":
            filter_logits = self.head(feat).squeeze(-1)  # (N,)
            if (
                training
                and self.train_filter_use_gt
                and gt_segment is not None
                and len(self.drop_classes) > 0
            ):
                segment = gt_segment
                if segment.dim() == 2 and segment.shape[1] == 1:
                    segment = segment.squeeze(1)
                is_drop = torch.zeros_like(segment, dtype=torch.bool)
                for cls_id in self.drop_classes:
                    is_drop |= segment == int(cls_id)
                keep_mask = ~is_drop
            else:
                keep_mask = filter_logits.sigmoid() < self.threshold
        else:  # energy_threshold
            keep_mask = feat[:, self.energy_channel] > self.energy_threshold
        return self._ensure_nonempty(keep_mask), filter_logits


# --------------------------------------------------------------------------- #
# Decoder                                                                      #
# --------------------------------------------------------------------------- #
class GenericMultiTaskDecoder(nn.Module):
    """Masked-cross-attention query decoder (loosely Mask2Former / OneFormer3D),
    generalized for multiple labels and arbitrary per-query heads + a point filter.
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
        pre_norm=True,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        pos_emb=True,
        enc_mode=False,
        query_type: Literal["learned"] = "learned",
        mlp_point_proj=False,
        supervise_attn_mask=True,
        point_filter_cfg=None,
    ):
        nn.Module.__init__(self)
        self.labels = tuple(label_specs.keys())
        self.label_specs = label_specs
        query_slices = {}
        query_start = 0
        max_classes = 1
        for label, spec in label_specs.items():
            query_end = query_start + int(spec["num_queries"])
            query_slices[label] = (query_start, query_end)
            query_start = query_end
            max_classes = max(max_classes, int(spec["num_classes"]))
        self.query_slices = query_slices
        self.total_queries = query_start

        # ---- shared decoder modules ------------------------------------- #
        self.full_in_channels = full_in_channels
        self.mask_channels = hidden_channels
        self.num_classes = max_classes
        self.enc_mode = enc_mode
        self.num_queries = self.total_queries
        self.query_type = query_type
        if query_type == "learned":
            self.query_feat = nn.Embedding(self.total_queries, hidden_channels)
            self.query_embed = nn.Embedding(self.total_queries, hidden_channels)
        self.pos_emb = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        ) if pos_emb else None

        # attention-mask annealing (configured via set_attn_mask_anneal)
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
                    is_last_block=(i == depth - 1),
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
        # No single-task class head; we use per-label heads instead.
        self.cls_pred = nn.Identity()
        self.cls_pred_by_label = nn.ModuleDict()   # primary PID head (matching)
        self.stuff_head_by_label = nn.ModuleDict()  # auxiliary per-label stuff loss
        self.head_by_key = nn.ModuleDict()          # extra categorical/continuous
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
                in_dim = hidden_channels
                if head["use_class_logits"]:
                    in_dim += int(spec["num_classes"])
                out_dim = int(head["dim"])
                self.head_by_key[head_key] = (
                    MLP(in_dim, hidden_channels, out_dim)
                    if head.get("head_mlp", True)
                    else nn.Linear(in_dim, out_dim)
                )

        # Optional pre-cross-attention point filter (global, shared across labels).
        self.point_filter = None
        if point_filter_cfg:
            cfg = deepcopy(point_filter_cfg)
            cfg.setdefault("full_in_channels", full_in_channels)
            cfg.setdefault("hidden_channels", hidden_channels)
            self.point_filter = PointFilter(**cfg)

        self.apply(self._init_weights)

    # ----- decoder core ---------------------------------------------------- #
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_attn_mask_anneal(
        self,
        enable: bool,
        anneal_steps: int = 10000,
        warmup_steps: int = 0,
        progressive: bool = False,
        progressive_delay: int = 0,
    ):
        """Configure attention-mask annealing for all blocks."""
        self.attn_mask_anneal = enable
        self.attn_mask_anneal_steps = anneal_steps
        self.attn_mask_warmup_steps = warmup_steps
        self.attn_mask_progressive = progressive
        self.attn_mask_progressive_delay = progressive_delay
        for i, block in enumerate(self.blocks):
            block.attn_mask_anneal = enable
            block.attn_mask_anneal_steps = anneal_steps
            if progressive and progressive_delay > 0:
                block.attn_mask_warmup_steps = warmup_steps + (i * progressive_delay)
            else:
                block.attn_mask_warmup_steps = warmup_steps

    def update_anneal_step(self, step: int):
        for block in self.blocks:
            block.set_anneal_step(step)

    def _get_queries(
        self, point: Point
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        batch_size = point.offset.shape[0]
        device = point.feat.device
        max_queries = self.num_queries
        if self.query_type == "learned":
            base_q = self.query_feat.weight  # [Q, C]
            base_pos = self.query_embed.weight if hasattr(self, "query_embed") else None
            q = base_q.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, Q, C]
            pos_q = base_pos.unsqueeze(0).repeat(batch_size, 1, 1) if base_pos is not None else None
            counts = torch.full((batch_size,), max_queries, dtype=torch.int32, device=device)
            valid_mask = torch.ones(batch_size, max_queries, dtype=torch.bool, device=device)
            return q, pos_q, counts, valid_mask
        raise NotImplementedError(f"Invalid query type: {self.query_type}")

    def _forward_decoder(self, point: Point, return_aux: bool = False):
        """Run the transformer decoder blocks. Returns out_q, final_mask_logits,
        query_counts, query_valid (+ aux lists if return_aux)."""
        point_proj = point.feat  # projection done in caller
        pos_k = self.pos_emb(point.coord) if self.pos_emb else None
        cu_seqlens_kv = torch.cat([point.offset.new_zeros(1), point.offset]).int()  # [B + 1]
        max_seqlen_kv = cu_seqlens_kv.diff().max()

        q, pos_q, query_counts, query_valid = self._get_queries(point)
        cu_seqlens_q = torch.cat([query_counts.new_zeros(1), query_counts.cumsum(dim=0)]).int()
        max_seqlen_q = int(query_counts.max().item()) if query_counts.numel() > 0 else 0

        q = q.reshape(-1, self.mask_channels)  # [B * Q, C]
        pos_q = pos_q.reshape(-1, self.mask_channels) if pos_q is not None else None
        query_valid = query_valid.reshape(-1, 1)
        query_valid_f = query_valid.to(q.dtype)

        aux_outputs = []
        aux_p_hat_list = []
        final_mask_logits = None
        for blk in self.blocks:
            q, mask_logits, mask_embed, mask_point_proj = blk(
                q, point_proj, cu_seqlens_q, cu_seqlens_kv,
                max_seqlen_q, max_seqlen_kv, pos_q, pos_k,
            )
            if mask_logits is not None:
                mask_logits = mask_logits * query_valid_f
                final_mask_logits = mask_logits
            q = q * query_valid_f
            if return_aux:
                aux_outputs.append(self.final_norm(q))
                aux_p_hat_list.append(mask_logits if self.supervise_attn_mask else None)

        outputs = {
            "out_q": self.final_norm(q),
            "final_mask_logits": final_mask_logits,
            "query_counts": query_counts.to(torch.long),
            "query_valid": query_valid.squeeze(-1).bool(),
        }
        if return_aux:
            outputs["aux_q_list"] = aux_outputs[:-1]
            outputs["aux_mask_logits_list"] = aux_p_hat_list[:-1]
        return outputs

    def up_cast(self, point):
        """enc_mode=False: no-op. enc_mode=True: walk the pooling hierarchy,
        concatenating multi-scale features up to point resolution."""
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

    # ----- per-query head predictions ------------------------------------- #
    def _compute_head_embeddings(self, label, q_features, class_embed):
        spec = self.label_specs[label]
        num_classes = int(spec["num_classes"])
        embeddings = {}
        for head in self.query_heads_by_label.get(label, []):
            head_input = q_features
            if head["use_class_logits"]:
                cls_input = class_embed[:, :num_classes]
                if head["detach_class_logits"]:
                    cls_input = cls_input.detach()
                head_input = torch.cat([q_features, cls_input], dim=-1)
            pred = self.head_by_key[self._head_key(label, head["name"])](head_input)
            if head.get("squeeze") and pred.dim() == 2 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            embeddings[head["name"]] = (head, pred)
        return embeddings

    def _compute_label_predictions(
        self, label, q_features, mask_logits, point, query_valid
    ):
        class_embed = self.cls_pred_by_label[label](q_features)
        head_embeddings = self._compute_head_embeddings(label, q_features, class_embed)

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
            valid_b = query_valid[q_start:q_end]

            mask_logits_b = mask_logits_b[valid_b]
            cls_b = cls_b[valid_b]

            pred_masks.append(mask_logits_b)
            pred_cls.append(cls_b)
            for name, (_, values) in head_embeddings.items():
                pred_heads[name].append(values[q_start:q_end][valid_b])

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

    # ----- point-filter plumbing ------------------------------------------ #
    def _scatter_mask_logits(self, mask_logits, keep_mask, num_points, dropped_mask_logit=-1e4):
        """Expand (Q_total, N_kept) mask logits back to (Q_total, N_full),
        filling dropped columns with a large-negative sentinel."""
        if mask_logits is None:
            return None
        full = mask_logits.new_full(
            (mask_logits.shape[0], num_points), dropped_mask_logit
        )
        full[:, keep_mask] = mask_logits
        return full

    def forward(self, point: Point):
        point_full = self.up_cast(point)
        num_points = point_full.feat.shape[0]

        # ---- optional pre-cross-attention point removal ------------------- #
        keep_mask = None
        filter_logits = None
        if self.point_filter is not None:
            gt_segment = getattr(point_full, "_filter_gt_segment", None)
            keep_mask, filter_logits = self.point_filter(
                point_full.feat, gt_segment=gt_segment, training=self.training
            )
            decoder_point = point_full[keep_mask]
        else:
            decoder_point = point_full.copy()
        decoder_point.feat = self.full_point_proj(decoder_point.feat)

        return_aux = self.training
        decoder_outputs = self._forward_decoder(decoder_point, return_aux=return_aux)
        query_valid = decoder_outputs["query_valid"]
        out_q = decoder_outputs["out_q"]

        final_mask_logits = decoder_outputs["final_mask_logits"]
        aux_q_list = decoder_outputs.get("aux_q_list", [])
        aux_mask_logits_list = decoder_outputs.get("aux_mask_logits_list", [])
        if keep_mask is not None:
            final_mask_logits = self._scatter_mask_logits(
                final_mask_logits, keep_mask, num_points
            )
            aux_mask_logits_list = [
                self._scatter_mask_logits(m, keep_mask, num_points)
                for m in aux_mask_logits_list
            ]

        outputs_by_label = {}
        for label in self.labels:
            predictions = self._compute_label_predictions(
                label, out_q, final_mask_logits, point_full, query_valid
            )
            if return_aux:
                aux_outputs = []
                for aux_q, aux_mask_logits in zip(aux_q_list, aux_mask_logits_list):
                    if aux_mask_logits is None:
                        continue
                    aux_outputs.append(
                        self._compute_label_predictions(
                            label, aux_q, aux_mask_logits, point_full, query_valid
                        )
                    )
                if aux_outputs:
                    predictions["aux_outputs"] = aux_outputs

            if label in self.stuff_head_by_label:
                stuff_logits = self.stuff_head_by_label[label](
                    point_full.feat
                ).squeeze(-1)
                predictions["stuff_logits"] = stuff_logits
                predictions["stuff_probs"] = stuff_logits.sigmoid()
            outputs_by_label[label] = predictions

        if filter_logits is not None:
            outputs_by_label["__filter_logits__"] = filter_logits
        return outputs_by_label


# --------------------------------------------------------------------------- #
# Detector                                                                     #
# --------------------------------------------------------------------------- #
@MODELS.register_module("detector-v4")
class UnifiedDetector(PointModel):
    """Unified Panda detector: arbitrary per-query heads, optional pre-attention
    point removal, configurable queries, and per-label overlap.

    Self-contained: owns its decoder stack and multi-label plumbing; imports only
    helpers (postprocess, registries, point utils)."""

    # ----- per-label config selection helpers ----------------------------- #
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

    # ----- shared plumbing ------------------------------------------------ #
    def up_cast(self, point: Point) -> Point:
        enc_mode = getattr(self.backbone, "enc_mode", True)
        if not enc_mode:
            return point
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    def update_anneal_step(self, step: int):
        self.decoder.update_anneal_step(step)

    def _segment_key(self, input_dict: Dict, spec: Dict) -> str:
        segment_key = spec["segment_key"]
        if segment_key in input_dict:
            return segment_key
        fallback = spec.get("segment_fallback_key")
        if fallback is not None and fallback in input_dict:
            return fallback
        return segment_key

    def _task_input(self, input_dict: Dict, label: str) -> Dict:
        spec = self.label_specs[label]
        instance_key = spec["instance_key"]
        segment_key = self._segment_key(input_dict, spec)
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
        if isinstance(segment, torch.Tensor) and segment.dim() == 2 and segment.shape[1] == 1:
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
        query_heads=None,
        overlap=None,  # None -> per-label `label_configs[...]["overlap"]` (default False);
                       # a bool here forces it for ALL labels (see _select_value)
        point_filter=None,
        filter_loss_weight=1.0,
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
        pre_norm=True,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        pos_emb=True,
        attn_mask_anneal=False,
        attn_mask_anneal_steps=10000,
        attn_mask_warmup_steps=0,
        attn_mask_progressive=False,
        attn_mask_progressive_delay=0,
        query_type: Literal["learned"] = "learned",
        use_stuff_head=False,
        stuff_classes=None,
        supervise_attn_mask=True,
        mlp_point_proj=False,
        stuff_threshold=0.5,
        mask_threshold=0.5,
        conf_threshold=0.5,
        nms_kernel="gaussian",
        nms_sigma=2.0,
        nms_pre=-1,
        nms_max=-1,
        min_points=2,
        fill_uncovered=False,
    ):
        PointModel.__init__(self)
        if isinstance(labels, str):
            labels = (labels,)
        self.labels = tuple(labels)
        if len(self.labels) == 0:
            raise ValueError("labels must contain at least one label name")

        label_configs = _with_query_heads(label_configs, self.labels, query_heads)
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

        self.point_filter_cfg = deepcopy(point_filter) if point_filter else None
        self.filter_loss_weight = float(filter_loss_weight)

        self.backbone = build_model(backbone)
        self.criteria_by_label = {
            label: build_criteria(self._criteria_cfg(label, criteria, criteria_by_label))
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
            pre_norm=pre_norm,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            pos_emb=pos_emb,
            enc_mode=False,
            query_type=query_type,
            mlp_point_proj=mlp_point_proj,
            supervise_attn_mask=supervise_attn_mask,
            point_filter_cfg=self.point_filter_cfg,
        )

        if attn_mask_anneal:
            self.decoder.set_attn_mask_anneal(
                enable=True,
                anneal_steps=attn_mask_anneal_steps,
                warmup_steps=attn_mask_warmup_steps,
                progressive=attn_mask_progressive,
                progressive_delay=attn_mask_progressive_delay,
            )

        self.postprocess_cfg = dict(
            stuff_threshold=stuff_threshold,
            mask_threshold=mask_threshold,
            conf_threshold=conf_threshold,
            nms_kernel=nms_kernel,
            nms_sigma=nms_sigma,
            nms_pre=nms_pre,
            nms_max=nms_max,
            min_points=min_points,
            fill_uncovered=fill_uncovered,
        )

    # ----- config / spec building ----------------------------------------- #
    def _criteria_cfg(self, label, criteria, criteria_by_label):
        """Auto-inject the per-label head specs + overlap flag into the loss cfg
        so the user does not have to duplicate them."""
        base = (
            criteria_by_label[label]
            if criteria_by_label is not None and label in criteria_by_label
            else criteria
        )
        spec = self.label_specs[label]

        # Instance-loss variants are all upgraded to the unified loss, which is a
        # superset (handles the primary mask/dice/cls plus arbitrary per-query
        # heads and the overlap path). Per-label head specs + overlap flag are
        # injected so configs don't have to duplicate them.
        _instance_types = {
            "InstanceSegmentationLoss",
            "FastInstanceSegmentationLoss",
            "FastInstanceSegmentationRegressionLoss",
            "FastUnifiedInstanceLoss",
        }

        def _inject(cfg):
            cfg = deepcopy(cfg)
            if isinstance(cfg, dict):
                if cfg.get("type") in _instance_types:
                    cfg["type"] = "FastUnifiedInstanceLoss"
                    # carry over regression_targets written the v3m3 way
                    legacy = cfg.pop("regression_targets", None)
                    heads = cfg.get("query_heads", spec.get("query_heads"))
                    if heads is None and legacy is not None:
                        heads = legacy
                    cfg["query_heads"] = heads
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
            spec = deepcopy(_DEFAULT_LABEL_SPECS.get(label, {}))
            spec.update(deepcopy(label_configs.get(label, {})))

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
            head["pred_key"]
            for head in self.label_specs[label].get("query_heads", [])
        ]

    # ----- forward -------------------------------------------------------- #
    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        point_full = self.up_cast(point)

        # Stash GT segment for an optional GT-driven point filter.
        if self.point_filter_cfg and self.training:
            eval_spec = self.label_specs[self.eval_label]
            seg_key = self._segment_key(input_dict, eval_spec)
            if seg_key in input_dict:
                point_full._filter_gt_segment = input_dict[seg_key]

        decoder_point = Point(point_full.copy())
        if hasattr(point_full, "_filter_gt_segment"):
            decoder_point._filter_gt_segment = point_full._filter_gt_segment
        outputs_by_label = self.decoder(decoder_point)
        filter_logits = outputs_by_label.pop("__filter_logits__", None)

        loss = None
        return_dict = {}
        emit_predictions = not self.training

        for label in self.labels:
            task_input = None
            spec = self.label_specs[label]
            segment_key = self._segment_key(input_dict, spec)
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

        # Optional point-filter supervision (global, label-agnostic).
        if filter_logits is not None and self.training:
            filter_loss = self._compute_filter_loss(filter_logits, input_dict)
            if filter_loss is not None:
                loss = (
                    filter_loss if loss is None else loss + filter_loss
                )
                return_dict["filter_loss"] = filter_loss

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
            primary_segment_key = self._segment_key(input_dict, primary_spec)
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

    def _compute_filter_loss(self, filter_logits, input_dict):
        """Binary BCE supervising the point filter to drop `drop_classes`."""
        drop_classes = self.point_filter_cfg.get("drop_classes") or []
        if not drop_classes:
            return None
        eval_spec = self.label_specs[self.eval_label]
        seg_key = self._segment_key(input_dict, eval_spec)
        if seg_key not in input_dict:
            return None
        segment = input_dict[seg_key]
        if segment.dim() == 2 and segment.shape[1] == 1:
            segment = segment.squeeze(1)
        target = torch.zeros_like(filter_logits)
        for cls_id in drop_classes:
            target[segment == int(cls_id)] = 1.0
        loss = F.binary_cross_entropy_with_logits(filter_logits, target)
        return self.filter_loss_weight * loss

    # ----- postprocess ---------------------------------------------------- #
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
    ):
        label = label or self.eval_label
        cfg = self.postprocess_cfg.copy()
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
        elif (
            "pred_masks_by_label" in forward_output
            and label in forward_output["pred_masks_by_label"]
        ):
            pred_masks = forward_output["pred_masks_by_label"][label]
            pred_logits = forward_output["pred_logits_by_label"][label]
            stuff_probs = None
            if "stuff_probs_by_label" in forward_output:
                stuff_probs = forward_output["stuff_probs_by_label"].get(label)
            pred_regression = None
            if "pred_regression_by_label" in forward_output:
                pred_regression = forward_output["pred_regression_by_label"].get(label)
            point_counts = forward_output.get("point_counts")
        else:
            pred_masks = forward_output["pred_masks"]
            pred_logits = forward_output["pred_logits"]
            stuff_probs = forward_output.get("stuff_probs")
            pred_regression = forward_output.get("pred_regression")
            point_counts = forward_output.get("point_counts")

        if point_counts is None:
            point_counts = forward_output.get(f"{label}_point_counts")

        # Single entry point, dispatched by the label's `overlap` config:
        #   overlap=False -> mutually-exclusive greedy panoptic fusion (per-point
        #     instance/class labels). UNCHANGED.
        #   overlap=True  -> overlap-preserving per-query rings (independent sigmoid
        #     masks), selected by mask/class confidence only (no physics dedup).
        if self.label_specs[label].get("overlap", False):
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
