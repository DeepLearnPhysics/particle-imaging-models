from __future__ import annotations

from copy import deepcopy
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

import torch.nn.functional as F
from timm.layers import trunc_normal_

from pimm.models.builder import MODELS, build_model
from pimm.models.losses import build_criteria
from pimm.models.modules import PointModel
from pimm.models.utils.misc import offset2batch, offset2bincount
from pimm.models.utils.structure import Point
from pimm.utils.comm import get_world_size, reduce_scalar_outputs_for_logging
from .postprocess import postprocess_batch
from .layers import LayerScale, SelfAttentionLayer, CrossAttentionLayer, MLP, Block

class MaskQueryDecoder(nn.Module):
    """Masked-cross-attention query decoder (loosely Mask2Former / OneFormer3D)."""

    __max_seqlen = 0

    def __init__(
        self,
        full_in_channels,
        hidden_channels,
        num_heads,
        num_classes,
        num_queries=32,
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
        upcast_attention=True,
        upcast_softmax=True,
        pos_emb=True,
        enc_mode=True,
        query_type: Literal["learned", "superpoint"] = "superpoint",
        mlp_point_proj=False,
        use_stuff_head=False,
        stuff_classes=None,
        supervise_attn_mask=True,
        train_filter_use_gt: bool = False,
    ):
        super().__init__()
        self.full_in_channels = full_in_channels
        self.mask_channels = hidden_channels
        self.num_classes = num_classes
        self.enc_mode = enc_mode
        self.num_queries = num_queries
        self.use_stuff_head = use_stuff_head
        self.stuff_classes = set(stuff_classes) if stuff_classes is not None else set()
        self.train_filter_use_gt = bool(train_filter_use_gt)

        self.query_type = query_type
        if self.query_type == "learned":
            self.query_feat = nn.Embedding(self.num_queries, hidden_channels)
            self.query_embed = nn.Embedding(self.num_queries, hidden_channels)
        self.pos_emb = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        ) if pos_emb else None

        # annealing parameters (can be set via set_attn_mask_anneal)
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
                    use_attn_mask=True,  # always True for instance segmentation
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
        # output FFN
        self.cls_pred = (
            MLP(hidden_channels, hidden_channels, num_classes + 1)
            if mlp_point_proj
            else nn.Linear(hidden_channels, num_classes + 1)
        )
        self.full_point_proj = (
            MLP(full_in_channels, hidden_channels, hidden_channels)
            if mlp_point_proj
            else nn.Linear(full_in_channels, hidden_channels)
        )

        # stuff head: point-wise binary classifier (stuff vs thing)
        if self.use_stuff_head:
            self.stuff_head = nn.Sequential(
                nn.Linear(full_in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1)
            )

        self.apply(self._init_weights)

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
        """Update annealing progress for all blocks."""
        for block in self.blocks:
            block.set_anneal_step(step)

    def _max_seqlen(self, seq_len: int) -> int:
        if seq_len > self.__max_seqlen:
            self.__max_seqlen = seq_len
        return self.__max_seqlen

    def _get_queries(self, point: Point) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        batch_size = point.offset.shape[0]
        device = point.feat.device
        max_queries = self.num_queries

        if self.query_type == "learned":
            base_q = self.query_feat.weight  # [Q, C]
            base_pos = self.query_embed.weight if hasattr(self, "query_embed") else None
            q = base_q.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, Q, C]
            pos_q = None
            if base_pos is not None:
                pos_q = base_pos.unsqueeze(0).repeat(batch_size, 1, 1)
            counts = torch.full((batch_size,), max_queries, dtype=torch.int32, device=device)
            valid_mask = torch.ones(batch_size, max_queries, dtype=torch.bool, device=device)
            return q, pos_q, counts, valid_mask
        else:
            raise NotImplementedError(f"Invalid query type: {self.query_type}")

    def _expand_masks(self, mask_logits: torch.Tensor, thing_mask: torch.Tensor) -> torch.Tensor:
        """Expand (Q, N_things) mask logits to (Q, N) over the full cloud; dropped
        points get a large-negative sentinel."""
        Q = mask_logits.shape[0]
        N = thing_mask.shape[0]
        mask_logits_full = mask_logits.new_full((Q, N), -1e4)
        mask_logits_full[:, thing_mask] = mask_logits
        return mask_logits_full

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
        final_mask_embed = None
        final_mask_point_proj = None
        for blk in self.blocks:
            q, mask_logits, mask_embed, mask_point_proj = blk(
                q,
                point_proj,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                pos_q,
                pos_k,
            )
            if mask_logits is not None:
                mask_logits = mask_logits * query_valid_f
                final_mask_logits = mask_logits
                final_mask_embed = mask_embed * query_valid_f if mask_embed is not None else None
                final_mask_point_proj = mask_point_proj
            q = q * query_valid_f
            if return_aux:
                aux_outputs.append(self.final_norm(q))
                if self.supervise_attn_mask:
                    aux_p_hat_list.append(mask_logits)
                else:
                    aux_p_hat_list.append(None)

        q_norm = self.final_norm(q)
        query_counts_long = query_counts.to(torch.long)
        query_valid_flat = query_valid.squeeze(-1).bool()

        outputs = {
            "out_q": q_norm,
            "point_proj": point_proj,
            "final_mask_logits": final_mask_logits,
            "final_mask_embed": final_mask_embed,
            "final_mask_point_proj": final_mask_point_proj,
            "query_counts": query_counts_long,
            "query_valid": query_valid_flat,
        }
        if return_aux:
            outputs["aux_q_list"] = aux_outputs[:-1]
            outputs["aux_mask_logits_list"] = aux_p_hat_list[:-1]
        return outputs

    def up_cast(self, point):
        """Upcast features to point-level resolution. enc_mode=False: no-op.
        enc_mode=True: walk the pooling hierarchy, concatenating multi-scale features."""
        if not self.enc_mode:
            return point
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point


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

@MODELS.register_module("detector-v1m1")
class Detector(PointModel):
    def __init__(
        self,
        num_classes,
        full_in_channels,
        hidden_channels,
        num_heads,
        num_queries=32,
        backbone=None,
        criteria=None,
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
        train_filter_use_gt: bool = False,
        mlp_point_proj=False,
        # postprocessing parameters
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
        super(Detector, self).__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

        self.decoder = MaskQueryDecoder(
            full_in_channels=full_in_channels,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_queries=num_queries,
            num_classes=num_classes,
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
            enc_mode=getattr(backbone, 'enc_mode', True),
            query_type=query_type,
            use_stuff_head=use_stuff_head,
            stuff_classes=stuff_classes,
            supervise_attn_mask=supervise_attn_mask,
            train_filter_use_gt=train_filter_use_gt,
            mlp_point_proj=mlp_point_proj,
        )
        
        # configure attention mask annealing
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
    
    def update_anneal_step(self, step: int):
        """Update attention mask annealing progress. Call this during training."""
        self.decoder.update_anneal_step(step)
    
    def _compute_stuff_loss(self, stuff_logits: torch.Tensor, input_dict: Dict) -> torch.Tensor:
        """
        Compute binary cross-entropy loss for stuff prediction.
        
        Args:
            stuff_logits: (N,) binary logits (high = stuff, low = thing)
            input_dict: contains 'segment' with per-point semantic labels
            
        Returns:
            loss: scalar BCE loss
        """
        # get semantic labels
        segment = input_dict["segment"]
        if isinstance(segment, torch.Tensor):
            if segment.dim() == 2 and segment.shape[1] == 1:
                segment = segment.squeeze(1)
        
        # create binary target: 1 for stuff classes, 0 for thing classes
        stuff_target = torch.zeros_like(stuff_logits)
        for stuff_class in self.decoder.stuff_classes:
            stuff_target[segment == stuff_class] = 1.0
        
        # binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(stuff_logits, stuff_target, reduction='mean')
        
        return loss

    def forward(self, input_dict, return_point=False):
        """based on DefaultSegmentorV2 in pimm/models/default.py"""
        point = Point(input_dict)
        point = self.backbone(point)
        point = self.decoder(point)

        return_dict = dict()
        if return_point:
            return_dict["point"] = point
        
        # train
        if self.training:
            loss, components = self.criteria(point.outputs, input_dict)
            return_dict.update(components)
            
            # add stuff loss if stuff head is enabled
            if self.decoder.use_stuff_head and "stuff_logits" in point.outputs:
                stuff_loss = self._compute_stuff_loss(point.outputs["stuff_logits"], input_dict)
                loss = loss + stuff_loss
                return_dict["stuff_loss"] = stuff_loss
            
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss, components = self.criteria(point.outputs, input_dict)
            return_dict.update(components)
            return_dict["loss"] = loss
            return_dict["seg_logits"] = point.pred_logits
            # also return raw outputs for QueryInsSegEvaluator
            if hasattr(point, 'outputs') and point.outputs is not None:
                return_dict["pred_logits"] = point.outputs.get("pred_logits")
                return_dict["pred_masks"] = point.outputs.get("pred_masks")
        # test
        else:
            return_dict["seg_logits"] = point.pred_logits
            # return raw outputs for QueryInsSegEvaluator
            if hasattr(point, 'outputs') and point.outputs is not None:
                return_dict["pred_logits"] = point.outputs.get("pred_logits")
                return_dict["pred_masks"] = point.outputs.get("pred_masks")

        # synchronize loss components across GPUs for consistent logging
        return_dict = reduce_scalar_outputs_for_logging(return_dict)
        return return_dict

    def postprocess(
        self,
        forward_output: dict,
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
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
        return postprocess_batch(
            pred_masks=forward_output["pred_masks"],
            pred_logits=forward_output["pred_logits"],
            stuff_probs=forward_output["stuff_probs"],
            point_counts=forward_output["point_counts"],
            stuff_classes=self.decoder.stuff_classes,
            **cfg,
        )


class MultiLabelMaskQueryDecoder(MaskQueryDecoder):
    """Mask-query decoder that batches queries for multiple instance targets."""

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
        upcast_attention=True,
        upcast_softmax=True,
        pos_emb=True,
        enc_mode=False,
        query_type: Literal["learned"] = "learned",
        mlp_point_proj=False,
        supervise_attn_mask=True,
    ):
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

        super().__init__(
            full_in_channels=full_in_channels,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_classes=max_classes,
            num_queries=self.total_queries,
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
            enc_mode=enc_mode,
            query_type=query_type,
            mlp_point_proj=mlp_point_proj,
            use_stuff_head=False,
            stuff_classes=None,
            supervise_attn_mask=supervise_attn_mask,
            train_filter_use_gt=False,
        )

        # Label-specific heads below replace the superclass's single class head.
        self.cls_pred = nn.Identity()
        self.cls_pred_by_label = nn.ModuleDict()
        self.stuff_head_by_label = nn.ModuleDict()
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
        self.apply(self._init_weights)

    def _compute_label_predictions(
        self,
        label,
        q_features,
        mask_logits,
        point,
        query_valid,
    ):
        class_embed = self.cls_pred_by_label[label](q_features)

        pred_masks = []
        pred_cls = []
        pred_logits = []

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

        return {
            "pred_masks": pred_masks,
            "pred_logits": pred_cls,
            "seg_logits": pred_logits,
        }

    def forward(self, point: Point, return_decoder: bool = False):
        point_full = self.up_cast(point)
        decoder_point = point_full.copy()
        decoder_point.feat = self.full_point_proj(point_full.feat)

        return_aux = self.training
        decoder_outputs = self._forward_decoder(decoder_point, return_aux=return_aux)
        final_mask_logits = decoder_outputs["final_mask_logits"]
        query_valid = decoder_outputs["query_valid"]
        out_q = decoder_outputs["out_q"]

        outputs_by_label = {}
        for label in self.labels:
            predictions = self._compute_label_predictions(
                label, out_q, final_mask_logits, point_full, query_valid
            )
            if return_aux:
                aux_outputs = []
                for aux_q, aux_mask_logits in zip(
                    decoder_outputs["aux_q_list"],
                    decoder_outputs["aux_mask_logits_list"],
                ):
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
                stuff_logits = self.stuff_head_by_label[label](point_full.feat).squeeze(-1)
                predictions["stuff_logits"] = stuff_logits
                predictions["stuff_probs"] = stuff_logits.sigmoid()
            outputs_by_label[label] = predictions

        if return_decoder:
            return outputs_by_label, decoder_outputs
        return outputs_by_label


@MODELS.register_module(["detector-v3", "detector-v3m1"])
class MultiLabelDetector(PointModel):
    """Shared-backbone Panda detector with one batched query decoder for all labels."""

    def __init__(
        self,
        full_in_channels,
        hidden_channels,
        num_heads,
        labels=("particle",),
        num_queries=32,
        num_classes=None,
        label_configs=None,
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
        use_stuff_head=True,
        stuff_classes=None,
        supervise_attn_mask=True,
        train_filter_use_gt: bool = False,
        mlp_point_proj=False,
        # postprocessing parameters for the primary/eval label
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
        super().__init__()
        if isinstance(labels, str):
            labels = (labels,)
        self.labels = tuple(labels)
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
        )
        self.eval_label = eval_label or self.labels[-1]
        if self.eval_label not in self.label_specs:
            raise ValueError(
                f"eval_label={self.eval_label!r} must be one of {self.labels}"
            )

        self.backbone = build_model(backbone)
        self.criteria_by_label = {
            label: build_criteria(self._criteria_cfg(label, criteria, criteria_by_label))
            for label in self.labels
        }

        if train_filter_use_gt:
            raise ValueError(
                "detector-v3 batches queries across labels and does not support "
                "per-label GT filtering inside cross-attention. Use stuff losses "
                "and postprocess thresholds instead."
            )

        self.decoder = MultiLabelMaskQueryDecoder(
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

    @classmethod
    def _build_label_specs(
        cls,
        labels,
        label_configs=None,
        num_queries=32,
        num_classes=None,
        stuff_classes=None,
        use_stuff_head=True,
        loss_weights=None,
    ):
        label_configs = label_configs or {}
        specs = {}
        for idx, label in enumerate(labels):
            spec = deepcopy(_DEFAULT_LABEL_SPECS.get(label, {}))
            spec.update(deepcopy(label_configs.get(label, {})))

            queries = cls._select_value(num_queries, labels, label, idx, None)
            classes = cls._select_value(num_classes, labels, label, idx, spec.get("num_classes"))
            stuff = cls._select_stuff_classes(
                stuff_classes, labels, label, idx, spec.get("stuff_classes", [])
            )
            task_use_stuff = cls._select_value(use_stuff_head, labels, label, idx, True)
            loss_weight = cls._select_value(loss_weights, labels, label, idx, 1.0)

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
            specs[label] = spec
        return specs

    @staticmethod
    def _criteria_cfg(label, criteria, criteria_by_label):
        if criteria_by_label is not None and label in criteria_by_label:
            return deepcopy(criteria_by_label[label])
        return deepcopy(criteria)

    def update_anneal_step(self, step: int):
        self.decoder.update_anneal_step(step)

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
    def _compute_stuff_loss(
        decoder: MultiLabelMaskQueryDecoder,
        label: str,
        stuff_logits: torch.Tensor,
        task_input: Dict,
    ) -> torch.Tensor:
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

    def forward(self, input_dict, return_point=False, return_decoder=False):
        point = Point(input_dict)
        point = self.backbone(point)
        point_full = self.up_cast(point)
        decoder_result = self.decoder(Point(point_full.copy()), return_decoder=return_decoder)
        if return_decoder:
            outputs_by_label, decoder_outputs = decoder_result
        else:
            outputs_by_label = decoder_result
            decoder_outputs = None

        loss = None
        return_dict = {}
        emit_predictions = not self.training

        for label in self.labels:
            task_input = None
            spec = self.label_specs[label]
            segment_key = self._segment_key(input_dict, spec)
            has_targets = spec["instance_key"] in input_dict and segment_key in input_dict
            if has_targets or self.training:
                task_input = self._task_input(input_dict, label)
            outputs = outputs_by_label[label]
            outputs_by_label[label] = outputs

            if emit_predictions:
                return_dict[f"{label}_seg_logits"] = outputs["seg_logits"]
                return_dict[f"{label}_pred_logits"] = outputs.get("pred_logits")
                return_dict[f"{label}_pred_masks"] = outputs.get("pred_masks")
                if "stuff_probs" in outputs:
                    return_dict[f"{label}_stuff_probs"] = outputs["stuff_probs"]

            if task_input is not None:
                loss_result = self.criteria_by_label[label](outputs, task_input)
                if isinstance(loss_result, tuple):
                    task_loss, components = loss_result
                else:
                    task_loss, components = loss_result, {}
                task_loss = task_loss * self.label_specs[label]["loss_weight"]
                loss = task_loss if loss is None else loss + task_loss
                return_dict[f"{label}_loss"] = task_loss
                for key, value in components.items():
                    return_dict[f"{label}_{key}"] = value

                if (
                    self.label_specs[label]["use_stuff_head"]
                    and "stuff_logits" in outputs
                ):
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
            if primary_spec["instance_key"] in input_dict:
                point_full.instance = input_dict[primary_spec["instance_key"]]
            primary_segment_key = self._segment_key(input_dict, primary_spec)
            if primary_segment_key in input_dict:
                point_full.segment = input_dict[primary_segment_key]

            if emit_predictions:
                return_dict["outputs_by_label"] = outputs_by_label
                return_dict["seg_logits_by_label"] = {
                    label: outputs["seg_logits"] for label, outputs in outputs_by_label.items()
                }
                return_dict["pred_logits_by_label"] = {
                    label: outputs["pred_logits"] for label, outputs in outputs_by_label.items()
                }
                return_dict["pred_masks_by_label"] = {
                    label: outputs["pred_masks"] for label, outputs in outputs_by_label.items()
                }
                return_dict["seg_logits"] = primary_outputs["seg_logits"]
                return_dict["pred_logits"] = primary_outputs.get("pred_logits")
                return_dict["pred_masks"] = primary_outputs.get("pred_masks")
                return_dict["stuff_probs"] = primary_outputs.get("stuff_probs")
                return_dict["point_counts"] = offset2bincount(point_full.offset)
                if return_decoder and decoder_outputs is not None:
                    return_dict["decoder"] = decoder_outputs

        if return_decoder and decoder_outputs is not None and "decoder" not in return_dict:
            return_dict["decoder"] = decoder_outputs

        if loss is not None:
            return_dict["loss"] = loss
        if return_point:
            return_dict["point"] = point_full
        return self._sync_scalar_outputs(return_dict)

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
            point_counts = forward_output.get("point_counts")
        else:
            pred_masks = forward_output["pred_masks"]
            pred_logits = forward_output["pred_logits"]
            stuff_probs = forward_output.get("stuff_probs")
            point_counts = forward_output.get("point_counts")

        if point_counts is None:
            point_counts = forward_output.get(f"{label}_point_counts")

        return postprocess_batch(
            pred_masks=pred_masks,
            pred_logits=pred_logits,
            stuff_probs=stuff_probs,
            point_counts=point_counts,
            stuff_classes=self.label_specs[label]["stuff_classes"],
            **cfg,
        )
