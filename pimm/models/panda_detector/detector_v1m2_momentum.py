from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

try:
    import flash_attn
except ImportError:
    flash_attn = None

import torch.nn.functional as F
import torch_scatter
from timm.layers import DropPath, trunc_normal_

from pimm.models.builder import MODELS, build_model
from pimm.models.losses import build_criteria
from pimm.models.modules import PointModel
from pimm.models.utils.misc import offset2batch, offset2bincount
from pimm.models.utils.structure import Point
from pimm.utils.comm import get_world_size, reduce_scalar_outputs_for_logging
from .postprocess import postprocess_batch

class MomentumRegressor(nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super().__init__()
        self.backbone = MLP(hidden_channels + num_classes, hidden_channels, 1)

    def forward(self, q_features, cls_logits):
        """
        Compute momentum regression with mixture of class-experts.

        Args:
            q_features: (N_q, hidden_channels)
            cls_logits: (N_q, num_classes+1) (include no-object class)

        Returns:
            momentum_pred: (N_q,) log p for each query
        """
        backbone_features = self.backbone(
            torch.cat([q_features, cls_logits[:, :-1]], dim=-1)
        )
        return backbone_features.squeeze(-1)


class IoUPredictor(nn.Module):
    """Predicts the IoU between predicted mask and GT mask for each query."""
    
    def __init__(self, hidden_channels):
        super().__init__()
        self.mlp = MLP(hidden_channels, hidden_channels, 1)
    
    def forward(self, q_features):
        """
        Predict IoU score for each query.
        
        Args:
            q_features: (N_q, hidden_channels) query features
            
        Returns:
            iou_pred: (N_q,) predicted IoU in [0, 1] (sigmoid applied)
        """
        logits = self.mlp(q_features).squeeze(-1)
        return logits.sigmoid()

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(channels, channels * 1, bias=qkv_bias)
        self.kv = nn.Linear(channels, channels * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.enable_flash = enable_flash
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos(self, qkv: torch.Tensor, q_pos: torch.Tensor) -> torch.Tensor:
        return qkv + q_pos if q_pos is not None else qkv

    def k(self, t: torch.Tensor) -> torch.Tensor:
        return F.linear(t, self.kv.weight[:self.channels, :], self.kv.bias[:self.channels])

    def v(self, t: torch.Tensor) -> torch.Tensor:
        return F.linear(t, self.kv.weight[self.channels:, :], self.kv.bias[self.channels:])
        
    def forward(
        self, qkv: torch.Tensor, q_pos: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        H = self.num_heads
        C = self.channels

        q = self.q(self.with_pos(qkv, q_pos))
        k = self.k(self.with_pos(qkv, q_pos))
        v = self.v(qkv)
        
        if self.enable_flash and flash_attn is not None and q.is_cuda:
            feat = flash_attn.flash_attn_varlen_func(
                q.to(torch.bfloat16).reshape(-1, H, C // H),
                k.to(torch.bfloat16).reshape(-1, H, C // H),
                v.to(torch.bfloat16).reshape(-1, H, C // H),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            )
            feat = feat.reshape(-1, C).to(q.dtype)
        else:
            q_dtype = q.dtype
            q = q.to(torch.bfloat16).reshape(-1, H, C // H)
            k = k.to(torch.bfloat16).reshape(-1, H, C // H)
            v = v.to(torch.bfloat16).reshape(-1, H, C // H)
            if self.upcast_attention:
                q = q.float()
                k = k.float()
                v = v.float()
            
            # create block-diagonal mask to prevent cross-batch attention
            N = qkv.shape[0]
            B = len(cu_seqlens) - 1
            attn_mask = torch.full((N, N), -1e4, dtype=q.dtype, device=q.device)
            
            for b in range(B):
                start = cu_seqlens[b].item()
                end = cu_seqlens[b+1].item()
                attn_mask[start:end, start:end] = 0.0
            
            # expand mask for SDPA: (1, 1, N, N) for broadcasting over batch and heads
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            
            # torch SDPA: reshape to (1, heads, seq_len, head_dim)
            q_sdpa = q.transpose(0, 1).unsqueeze(0).contiguous()  # (1, H, N, head_dim)
            k_sdpa = k.transpose(0, 1).unsqueeze(0).contiguous()  # (1, H, N, head_dim)
            v_sdpa = v.transpose(0, 1).unsqueeze(0).contiguous()  # (1, H, N, head_dim)
            
            feat = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
            )
            
            # feat: (1, H, N, head_dim) -> (N, H, head_dim) -> (N, C)
            feat = feat.squeeze(0).transpose(0, 1).reshape(-1, C).to(qkv.dtype)
            if self.upcast_attention:
                feat = feat.to(qkv.dtype)
        
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        return feat


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.k_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.v_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.enable_flash = enable_flash
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        H = self.num_heads
        C = self.channels

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # reshape to (batch*seq, heads, head_dim) 
        q = q.reshape(-1, H, C // H)
        k = k.reshape(-1, H, C // H)
        v = v.reshape(-1, H, C // H)
        
        if self.upcast_attention:
            q = q.float()
            k = k.float()
            v = v.float()
        
        # prepare attention mask for SDPA
        # SDPA expects mask broadcastable to (batch, heads, seq_q, seq_kv)
        sdpa_mask = None
        if attn_mask is not None:
            # attn_mask is (N_q, N_kv) additive mask
            # expand to (1, 1, N_q, N_kv) for broadcasting over batch and heads
            sdpa_mask = attn_mask.unsqueeze(0).unsqueeze(0).to(q.dtype)
        
        # torch SDPA with 3D inputs (batch*seq, heads, head_dim)
        # will internally handle as (1, heads, seq, head_dim)
        q_sdpa = q.transpose(0, 1).unsqueeze(0).contiguous()  # (1, H, N_q, head_dim)
        k_sdpa = k.transpose(0, 1).unsqueeze(0).contiguous()  # (1, H, N_kv, head_dim)
        v_sdpa = v.transpose(0, 1).unsqueeze(0).contiguous()  # (1, H, N_kv, head_dim)
        
        feat = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=sdpa_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )
        
        # feat: (1, H, N_q, head_dim) -> (N_q, H, head_dim) -> (N_q, C)
        feat = feat.squeeze(0).transpose(0, 1).reshape(-1, C)
        if self.upcast_attention:
            feat = feat.to(self.q_proj.weight.dtype)
        
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        return feat


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.RMSNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=True,
        use_attn_mask=False,
        attn_mask_eps=1e-6,
        attn_mask_anneal=False,
        attn_mask_anneal_steps=10000,
        attn_mask_warmup_steps=0,
        supervise_attn_mask=True,
        is_last_block=False,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.use_attn_mask = use_attn_mask
        self.attn_mask_eps = attn_mask_eps
        self.attn_mask_anneal = attn_mask_anneal
        self.attn_mask_anneal_steps = attn_mask_anneal_steps
        self.attn_mask_warmup_steps = attn_mask_warmup_steps
        self.supervise_attn_mask = supervise_attn_mask
        self.is_last_block = is_last_block
        
        # annealing progress: 0.0 (full mask) -> 1.0 (no mask)
        self.register_buffer('anneal_progress', torch.tensor(0.0))
        self._current_step = 0

        self.norm1 = norm_layer(channels)
        self.ls1 = (
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
        self.norm_kv = norm_layer(channels)
        self.self_attn = SelfAttentionLayer(
            channels,
            num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = norm_layer(channels)
        self.ls2 = (
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )

        self.cross_attn = CrossAttentionLayer(
            channels,
            num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm3 = norm_layer(channels)
        self.ls3 = (
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
        self.mlp = MLP(
            in_channels=channels,
            hidden_channels=int(channels * mlp_ratio),
            out_channels=channels,
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # create mask_mlp if:
        # - supervise_attn_mask=True (all blocks need it for per-layer supervision)
        # - supervise_attn_mask=False AND is_last_block (only last block computes masks)
        if self.use_attn_mask and (self.supervise_attn_mask or self.is_last_block):
            self.mask_mlp = MLP(channels, channels, channels)

    def set_anneal_step(self, step: int):
        """Update annealing progress based on training step."""
        if self.attn_mask_anneal and self.attn_mask_anneal_steps > 0:
            self._current_step = step
            # account for warmup: no annealing during warmup
            if step < self.attn_mask_warmup_steps:
                progress = 0.0
            else:
                # start annealing after warmup
                effective_step = step - self.attn_mask_warmup_steps
                progress = min(effective_step / self.attn_mask_anneal_steps, 1.0)
            self.anneal_progress.fill_(progress)
    
    def get_anneal_factor(self) -> float:
        """Get current annealing factor: 1.0 (full mask) -> 0.0 (no mask)."""
        if not self.attn_mask_anneal:
            return 1.0
        # during warmup, keep full mask strength
        if self._current_step < self.attn_mask_warmup_steps:
            return 1.0
        # cosine decay for smoother transition
        return 0.5 * (1.0 + torch.cos(self.anneal_progress * 3.14159)).item()
    
    @staticmethod
    def with_pos(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return x + pos if pos is not None else x

    def _compute_attn_mask(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ):
        """
        Compute dynamic attention mask based on predicted point-to-prototype assignments.
        
        Args:
            q: query features (N_q, D) where N_q is total queries across batch
            kv: key/value features (N_kv, D) where N_kv is total points across batch
            cu_seqlens_q: cumulative sequence lengths for queries [0, K, 2K, ..., B*K]
            cu_seqlens_kv: cumulative sequence lengths for points [0, P1, P1+P2, ..., sum(P)]
        
        Returns:
            attn_mask: (N_q, N_kv) additive attention mask
            z: (N_q, N_kv) point-to-prototype assignment logits for supervision
        """
        # compute mask embeddings from queries
        m_k = self.mask_mlp(q)  # (N_q, D)
        e_i = kv  # (N_kv, D)
        
        # compute point-to-prototype assignment logits: z_ik = e_i^T @ m_k
        z = torch.matmul(e_i, m_k.t())  # (N_kv, N_q)
        z = z.t()  # (N_q, N_kv)
        
        # compute assignment probabilities: p_hat_ik = sigmoid(z_ik)
        p_hat = torch.sigmoid(z)  # (N_q, N_kv)
        
        # compute attention mask: A_ik = log(p_hat_ik + eps)
        # detach p_hat for mask to prevent gradient feedback through attention
        attn_mask = torch.log(p_hat.detach() + self.attn_mask_eps)  # (N_q, N_kv)
        
        # apply annealing: gradually reduce mask strength during training
        if self.attn_mask_anneal:
            anneal_factor = self.get_anneal_factor()
            attn_mask = attn_mask * anneal_factor
        
        # mask out cross-batch attention: queries from batch b should not attend to points from batch b'
        B = len(cu_seqlens_kv) - 1
        
        for b in range(B):
            # queries for this batch
            start_q = cu_seqlens_q[b].item()
            end_q = cu_seqlens_q[b+1].item()
            
            # points for this batch
            start_kv = cu_seqlens_kv[b].item()
            end_kv = cu_seqlens_kv[b+1].item()
            
            # mask out all cross-batch attention (set to large value, not affected by annealing)
            attn_mask[start_q:end_q, :start_kv] = -1e4
            attn_mask[start_q:end_q, end_kv:] = -1e4
        
        # return logits z for supervision (loss expects logits and applies sigmoid internally)
        return attn_mask, z

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        pos_q: torch.Tensor = None,
        pos_k: torch.Tensor = None,
    ):
        """
        Cross attention:
            Allow adding positional encodings to q and k, but not v, following Mask2Former.
        
        Returns:
            q: updated query features
            mask_logits: (N_q, N_kv) assignment logits (None if use_attn_mask=False)
        """
        kv_n = self.norm_kv(kv.float()).to(kv.dtype)
        
        # compute attention masks if this block has mask_mlp
        # supervise_attn_mask=True: all blocks have mask_mlp, compute masks for attention + supervision
        # supervise_attn_mask=False: only last block has mask_mlp, computes masks for prediction only
        attn_mask = None
        mask_logits = None
        if self.use_attn_mask and hasattr(self, 'mask_mlp'):
            compute_attn = self.supervise_attn_mask  # only use as attention mask if supervising
            if compute_attn:
                attn_mask, mask_logits = self._compute_attn_mask(q, kv_n, cu_seqlens_q, cu_seqlens_kv)
            else:
                # last block in unsupervised mode: compute masks but don't use for attention
                _, mask_logits = self._compute_attn_mask(q, kv_n, cu_seqlens_q, cu_seqlens_kv)
        
        if self.pre_norm:
            # cross-attention
            shortcut = q
            q_n = self.norm1(q.float()).to(q.dtype)
            q = self.drop_path(
                self.ls1(
                    self.cross_attn(
                        q=self.with_pos(q_n, pos_q),
                        k=self.with_pos(kv_n, pos_k),
                        v=kv_n,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_kv=max_seqlen_kv,
                        attn_mask=attn_mask,
                    )
                )
            )
            q += shortcut

            # self-attention
            q_n = self.norm2(q.float()).to(q.dtype)
            q = q + self.drop_path(
                self.ls2(
                    self.self_attn(
                        q_n, pos_q, cu_seqlens_q, max_seqlen_q
                    )
                )
            )

            # mlp
            shortcut = q
            q_n = self.norm3(q.float()).to(q.dtype)
            q = q + self.drop_path(self.ls3(self.mlp(q_n)))
        else:
            # cross-attention
            q += self.drop_path(
                self.ls1(
                    self.cross_attn(
                        q=self.with_pos(q, pos_q),
                        k=self.with_pos(kv_n, pos_k),
                        v=kv_n,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_kv=max_seqlen_kv,
                        attn_mask=attn_mask,
                    )
                )
            )
            q = self.norm1(q.float()).to(q.dtype)

            # self-attention
            q += self.drop_path(
                self.ls2(
                    self.self_attn(q, pos_q, cu_seqlens_q, max_seqlen_q)
                )
            )
            q = self.norm2(q.float()).to(q.dtype)

            # mlp
            q += self.drop_path(self.ls3(self.mlp(q)))
            q = self.norm3(q.float()).to(q.dtype)
        return q, mask_logits


class MaskQueryDecoder(nn.Module):
    """Based loosely on Mask2former and Oneformer3D"""

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
        predict_iou: bool = False,
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

        # decoder_proj no longer needed - using full resolution features
        # self.decoder_proj = nn.Linear(upcast_in_channels, hidden_channels)
        self.query_type = query_type
        if self.query_type == "learned":
            self.query_feat = nn.Embedding(self.num_queries, hidden_channels)
            self.query_embed = nn.Embedding(self.num_queries, hidden_channels)
        self.pos_emb = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        ) if pos_emb else None

        # annealing parameters (can be set via config)
        self.attn_mask_anneal = False
        self.attn_mask_anneal_steps = 10000
        self.attn_mask_warmup_steps = 0
        self.attn_mask_progressive = False  # whether to use progressive annealing
        self.attn_mask_progressive_delay = 0  # delay between blocks (in steps)
        
        # supervise_attn_mask controls mask computation and supervision:
        #   True: each block has mask_mlp, masks constrain cross-attention, supervised at all layers
        #   False: only last block has mask_mlp, full cross-attention (no masking), only final layer supervised
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
                    is_last_block=(i == depth - 1),  # only last block needs mask_mlp when supervise_attn_mask=False
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

        # momentum regression head: mixture of class-experts
        # shared MLP backbone processes queries
        self.momentum_regressor = MomentumRegressor(hidden_channels, num_classes)
        
        # IoU prediction head: predicts mask quality (IoU with GT)
        self.predict_iou = predict_iou
        if self.predict_iou:
            self.iou_pred = IoUPredictor(hidden_channels)

        # stuff head: point-wise binary classifier (stuff vs thing)
        if self.use_stuff_head:
            self.stuff_head = nn.Sequential(
                nn.Linear(full_in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1)  # binary logits (stuff vs thing)
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
        """
        Configure attention mask annealing for all blocks.
        
        Args:
            enable: Enable annealing
            anneal_steps: Number of steps to complete annealing
            warmup_steps: Number of steps before starting annealing
            progressive: If True, start annealing earlier blocks before later blocks
            progressive_delay: Steps of delay between successive blocks (only if progressive=True)
        """
        self.attn_mask_anneal = enable
        self.attn_mask_anneal_steps = anneal_steps
        self.attn_mask_warmup_steps = warmup_steps
        self.attn_mask_progressive = progressive
        self.attn_mask_progressive_delay = progressive_delay
        
        for i, block in enumerate(self.blocks):
            block.attn_mask_anneal = enable
            block.attn_mask_anneal_steps = anneal_steps
            
            # progressive annealing: earlier blocks start annealing sooner
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
            base_pos = self.query_embed.weight if hasattr(self, "query_embed") else None # [Q, C]
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
        """
        Expand mask logits from thing-only points to full (thing + stuff) point cloud.

        The goal here is to only perform attention between queries and only thing points,
        not any stuff points.
        
        Args:
            mask_logits: (Q, N_things) mask logits over thing points only
            thing_mask: (N,) boolean mask indicating thing points
            
        Returns:
            mask_logits_full: (Q, N) mask logits over all points (stuff points get -inf)
        """
        Q = mask_logits.shape[0]
        N = thing_mask.shape[0]
        
        # create full mask with very negative values for stuff points
        mask_logits_full = mask_logits.new_full((Q, N), -1e4)
        
        # fill in thing points with actual logits
        mask_logits_full[:, thing_mask] = mask_logits
        
        return mask_logits_full

    def _forward_decoder(self, point: Point, return_aux: bool = False):
        """
        Internal helper to run the transformer decoder blocks.

        Args:
            point: Input point cloud features (after thing filtering/projection to transformer embedding size).
            return_aux: Whether to return auxiliary outputs from intermediate layers.

        Returns:
            dict: Dictionary containing:
                - "out_q": Final queries.
                - "point_proj": Projected point features used in decoder.
                - "final_mask_logits": Mask logits from the last decoder layer.
                - "query_counts": Number of queries per batch.
                - "query_valid": Validity mask for queries.
                - "aux_q_list" (optional): List of query features from intermediate layers.
                - "aux_mask_logits_list" (optional): List of mask logits from intermediate layers.
        """
        point_proj = point.feat # projection done in forward
        pos_k = self.pos_emb(point.coord) if self.pos_emb else None
        # cu_seqlens for kv
        cu_seqlens_kv = torch.cat([point.offset.new_zeros(1), point.offset]).int() # [B + 1]
        max_seqlen_kv = cu_seqlens_kv.diff().max()

        # queries and their positional encodings
        q, pos_q, query_counts, query_valid = self._get_queries(point)
        cu_seqlens_q = torch.cat([query_counts.new_zeros(1), query_counts.cumsum(dim=0)]).int()
        max_seqlen_q = int(query_counts.max().item()) if query_counts.numel() > 0 else 0

        q = q.reshape(-1, self.mask_channels) # [B * Q, C]
        pos_q = pos_q.reshape(-1, self.mask_channels) if pos_q is not None else None # [B * Q, C]
        query_valid = query_valid.reshape(-1, 1)
        query_valid_f = query_valid.to(q.dtype)

        # auxiliary outputs from each decoder layer (not initial embeddings)
        aux_outputs = []
        aux_p_hat_list = []

        # pass through blocks
        final_mask_logits = None
        for blk in self.blocks:
            q, mask_logits = blk(
                q,
                point_proj,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                pos_q,
                pos_k,
            )
            # mask_logits can be None if use_attn_mask=False
            if mask_logits is not None:
                mask_logits = mask_logits * query_valid_f
                final_mask_logits = mask_logits
            q = q * query_valid_f
            if return_aux:
                aux_outputs.append(self.final_norm(q))
                # only include mask_logits in aux outputs if supervision is enabled
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
            "query_counts": query_counts_long,
            "query_valid": query_valid_flat,
        }
        if return_aux:
            outputs["aux_q_list"] = aux_outputs[:-1]
            outputs["aux_mask_logits_list"] = aux_p_hat_list[:-1]
        return outputs

    def up_cast(self, point):
        """
        Upcast features to point-level resolution.
        - If decoder present (enc_mode=False): features already point-level, return as-is
        - If encoder-only (enc_mode=True): walk up pooling hierarchy, concatenate multi-scale features
        """
        if not self.enc_mode:
            # decoder output is already at point-level, no upcasting needed
            return point
        
        # encoder-only: upcast by walking pooling hierarchy and concatenating features
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    def _compute_predictions(self, q_features, mask_logits, point, query_counts, query_valid):
        """
        Helper to compute predictions from query features and mask logits.

        Args:
            q_features: (N_q, D) normalized query features
            mask_logits: (N_q, N_kv) point-to-prototype assignment logits from decoder
            point: Point structure with offset and batch info
            query_counts: Tensor of shape (batch_size,) with per-batch query counts
        """
        class_embed = self.cls_pred(q_features)
        momentum_pred = self.momentum_regressor(q_features, class_embed)
        iou_pred = self.iou_pred(q_features) if self.predict_iou else None

        pred_masks = []
        pred_cls = []
        pred_logits = []
        pred_momentum = []
        pred_iou = [] if self.predict_iou else None

        C = self.num_classes

        # reshape mask logits and class predictions by batch
        B = point.offset.shape[0]
        counts = offset2bincount(point.offset).to(torch.long)
        query_counts = query_counts.to(torch.long)

        query_offsets = torch.cat([query_counts.new_zeros(1), query_counts.cumsum(dim=0)])
        point_offsets = torch.cat([counts.new_zeros(1), point.offset])

        for b in range(B):
            P_b = counts[b].item()
            q_start, q_end = query_offsets[b], query_offsets[b + 1]
            p_start, p_end = point_offsets[b], point_offsets[b + 1]

            mask_logits_b = mask_logits[q_start:q_end, p_start:p_end]
            cls_b = class_embed[q_start:q_end]
            momentum_b = momentum_pred[q_start:q_end]
            valid_b = query_valid[q_start:q_end]

            mask_logits_b = mask_logits_b[valid_b]
            cls_b = cls_b[valid_b]
            momentum_b = momentum_b[valid_b]

            pred_masks.append(mask_logits_b)
            pred_cls.append(cls_b)
            pred_momentum.append(momentum_b)
            
            if self.predict_iou:
                iou_b = iou_pred[q_start:q_end][valid_b]
                pred_iou.append(iou_b)

            if mask_logits_b.shape[0] > 0:
                s = mask_logits_b.transpose(0, 1).unsqueeze(-1)
                c = cls_b[:, :C].unsqueeze(0)
                logits_b = torch.logsumexp(s + c, dim=1)
            else:
                logits_b = mask_logits.new_zeros((P_b, C))
            pred_logits.append(logits_b)

        pred_logits = torch.cat(pred_logits, dim=0) if pred_logits else mask_logits.new_zeros((0, C))

        return {
            "pred_masks": pred_masks,  # List[Tensor(Q_b, P_b)] for loss
            "pred_logits": pred_cls,  # List[Tensor(Q_b, C+1)] for classification loss
            "pred_momentum": pred_momentum,  # List[Tensor(Q_b,)]
            "pred_iou": pred_iou,  # List[Tensor(Q_b,)] predicted IoU scores
            "seg_logits": pred_logits,  # Tensor(N, C) for per-point prediction
        }

    def forward(self, point: Point, return_all=False, gt_segment: Optional[torch.Tensor] = None):
        # get point-level features (either from decoder output or by upcasting encoder)
        point_full = self.up_cast(point)

        # stuff prediction if enabled
        stuff_logits = None
        thing_mask = None
        stuff_probs = None
        
        if self.use_stuff_head:
            # predict stuff with point-wise classifier (always compute for supervision/outputs)
            stuff_logits = self.stuff_head(point_full.feat).squeeze(-1)  # (N,)
            stuff_probs = stuff_logits.sigmoid()

            # choose filtering source: GT during training if enabled and available; otherwise predicted stuff probs
            # the hope is that stuff probs will approach the GT by the end of training.
            if (
                self.training
                and self.train_filter_use_gt
                and gt_segment is not None
                and len(self.stuff_classes) > 0
            ):
                segment = gt_segment
                if isinstance(segment, torch.Tensor) and segment.dim() == 2 and segment.shape[1] == 1:
                    segment = segment.squeeze(1)
                # build stuff mask from GT classes
                is_stuff = torch.zeros_like(segment, dtype=torch.bool)
                for cls_id in self.stuff_classes:
                    is_stuff |= (segment == int(cls_id))
                thing_mask = ~is_stuff
            else:
                # threshold at 0.5: points with prob > 0.5 are stuff, rest are things
                thing_mask = stuff_probs < 0.5
            assert thing_mask.any(), "No thing points found"
            point_for_decoder = point_full[thing_mask]
            point_for_decoder.feat = self.full_point_proj(point_for_decoder.feat)
        else:
            # no stuff filtering - use all points
            full_point_proj = self.full_point_proj(point_full.feat)
            point_for_decoder = point_full.copy()
            point_for_decoder.feat = full_point_proj
        
        # forward decoder with auxiliary outputs if training
        # decoder uses full resolution point features for cross-attention
        return_aux = self.training
        decoder_outputs = self._forward_decoder(point_for_decoder, return_aux=return_aux)
        final_mask_logits = decoder_outputs["final_mask_logits"]
        query_counts = decoder_outputs["query_counts"]
        query_valid = decoder_outputs["query_valid"]
        out_q = decoder_outputs["out_q"]

        # compute predictions for final layer
        if self.use_stuff_head and thing_mask is not None:
            # out points are thing points only so we need to go back to the full point cloud
            # non-thing points get -inf (ish) logits
            final_mask_logits = self._expand_masks(final_mask_logits, thing_mask)
        predictions = self._compute_predictions(out_q, final_mask_logits, point_full, query_counts, query_valid)
        
        if return_aux:
            aux_outputs = []
            # go thru each layer and compute predictions
            for aux_q, aux_mask_logits in zip(decoder_outputs["aux_q_list"], decoder_outputs["aux_mask_logits_list"]):
                if aux_mask_logits is None:  # supervise_attn_mask=False
                    continue
                if self.use_stuff_head and thing_mask is not None:
                    aux_mask_logits = self._expand_masks(aux_mask_logits, thing_mask)
                aux_out = self._compute_predictions(aux_q, aux_mask_logits, point_full, query_counts, query_valid)
                aux_outputs.append(aux_out)
            # only add aux_outputs if we have any (will be empty if supervise_attn_mask=False)
            if aux_outputs:
                predictions["aux_outputs"] = aux_outputs

        # add stuff predictions to outputs
        if self.use_stuff_head and stuff_logits is not None:
            predictions["stuff_logits"] = stuff_logits  # (N,) binary logits
            predictions["stuff_probs"] = stuff_probs    # (N,) probabilities

        # store in point for compatibility
        point_full.pred_cls = predictions["pred_logits"]        # List[Tensor(Q, C+1)]
        point_full.pred_masks = predictions["pred_masks"]       # List[Tensor(Q, P_b)]
        point_full.pred_logits = predictions["seg_logits"]      # Tensor(N, C) for evaluation
        point_full.outputs = predictions
        
        return point_full


@MODELS.register_module("detector-v1m2")
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
        predict_iou=False,  # predict mask IoU and use in scoring (Mask Scoring R-CNN style)
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
            predict_iou=predict_iou,
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
            
            # note: IoU loss is computed in the criteria, same as momentum loss
            # pred_iou is included in outputs for the criteria to use
            
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss, components = self.criteria(point.outputs, input_dict)
            return_dict.update(components)
            return_dict["loss"] = loss
            return_dict["seg_logits"] = point.pred_logits
            # also return raw outputs for evaluator
            if hasattr(point, 'outputs') and point.outputs is not None:
                return_dict["pred_logits"] = point.outputs.get("pred_logits")
                return_dict["pred_momentum"] = point.outputs.get("pred_momentum")
                return_dict["pred_masks"] = point.outputs.get("pred_masks")
                return_dict["pred_iou"] = point.outputs.get("pred_iou")
                return_dict["stuff_probs"] = point.outputs.get("stuff_probs")
        # test
        else:
            return_dict["seg_logits"] = point.pred_logits
            # return raw outputs for evaluator
            if hasattr(point, 'outputs') and point.outputs is not None:
                return_dict["pred_logits"] = point.outputs.get("pred_logits")
                return_dict["pred_momentum"] = point.outputs.get("pred_momentum")
                return_dict["pred_masks"] = point.outputs.get("pred_masks")
                return_dict["pred_iou"] = point.outputs.get("pred_iou")
                return_dict["stuff_probs"] = point.outputs.get("stuff_probs")

        # synchronize loss components
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
            stuff_probs=forward_output.get("stuff_probs"),
            point_counts=forward_output["point_counts"],
            pred_momentum=forward_output.get("pred_momentum"),
            pred_iou=forward_output.get("pred_iou"),
            stuff_classes=self.decoder.stuff_classes,
            **cfg,
        )