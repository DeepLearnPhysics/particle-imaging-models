"""Low-level transformer-decoder layers shared across the panda detector family.

Pure building blocks -- LayerScale, flash/SDPA Self- and Cross-AttentionLayer, MLP,
and the Mask2Former-style decoder ``Block`` (cross-attn -> self-attn -> FFN with the
per-layer mask head). These are composed by each detector's decoder; you do not need
to read this file to follow a detector's control flow.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from timm.layers import DropPath

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


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

        if self.enable_flash and q.is_cuda:
            feat = flash_attn_varlen_func(
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
        sdpa_mask = None
        if attn_mask is not None:
            # attn_mask is (N_q, N_kv) additive mask
            sdpa_mask = attn_mask.unsqueeze(0).unsqueeze(0).to(q.dtype)

        # torch SDPA with 3D inputs (batch*seq, heads, head_dim)
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
        """Compute dynamic attention mask from predicted point-to-prototype assignments.

        Returns (attn_mask, z, m_k): a bounded additive mask log(sigmoid(z)+eps)
        (never -inf), the assignment logits z for supervision, and the mask embeddings.
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

        # mask out cross-batch attention
        B = len(cu_seqlens_kv) - 1
        for b in range(B):
            start_q = cu_seqlens_q[b].item()
            end_q = cu_seqlens_q[b+1].item()
            start_kv = cu_seqlens_kv[b].item()
            end_kv = cu_seqlens_kv[b+1].item()
            attn_mask[start_q:end_q, :start_kv] = -1e4
            attn_mask[start_q:end_q, end_kv:] = -1e4

        # return logits z for supervision (loss expects logits and applies sigmoid internally)
        return attn_mask, z, m_k

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
        """Cross-attn (q<-kv) -> self-attn (q<->q) -> MLP, Mask2Former-style.

        Positional encodings are added to q and k but not v.
        Returns (q, mask_logits, mask_embed, mask_point_proj).
        """
        kv_n = self.norm_kv(kv.float()).to(kv.dtype)

        attn_mask = None
        mask_logits = None
        mask_embed = None
        mask_point_proj = None
        if self.use_attn_mask and hasattr(self, 'mask_mlp'):
            mask_point_proj = kv_n
            compute_attn = self.supervise_attn_mask  # only use as attention mask if supervising
            if compute_attn:
                attn_mask, mask_logits, mask_embed = self._compute_attn_mask(q, kv_n, cu_seqlens_q, cu_seqlens_kv)
            else:
                # last block in unsupervised mode: compute masks but don't use for attention
                _, mask_logits, mask_embed = self._compute_attn_mask(q, kv_n, cu_seqlens_q, cu_seqlens_kv)

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
        return q, mask_logits, mask_embed, mask_point_proj
