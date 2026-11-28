"""Masked query-decoder layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pimm.models.utils.misc import DropPath
from pimm.ops import varlen_attention
from pimm.ops.attention import get_attention_backend


def _attend(
    layer,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    attn_mask=None,
):
    """``layer``'s attention on ``[rows, heads, dim]`` inputs, as ``[rows, channels]``: with the ``"flash"``
    attention backend FlashAttention, or the masked Triton kernel for a mask, in the run's AMP dtype (bf16 without
    AMP); with ``"torch"`` PyTorch's, in the inputs' dtype (fp32 with ``upcast_attention``). ``attn_mask`` is jagged
    (``pimm.ops.varlen_attention``)."""
    if get_attention_backend() == "flash":
        dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else torch.bfloat16
        )
    else:
        dtype = torch.float32 if layer.upcast_attention else q.dtype
    feat = varlen_attention(
        q.to(dtype),
        k.to(dtype),
        v.to(dtype),
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        softmax_scale=layer.scale,
        dropout_p=layer.attn_drop.p if layer.training else 0.0,
        attn_mask=attn_mask,
    )
    return feat.reshape(len(feat), -1).to(q.dtype)


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
        upcast_attention=True,
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
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.upcast_attention = upcast_attention
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos(self, qkv: torch.Tensor, q_pos: torch.Tensor) -> torch.Tensor:
        return qkv + q_pos if q_pos is not None else qkv

    def k(self, t: torch.Tensor) -> torch.Tensor:
        return F.linear(
            t,
            self.kv.weight[: self.channels, :],
            None if self.kv.bias is None else self.kv.bias[: self.channels],
        )

    def v(self, t: torch.Tensor) -> torch.Tensor:
        return F.linear(
            t,
            self.kv.weight[self.channels :, :],
            None if self.kv.bias is None else self.kv.bias[self.channels :],
        )

    def forward(
        self,
        qkv: torch.Tensor,
        q_pos: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        H = self.num_heads
        C = self.channels

        q = self.q(self.with_pos(qkv, q_pos))
        k = self.k(self.with_pos(qkv, q_pos))
        v = self.v(qkv)
        q = q.reshape(-1, H, C // H)
        k = k.reshape(-1, H, C // H)
        v = v.reshape(-1, H, C // H)

        feat = _attend(self, q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
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
        upcast_attention=True,
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
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.upcast_attention = upcast_attention
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

        q = q.reshape(-1, H, C // H)
        k = k.reshape(-1, H, C // H)
        v = v.reshape(-1, H, C // H)

        feat = _attend(
            self,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            attn_mask,
        )
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


# Masked cross-attention follows Mask2Former (github.com/facebookresearch/Mask2Former).
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
        upcast_attention=False,
        attn_mask_eps=1e-6,
    ):
        super().__init__()
        self.channels = channels
        self.attn_mask_eps = attn_mask_eps

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
            upcast_attention=upcast_attention,
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
            upcast_attention=upcast_attention,
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

        self.mask_mlp = MLP(channels, channels, channels)

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
        The mask is jagged: each event's (N_q_b, N_kv_b) block, flattened. z is dense
        (N_q, N_kv).
        """
        m_k = self.mask_mlp(q)  # (N_q, D)
        e_i = kv  # (N_kv, D)

        z = torch.matmul(e_i, m_k.t()).t()

        p_hat = torch.sigmoid(z)  # (N_q, N_kv)

        # detach p_hat for mask to prevent gradient feedback through attention
        attn_mask = torch.log(p_hat.detach() + self.attn_mask_eps)  # (N_q, N_kv)

        q_bounds, kv_bounds = cu_seqlens_q.tolist(), cu_seqlens_kv.tolist()
        attn_mask = torch.cat(
            [
                attn_mask[a:b, c:d].flatten()
                for a, b, c, d in zip(q_bounds, q_bounds[1:], kv_bounds, kv_bounds[1:])
            ]
        )

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

        Learned query encodings are added before projection.
        Returns (q, mask_logits, mask_embed, mask_point_proj).
        """
        kv_n = self.norm_kv(kv.float()).to(kv.dtype)

        mask_point_proj = kv_n
        attn_mask, mask_logits, mask_embed = self._compute_attn_mask(
            q, kv_n, cu_seqlens_q, cu_seqlens_kv
        )

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

        q_n = self.norm2(q.float()).to(q.dtype)
        q = q + self.drop_path(
            self.ls2(
                self.self_attn(
                    q_n,
                    pos_q,
                    cu_seqlens_q,
                    max_seqlen_q,
                )
            )
        )

        q_n = self.norm3(q.float()).to(q.dtype)
        q = q + self.drop_path(self.ls3(self.mlp(q_n)))
        return q, mask_logits, mask_embed, mask_point_proj
