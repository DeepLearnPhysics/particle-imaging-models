"""Packed variable-length attention on PyTorch 2.10.

Mirrors flash_attn.flash_attn_varlen_func / flash_attn_varlen_qkvpacked_func
on top of the PyTorch 2.10 ``torch.nn.attention.varlen_attn`` API.
"""

import torch
from torch.nn.attention.varlen import varlen_attn


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Attention over packed variable-length sequences.

    q: (total_q, heads, head_dim); k, v: (total_k, heads, head_dim).
    Returns (total_q, heads, head_dim).
    """
    if dropout_p:
        raise NotImplementedError(
            "torch.nn.attention.varlen_attn does not support dropout; "
            "set attn_drop to 0 or disable the flash attention path"
        )
    # torch's varlen kernels require head_dim % 8 == 0. Zero-padding the head
    # dimension is exact: padded dims contribute nothing to q.k scores, and the
    # padded value dims are sliced off. PyTorch 2.10 has no explicit scale
    # argument, so rescale q to preserve either the requested scale or the
    # original head dimension's default after padding.
    head_dim = q.shape[-1]
    pad = -head_dim % 8
    target_scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    if pad:
        q, k, v = (torch.nn.functional.pad(tensor, (0, pad)) for tensor in (q, k, v))
    kernel_scale = q.shape[-1] ** -0.5
    if target_scale != kernel_scale:
        q = q * (target_scale / kernel_scale)
    out = varlen_attn(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal=causal,
    )
    if pad:
        out = out[..., :head_dim]
    return out


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Self-attention over packed qkv of shape (total, 3, heads, head_dim)."""
    # pad the packed tensor before unbinding: one allocation and one backward
    # op instead of three (see the head_dim note in flash_attn_varlen_func)
    head_dim = qkv.shape[-1]
    pad = -head_dim % 8
    if pad:
        if softmax_scale is None:
            softmax_scale = head_dim**-0.5
        qkv = torch.nn.functional.pad(qkv, (0, pad))
    q, k, v = qkv.unbind(dim=1)
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
    )
    if pad:
        out = out[..., :head_dim]
    return out
