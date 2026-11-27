"""Dropout-free variable-length FlashAttention on Turing, with padded head dimensions."""

import torch
import torch.nn.functional as F

try:
    import flash_attn_turing
except ImportError as error:
    raise ImportError(
        "FlashAttention on Turing requires flash_attn_turing; "
        'set attention_backend="torch" (or pass fused=False) for PyTorch\'s attention'
    ) from error

HEAD_DIMS = (32, 64, 96, 128)


def _pad(tensor, size):
    return (
        F.pad(tensor, (0, size - tensor.shape[-1]))
        if tensor.shape[-1] < size
        else tensor.contiguous()
    )


# FlashAttention-2 kernels from flash-attention-turing.
class _Varlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_k, max_q, max_k, scale):
        dim = q.shape[-1]
        if dim > HEAD_DIMS[-1]:
            raise ValueError(
                f"Turing FlashAttention takes head dimensions up to {HEAD_DIMS[-1]}, got {dim}"
            )
        size = next(size for size in HEAD_DIMS if size >= dim)
        out, lse = flash_attn_turing.varlen_fwd(
            _pad(q, size),
            _pad(k, size),
            _pad(v, size),
            cu_q.int(),
            cu_k.int(),
            max_q,
            max_k,
            scale,
            False,
        )
        out = out[..., :dim].contiguous()
        ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k)
        ctx.settings = max_q, max_k, scale, size
        return out

    @staticmethod
    def backward(ctx, grad):
        q, k, v, out, lse, cu_q, cu_k = ctx.saved_tensors
        max_q, max_k, scale, size = ctx.settings
        gradients = flash_attn_turing.varlen_bwd(
            _pad(q, size),
            _pad(k, size),
            _pad(v, size),
            _pad(out, size),
            lse,
            _pad(grad, size),
            cu_q.int(),
            cu_k.int(),
            max_q,
            max_k,
            scale,
            False,
        )
        return (
            *[g[..., : q.shape[-1]].contiguous() for g in gradients],
            None,
            None,
            None,
            None,
            None,
        )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
):
    """Attention over packed FP16 queries, keys and values; dropout is unsupported."""
    if dropout_p:
        raise NotImplementedError("the Turing kernels run without dropout")
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    return _Varlen.apply(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale
    )
