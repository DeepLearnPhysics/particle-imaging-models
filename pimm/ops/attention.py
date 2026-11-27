"""Variable-length attention using FlashAttention, masked Triton kernels or PyTorch."""

import torch
import torch.nn.functional as F

from ._fused import require_fused
from .rope import apply_rope

_BACKENDS = ("flash", "torch")
_backend = "flash"


def set_attention_backend(name):
    """Choose the attention of this process's ``varlen_attention`` and ``serialized_attention`` calls without
    ``fused``: ``"flash"`` (the default) is ``fused=True`` and ``"torch"`` is ``fused=False``, PyTorch's attention on any
    GPU. Training and test runs set it from the config key ``attention_backend``."""
    global _backend
    if name not in _BACKENDS:
        raise ValueError(f"attention_backend must be one of {_BACKENDS}, got {name!r}")
    _backend = name


def get_attention_backend():
    """The name given to ``set_attention_backend``, ``"flash"`` if it was never called."""
    return _backend


_FIX = 'set attention_backend="torch" (or pass fused=False) for PyTorch\'s attention'


def _flash(tensor):
    """The module with flash_attn's variable-length functions for ``tensor``'s GPU: flash_attn on Ampere or newer,
    the flash-attention-turing port on Turing."""
    capability = (
        torch.cuda.get_device_capability(tensor.device) if tensor.is_cuda else None
    )
    if capability is not None and capability >= (8, 0):
        try:
            from flash_attn import flash_attn_interface
        except ImportError as error:
            raise ImportError(f"FlashAttention requires the flash-attn package; {_FIX}") from error
        return flash_attn_interface
    if capability == (7, 5):
        from . import _attention_turing

        return _attention_turing
    raise RuntimeError(
        f"FlashAttention runs on Turing (sm_75) and Ampere or newer GPUs, not on {tensor.device} "
        f"(capability {capability}); {_FIX}"
    )


def _pad(rows, valid):
    """``[total, ...]`` rows to zero-padded ``[batch, length, ...]``; scattering keeps the backward a masked select."""
    padded = rows.new_zeros(valid.shape + rows.shape[1:])
    padded[valid] = rows
    return padded


def varlen_attention(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *,
    softmax_scale=None,
    dropout_p=0.0,
    attn_mask=None,
    fused=None,
):
    """Softmax attention within sequences, with the same API as flash_attn_varlen_func.

    ``q`` is ``[total_q, heads, dim]`` and ``k`` and ``v`` are ``[total_k, heads, dim]``. The rows of sequence ``b``
    are ``cu_seqlens[b]:cu_seqlens[b + 1]`` (int32), and ``max_seqlen_*`` are upper bounds on the lengths. The
    attention is computed in the inputs' dtype, and the output has the layout of ``q``.

    ``fused=True`` runs FlashAttention (flash_attn on Ampere or newer, the Turing port on sm_75) or raises;
    ``fused=False`` runs PyTorch's attention, one SDPA call over the sequences padded into a batch. ``attn_mask`` is
    additive and jagged: each sequence's ``[len_q_b, len_k_b]`` block, row-major, concatenated.
    With a mask, ``fused=True`` runs a Triton kernel over each sequence's block instead of padding (fp32 accumulation,
    the mask rounded to the inputs' dtype as the reference rounds it); the mask gets no gradient.
    Without ``fused``, the attention backend decides: ``"flash"`` is ``fused=True``, ``"torch"`` is ``fused=False``
    (``set_attention_backend``).
    """
    if fused is None:
        fused = _backend == "flash"
    if fused and attn_mask is not None:
        require_fused(
            "varlen_attention",
            _masked_unsupported(q, k, attn_mask, dropout_p, max_seqlen_q, max_seqlen_k),
        )
        scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        return _masked_forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            float(scale),
        )[0]
    if fused:
        return _flash(q).flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
        )
    dim = q.shape[-1]
    if dim % 8:  # PyTorch's memory-efficient kernel needs widths divisible by 8; zero channels change nothing
        softmax_scale = dim**-0.5 if softmax_scale is None else softmax_scale
        q, k, v = (F.pad(t, (0, -dim % 8)) for t in (q, k, v))
    q_valid = torch.arange(max_seqlen_q, device=q.device) < cu_seqlens_q.diff()[:, None]
    k_valid = torch.arange(max_seqlen_k, device=q.device) < cu_seqlens_k.diff()[:, None]
    bias = q.new_zeros(q_valid.shape + k_valid.shape[1:]).masked_fill(
        ~k_valid[:, None], -1e4
    )  # padded keys
    if (
        attn_mask is not None
    ):  # the jagged layout is the row-major order of the valid (query, key) pairs
        bias = bias.masked_scatter(
            q_valid[:, :, None] & k_valid[:, None], attn_mask.to(q.dtype)
        )
    output = F.scaled_dot_product_attention(
        _pad(q, q_valid).transpose(1, 2),
        _pad(k, k_valid).transpose(1, 2),
        _pad(v, k_valid).transpose(1, 2),
        attn_mask=bias[:, None],
        dropout_p=dropout_p,
        scale=softmax_scale,
    )
    return output.transpose(1, 2)[q_valid][..., :dim]


def _masked_unsupported(q, k, attn_mask, dropout_p, max_seqlen_q, max_seqlen_k):
    """Why the masked kernel would compute something other than the reference on this call, or None."""
    if dropout_p != 0:
        return "the masked kernel has no dropout"
    if attn_mask.requires_grad:
        return "the masked kernel gives attn_mask no gradient"
    if k.shape[1] != q.shape[1]:
        return "the masked kernel takes as many key heads as query heads"
    if max_seqlen_q * max_seqlen_k >= 2**31:
        return "a sequence's mask block exceeds 32-bit addressing"
    return None


@torch.library.custom_op("pimm::masked_attention", mutates_args=())
def _masked_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .triton.attention import masked_forward

    return masked_forward(
        q, k, v, mask, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale
    )


@_masked_forward.register_fake
def _masked_forward_fake(
    q, k, v, mask, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale
):
    return torch.empty_like(q), q.new_empty(
        (q.shape[1], q.shape[0]), dtype=torch.float32
    )


@torch.library.custom_op("pimm::masked_attention_backward", mutates_args=())
def _masked_backward(
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .triton.attention import masked_backward

    return masked_backward(
        grad,
        q,
        k,
        v,
        mask,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scale,
    )


@_masked_backward.register_fake
def _masked_backward_fake(
    grad,
    q,
    k,
    v,
    mask,
    out,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    scale,
):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _masked_setup_context(ctx, inputs, output):
    q, k, v, mask, cu_seqlens_q, cu_seqlens_k, *settings = inputs
    ctx.save_for_backward(q, k, v, mask, *output, cu_seqlens_q, cu_seqlens_k)
    ctx.settings = settings
    ctx.mark_non_differentiable(output[1])


def _masked_autograd_backward(ctx, grad, unused):
    return (
        *_masked_backward(grad, *ctx.saved_tensors, *ctx.settings),
        None,
        None,
        None,
        None,
        None,
        None,
    )


_masked_forward.register_autograd(
    _masked_autograd_backward, setup_context=_masked_setup_context
)


def serialized_attention(
    qkv, gather, scatter, cos, sin, query_offsets, offsets, max_seqlen, heads, fused=None
):
    """Gather QKV into shifted windows, rotate in FP32, and attend in FP16.

    ``scatter`` selects each point's primary slot; each window's primary queries form a suffix.
    With ``fused=True``, Ampere and newer fuse the producer and omit duplicate queries, and Turing uses the same
    FlashAttention adapter as ``varlen_attention``; with ``fused=False``, PyTorch's attention runs on any GPU.
    Without ``fused``, the attention backend decides, as in ``varlen_attention``.
    Returned rows are in point order and the input dtype.
    """
    if fused is None:
        fused = _backend == "flash"
    if fused and qkv.is_cuda and torch.cuda.get_device_capability(qkv.device) >= (8, 0):
        return _serialized_forward(
            qkv.contiguous(),
            gather,
            scatter,
            cos,
            sin,
            query_offsets,
            offsets,
            max_seqlen,
            heads,
        )[0]
    packed = qkv[gather].unflatten(1, (3, heads, -1))
    output = varlen_attention(
        apply_rope(packed[:, 0].float(), cos, sin).half(),
        apply_rope(packed[:, 1].float(), cos, sin).half(),
        packed[:, 2].half(),
        offsets,
        offsets,
        max_seqlen,
        max_seqlen,
        fused=fused,
    )
    return output.flatten(1)[scatter].to(qkv.dtype)


@torch.library.custom_op("pimm::serialized_attention", mutates_args=())
def _serialized_forward(
    qkv: torch.Tensor,
    gather: torch.Tensor,
    scatter: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_offsets: torch.Tensor,
    offsets: torch.Tensor,
    max_seqlen: int,
    heads: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    from .triton.window_attention import forward

    return forward(
        qkv, gather, scatter, cos, sin, query_offsets, offsets, max_seqlen, heads
    )


@_serialized_forward.register_fake
def _serialized_forward_fake(
    qkv, gather, scatter, cos, sin, query_offsets, offsets, max_seqlen, heads
):
    rows, slots = qkv.shape[0], gather.shape[0]
    dim = qkv.shape[1] // (3 * heads)
    padded = (dim + 7) // 8 * 8
    return (
        qkv.new_empty((rows, heads * dim)),
        qkv.new_empty((rows, heads, padded), dtype=torch.float16),
        qkv.new_empty((2, slots, heads, padded), dtype=torch.float16),
        qkv.new_empty((rows, heads, padded), dtype=torch.float16),
        qkv.new_empty((heads, rows), dtype=torch.float32),
        torch.empty_like(scatter),
        torch.empty_like(scatter),
    )


@torch.library.custom_op("pimm::serialized_attention_backward", mutates_args=())
def _serialized_backward(
    grad: torch.Tensor,
    query: torch.Tensor,
    kv: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    query_map: torch.Tensor,
    duplicates: torch.Tensor,
    scatter: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_offsets: torch.Tensor,
    offsets: torch.Tensor,
    max_seqlen: int,
    heads: int,
) -> torch.Tensor:
    from .triton.window_attention import backward

    return backward(
        grad,
        query,
        kv,
        output,
        lse,
        query_map,
        duplicates,
        scatter,
        cos,
        sin,
        query_offsets,
        offsets,
        max_seqlen,
        heads,
    )


@_serialized_backward.register_fake
def _serialized_backward_fake(
    grad,
    query,
    kv,
    output,
    lse,
    query_map,
    duplicates,
    scatter,
    cos,
    sin,
    query_offsets,
    offsets,
    max_seqlen,
    heads,
):
    return grad.new_empty((grad.shape[0], 3 * grad.shape[1]))


def _serialized_setup_context(ctx, inputs, output):
    _, _, scatter, cos, sin, query_offsets, offsets, *settings = inputs
    ctx.save_for_backward(*output[1:], scatter, cos, sin, query_offsets, offsets)
    ctx.settings = settings
    ctx.mark_non_differentiable(*output[1:])
    ctx.set_materialize_grads(False)


def _serialized_autograd_backward(ctx, grad, *unused):
    if grad is None:
        return (None,) * 9
    return (_serialized_backward(grad, *ctx.saved_tensors, *ctx.settings),) + (
        None,
    ) * 8


_serialized_forward.register_autograd(
    _serialized_autograd_backward, setup_context=_serialized_setup_context
)
