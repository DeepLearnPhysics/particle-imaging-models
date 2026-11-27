"""spconv's SubMConv3d with asynchronous map construction and a four-channel Native fp16 kernel."""

import math

import spconv
import spconv.pytorch as spconv_torch
from spconv import constants
from spconv.core import ConvAlgo
import torch

from ._fused import require_fused

__all__ = ["SubMConv3d"]


def _unsupported_module(module):
    """Why the fused path can't serve this convolution, or None; checked once, when it is built."""
    from cumm import tensorview as tv

    if not spconv.__version__.startswith("2.3.") or not constants.ALL_WEIGHT_IS_KRSC:
        return "it calls spconv 2.3's internals and needs its KRSC weight layout"
    if module.fp32_accum is not None or module.act_type != tv.gemm.Activation.None_:
        return "it has no fp32-accumulating or fused-activation variant"
    if module.algo == ConvAlgo.Native and all(size % 2 for size in module.kernel_size):
        if module.in_channels != 4:
            return "Native fusion preserves spconv's reduction order only with four input channels"
        return None  # the Triton fp16 Native convolution
    if (
        module.algo == ConvAlgo.MaskImplicitGemm
        and math.prod(module.kernel_size) <= 32
        and module.indice_key
    ):
        return None  # spconv's implicit GEMM on a map built without the host wait
    return "it covers four-channel odd Native kernels and implicit GEMM with up to 32 offsets and an indice_key"


def _unsupported_call(module, sparse, add_input):
    """Why the fused path would compute something other than spconv on this training call, or None."""
    autocast = torch.is_autocast_enabled("cuda")
    if add_input is not None:
        return "it has no residual input"
    if torch.are_deterministic_algorithms_enabled():
        return "it has no deterministic mode"
    if (
        module.algo == ConvAlgo.Native
        and not autocast
        and sparse.features.dtype != torch.float16
    ):
        return "its Native convolution computes in float16, so it needs autocast or float16 inputs"
    if module.algo == ConvAlgo.MaskImplicitGemm and (
        autocast or sparse.features.dtype != torch.float32
    ):
        return "its implicit GEMM path is float32 without autocast"
    if len(sparse.features) * math.prod(module.kernel_size) >= 2**31:
        return "its maps would exceed 32-bit addressing"
    return None


class SubMConv3d(spconv_torch.SubMConv3d):
    """spconv's SubMConv3d (same arguments, parameters and checkpoint keys) plus ``fused``.

    With ``fused=True`` a training call builds its maps without spconv's host wait (implicit GEMM with up to 32
    offsets and an ``indice_key``) or runs the Triton fp16 Native convolution (four input channels and odd kernels);
    it raises when built, or called, outside those cases. Evaluation always runs spconv.
    """

    def __init__(self, *args, fused=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fused = fused
        if fused:
            require_fused("SubMConv3d", _unsupported_module(self))

    def forward(self, input, add_input=None):
        if not (self.fused and self.training):
            return super().forward(input, add_input)
        require_fused("SubMConv3d", _unsupported_call(self, input, add_input))
        mapped, data = _build_maps(self, input)
        if self.algo == ConvAlgo.Native:
            from .triton.submconv3d import NativeSubmConvFp16

            features = NativeSubmConvFp16.apply(
                input.features.contiguous(),
                self.weight,
                data.indice_pairs,
                data.indice_pair_num,
            )
            if self.bias is not None:
                features += self.bias
            return mapped.replace_feature(features)
        return super().forward(mapped)


def _sort_masks_(mask, order):
    """In place, spconv's stable sort of a map's mask words as unsigned keys, carrying ``order``, without its
    host wait."""
    words = mask.view(-1)
    permutation = (words.long() & 0xFFFFFFFF).argsort(stable=True)
    words.copy_(words[permutation])
    order.copy_(order[permutation])


def _build_maps(module, sparse):
    """Reuse or build a training map, without mutating the caller's indice dictionary."""
    from spconv.pytorch import ops
    from spconv.pytorch.core import ImplicitGemmIndiceData, IndiceData

    cached = sparse.indice_dict.copy()
    data = cached.get(module.indice_key) if module.indice_key is not None else None
    native = module.algo == ConvAlgo.Native
    expected = IndiceData if native else ImplicitGemmIndiceData
    if data is not None:
        if not isinstance(data, expected) or data.algo != module.algo:
            raise ValueError(
                "Convolutions sharing an indice_key must use the same algorithm"
            )
        module._check_subm_reuse_valid(sparse, sparse.spatial_shape, data)
        if not native and not data.pair_bwd.numel():
            data = None  # An inference-only producer has no input-gradient map.
    if data is None:
        metadata = (
            sparse.spatial_shape,
            sparse.spatial_shape,
            True,
            module.algo,
            module.kernel_size,
            module.stride,
            module.dilation,
            module.padding,
        )
        if native:
            outids, pairs, counts = ops.get_indice_pairs(
                sparse.indices,
                sparse.batch_size,
                sparse.spatial_shape,
                module.algo,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.output_padding,
                True,
                False,
            )
            data = IndiceData(outids, sparse.indices, pairs, counts, *metadata)
        else:
            sort = ops.SPCONV_DO_SORT
            result = ops.get_indice_pairs_implicit_gemm(
                sparse.indices,
                sparse.batch_size,
                sparse.spatial_shape,
                module.algo,
                ksize=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                out_padding=module.output_padding,
                subm=True,
                transpose=False,
                is_train=True,
                alloc=sparse.thrust_allocator,
                timer=sparse._timer,
                **({"do_sort": False} if sort else {}),
            )
            (
                outids,
                _,
                pair_fwd,
                pair_bwd,
                masks_fwd,
                masks_bwd,
                order_fwd,
                order_bwd,
                masks,
            ) = result
            if sort:
                for mask, order in zip(masks_fwd, order_fwd):
                    _sort_masks_(mask, order)
            data = ImplicitGemmIndiceData(
                outids,
                sparse.indices,
                pair_fwd,
                pair_bwd,
                masks_fwd,
                masks_bwd,
                order_fwd,
                order_bwd,
                masks,
                *metadata,
            )
        if module.indice_key is not None:
            cached[module.indice_key] = data
    output = sparse.replace_feature(sparse.features)
    output.indice_dict = cached
    return output, data
