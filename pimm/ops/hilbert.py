"""Hilbert serialization codes of integer point coordinates, as a Triton kernel."""

__all__ = ["hilbert_encode"]


def hilbert_encode(grid_coord, depth):
    """``[N, 3]`` non-negative integer coordinates below ``2**depth``, on a CUDA device, to ``[N]`` int64 Hilbert codes.

    The codes equal those of the PyTorch encoder in ``pimm.models.utils.serialization`` (the same curve, bit for bit),
    which the models use for CPU tensors.
    """
    from .triton.hilbert import hilbert_codes

    return hilbert_codes(grid_coord, depth)
