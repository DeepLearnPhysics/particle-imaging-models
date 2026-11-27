# Ported from https://github.com/WireCell/wire-cell-fm (wcfm/model/backbones/polarmae/cnms_kernel.py).
"""The greedy pass of centrality-based non-maximum suppression, one program per event and one candidate at a time.

Why this exists: in the PoLAr-MAE tokenizer (``pimm/models/polarmae/ops.py``, ``cnms``), the candidate group centres of
every event are visited in descending neighbour count, and a candidate not yet suppressed is retained and suppresses
its ball. The visit is sequential, and in the PyTorch version (``ops._greedy_rounds``) it is run in rounds of scatters
over the whole neighbour table. Here the candidates of one event are visited in order by one program, with the
suppressed flags in a scratch row.

Reached through ``pimm.models.polarmae.ops.cnms`` on CUDA tensors.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _greedy_kernel(
    order_ptr, nb_ptr, lengths_ptr, retain_ptr, suppressed_ptr, P, K, BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.load(lengths_ptr + b)
    cols = tl.arange(0, BLOCK_K)
    row = b.to(tl.int64) * P
    for i in range(0, n):
        c = tl.load(order_ptr + row + i)
        if c < n:
            if tl.load(suppressed_ptr + row + c) == 0:
                tl.store(retain_ptr + row + c, 1)
                nb = tl.load(nb_ptr + (row + c) * K + cols, mask=cols < K, other=-1)
                tl.store(suppressed_ptr + row + nb, 1, mask=nb >= 0)
        # The next candidate's suppressed flag may have been written by any thread above.
        tl.debug_barrier()


def greedy_retain(order: Tensor, idx: Tensor, lengths: Tensor) -> Tensor:
    """`(B, P)` bool: the candidates retained by the sequential greedy pass.

    `order` is `(B, P)`, each event's candidates by descending neighbour count; `idx` is the
    `(B, P, K)` neighbour lists padded with -1. A candidate is retained when no retained
    candidate before it in `order` holds it in its list, and every candidate in a retained one's
    list is suppressed. Candidates at or past `lengths` are never retained.
    """
    B, P, K = idx.shape
    retain = torch.zeros(B, P, dtype=torch.int8, device=idx.device)
    if B == 0 or P == 0:
        return retain.bool()
    suppressed = torch.zeros_like(retain)
    _greedy_kernel[(B,)](
        order.to(torch.int32).contiguous(),
        idx.to(torch.int32).contiguous(),
        lengths.to(device=idx.device, dtype=torch.int32).contiguous(),
        retain,
        suppressed,
        P,
        K,
        BLOCK_K=triton.next_power_of_2(max(K, 1)),
        num_warps=4,
    )
    return retain.bool()
