"""
General Utils for Models

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from itertools import chain

import torch
import torch.nn as nn


# Stochastic depth from timm (huggingface/pytorch-image-models).
def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
    """Drop residual paths independently for each sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Per-sample stochastic depth with timm-compatible behavior."""

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(
            x,
            self.drop_prob,
            self.training,
            self.scale_by_keep,
        )

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.no_grad()
def bincount2offset(bincount):
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset2batch(offset, size=None):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(
        bincount, output_size=size
    )  # a known row count skips reading the total back


@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# Mean pooling from PyG (pyg-team/pytorch_geometric, utils/_scatter.py).
def scatter_mean(src, index, dim_size=None):
    """Average feature rows by batch index; empty batches remain zero."""
    size = dim_size
    if size is None:
        size = int(index.max()) + 1 if index.numel() else 0
    count = src.new_zeros(size).scatter_add_(0, index, src.new_ones(index.numel()))
    count.clamp_(min=1)
    out = src.new_zeros(size, src.shape[1])
    return out.scatter_add_(0, index[:, None].expand_as(src), src) / count[:, None]
