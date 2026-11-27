"""Panda V2 encodes point clouds with sparse convolutions and local attention.

The first ``conv_stages`` of four stages are sparse convolutions, which preserve fine detail; in the other stages,
nearby points attend to each other in windows of ``window_size`` serialized points with 3D rotary positions. Between
stages, points are pooled into voxels ``strides`` times larger. As in PTv3, the coarsest stage is returned as a Point:
each pooled Point keeps the finer one in ``pooling_parent`` and its cluster indices in ``pooling_inverse``, and with
``Point.upcast()`` the four stages' features are joined at each input point (1170 channels by default). Inputs contain
feat, coord, and batch or offset; grid_coord is optional and otherwise computed using grid_size.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import spconv.pytorch as spconv
from spconv.core import ConvAlgo

from pimm.models.builder import MODELS
from pimm.models.utils.checkpoint import recompute_activation
from pimm.models.utils.misc import DropPath
from pimm.models.utils.structure import Point
from pimm.ops import SubMConv3d, serialized_attention
from pimm.ops.rope import rope_tables

__all__ = ["PandaV2Encoder"]

ORDERS = ("hilbert", "hilbert-trans", "z", "z-trans")


def pool(grid, batch, stride):
    """Voxels ``stride`` times larger, sorted by batch and position, and the voxel of each point.

    Coordinates are packed with the batch index into one int64 key, 16 bits per axis."""
    grid = torch.div(grid.long(), stride, rounding_mode="trunc")
    torch._assert_async(((grid >= 0) & (grid < 1 << 16)).all(), "grid coordinates must be in [0, 65536)")
    key = batch.long() << 48 | grid[:, 0] << 32 | grid[:, 1] << 16 | grid[:, 2]
    key, cluster = torch.unique(key, return_inverse=True)
    bits = (1 << 16) - 1
    return torch.stack([key >> 32 & bits, key >> 16 & bits, key & bits], 1), key >> 48, cluster


def window_indices(batch, size):
    """Split each event into windows of ``size`` points, shifting the final window to cover its end."""
    count = torch.bincount(batch)
    first = count.cumsum(0) - count  # each event's first serialized position
    width = count.clamp(max=size)
    per_event = (count + size - 1) // size
    first_window = per_event.cumsum(0) - per_event
    event = torch.repeat_interleave(per_event)  # the event of each window
    j = torch.arange(len(event), device=batch.device) - first_window[event]
    start = first[event] + torch.minimum(j * size, count[event] - width[event])
    offsets = F.pad(width[event].cumsum(0), (1, 0))
    index = torch.repeat_interleave(start - offsets[:-1], width[event])
    index += torch.arange(len(index), device=batch.device)
    t = torch.arange(len(batch), device=batch.device) - first[batch]
    window = first_window[batch] + torch.minimum(t // size, per_event[batch] - 1)
    where = offsets[window] + first[batch] + t - start[window]
    query_width = (count[event] - j * size).clamp(max=size)
    query_offsets = F.pad(query_width.cumsum(0), (1, 0)).int()
    return index, offsets.int(), where, query_offsets


@torch.no_grad()
def plan_attention(point, head_dim, shuffle, window):
    """Window indices and rotary tables for the four point orders, shuffled only during training."""
    point.serialization(order=ORDERS, shuffle_orders=shuffle)
    index, offsets, where, query_offsets = window_indices(point.batch, window)
    plans = []
    for order, inverse in zip(point.serialized_order, point.serialized_inverse):
        gather = order[index]
        plans.append(
            (
                gather,
                where[inverse],
                *rope_tables(point.grid_coord[gather], head_dim),
                query_offsets,
                offsets,
                window,
            )
        )
    return plans


class Attention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.heads = heads
        self.qkv, self.proj = (
            nn.Linear(channels, 3 * channels),
            nn.Linear(channels, channels),
        )

    def forward(self, x, plan):
        qkv = self.qkv(x)
        with torch.autocast("cuda", enabled=False):
            out = serialized_attention(qkv, *plan, self.heads)
        return self.proj(out.to(x.dtype))


class AttnBlock(nn.Module):
    def __init__(self, channels, heads, drop, order_index):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(channels), Attention(channels, heads)
        self.norm2, self.mlp = (
            nn.LayerNorm(channels),
            nn.Sequential(
                nn.Linear(channels, 4 * channels),
                nn.GELU(),
                nn.Linear(4 * channels, channels),
            ),
        )
        self.drop_path = DropPath(drop, scale_by_keep=False)
        self.order_index = order_index

    def forward(self, x, plans):
        keep = 1 - self.drop_path.drop_prob if self.training else 1
        attn = self.attn(self.norm1(x), plans[self.order_index])
        # Scaling after multiplication preserves BF16 rounding.
        x = x + self.drop_path(attn) / keep
        fc1, act, fc2 = self.mlp
        mlp = recompute_activation(act, fc1(self.norm2(x)), fc2)  # fc2's backward recomputes the GELU
        return x + self.drop_path(mlp) / keep


class AttnStage(nn.ModuleList):
    """Apply each attention block using its assigned point order."""

    def forward(self, x, plans):
        for block in self:
            x = block(x, plans)
        return x


class SparseConv(SubMConv3d):
    """Keep spconv for deterministic training and full precision Native calls."""

    def forward(self, sparse):
        if torch.are_deterministic_algorithms_enabled() or (
            self.algo == ConvAlgo.Native
            and not torch.is_autocast_enabled("cuda")
            and sparse.features.dtype != torch.float16
        ):
            return spconv.SubMConv3d.forward(self, sparse)
        return super().forward(sparse)


class ConvBlock(nn.Module):
    def __init__(self, channels, key):
        super().__init__()
        self.conv = SparseConv(
            channels, channels, 3, bias=True, indice_key=key, fused=True
        )
        self.linear, self.norm = nn.Linear(channels, channels), nn.LayerNorm(channels)

    def forward(self, point):
        with torch.autocast("cuda", enabled=False):
            x = point.feat.float()
            point.sparse_conv_feat = self.conv(point.sparse_conv_feat.replace_feature(x))
            point.feat = (x + self.norm(self.linear(point.sparse_conv_feat.features))).to(
                point.feat.dtype
            )


class Pool(nn.Module):
    """Merge points in larger voxels with a projection, max pooling, BatchNorm and GELU."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.proj, self.norm, self.stride = (
            nn.Linear(cin, cout),
            nn.BatchNorm1d(cout, eps=1e-3, momentum=0.01),
            stride,
        )

    def forward(self, point):
        if self.stride == 1 and "pooling_parent" in point.keys():  # a pooled level is unique and sorted
            grid, batch = point.grid_coord, point.batch
            cluster = torch.arange(len(grid), device=grid.device)
        else:
            grid, batch, cluster = pool(point.grid_coord, point.batch, self.stride)
        # Rows sorted stably by cluster: sums run in row order (reproducible), and the max of each cluster takes the
        # gradient to its first maximal row, ties included.
        order = torch.argsort(cluster, stable=True)
        count = torch.bincount(cluster, minlength=len(grid))
        ptr = torch.cat([count.new_zeros(1), torch.cumsum(count, 0)])
        feat = torch_scatter.segment_csr(self.proj(point.feat).index_select(0, order), ptr, reduce="max")
        pooled = Point(
            feat=F.gelu(self.norm(feat)),
            grid_coord=grid,
            batch=batch,
            pooling_parent=point,
            pooling_inverse=cluster,
        )
        with torch.no_grad():
            for key in ("coord", "origin_coord"):
                if key in point.keys():
                    total = torch_scatter.segment_csr(point[key].index_select(0, order), ptr, reduce="sum")
                    pooled[key] = total / count[:, None].to(total.dtype)
        return pooled


@MODELS.register_module("panda-encoder-v2m1")
class PandaV2Encoder(nn.Module):
    """Return the coarsest stage's Point; heads gives per-stage counts (18 channels per head by default).
    Attention runs with the process's attention backend (``pimm.ops.set_attention_backend``).
    With ``mask_token``, the stem features of points flagged by a boolean ``mask`` are replaced by a learned token."""

    def __init__(
        self,
        in_channels=4,
        drop_path=0.0,
        grid_size=0.001,
        channels=(54, 108, 432, 576),
        depths=(3, 6, 12, 6),
        heads=(3, 6, 24, 32),
        strides=(2, 1, 4),
        window_size=1024,
        conv_stages=2,
        mask_token=False,
    ):
        super().__init__()
        self.grid_size = grid_size
        if not len(channels) == len(depths) == len(heads) == 4:
            raise ValueError(
                "channels, depths, and heads must each describe four stages"
            )
        if any(
            c <= 0 or d <= 0 or h <= 0 or c % h
            for c, d, h in zip(channels, depths, heads)
        ):
            raise ValueError(
                "Stage sizes must be positive and channels divisible by heads"
            )
        strides = tuple(strides)
        if len(strides) != 3 or any(
            not isinstance(s, int) or s <= 0 for s in strides
        ):
            raise ValueError(
                "strides must be three positive integers, one per pooling step"
            )
        if not 0 <= conv_stages <= 4:
            raise ValueError("conv_stages must be between 0 and 4")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.head_dims = [c // h for c, h in zip(channels, heads)]
        if any(width % 6 for width in self.head_dims[conv_stages:]):
            raise ValueError(
                "Attention head widths must be divisible by six for 3D rotary positions"
            )
        self.strides, self.window_size, self.conv_stages = strides, window_size, conv_stages
        # The Native kernel preserves spconv's reduction order for four input channels.
        self.stem = SparseConv(
            in_channels,
            channels[0],
            5,
            padding=1,
            bias=False,
            indice_key="stem",
            fused=in_channels == 4,
        )
        self.stem_norm = nn.LayerNorm(channels[0])
        self.mask_token = nn.Parameter(torch.zeros(1, channels[0])) if mask_token else None
        self.pools = nn.ModuleList(
            Pool(channels[s], channels[s + 1], stride)
            for s, stride in enumerate(strides)
        )
        drops = torch.linspace(0, drop_path, sum(depths)).tolist()
        self.blocks = nn.ModuleList(
            (
                nn.ModuleList(ConvBlock(channels[s], f"stage{s}") for _ in range(depth))
                if s < conv_stages
                else AttnStage(
                    AttnBlock(channels[s], heads[s], drops[sum(depths[:s]) + i], i % 4)
                    for i in range(depth)
                )
            )
            for s, depth in enumerate(depths)
        )

    def forward(self, data_dict):
        point = data_dict if isinstance(data_dict, Point) else Point(data_dict)
        if "grid_coord" not in point.keys():
            coord = point.coord.float()
            point.grid_coord = torch.div(
                coord - coord.min(0).values, self.grid_size, rounding_mode="trunc"
            ).long()
        point.sparsify()
        point.feat = F.gelu(self.stem_norm(self.stem(point.sparse_conv_feat).features))
        if "mask" in point.keys():
            point.feat = torch.where(
                point.mask[:, None], self.mask_token.to(point.feat.dtype), point.feat
            )
        for s, blocks in enumerate(self.blocks):
            if s > 0:
                point = self.pools[s - 1](point)
            if s < self.conv_stages:
                point.sparsify()
                for block in blocks:
                    block(point)
            else:
                point.feat = blocks(
                    point.feat,
                    plan_attention(point, self.head_dims[s], self.training, self.window_size),
                )
        return point

