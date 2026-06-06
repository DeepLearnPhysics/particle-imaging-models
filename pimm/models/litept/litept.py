"""LitePT backbone for pimm.

This is a pimm-local port of the Pointcept-style LitePT implementation. The
LitePT-specific helpers are kept in this file so the backbone can move as a
single module without importing implementation details from PTv3.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch_scatter
from addict import Dict
from timm.layers import DropPath

import flash_attn
from pointrope import PointROPE

from pimm.models.builder import MODELS
from pimm.models.modules import PointModule, PointSequential
from pimm.models.utils.misc import offset2bincount
from pimm.models.utils.structure import Point


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
        mask_token=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")
        if mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, embed_channels))
        else:
            self.mask_token = None

    def forward(self, point: Point):
        point = self.stem(point)
        if self.mask_token is not None and "mask" in point.keys():
            point.feat = torch.where(
                point.mask.unsqueeze(-1),
                self.mask_token.to(dtype=point.feat.dtype),
                point.feat,
            )
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class GridPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,
        re_serialization=False,
        serialization_order="z",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable
        self.re_serialization = re_serialization
        self.serialization_order = serialization_order

        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = PointSequential(norm_layer(out_channels)) if norm_layer else None
        self.act = PointSequential(act_layer()) if act_layer else None

    def forward(self, point: Point):
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError(
                "[grid_coord] or [coord, grid_size] should be included in the Point"
            )
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")
        grid_coord = grid_coord | point.batch.view(-1, 1) << 48
        grid_coord, cluster, counts = torch.unique(
            grid_coord,
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        grid_coord = grid_coord & ((1 << 48) - 1)
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
        )
        if "origin_coord" in point.keys():
            point_dict["origin_coord"] = torch_scatter.segment_csr(
                point.origin_coord[indices], idx_ptr, reduce="mean"
            )
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context
        if "name" in point.keys():
            point_dict["name"] = point.name
        if "split" in point.keys():
            point_dict["split"] = point.split
        if "color" in point.keys():
            point_dict["color"] = torch_scatter.segment_csr(
                point.color[indices], idx_ptr, reduce="mean"
            )
        if "segment_motif" in point.keys():
            point_dict["segment_motif"] = point.segment_motif[head_indices]
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride
        if "mask" in point.keys():
            point_dict["mask"] = (
                torch_scatter.segment_csr(
                    point.mask[indices].float(), idx_ptr, reduce="mean"
                )
                > 0.5
            )
        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point

        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        if self.re_serialization:
            point.serialization(
                order=self.serialization_order, shuffle_orders=self.shuffle_orders
            )
        point.sparsify()
        return point


class GridUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))
        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))
        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())
        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pooling_inverse
        feat = point.feat
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + self.proj(point).feat[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)
        if self.traceable:
            point.feat = feat
            parent["unpooling_parent"] = point
            parent["unpooling_inverse"] = inverse
        return parent


class PointROPEAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        rope_freq,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.patch_size = patch_size
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = PointROPE(freq=rope_freq)

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            if self.patch_size == -1:
                cu_seqlens = torch.cat([point.offset.new_zeros(1), point.offset]).int()
                point[pad_key] = None
                point[unpad_key] = None
                point[cu_seqlens_key] = cu_seqlens
                return point[pad_key], point[unpad_key], point[cu_seqlens_key]

            offset = point.offset
            bincount = offset2bincount(offset)
            patch_size = self.patch_size
            device = offset.device

            bincount_pad = torch.div(
                bincount + patch_size - 1, patch_size, rounding_mode="trunc"
            ) * patch_size
            mask_pad = bincount > patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad

            offset_ = F.pad(offset, (1, 0))
            offset_pad = F.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            n_total = offset_[-1]
            n_pad_total = offset_pad[-1]
            shift = offset_pad[:-1] - offset_[:-1]

            idx_unpad = torch.arange(n_total, device=device)
            unpad = idx_unpad + shift[torch.searchsorted(offset, idx_unpad, right=True)]

            pad = torch.arange(n_pad_total, device=device)
            remainder = bincount % patch_size
            needs_pad = mask_pad & (remainder != 0)
            if needs_pad.any():
                pad_views = torch.where(needs_pad)[0]
                r = remainder[pad_views]
                copy_lens = patch_size - r
                total_copies = copy_lens.sum()
                view_of_copy = torch.arange(
                    len(pad_views), device=device
                ).repeat_interleave(copy_lens)
                local_idx = torch.arange(total_copies, device=device)
                copy_cumsum = F.pad(torch.cumsum(copy_lens, dim=0), (1, 0))
                local_idx = local_idx - copy_cumsum[view_of_copy]
                dst = (
                    offset_pad[pad_views[view_of_copy] + 1]
                    - patch_size
                    + r[view_of_copy]
                    + local_idx
                )
                pad[dst] = pad[dst - patch_size]

            offset_pad_cumsum = torch.cumsum(bincount_pad, dim=0)
            idx_pad = torch.arange(n_pad_total, device=device)
            pad = pad - shift[
                torch.searchsorted(offset_pad_cumsum, idx_pad, right=True)
            ]

            patches_per_view = torch.div(
                bincount_pad + patch_size - 1, patch_size, rounding_mode="trunc"
            ).int()
            total_patches = patches_per_view.sum()
            patches_cumsum = F.pad(torch.cumsum(patches_per_view, dim=0), (1, 0))
            patch_idx = torch.arange(total_patches, device=device, dtype=torch.int32)
            patch_view = torch.searchsorted(
                patches_cumsum[1:], patch_idx, right=True
            ).int()
            patch_local = patch_idx - patches_cumsum[patch_view].int()
            cu_seqlens_vals = offset_pad[patch_view].int() + patch_local * patch_size

            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = F.pad(
                cu_seqlens_vals.int(), (0, 1), value=int(n_pad_total)
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        heads = self.num_heads
        channels = self.channels
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        max_seqlen = int(cu_seqlens[-1]) if self.patch_size == -1 else self.patch_size

        order = point.serialized_order[self.order_index]
        if pad is not None:
            order = order[pad]
            inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)[order]
        pos = point.grid_coord[order].reshape(-1, 3).unsqueeze(0)
        q, k, v = qkv.half().chunk(3, dim=-1)
        q = q.reshape(-1, heads, channels // heads).transpose(0, 1)[None]
        k = k.reshape(-1, heads, channels // heads).transpose(0, 1)[None]
        q = self.rope(q.float(), pos).to(q.dtype)
        k = self.rope(k.float(), pos).to(k.dtype)
        qkv_rotated = torch.stack(
            [
                q.squeeze(0).transpose(0, 1),
                k.squeeze(0).transpose(0, 1),
                v.reshape(-1, heads, channels // heads),
            ],
            dim=1,
        )
        feat = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv_rotated,
            cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=self.attn_drop if self.training else 0,
            softmax_scale=self.scale,
        ).reshape(-1, channels)
        feat = feat.to(qkv.dtype)
        if pad is not None:
            feat = feat[inverse]
        point.feat = self.proj_drop(self.proj(feat))
        return point


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

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_conv=True,
        enable_attn=True,
        rope_freq=100.0,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.enable_conv = enable_conv
        self.enable_attn = enable_attn

        if self.enable_conv:
            self.conv = PointSequential(
                spconv.SubMConv3d(
                    channels,
                    channels,
                    kernel_size=3,
                    bias=True,
                    indice_key=cpe_indice_key,
                ),
                nn.Linear(channels, channels),
                norm_layer(channels),
            )
        else:
            self.norm0 = PointSequential(norm_layer(channels))

        if self.enable_attn:
            self.norm1 = PointSequential(norm_layer(channels))
            self.attn = PointROPEAttention(
                channels=channels,
                patch_size=patch_size,
                rope_freq=rope_freq,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                order_index=order_index,
            )
            self.norm2 = PointSequential(norm_layer(channels))
            self.mlp = PointSequential(
                MLP(
                    in_channels=channels,
                    hidden_channels=int(channels * mlp_ratio),
                    out_channels=channels,
                    act_layer=act_layer,
                    drop=proj_drop,
                )
            )
            self.drop_path = PointSequential(
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

    def forward(self, point: Point):
        if self.enable_conv:
            shortcut = point.feat
            point = self.conv(point)
            point.feat = shortcut + point.feat
        else:
            point = self.norm0(point)

        if self.enable_attn:
            shortcut = point.feat
            if self.pre_norm:
                point = self.norm1(point)
            point = self.drop_path(self.attn(point))
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm1(point)

            shortcut = point.feat
            if self.pre_norm:
                point = self.norm2(point)
            point = self.drop_path(self.mlp(point))
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm2(point)

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


@MODELS.register_module("LitePT")
class LitePT(PointModule):
    def __init__(
        self,
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
        traceable=True,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.enc_mode = enc_mode
        self.freeze_encoder = freeze_encoder
        self.shuffle_orders = shuffle_orders
        self.enc_conv = enc_conv
        self.enc_attn = enc_attn
        self.dec_conv = dec_conv
        self.dec_attn = dec_attn

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1
        assert self.enc_mode or self.num_stages == len(dec_num_head) + 1
        assert self.enc_mode or self.num_stages == len(dec_patch_size) + 1

        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
            mask_token=mask_token,
        )

        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        traceable=traceable,
                        re_serialization=enc_attn[s],
                        serialization_order=self.order,
                        shuffle_orders=self.shuffle_orders,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_conv=enc_conv[s],
                        enable_attn=enc_attn[s],
                        rope_freq=enc_rope_freq[s],
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        if not self.enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        traceable=traceable,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_conv=dec_conv[s],
                            enable_attn=dec_attn[s],
                            rope_freq=dec_rope_freq[s],
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

        if self.freeze_encoder:
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.enc.parameters():
                p.requires_grad = False

    def forward(self, data_dict):
        point = Point(data_dict)
        if self.enc_attn[0]:
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)
        return point
