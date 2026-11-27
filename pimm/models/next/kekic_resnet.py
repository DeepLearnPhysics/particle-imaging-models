"""The sparse ResNet classifier of Kekic et al. 2021 (arXiv 2009.10783), ported from SparseConvNet to spconv
(github.com/mmkekic/NEXT_torch @1c43507: src/models/ResNet_sparse.py ``Feature_extr`` / ``SparseResNet``, nets.py
``Classifier``). The defaults are the published model on its 10 x 10 x 5 mm voxels; the native 3 mm variant changes only
what the resolution forces: an isotropic stem and first downsample, a larger input grid and a mean over each event's
sites at the end (the last map is no longer 1 x 1 x 1). Input: ``grid_coord``, ``feat`` (energy fraction), ``offset``;
wrap in ``SparseEventClassifier`` for the loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

from pimm.models.builder import MODELS
from pimm.models.utils.misc import offset2batch
from pimm.ops import SubMConv3d


# Adapted from mmkekic/NEXT_torch @1c43507, src/models/ResNet_sparse.py and nets.py.
# SCN momentum is the share of the running statistic kept.
def bn_relu(channels, mom):
    return spconv.SparseSequential(
        nn.BatchNorm1d(channels, eps=1e-5, momentum=1 - mom), nn.ReLU()
    )


class Stage(nn.Module):
    """SparseResNet's two pre-activated basic blocks, the first strided by 2. Its two strided convs (the branch's and the
    residual's) run as one conv with both outputs, so they share output sites: spconv orders a strided conv's sites
    afresh on every call."""

    def __init__(self, c_in, c, key, mom, fused):
        super().__init__()
        subm = lambda: SubMConv3d(
            c, c, 3, bias=False, indice_key=f"subm{key}", fused=fused
        )
        self.norm = bn_relu(c_in, mom)
        self.strided = spconv.SparseConv3d(
            c_in, 2 * c, 3, stride=2, bias=False
        )  # [branch | residual]
        self.branch1 = spconv.SparseSequential(bn_relu(c, mom), subm())
        self.branch2 = spconv.SparseSequential(
            bn_relu(c, mom), subm(), bn_relu(c, mom), subm()
        )

    def forward(self, x):
        y = self.strided(self.norm(x))
        branch, residual = y.features.chunk(2, dim=1)
        x = self.branch1(y.replace_feature(branch.contiguous()))
        x = x.replace_feature(x.features + residual)
        return x.replace_feature(x.features + self.branch2(x).features)


@MODELS.register_module("KekicSparseResNet")
class KekicSparseResNet(nn.Module):
    def __init__(
        self,
        n_initial_filters=15,
        mom=0.7,
        fused=False,
        stem_kernel=(3, 3, 5),
        down_kernel=(5, 5, 15),
        down_stride=(2, 2, 4),
        spatial_size=(97, 97, 199),
        global_pool=False,
    ):
        super().__init__()
        n0 = n_initial_filters
        self.spatial_size, self.global_pool = spatial_size, global_pool
        self.stem = SubMConv3d(
            1,
            n0,
            stem_kernel,
            bias=False,
            indice_key="stem",
            fused=fused and tuple(stem_kernel) == (3, 3, 3),
        )
        self.relu1 = bn_relu(n0, mom)
        self.downsample = spconv.SparseConv3d(
            n0, 2 * n0, down_kernel, stride=down_stride, bias=False
        )
        widths = [2 * n0, 4 * n0, 8 * n0, 16 * n0, 32 * n0]
        self.stages = nn.ModuleList(
            Stage(a, b, i, mom, fused)
            for i, (a, b) in enumerate(zip(widths, widths[1:]))
        )
        self.final = bn_relu(widths[-1], mom)
        self.head = nn.Sequential(
            nn.Dropout(),
            nn.Linear(widths[-1], 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 2),
        )

    def forward(self, input_dict):
        batch = offset2batch(input_dict["offset"])
        size = int(batch.max()) + 1
        indices = torch.cat([batch[:, None], input_dict["grid_coord"]], 1).int()
        x = spconv.SparseConvTensor(
            input_dict["feat"], indices, self.spatial_size, size
        )
        x = self.downsample(self.relu1(self.stem(x)))
        for stage in self.stages:
            x = stage(x)
        x = self.final(x)
        if self.global_pool:
            b = x.indices[:, 0].long()
            feat = x.features.new_zeros(size, x.features.shape[1]).index_add_(
                0, b, x.features
            )
            feat = feat / torch.bincount(b, minlength=size).clamp(min=1)[:, None]
        else:  # SCN AveragePooling(3, 2, 1) on the 2 x 2 x 2 map, then SparseToDense (empty sites are zeros)
            feat = F.avg_pool3d(x.dense(), 2, 1).flatten(1)
        return self.head(feat)
