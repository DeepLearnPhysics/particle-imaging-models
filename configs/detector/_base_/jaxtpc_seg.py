# Base dataset config for JAXTPC 3D seg data.
#
# Set JAXTPC_DATA_ROOT environment variable or override data_root in child config.
# Expected directory layout:
#   {data_root}/seg/{split}/{dataset_name}_seg_NNNN.h5
#   or: {data_root}/seg/{dataset_name}_seg_NNNN.h5  (flat, split ignored)

import os

_data_root = os.environ.get("JAXTPC_DATA_ROOT", "/path/to/jaxtpc/production")

# Coordinate normalization center and scale.
# Default is for SBND-scale dual-TPC: x in [-2160, 2160], y/z in [-2160, 2160] mm.
_center = [0.0, 0.0, 0.0]
_scale = 2160.0 * 3 ** 0.5  # ~3741 mm — normalizes to roughly [-1, 1]

grid_size = 0.001  # after normalization

transform = [
    dict(type="PDGToSemantic", scheme="motif_5cls"),
    dict(type="NormalizeCoord", center=_center, scale=_scale),
    dict(type="LogTransform", min_val=0.01, max_val=20.0),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
    dict(type="Copy", keys_dict={"segment_motif": "segment"}),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment"),
        feat_keys=("coord", "energy"),
    ),
]

test_transform = [
    dict(type="PDGToSemantic", scheme="motif_5cls"),
    dict(type="NormalizeCoord", center=_center, scale=_scale),
    dict(type="LogTransform", min_val=0.01, max_val=20.0),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="Copy", keys_dict={"segment_motif": "segment"}),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment"),
        feat_keys=("coord", "energy"),
    ),
]

data = dict(
    num_classes=5,
    ignore_index=-1,
    names=["shower", "track", "michel", "delta", "led"],
    train=dict(
        type="JAXTPCDataset",
        data_root=_data_root,
        split="train",
        dataset_name="sim",
        modalities=("seg",),
        transform=transform,
        min_deposits=1024,
        max_len=-1,
    ),
    val=dict(
        type="JAXTPCDataset",
        data_root=_data_root,
        split="val",
        dataset_name="sim",
        modalities=("seg",),
        transform=test_transform,
        min_deposits=1024,
        max_len=1000,
    ),
)
