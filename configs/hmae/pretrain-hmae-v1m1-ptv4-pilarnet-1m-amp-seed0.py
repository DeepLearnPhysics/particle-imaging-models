"""
Configuration for pretraining HMAE (Hierarchical Masked Autoencoder) on PILArNet dataset

This config implements MAE-style pretraining where:
1. Point clouds are grouped into patches at the coarsest encoder level
2. A fraction of patches are randomly masked
3. Only visible patches are encoded
4. A decoder reconstructs masked patches via cross-attention
5. Chamfer loss measures reconstruction quality
"""

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 48 * 4
num_worker = 16
batch_size_val = 64
mix_prob = 0
clip_grad = 0.1
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
evaluate = True  # enable PretrainEvaluator
find_unused_parameters = False
detect_anomaly = False
matmul_precision = "high"
deterministic = False
seed = 0
num_events = 1_000_000

# wandb settings
use_wandb = True
wandb_project = "Pretraining-HMAE-PILArNet"

# grid and patch settings
grid_size = 0.001
patch_size = 0.016  # coarsest level: grid_size * 2^4 = 0.016
mask_ratio = 0.6
points_per_patch = 64

warmup_ratio = 0.025

# model settings
model = dict(
    type="HMAE-v1m1",
    backbone=dict(
        type="PT-v4m1",
        in_channels=1,  # xyz+E only
        # order=("hilbert", "hilbert-trans", "z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 3, 13), # 1 CPE + 12 attn
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=8,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        layer_scale=1e-5,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        traceable=True,
        enc_mode=True,  # encoder only
        mask_token=False,  # we handle masking ourselves
    ),
    decoder_channels=512,
    decoder_num_heads=8,
    decoder_num_layers=4,
    decoder_mlp_ratio=4.0,
    points_per_patch=points_per_patch,
    patch_size=patch_size,
    mask_ratio=mask_ratio,
    coord_loss_weight=1.0,
    energy_loss_weight=0.0,
    drop_path=0.1,
)

# scheduler settings
epoch = 200 * (1_000_000 // num_events)
wandb_run_name = f"hmae-v1m1-ptv4-pilarnet-pretrain-{num_events}ev-amp-seed{seed}-{epoch}epoch"
eval_epoch = epoch  # no eval for MAE

base_lr = 0.0001
lr_decay = 1.0

base_wd = 0.05

# layer-wise lr decay for encoder
dec_depths = model["backbone"]["enc_depths"]
param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=base_lr * lr_decay ** (sum(dec_depths) - sum(dec_depths[:e]) - b - 1),
    )
    for e in range(len(dec_depths))
    for b in range(dec_depths[e])
]
del dec_depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=warmup_ratio,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings - masking done in transform
transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(
        type="LogTransform",
        min_val=0.01,
        max_val=20.0,
        log=True,
        keys=("energy",),
    ),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train"),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5, axes=("x", "y", "z")),
    # hierarchical masking at coarsest level
    dict(
        type="HierarchicalMaskGenerator",
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        points_per_patch=points_per_patch,
        min_points_per_patch=4,
        view_keys=("coord", "origin_coord", "energy"),
    ),
    # pad targets to fixed K points
    dict(
        type="HMAECollate",
        points_per_patch=points_per_patch,
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=(
            "visible_coord",
            "visible_origin_coord",
            "visible_energy",
            "masked_centroids",
            "masked_point_counts",
            "target_coords_padded",
            "target_energy_padded",
            "target_mask",
            "n_visible_patches",
            "n_masked_patches",
            "hmae_valid",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(
            visible_offset="visible_coord",
            masked_offset="masked_centroids",
        ),
        visible_feat_keys=("visible_energy",),
    ),
]

data = dict(
    num_classes=5,
    names=["shower", "track", "michel", "delta", "led"],
    train=dict(
        type="PILArNetH5Dataset",
        split="train",
        data_root="/sdf/home/y/youngsam/data/dune/larnet/h5/reprocessed/",
        revision="v2",
        transform=transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=num_events,
        remove_low_energy_scatters=False,
        loop=1,
    ),
    test=dict(
        type="PILArNetH5Dataset",
        split="val",
        data_root="/sdf/home/y/youngsam/data/dune/larnet/h5/reprocessed/",
        revision="v2",
        transform=[
            dict(
                type="NormalizeCoord",
                center=[384.0, 384.0, 384.0],
                scale=768.0 * 3**0.5 / 2,
            ),
            dict(
                type="LogTransform",
                min_val=0.01,
                max_val=20.0,
                log=True,
                keys=("energy",),
            ),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(
                type="Copy",
                keys_dict={"coord": "origin_coord", "segment_motif": "segment"},
            ),
            # no augmentation for validation - deterministic masking
            dict(
                type="HierarchicalMaskGenerator",
                patch_size=patch_size,
                mask_ratio=mask_ratio,
                points_per_patch=points_per_patch,
                min_points_per_patch=4,
                view_keys=("coord", "origin_coord", "energy"),
            ),
            dict(
                type="HMAECollate",
                points_per_patch=points_per_patch,
            ),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=(
                    "visible_coord",
                    "visible_origin_coord",
                    "visible_energy",
                    "masked_centroids",
                    "masked_point_counts",
                    "target_coords_padded",
                    "target_energy_padded",
                    "target_mask",
                    "n_visible_patches",
                    "n_masked_patches",
                    "hmae_valid",
                    "grid_size",
                    "name",
                    # for pretrain evaluator
                    "coord",
                    "grid_coord",
                    "energy",
                    "inverse",
                    "segment",
                ),
                offset_keys_dict=dict(
                    visible_offset="visible_coord",
                    masked_offset="masked_centroids",
                ),
                visible_feat_keys=("visible_energy",),
            ),
        ],
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1000,  # limit val set size for faster evaluation
        remove_low_energy_scatters=False,
        loop=1,
    ),
    val=dict(
        type="PILArNetH5Dataset",
        split="val",
        data_root="/sdf/home/y/youngsam/data/dune/larnet/h5/reprocessed/",
        revision="v2",
        transform=[
            dict(
                type="NormalizeCoord",
                center=[384.0, 384.0, 384.0],
                scale=768.0 * 3**0.5 / 2,
            ),
            dict(
                type="LogTransform",
                min_val=0.01,
                max_val=20.0,
                log=True,
                keys=("energy",),
            ),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="Copy", keys_dict={"segment_motif": "segment"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "energy", "inverse", "segment"),
                feat_keys=("energy",),
            ),
        ],
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=-1,
        remove_low_energy_scatters=False,
        loop=1,
    ),
)

class_freqs = [1926651899, 2038240940, 34083197, 92015482, 1145363125]
class_weights = [sum(class_freqs) / f for f in class_freqs]

hooks = [
    dict(
        type="ParameterCounter",
        show_details=True,
        show_gradients=False,
        sort_by_params=True,
        min_params=1,
    ),
    dict(
        type="WeightDecayExclusion",
        exclude_bias_from_wd=True,
        exclude_norm_from_wd=True,
        exclude_gamma_from_wd=True,
        exclude_token_from_wd=True,
        exclude_ndim_1_from_wd=True,
    ),
    dict(type="CheckpointLoader"),
    dict(
        type="DtypeOverrider",
        class_patterns=["LayerNorm"],
        dtype="float32",
        override_parameters=True,
        methods_to_override=["forward"],
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="HMAEEvaluator", every_n_steps=1000, max_batches=100),
    dict(
        type="PretrainEvaluator",
        write_cls_iou=True,
        class_weights=class_weights,
        every_n_steps=1000,
        # max_samples_per_class=15000,
    ),
    dict(type="CheckpointSaverIteration", save_freq=5),
    dict(type="GradientNormLogger", log_frequency=10, log_per_layer=True),
    # dict(type="RuntimeProfiler", forward=True, backward=True, interrupt=True, warm_up=2, sort_by="cuda_time_total", row_limit=30, memory=True),
]

