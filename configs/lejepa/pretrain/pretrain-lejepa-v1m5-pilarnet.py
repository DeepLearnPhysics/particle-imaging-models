"""
Configuration for pretraining a LeJEPA v1m5 model on PILArNet dataset.

v1m5: SIGReg + masked prediction (mask_loss, roll_mask_loss) + local-global.
  No teacher-student. mask_token=True in backbone.
"""

_base_ = ["../../_base_/default_runtime.py"]

# misc custom setting
batch_size = 48   # total effective bs across all gpus
num_worker = 24
batch_size_val = 32
mix_prob = 0
clip_grad = 1.0
empty_cache = False
sync_bn=True
enable_amp = True
amp_dtype = "bfloat16"
evaluate = True
find_unused_parameters = False
detect_anomaly = False
matmul_precision = "high"
deterministic = False
seed = 0
use_wandb = True
wandb_project = "Pretraining-LeJEPA-PILArNet"

grid_size = 0.001
warmup_ratio = 0.05

# model settings
model = dict(
    type="LeJEPA-v1m5",
    backbone=dict(
        type="PT-v3m2",
        in_channels=4,
        order=("hilbert", "hilbert-trans", "z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 9, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        enable_cpe=True,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        layer_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        enc_mode=True,
        mask_token=True,
        cpe_first_layer_only=False,
        cpe_shared_weight=False,
    ),
    head_in_channels=512,
    proj_hidden_channels=(2048, 2048),
    proj_dim=32,
    lamb=0.02,
    mask_weight=0,
    roll_mask_weight=1/2,
    unmask_weight=1/2,
    num_global_view=2,
    num_local_view=6,
    up_cast_level=0,
    sigreg_knots=17,
    sigreg_num_slices=256,
    # Masking schedule (set start==base to disable warmup)
    # mask_ratio_start=0.6,
    # mask_ratio_base=0.7,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=warmup_ratio,
    mask_size_start=0.01,
    mask_size_base=0.075,
    mask_size_warmup_ratio=warmup_ratio,
    mask_jitter=grid_size / 4,
    match_max_r=2 * grid_size,
)

# scheduler settings
epoch = 100
base_lr = 0.0005
lr_decay = 0.9

base_wd = 0.04
final_wd = 0.04

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

# dataset settings
transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", sum_keys=("energy",)),
    dict(
        type="LogTransform",
        min_val=0.01,
        max_val=20.0,
        log=True,
        keys=("energy",),
    ),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord", "energy", "segment_motif"),
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=6,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[
            dict(
                type="MultiplicativeRandomJitter",
                sigma=0.05,
                clip=0.05,
                keys=("energy"),
                p=0.8,
            ),
        ],
        global_transform=[
            dict(type="CenterShift", apply_z=False, axes=("x", "y", "z")),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
            dict(type="RandomFlip", p=0.5, axes=("x", "y", "z")),
            dict(
                type="RandomJitter",
                sigma=grid_size / 4,
                clip=grid_size,
                keys=("coord",),
            ),
        ],
        local_transform=[
            dict(type="CenterShift", apply_z=False, axes=("x", "y", "z")),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
            dict(type="RandomFlip", p=0.5, axes=("x", "y", "z")),
            dict(
                type="RandomJitter",
                sigma=grid_size / 4,
                clip=grid_size,
                keys=("coord",),
            ),
        ],
        max_size=30000,
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_energy",
            "global_segment_motif",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_energy",
            "local_segment_motif",
            "local_offset",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_energy",),
        local_feat_keys=("local_coord", "local_energy",),
    ),
]

data = dict(
    num_classes=5,
    names=["shower", "track", "michel", "delta", "led"],
    train=dict(
        type="PILArNetH5Dataset",
        revision="v1",
        split="train",
        transform=transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1_000_000,
        remove_low_energy_scatters=False,
        loop=1,
    ),
    val=dict(
        type="PILArNetH5Dataset",
        revision="v1",
        split="val",
        transform=[
            dict(
                type="NormalizeCoord",
                center=[384.0, 384.0, 384.0],
                scale=768.0 * 3**0.5 / 2,
            ),
            dict(type="LogTransform", min_val=0.01, max_val=20.0, log=True, keys=("energy",)),
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
                feat_keys=("coord", "energy",),
            ),
        ],
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=10000,
        remove_low_energy_scatters=False,
        loop=1,
    ),
)

class_freqs = [1926651899, 2038240940, 34083197, 92015482, 1145363125]
class_weights = [sum(class_freqs) / f for f in class_freqs]

hooks = [
    dict(
        type="WandbNamer",
        keys=("model.type", "data.train.max_len", "amp_dtype", "seed"),
        sep="-",
    ),
    dict(
        type="ParameterCounter",
        show_details=False,
        show_gradients=False,
        sort_by_params=True,
        min_params=1,
    ),
    dict(type="CheckpointLoader"),
    dict(
        type="DtypeOverrider",
        class_patterns=["LayerNorm"],
        dtype="float32",
        override_parameters=True,
        methods_to_override=["forward"],
    ),
    dict(type="ModelHook"),
    dict(
        type="WeightDecayExclusion",
        exclude_bias_from_wd=True,
        exclude_norm_from_wd=True,
        exclude_gamma_from_wd=True,
        exclude_token_from_wd=True,
        exclude_ndim_1_from_wd=True,
    ),
    dict(
        type="WeightDecayScheduler",
        base_value=base_wd,
        final_value=final_wd,
        warmup_ratio=1.0,
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaverIteration", save_freq=1000),
    dict(type="GradientNormLogger", log_frequency=10, log_per_layer=True),
    dict(
        type="OnlineLinearProbe",
        num_classes=5,
        lr=1e-3,
        weight_decay=1e-6,
        log_frequency=10,
        prefix="online_probe",
        class_names=["shower", "track", "michel", "delta", "led"],
        weight=class_weights,
    ),
    dict(
        type="PretrainEvaluator",
        write_cls_iou=True,
        every_n_steps=1000,
        train_config=dict(
            criteria=[dict(type="CrossEntropyLoss", weight=class_weights)],
        ),
    ),
    dict(
        type="ResourceUtilizationLogger",
        log_frequency=10,
        prefix="resources",
        log_per_gpu=True,
        log_cpu=True,
        log_system_memory=True,
    ),
]
