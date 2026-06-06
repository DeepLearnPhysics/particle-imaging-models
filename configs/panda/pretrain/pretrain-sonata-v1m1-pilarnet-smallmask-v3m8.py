"""
Configuration for pretraining a SONATA model on PILArNet dataset
with PT-v3m8 backbone (bottleneck CPE variant of Utonia)
"""

_base_ = ["../../_base_/default_runtime.py"]

# misc custom setting
batch_size = 48  # total effective bs across all gpus
num_worker = 24
batch_size_val = 36
mix_prob = 0
clip_grad = 3.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
evaluate = True
find_unused_parameters = False
detect_anomaly = False
matmul_precision = "high"
deterministic = False
seed = 0
# Weights & Biases specific settings
use_wandb = True  # Enable Weights & Biases logging
wandb_project = "Pretraining-Sonata-PILArNet-M"  # Change to your desired project name

grid_size = 0.001
warmup_ratio = 0.05

# model settings
model = dict(
    type="Sonata-v1m1",
    # backbone - student & teacher
    backbone=dict(
        type="PT-v3m8",
        in_channels=4,  # [xyz, energy]
        order=("hilbert", "hilbert-trans", "z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 9, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        enc_cpe_channels=(54, 108, 108, 108, 108),
        mlp_ratio=4,
        qk_norm=False,
        qkv_bias=True,
        qk_scale=None,
        layer_scale=1e-5,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        enable_cpe=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        enc_mode=True,
        mask_token=True,
        cpe_first_layer_only=False,
        rope_base=10,
        rope_jitter=1.1,
        rope_rescale=1.2,
    ),
    teacher_custom=dict(
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ),
    head_in_channels=576,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=6,
    mask_size_start=0.01,
    mask_size_base=0.075,
    mask_size_warmup_ratio=warmup_ratio,
    mask_ratio_start=0.5,
    mask_ratio_base=0.9,
    mask_ratio_warmup_ratio=warmup_ratio,
    mask_jitter=grid_size / 2,  # usually grid_size / 2
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=warmup_ratio,
    student_temp=0.10,
    mask_loss_weight=2 / 8,
    roll_mask_loss_weight=2 / 8,
    unmask_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1.0,
    match_max_r=2 * grid_size,
    up_cast_level=0,
)

# scheduler settings
# epoch: set directly here or via cli with --options epoch=X
epoch = 100
base_lr = 0.0026 #* (batch_size / 48) ** 0.5
lr_decay = 0.9  # layer-wise lr decay

base_wd = 0.04  # wd scheduler enable in hooks
final_wd = 0.2  # wd scheduler enable in hooks

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
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3 ** 0.5 / 2),
    dict(type="RandomScale", scale=[0.9, 1.2]),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        sum_keys=("energy",),
    ),
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
        view_keys=("coord", "origin_coord", "energy"),
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=6,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[
            dict(
                type="MultiplicativeRandomJitter",
                sigma=0.05,
                clip=0.1,
                keys=("energy",),
                p=0.8,
            ),
        ],
        global_transform=[
            dict(type="CenterShift", axes=("x", "y", "z")),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
            dict(type="RandomFlip", p=0.5, axes=("x", "y", "z")),
            dict(
                type="RandomJitter",
                sigma=grid_size / 8,
                clip=grid_size,
                keys=("coord",),
            ),
        ],
        local_transform=[
            dict(type="CenterShift", axes=("x", "y", "z")),
            dict(type="RandomScale", scale=[0.9, 1.1]),
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
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_energy",
            "local_offset",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=(
            "global_coord",
            "global_energy",
        ),
        local_feat_keys=(
            "local_coord",
            "local_energy",
        ),
    ),
]

data = dict(
    num_classes=5,
    names=["shower", "track", "michel", "delta", "led"],
    train=dict(
        type="PILArNetH5Dataset",
        revision="v2",
        split="train",
        transform=transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1_000_000,  # override via --options data.train.max_len=X
        remove_low_energy_scatters=False,
        loop=1,
    ),
    val=dict(
        type="PILArNetH5Dataset",
        revision="v2",
        split="val",
        transform=[
            dict(
                type="NormalizeCoord",
                center=[384.0, 384.0, 384.0],
                scale=768.0 * 3**0.5 / 2,
            ),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
                sum_keys=("energy",),
            ),
            dict(type="LogTransform", min_val=0.01, max_val=20.0, log=True, keys=("energy",)),
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
        max_len=10_000,
        remove_low_energy_scatters=False,
        loop=1,
    ),
)

class_freqs = [1926651899, 2038240940, 34083197, 92015482, 1145363125]
class_weights = [sum(class_freqs) / f for f in class_freqs]

hooks = [
    # auto-generate wandb run name from config values
    dict(
        type="WandbNamer",
        keys=("model.type", "model.backbone.type", "data.train.max_len", "amp_dtype", "seed"),
    ),
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
    dict(type="ModelHook"),
    dict(
        type="WeightDecayScheduler",
        base_value=base_wd,
        final_value=final_wd,
        warmup_ratio=1.0,
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaverIteration", save_freq=1000),
    dict(type="GradientNormLogger", log_frequency=10, log_per_layer=False),
    dict(
        type="PretrainEvaluator",
        write_cls_iou=True,
        every_n_steps=1000,
        train_config=dict(
            criteria=[dict(type="CrossEntropyLoss", weight=class_weights)],
        )
    ),
    dict(
        type="PrototypeUsageLogger",
        log_frequency=10,
        prefix="prototypes",
    ),
    dict(
        type="FeatureStdMonitor",
        log_frequency=10,
        prefix="feature_std",
        monitor_student=True,
        monitor_teacher=True,
        track_channels=False,
    ),
]
