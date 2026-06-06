_base_ = ["../../_base_/default_runtime.py"]

# misc custom setting
batch_size = 48  # bs: total bs in all gpus
num_worker = 12
mix_prob = 0.0
clip_grad = None
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
matmul_precision = "high"
seed = 0
evaluate = True

# Weights & Biases specific settings
use_wandb = True  # Enable Weights & Biases logging
wandb_project = "PartSeg-Sonata-PILArNet-M"  # Change to your desired project name


class_freqs = [1926651899, 2038240940, 34083197, 92015482, 1145363125]
class_weights = [sum(class_freqs) / f for f in class_freqs]

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=5,
    backbone_out_channels=1232,
    backbone=dict(
        type="PT-v3m2",
        in_channels=4,  # [xyz, energy]
        order=("hilbert", "hilbert-trans", "z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 9, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(256, 256, 256, 256),
        layer_scale=0.0,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
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
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
    ),
    criteria=[
        # dict(type="FocalLoss", loss_weight=1.0, weight=class_weights, ignore_index=-1),
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0/20.0, ignore_index=-1,
             class_weights=None),
    ],
    freeze_backbone=True,
)

# scheduler settings
epoch = 20
eval_epoch = 20
optimizer = dict(type="AdamW", lr=0.0015, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0015, 0.00015],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.00015)]
# dataset settings
grid_size = 0.001  # ~ 0.001/(1 / (768.0 * 3**0.5 / 2))
transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="LogTransform", min_val=1e-2, max_val=20.0, log=True),
    dict(
        type="GridSample",
        grid_size=0.001,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        # return_inverse=True,
    ),
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=grid_size / 8, clip=grid_size),
    dict(type="Copy", keys_dict={"segment_motif": "segment"}),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "energy", "segment"),
        feat_keys=("coord", "energy"),
    ),
]
test_transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="LogTransform", min_val=1e-2, max_val=20.0, log=True),
    dict(type="GridSample", grid_size=0.001, hash_type="fnv", mode="train", return_grid_coord=True,
        # return_inverse=True,
    ),
    # dict(type="CenterShift", apply_z=True),
    dict(type="Copy", keys_dict={"segment_motif": "segment"}),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "energy", "segment"),
        feat_keys=("coord", "energy"),
    ),
]

data = dict(
    num_classes=5,
    ignore_index=-1,
    names=["shower", "track", "michel", "delta", "led"],
    train=dict(
        # PILArNet
        type="PILArNetH5Dataset",
        revision="v1",
        split="train",
        # data_root="/path/to/pilarnet-m/",
        transform=transform,
        test_mode=False,
        energy_threshold=0.0,
        min_points=1024,
        max_len=1_000_000,  # override via --options data.train.max_len=X
        remove_low_energy_scatters=False,
        loop=1,
    ),
    val=dict(
        type="PILArNetH5Dataset",
        revision="v1",
        split="val",
        # data_root="/path/to/pilarnet-m/",
        transform=transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1000,
        remove_low_energy_scatters=False,
        loop=1,
    ),
    test=dict(
        type="PILArNetH5Dataset",
        revision="v1",
        split="test",
        # data_root="/path/to/pilarnet-m/",
        transform=test_transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=10_000,
    ),
)


# hook
hooks = [
    # auto-generate wandb run name from config values
    dict(
        type="WandbNamer",
        keys=("model.type", "data.train.max_len", "amp_dtype", "seed"),
        extra="lin"
    ),
    dict(
        type="DtypeOverrider",
        class_patterns=["LayerNorm"],
        dtype="float32",
        override_parameters=True,
        methods_to_override=["forward"],
    ),
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
        strict=False,
    ),
    dict(type="GradientNormLogger", log_frequency=10),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", every_n_steps=1000, write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None, evaluator_every_n_steps=1000),
    dict(type="FinalEvaluator", test_last=False),
]
