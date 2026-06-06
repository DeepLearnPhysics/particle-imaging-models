"""
PTv3 semantic segmentation on JAXTPC 3D data.

Drop-in replacement for PILArNet semseg — same model, different data source.
"""

_base_ = [
    "../../../configs/_base_/default_runtime.py",
    "../_base_/jaxtpc_seg.py",
]

# --- training ---
batch_size = 48
num_worker = 24
mix_prob = 0.0
clip_grad = None
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
matmul_precision = "high"
seed = 0
evaluate = True

use_wandb = True
wandb_project = "SemSeg-JAXTPC"

class_weights = None  # set from data statistics if needed

# --- model ---
model = dict(
    type="DefaultSegmentorV2",
    num_classes=5,
    backbone_out_channels=64,
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
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        layer_scale=0.0,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0 / 20.0,
            ignore_index=-1,
        ),
    ],
    freeze_backbone=False,
)

# --- scheduler ---
epoch = 20
eval_epoch = 20
base_lr = 0.0026
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.04)
param_dicts = None

scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# --- hooks ---
hooks = [
    dict(type="WeightDecayExclusion",
         exclude_bias_from_wd=True, exclude_norm_from_wd=True,
         exclude_gamma_from_wd=True, exclude_token_from_wd=True,
         exclude_ndim_1_from_wd=True),
    dict(type="CheckpointLoader"),
    dict(type="GradientNormLogger", log_frequency=10),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", every_n_steps=1000, write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None, evaluator_every_n_steps=1000),
    dict(type="PreciseEvaluator", test_last=False),
]
