"""
PoLAr-MAE Semantic Segmentation - Full Fine-Tuning (FFT)

Usage:
    sh scripts/train.sh -g 1 -d polarmae/semseg -c semseg-polarmae-pilarnet-fft -n polarmae_fft \
        -w /path/to/polarmae_pretrain.ckpt

Pretrained checkpoint:
    wget https://github.com/DeepLearnPhysics/PoLAr-MAE/releases/download/weights/polarmae_pretrain.ckpt
"""

_base_ = ["../../_base_/default_runtime.py"]

# misc custom setting
batch_size = 32  # PoLAr-MAE used smaller batches
num_worker = 16
mix_prob = 0.0
clip_grad = None
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
matmul_precision = "high"
seed = 0
evaluate = True

# Weights & Biases
use_wandb = True
wandb_project = "PoLArMAE-SemSeg-PILArNet"

# Model settings (architecture matches polarmae_fft_segsem.ckpt)
model = dict(
    type="PoLArMAE-SemSeg",
    num_classes=4,  # shower, track, Michel, delta
    arch="vit_small",  # 384 dim, 12 layers, 6 heads
    voxel_size=5.0,
    num_channels=4,  # xyz + energy
    seg_head_fetch_layers=[3, 7, 11],  # Multi-scale feature aggregation
    seg_head_combination_method="mean",
    seg_head_dim=512,
    seg_head_dropout=0.5,
    freeze_encoder=False,  # FFT mode - train entire model
    apply_encoder_postnorm=False,
    condition_global_features=True,
    upsampling_k=5,
    # Coordinate normalization (PoLAr-MAE defaults)
    center=[384.0, 384.0, 384.0],
    scale=1.0 / (768 * (3**0.5) / 2),  # ~1/665
    transformer_kwargs=dict(
        postnorm=False,
        add_pos_at_every_layer=True,
        drop_rate=0.0,
        attn_drop_rate=0.05,
        drop_path_rate=0.25,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=0.05, ignore_index=-1),
    ],
)

# Training schedule (matching PoLAr-MAE paper)
epoch = 100
eval_epoch = 100
base_lr = 1e-3
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.05)

# Layerwise LR decay for encoder
lr_decay = 0.75
param_dicts = []

# Apply LR decay to encoder blocks (smaller LR for earlier layers)
for i in range(12):  # 12 transformer blocks
    exp = 11 - i  # Earlier layers get smaller LR
    param_dicts.append(
        dict(
            keyword=f"encoder.blocks.{i}.",
            lr=base_lr * (lr_decay ** exp),
        )
    )

scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# Dataset settings
# Note: PoLAr-MAE handles tokenization internally, so we use minimal transforms
# No GridSample needed - PoLAr-MAE does its own FPS-based tokenization
transform = [
    dict(type="LogTransform", min_val=0.13, max_val=20.0),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
    dict(type="Copy", keys_dict={"segment_motif": "segment"}),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "segment"),
        feat_keys=("coord", "energy"),
    ),
]

test_transform = [
    dict(type="LogTransform", min_val=0.13, max_val=20.0),
    dict(type="Copy", keys_dict={"segment_motif": "segment"}),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "segment"),
        feat_keys=("coord", "energy"),
    ),
]

data = dict(
    num_classes=4,
    ignore_index=-1,
    names=["shower", "track", "michel", "delta"],
    train=dict(
        type="PILArNetH5Dataset",
        revision="v1",
        split="train",
        transform=transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1_000_000,
        remove_low_energy_scatters=True,
    ),
    val=dict(
        type="PILArNetH5Dataset",
        revision="v1",
        split="val",
        transform=test_transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1000,
        remove_low_energy_scatters=True,
    ),
    test=dict(
        type="PILArNetH5Dataset",
        revision="v1",
        split="test",
        transform=test_transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1000,
        remove_low_energy_scatters=True,
    ),
)

# Hooks
hooks = [
    dict(
        type="WandbNamer",
        keys=("model.type", "model.arch", "data.train.max_len", "seed"),
        extra="fft",
    ),
    # Load PoLAr-MAE pretrained weights (from original library's Lightning ckpt)
    # Maps: student.tokenizer.{grouping,embedding}.* → {grouping,embedding}.*
    #        student.pos_embedding.* → pos_embed.*
    #        student.encoder.{blocks,norm}.* → encoder.{blocks,norm}.*
    dict(
        type="WeightDecayExclusion",
        exclude_bias_from_wd=True,
        exclude_norm_from_wd=True,
        exclude_gamma_from_wd=True,
        exclude_token_from_wd=True,
        exclude_ndim_1_from_wd=True,
    ),
    dict(type="CheckpointLoader", keywords="student.tokenizer.", replacement=""),
    dict(type="CheckpointLoader", keywords="student.pos_embedding.", replacement="pos_embed."),
    dict(type="CheckpointLoader", keywords="student.encoder.", replacement="encoder."),
    dict(type="GradientNormLogger", log_frequency=10),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", every_n_steps=1000, write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None, evaluator_every_n_steps=1000),
    dict(type="FinalEvaluator", test_last=False),
]
