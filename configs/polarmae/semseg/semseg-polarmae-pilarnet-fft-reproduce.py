"""
PoLAr-MAE Semantic Segmentation - Reproduce fine-tuned checkpoint results.

Usage (evaluate only):
    sh scripts/train.sh -g 1 -d polarmae/semseg -c semseg-polarmae-pilarnet-fft-reproduce \
        -n reproduce_fft -w polarmae_fft_segsem.ckpt \
        -- --options epoch=1 evaluate=True

Fine-tuned checkpoint:
    wget https://github.com/DeepLearnPhysics/PoLAr-MAE/releases/download/weights/polarmae_fft_segsem.ckpt
"""

_base_ = ["../../_base_/default_runtime.py"]

batch_size = 32
num_worker = 16
mix_prob = 0.0
clip_grad = None
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
matmul_precision = "high"
seed = 0
evaluate = True

use_wandb = False

model = dict(
    type="PoLArMAE-SemSeg",
    num_classes=4,  # shower, track, Michel, delta
    arch="vit_small",
    voxel_size=5.0,
    num_channels=4,
    seg_head_fetch_layers=[3, 7, 11],
    seg_head_combination_method="mean",
    seg_head_dim=512,
    seg_head_dropout=0.5,
    freeze_encoder=False,
    apply_encoder_postnorm=False,
    condition_global_features=True,
    upsampling_k=5,
    center=[384.0, 384.0, 384.0],
    scale=1.0 / (768 * (3**0.5) / 2),
    transformer_kwargs=dict(
        postnorm=False,
        add_pos_at_every_layer=True,
        drop_rate=0.0,
        attn_drop_rate=0.05,
        drop_path_rate=0.25,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
    ],
)

epoch = 1
eval_epoch = 1
base_lr = 1e-4
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=base_lr,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# Transforms: no rotation for evaluation
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
        revision="v3_extra",
        split="train",
        transform=test_transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=500,
        remove_low_energy_scatters=True,
    ),
    val=dict(
        type="PILArNetH5Dataset",
        revision="v3_extra",
        split="val",
        transform=test_transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=10000,
        remove_low_energy_scatters=True,
    ),
    test=dict(
        type="PILArNetH5Dataset",
        revision="v3_extra",
        split="test",
        transform=test_transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=10000,
        remove_low_energy_scatters=True,
    ),
)

hooks = [
    # Load fine-tuned checkpoint (encoder.tokenizer.* → embedding.*, etc.)
    dict(type="CheckpointLoader", keywords="encoder.tokenizer.", replacement=""),
    dict(type="CheckpointLoader", keywords="encoder.pos_embed.", replacement="pos_embed."),
    dict(type="CheckpointLoader", keywords="encoder.transformer.", replacement="encoder."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", every_n_steps=0, write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
]
