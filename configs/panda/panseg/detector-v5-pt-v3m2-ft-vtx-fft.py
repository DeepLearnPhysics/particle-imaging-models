_base_ = ["../../_base_/default_runtime.py"]

# misc custom setting
batch_size = 48  # bs: total bs in all gpus
num_worker = 24
val_batch_size = 1
mix_prob = 0.0
clip_grad = 1.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
matmul_precision = "high"
seed = 0
evaluate = True
find_unused_parameters = False

# Weights & Biases specific settings
use_wandb = True  # Enable Weights & Biases logging
wandb_project = "InsSeg-VTX-Sonata-PILArNet-M"  # Change to your desired project name

epoch = 20
eval_epoch = 20

# detector-v5 model settings
model = dict(
    type="detector-v5",
    label_configs=dict(
        interaction=dict(
            num_queries=12,
            num_classes=2,  # stuff, thing
            instance_key="instance_interaction",
            segment_key="segment_interaction",
            stuff_classes=[0],
            use_stuff_head=True,
            loss_weight=1.0,
            overlap=False,
            query_heads=[],
            criterion=dict(
                type="FastUnifiedInstanceLoss",
                cost_mask=1.0,
                cost_dice=1.0,
                cost_class=1.0,
                loss_weight_focal=2.0,
                loss_weight_dice=5.0,
                cls_weight_matched=2.0,
                cls_weight_noobj=0.5,
                focal_alpha=0.25,
                focal_gamma=2.0,
                aux_loss_weight=1.0,
                num_points=100_000,
            ),
        ),
    ),
    supervise_attn_mask=True,
    backbone=dict(
        type="PT-v3m2",
        in_channels=4,  # [xyz, energy]
        order=("hilbert", "hilbert-trans", "z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 9, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        mlp_ratio=4,
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
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=True,  # encoder only
        freeze_encoder=False,
    ),
    full_in_channels=1232,
    hidden_channels=256,
    num_heads=16,
    depth=3,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.0,
    layer_scale=None,
    pre_norm=True,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
    pos_emb=True,
    postprocess=dict(
        stuff_threshold=0.5,
        mask_threshold=0.5,
        conf_threshold=0.5,
        nms_kernel="gaussian",
        nms_sigma=2.0,
        nms_pre=-1,
        nms_max=-1,
        min_points=2,
        fill_uncovered=False,
    ),
)

# scheduler settings
lr_decay = 0.97
base_lr = 2e-4
base_wd = 0.01
backbone_mult = 1.0

# encoder/decoder depths
enc_depths = model["backbone"]["enc_depths"]
dec_depth = model["depth"]

param_dicts = []

# encoder: smallest LR at first encoder block
for e in range(len(enc_depths)):
    for b in range(enc_depths[e]):
        exp = (sum(enc_depths) - sum(enc_depths[:e]) - b - 1) + dec_depth
        param_dicts.append(
            dict(
                keyword=f"enc{e}.block{b}.",
                lr=base_lr * (lr_decay**exp) * backbone_mult,
            )
        )

# decoder: highest LR at last decoder block
for b in range(dec_depth):
    exp = dec_depth - b - 1
    param_dicts.append(
        dict(
            keyword=f"decoder.blocks.{b}.",
            lr=base_lr * (lr_decay ** exp),
        )
    )

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
grid_size = 0.001  # ~ 0.001/(1 / (768.0 * 3**0.5 / 2))
transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="LogTransform", min_val=1.0e-2, max_val=20.0),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment_interaction", "instance_interaction"),
        feat_keys=("coord", "energy"),
    ),
]
test_transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="LogTransform", min_val=1.0e-2, max_val=20.0),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment_interaction", "instance_interaction"),
        feat_keys=("coord", "energy"),
    ),
]


data = dict(
    num_classes=2,
    ignore_index=-1,
    names=["stuff", "thing"],
    train=dict(
        type="PILArNetH5Dataset",
        revision="v3",
        split="train",
        # data_root="/path/to/pilarnet-m/",
        transform=transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1_000_000,  # override via --options data.train.max_len=X
        remove_low_energy_scatters=False,
    ),
    val=dict(
        type="PILArNetH5Dataset",
        revision="v3",
        split="val",
        # data_root="/path/to/pilarnet-m/",
        transform=test_transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1000,
        remove_low_energy_scatters=False,
    ),
    test=dict(
        type="PILArNetH5Dataset",
        revision="v3",
        split="test",
        # data_root="/path/to/pilarnet-m/",
        transform=test_transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1000,
        remove_low_energy_scatters=False,
    ),
)


# hook
hooks = [
    # auto-generate wandb run name from config values
    dict(
        type="WandbNamer",
        keys=("model.type", "data.train.max_len", "amp_dtype", "seed"),
        extra="fft",
    ),
    dict(
        type="WeightDecayExclusion",
        exclude_bias_from_wd=True,
        exclude_norm_from_wd=True,
        exclude_gamma_from_wd=True,
        exclude_token_from_wd=True,
        exclude_ndim_1_from_wd=True,
    ),
    dict(
        type="CheckpointLoader",
        replacements={
            "module.student.backbone": "module.backbone",
            # Portable Panda-Base exports use bare backbone keys.
            "embedding": "backbone.embedding",
            "enc": "backbone.enc",
        },
    ),
    dict(
        type="ParameterCounter",
        show_details=False,
        show_gradients=False,
        sort_by_params=False,
        min_params=1,
    ),
    dict(type="GradientNormLogger", log_frequency=10, log_per_layer=True),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="InstanceSegmentationEvaluator",
        every_n_steps=1000,
        stuff_threshold=0.5,
        mask_threshold=0.5,
        stuff_classes=[0],
        iou_thresh=0.5,
        class_names=["thing"],
        require_class_for_match=False,
        labels=("interaction",),
    ),
    dict(type="CheckpointSaver", save_freq=None, evaluator_every_n_steps=1000),
    dict(
        type="AttentionMaskAnnealingHook",
        log_frequency=100,
        log_per_layer=False,
        prefix="anneal",
    ),
    dict(type="FinalEvaluator", test_last=True),
]


test = dict(
    type="InstanceSegTester",
    class_names=["thing"],
    stuff_classes=[0],
    require_class_for_match=False,
    labels=("interaction",),
)
