_base_ = ["../_base_/default_runtime.py"]

# Labeled training events: 100, 1000, 10000, 100000 or the full split (at least 16).
num_train_events = 1000
assert num_train_events >= 16

schedule_total_steps = 625000
epoch = eval_epoch = {100: 2000, 1000: 1000, 10000: 300}.get(
    num_train_events, schedule_total_steps // (num_train_events // 16)
)
check = {100: 625, 1000: 1250, 10000: 3750}.get(
    num_train_events, 6250 if num_train_events <= 100000 else 12500
)
batch_size = 16
batch_size_val = batch_size_test = 8
num_worker = 12
seed = 0
clip_grad = 1.0
enable_amp = True
amp_dtype = "bfloat16"
matmul_precision = None
mix_prob = 0.0
use_wandb = False
# Task evaluators are configured through hooks.
test = None
evaluate = True
decoder_lr, lora_lr = (0.0001, 0.0001)
optimizer = dict(type="AdamW", fused=True, lr=decoder_lr, weight_decay=0.01)
param_dicts = [dict(keyword="lora_", lr=lora_lr)]
scheduler = dict(
    type="OneCycleLR",
    max_lr=[decoder_lr, lora_lr],
    pct_start=625,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
save_path = f"exp/pilarnet-m/semseg-panda-v2m1-0a-lora/k{num_train_events}"

weight = "weights/pilarnet-m.pth"
encoder = dict(
    type="panda-encoder-v2m1",
    in_channels=4,
    drop_path=0.0,
    grid_size=0.001,
    depths=(3, 6, 12, 6),
    conv_stages=2,
)
# The encoder's linear layers, by stage type, and its pooling projections.
lora_targets = [
    f"backbone.blocks.{s}.{b}.{m}"
    for s, depth in enumerate(encoder["depths"])
    for b in range(depth)
    for m in (
        ("linear",)
        if s < encoder["conv_stages"]
        else ("attn.qkv", "attn.proj", "mlp.0", "mlp.2")
    )
] + [f"backbone.pools.{i}.proj" for i in range(len(encoder["depths"]) - 1)]
joint_combinations = ((0, 0), (0, 1), (1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 5))
joint_names = [
    "shower+photon",
    "shower+electron",
    "track+muon",
    "track+pion",
    "track+proton",
    "michel+electron",
    "delta+electron",
    "led+none",
]
head = dict(
    label="segment",
    num_classes=8,
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=0.05,
            ignore_index=-1,
            class_weights=None,
        ),
    ],
)
model = dict(
    type="LoRAAdapter",
    rank=16,
    alpha=32,
    dropout=0.0,
    target_modules=lora_targets,
    freeze_batchnorm=True,
    trainable_keywords=("seg_head",),
    model=dict(
        type="DefaultSegmentorV2",
        backbone=encoder,
        backbone_out_channels=1170,
        mlp_head=True,
        mlp_hidden_channels=256,
        freeze_backbone=True,
        num_classes=8,
        heads=dict(mlp=head),
    ),
)

transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5, axes=("x", "y", "z")),
    dict(
        type="GridSample",
        grid_size=0.001,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        sum_keys=("energy",),
    ),
    dict(type="LogTransform", min_val=0.01, max_val=20.0, keys=("energy",)),
    dict(
        type="JointLabel",
        keys=("segment_motif", "segment_pid"),
        combinations=joint_combinations,
        ignore_index=-1,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment"),
        feat_keys=("coord", "energy"),
    ),
]
eval_transform = [
    transform[0],
    transform[7],
    dict(type="Copy", keys_dict=dict(segment="origin_segment", segment_motif="origin_segment_motif")),
    dict(transform[5], mode="deterministic", return_inverse=True),
    transform[6],
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "origin_segment", "origin_segment_motif", "inverse"),
        feat_keys=("coord", "energy"),
        offset_keys_dict=dict(offset="coord", origin_offset="inverse"),
    ),
]
dataset = dict(
    type="PILArNetH5Dataset",
    revision="v3_extra",
    data_root="data/pilarnet-m",
    energy_threshold=0.13,
    min_points=0,
    remove_low_energy_scatters=False,
    test_mode=False,
    loop=1,
)
data = dict(
    num_classes=8,
    ignore_index=-1,
    names=joint_names,
    train=dict(
        dataset,
        split="train",
        max_len=num_train_events,
        transform=transform,
        overlay_n_events=2,
        overlay_prob=0.5,
        overlay_rotate_90=False,
        overlay_dedup=False,
    ),
    val=dict(dataset, split="val", max_len=1000, transform=eval_transform),
    test=dict(dataset, split="test", max_len=-1, transform=eval_transform),
)

hooks = [
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
            "stem": "model.backbone.stem",
            "pools.": "model.backbone.pools.",
            "blocks.": "model.backbone.blocks.",
        },
        strict=False,
    ),
    dict(type="ParameterCounter", show_details=True, show_gradients=True),
    dict(type="GradientNormLogger", log_frequency=100),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="PointClassificationEvaluator",
        every_n_steps=check,
        heads=dict(mlp="segment"),
        names=dict(segment=joint_names),
    ),
    dict(
        type="CheckpointSaver",
        save_freq=check,
        evaluator_every_n_steps=check,
        save_iter_checkpoints=False,
    ),
]
