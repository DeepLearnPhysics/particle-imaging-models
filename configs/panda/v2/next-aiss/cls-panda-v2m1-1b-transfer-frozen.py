_base_ = ["../_base_/default_runtime.py"]

# Labeled training events: 100, 1000, 10000, 100000 or all 416993.
num_train_events = 1000
assert num_train_events in (100, 1000, 10000, 100000, 416993)

# Pretrained encoder: lartpc, sphenix, wc or next.
source = "next"
assert source in ("lartpc", "sphenix", "wc", "next")

schedule_total_steps = 625000
epoch = eval_epoch = {100: 2000, 1000: 1000, 10000: 300}.get(
    num_train_events, schedule_total_steps // (num_train_events // 16)
)
check = {100: 250, 1000: 1240, 10000: 3750, 100000: 6250, 416993: 13031}[
    num_train_events
]
checkpoint_every = min(check, 500)
batch_size = 16
batch_size_val = 16
batch_size_test = None
num_worker = 8
seed = 0
clip_grad = 1.0
enable_amp = True
amp_dtype = "bfloat16"
matmul_precision = None
mix_prob = 0.0
use_wandb = False
# Task evaluators are configured through hooks.
test = None
optimizer = dict(type="AdamW", fused=True, lr=0.0003, weight_decay=0.01)
param_dicts = [dict(keyword="backbone.stem", lr=0.0001)]
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0003, 0.0001],
    pct_start=625,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
save_path = f"exp/next-aiss/cls-panda-v2m1-1b-transfer-frozen/from-{source}/k{num_train_events}"

weight = f"weights/{source}.pth"
encoder = dict(
    type="panda-encoder-v2m1",
    in_channels=4,
    drop_path=0.0,
    grid_size=0.0008617168196860086,
    depths=(3, 6, 12, 6),
    conv_stages=2,
)
model = dict(
    type="LoRAAdapter",
    rank=16,
    alpha=32,
    dropout=0.0,
    target_modules=[],
    trainable_keywords=("cls_head", "backbone.stem"),
    freeze_batchnorm=False,
    model=dict(
        type="EventClassifier",
        num_classes=2,
        backbone_out_channels=1170,
        backbone=encoder,
        pool="mean",
        freeze_backbone=False,
        criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0)],
    ),
)

data_root = "data/next"
sources = [
    dict(name="Bi", label=0, pattern="Bi_part_*"),
    dict(name="0nubb", label=1, pattern="0nubb_part_*"),
]
inputs = [
    dict(type="NormalizeCoord", center=[0.0, 0.0, 1809.0], scale=3133.279910892099),
    dict(type="CenterShift", apply_z=True),
    dict(type="RescaleTotal", key="energy", total=2.458),
]
log_energy = dict(
    type="LogTransform", min_val=2.48e-05, max_val=0.025, keys=("energy",)
)
collect = [
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "event_label"),
        feat_keys=("coord", "energy"),
    ),
]
train_transform = [
    *inputs,
    *(
        dict(type="RandomRotate", angle=[-1, 1], axis=a, center=[0, 0, 0], p=1.0)
        for a in ("x", "y", "z")
    ),
    dict(type="RandomFlip", p=0.5, axes=("x", "y", "z")),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(
        type="GridSample",
        grid_size=0.0008617168196860086,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        sum_keys=("energy",),
    ),
    log_energy,
    *collect,
]
eval_transform = [
    *inputs,
    dict(
        type="GridSample",
        grid_size=0.0008617168196860086,
        hash_type="fnv",
        mode="deterministic",
        return_grid_coord=True,
        sum_keys=("energy",),
    ),
    log_energy,
    *collect,
]
dataset = dict(
    type="NEXTH5Dataset",
    data_root=data_root,
    configs=sources,
    holdout_events_per_config=5000,
    min_points=512,
    event_filter=[dict(type="NEXTTopologyCut")],
    index_workers=16,
    preload=True,
    index_cache_dir="exp/cache/next",
)
pool = 1000000  # the events the encoders were pretrained on; the test set is every train event after them
data = dict(
    num_classes=2,
    names=["Bkg", "Signal"],
    train=dict(
        **dataset,
        split="train",
        subset=(0, 0, pool),
        max_per_source=(0, num_train_events // 2) if num_train_events <= 10000 else -1,
        max_len=num_train_events if num_train_events == 100000 else -1,
        transform=train_transform
    ),
    val=dict(**dataset, split="val", transform=eval_transform),
    test=dict(**dataset, split="train", subset=(0, pool, None), transform=eval_transform),
)
data["train"]["preload"] = num_train_events != 416993

hooks = [
    dict(
        type="CheckpointLoader",
        # A stem built for other input channels stays unloaded.
        replacements={
            "stem": "model.backbone.stem",
            "pools.": "model.backbone.pools.",
            "blocks.": "model.backbone.blocks.",
            **(
                {"stem.weight": "unloaded_source_stem.weight"} if source == "wc" else {}
            ),
        },
        strict=False,
    ),
    dict(type="ParameterCounter", show_details=False, show_gradients=True),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="EventClassificationEvaluator",
        every_n_steps=check,
        metric="auroc",
    ),
    dict(
        type="CheckpointSaver",
        save_freq=checkpoint_every,
        evaluator_every_n_steps=check,
        save_iter_checkpoints=False,
    ),
]
