_base_ = ["../lora.py"]
from os import environ as __env

K = int(__env.get("K", "1000"))
assert K in (100, 1000, 10000, 100000, 416993)
weight = "weights/next.pth"
encoder = dict(in_channels=4, grid_size=0.0008617168196860086)
check = {100: 250, 1000: 1240, 10000: 3750, 100000: 6250, 416993: 13031}[K]
checkpoint_every = min(check, 500)

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
        max_per_source=(0, K // 2) if K <= 10000 else -1,
        max_len=K if K == 100000 else -1,
        transform=train_transform
    ),
    val=dict(**dataset, split="val", transform=eval_transform),
    test=dict(**dataset, split="train", subset=(0, pool, None), transform=eval_transform),
)
data["train"]["preload"] = K != 416993
batch_size_val = 16
num_worker = 8
seed = 0
