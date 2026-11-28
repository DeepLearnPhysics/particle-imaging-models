_base_ = ["../_base_/default_runtime.py"]

# Labeled training events: 100, 1000, 10000, 100000 or all 416993.
num_train_events = 1000
assert num_train_events in (100, 1000, 10000, 100000, 416993)
check = {100: 40, 1000: 20, 10000: 54, 100000: 97, 416993: 1000}[num_train_events]

data_root = "data/next-aiss"
sources = [
    dict(name="Bi", label=0, pattern="Bi_part_*"),
    dict(name="0nubb", label=1, pattern="0nubb_part_*"),
]
voxelize = dict(type="KekicVoxelize", voxel=(3.0, 3.0, 3.0), box=(900.0, 900.0, 900.0))
collect = [
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("grid_coord", "event_label"),
        offset_keys_dict=dict(offset="grid_coord"),
        feat_keys=("energy",),
    ),
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
    index_cache_dir="exp/cache/next-aiss",
)
data = dict(
    num_classes=2,
    names=["Bkg", "Signal"],
    train=dict(
        **dataset,
        split="train",
        subset=(0, 0, 1000000),
        **dict(max_per_source=(0, num_train_events // 2)) if num_train_events <= 10000 else {},
        **dict(max_len=num_train_events) if num_train_events == 100000 else {},
        transform=[dict(**voxelize, augment=True), *collect]
    ),
    val=dict(**dataset, split="val", transform=[voxelize, *collect]),
)
model = dict(
    type="SparseEventClassifier",
    backbone=dict(
        type="KekicSparseResNet",
        n_initial_filters=15,
        mom=0.7,
        stem_kernel=(3, 3, 3),
        down_kernel=(5, 5, 5),
        down_stride=(2, 2, 2),
        spatial_size=(301, 301, 301),
        global_pool=True,
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            weight=(
                [0.5, 0.5]
                if num_train_events <= 10000
                else (
                    [0.90657, 0.09343]
                    if num_train_events == 100000
                    else [377839 / 416993, 1 - 377839 / 416993]
                )
            ),
        ),
    ],
)
batch_size = min(1024, num_train_events)
batch_size_val = 256
enable_amp = True
amp_dtype = "float16"
num_worker = 24
epoch = eval_epoch = {100: 2000, 1000: 1000, 10000: 300, 100000: 50, 416993: 246}[num_train_events]
optimizer = dict(type="Adam", fused=True, lr=0.0001, weight_decay=0.0001)
scheduler = dict(type="MultiStepLR", milestones=[], gamma=1.0)
sync_bn = True
seed = 0
save_path = f"exp/next-aiss/cls-sparse-resnet-0-base/k{num_train_events}"
test = None
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="EventClassificationEvaluator", every_n_steps=check, metric="auroc"),
    dict(
        type="CheckpointSaver",
        save_freq=min(check, 500),
        evaluator_every_n_steps=check,
        save_iter_checkpoints=False,
    ),
]
