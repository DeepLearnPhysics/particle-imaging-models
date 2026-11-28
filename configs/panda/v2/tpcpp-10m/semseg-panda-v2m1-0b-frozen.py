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
seed = {100: 46990708, 1000: 6272337, 10000: 7618066, 70000: 21526894}.get(
    num_train_events, 6272337
)
clip_grad = None
enable_amp = True
amp_dtype = "float16"
matmul_precision = None
mix_prob = 0.0
use_wandb = False
# Task evaluators are configured through hooks.
test = None
evaluate = True
optimizer = dict(type="AdamW", fused=True, lr=0.0003, weight_decay=0.01)
param_dicts = [dict(keyword="lora_", lr=0.0001)]
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0003, 0.0001],
    pct_start=625,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
save_path = f"exp/tpcpp-10m/semseg-panda-v2m1-0b-frozen/k{num_train_events}"

weight = "weights/sphenix.pth"
encoder = dict(
    type="panda-encoder-v2m1",
    in_channels=4,
    drop_path=0.0,
    grid_size=0.003,
    depths=(3, 6, 12, 6),
    conv_stages=2,
)
pid_names = ["other", "pion", "kaon", "proton", "electron"]
noise_names = ["signal", "noise"]
pid_counts = [275066, 1483209, 61971, 159714, 774345]
lovasz = dict(
    type="LovaszLoss",
    mode="multiclass",
    loss_weight=0.05,
    ignore_index=-1,
    class_weights=None,
)
pid_head = dict(
    label="segment_pid",
    num_classes=5,
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=-1,
            weight=[sum(pid_counts) / count for count in pid_counts],
        ),
        lovasz,
    ],
)
noise_head = dict(
    label="segment_noise",
    num_classes=2,
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1), lovasz],
)
model = dict(
    type="LoRAAdapter",
    rank=16,
    alpha=32,
    dropout=0.0,
    target_modules=[],
    freeze_batchnorm=True,
    trainable_keywords=("seg_head",),
    model=dict(
        type="DefaultSegmentorV2",
        backbone=encoder,
        backbone_out_channels=1170,
        mlp_head=True,
        mlp_hidden_channels=256,
        freeze_backbone=True,
        num_classes=5,
        heads=dict(
            mlp_pid=pid_head,
            mlp_noise=noise_head,
            linear_pid=dict(pid_head, mlp=False),
            linear_noise=dict(noise_head, mlp=False),
        ),
    ),
)

pid = dict(
    type="JointLabel",
    keys=("segment_pid",),
    combinations=tuple(((c,) for c in range(5))),
    out_key="segment_pid",
    ignore_index=-1,
)
transform = [
    dict(type="NormalizeCoord", center=[0.0, 0.0, 0.0], scale=180.0),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5, axes=("x", "y", "z")),
    dict(
        type="GridSample",
        grid_size=0.003,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        sum_keys=("energy",),
    ),
    dict(type="LogTransform", min_val=20.0, max_val=2000.0, log=True, keys=("energy",)),
    dict(type="BinaryLabel", key="segment_motif", values=(4,), out_key="segment_noise"),
    pid,
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment_pid", "segment_noise"),
        feat_keys=("coord", "energy"),
    ),
]
eval_transform = [
    transform[0],
    dict(
        type="Copy",
        keys_dict=dict(
            segment_pid="origin_segment_pid", segment_motif="origin_segment_motif"
        ),
    ),
    dict(transform[5], mode="deterministic", return_inverse=True),
    transform[6],
    dict(
        type="BinaryLabel",
        key="segment_motif",
        values=(4,),
        out_key="segment_noise",
    ),
    dict(
        type="BinaryLabel",
        key="origin_segment_motif",
        values=(4,),
        out_key="origin_segment_noise",
    ),
    pid,
    dict(pid, keys=("origin_segment_pid",), out_key="origin_segment_pid"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=(
            "coord",
            "grid_coord",
            "segment_pid",
            "segment_noise",
            "origin_segment_pid",
            "origin_segment_noise",
            "inverse",
        ),
        feat_keys=("coord", "energy"),
        offset_keys_dict=dict(offset="coord", origin_offset="inverse"),
    ),
]
dataset = dict(
    type="PILArNetH5Dataset",
    revision="v2",
    data_root="data/sphenix",
    energy_threshold=0.0,
    min_points=0,
    remove_low_energy_scatters=False,
    test_mode=False,
    loop=1,
)
data = dict(
    num_classes=5,
    ignore_index=-1,
    names=pid_names,
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
    val=dict(
        dataset, split="val", max_len=1000, max_points=3200, transform=eval_transform
    ),
    test=dict(
        dataset, split="test", max_len=-1, max_points=3200, transform=eval_transform
    ),
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
        heads=dict(
            linear_pid="segment_pid",
            linear_noise="segment_noise",
            mlp_pid="segment_pid",
            mlp_noise="segment_noise",
        ),
        groups=dict(
            linear=["linear_pid", "linear_noise"], mlp=["mlp_pid", "mlp_noise"]
        ),
        names=dict(segment_pid=pid_names, segment_noise=noise_names),
    ),
    dict(
        type="CheckpointSaver",
        save_freq=check,
        evaluator_every_n_steps=check,
        save_iter_checkpoints=False,
    ),
]
