_base_ = ["_base.py"]

weight = "weights/sphenix.pth"
encoder = dict(grid_size=0.003)
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
linear_heads = dict(
    linear_pid=dict(pid_head, mlp=False), linear_noise=dict(noise_head, mlp=False)
)
model = dict(
    model=dict(
        num_classes=5,
        backbone=dict(grid_size=0.003),
        heads=dict(mlp_pid=pid_head, mlp_noise=noise_head),
    ),
)
hooks_override = dict(
    PointClassificationEvaluator=dict(
        heads=dict(mlp_pid="segment_pid", mlp_noise="segment_noise"),
        names=dict(segment_pid=pid_names, segment_noise=noise_names),
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
        max_len={{_base_.K}},
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
