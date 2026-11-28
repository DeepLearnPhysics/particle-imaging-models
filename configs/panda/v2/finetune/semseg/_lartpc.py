_base_ = ["_base.py"]

weight = "weights/lartpc.pth"
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
linear_heads = dict(linear=dict(head, mlp=False))
model = dict(model=dict(num_classes=8, heads=dict(mlp=head)))
hooks_override = dict(
    PointClassificationEvaluator=dict(
        heads=dict(mlp="segment"), names=dict(segment=joint_names)
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
    data_root="data/lartpc",
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
        max_len={{_base_.K}},
        transform=transform,
        overlay_n_events=2,
        overlay_prob=0.5,
        overlay_rotate_90=False,
        overlay_dedup=False,
    ),
    val=dict(dataset, split="val", max_len=1000, transform=eval_transform),
    test=dict(dataset, split="test", max_len=-1, transform=eval_transform),
)
