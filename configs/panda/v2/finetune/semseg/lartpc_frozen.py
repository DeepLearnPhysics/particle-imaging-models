_base_ = ["lartpc.py"]
from os import environ as __env

seed = {100: 5917844, 1000: 21487196, 10000: 35706528}.get(
    int(__env.get("K", "1000")), 21487196
)
joint_names = ["shower", "track", "michel", "delta", "led"]
head = dict(
    label="segment",
    num_classes=5,
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
model = dict(
    target_modules=[],
    model=dict(
        num_classes=5, heads=dict(_delete_=True, mlp=head, linear=dict(head, mlp=False))
    ),
)
clip_grad = None
optimizer = dict(lr=0.0003)
scheduler = dict(max_lr=[0.0003, 0.0001])
amp_dtype = "float16"
hooks_override = dict(
    PointClassificationEvaluator=dict(
        heads=dict(linear="segment", mlp="segment"),
        names=dict(segment=joint_names),
        groups=dict(linear=["linear"], mlp=["mlp"]),
    ),
)

motif = dict(
    type="JointLabel",
    keys=("segment_motif",),
    combinations=((0,), (1,), (2,), (3,), (4,)),
    ignore_index=-1,
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
    motif,
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment"),
        feat_keys=("coord", "energy"),
    ),
]
eval_transform = [
    transform[0],
    motif,
    dict(type="Copy", keys_dict=dict(segment="origin_segment")),
    dict(transform[5], mode="deterministic", return_inverse=True),
    transform[6],
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
        feat_keys=("coord", "energy"),
        offset_keys_dict=dict(offset="coord", origin_offset="inverse"),
    ),
]
data = dict(
    num_classes=5,
    names=joint_names,
    train=dict(transform=transform),
    val=dict(transform=eval_transform),
    test=dict(transform=eval_transform),
)
save_path = "exp/finetune/semseg/lartpc_frozen" + "/k" + __env.get("K", "1000")
