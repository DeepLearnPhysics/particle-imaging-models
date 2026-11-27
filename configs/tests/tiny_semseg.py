_base_ = ["../_base_/default_runtime.py"]

seed = 0
deterministic = False
use_wandb = False
evaluate = True
epoch = 1
eval_epoch = 1
batch_size = 4
batch_size_val = 1
batch_size_test = 1
num_worker = 0
mix_prob = 0.0
enable_amp = False
matmul_precision = "high"
checkpoint_format = "standard"

model = dict(
    type="DefaultSegmentorV2",
    num_classes=5,
    backbone_out_channels=8,
    backbone=dict(
        type="PT-v3m2",
        in_channels=4,
        order=("z",),
        stride=(2,),
        enc_depths=(1, 1),
        enc_channels=(8, 16),
        enc_num_head=(1, 2),
        enc_patch_size=(32, 32),
        dec_depths=(1,),
        dec_channels=(8,),
        dec_num_head=(1,),
        dec_patch_size=(32,),
        mlp_ratio=2,
        drop_path=0.0,
        shuffle_orders=False,
        enable_rpe=False,
        enable_flash=False,
        enable_cpe=True,
        upcast_attention=True,
        upcast_softmax=True,
        enc_mode=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
    ],
)

optimizer = dict(type="AdamW", lr=1.0e-3, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=1.0e-3,
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

transform = [
    dict(
        type="NormalizeCoord",
        center=[384.0, 384.0, 384.0],
        scale=768.0 * 3**0.5 / 2,
    ),
    dict(type="LogTransform", min_val=0.01, max_val=20.0),
    dict(
        type="GridSample",
        grid_size=0.001,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="Copy", keys_dict={"segment_motif": "segment"}),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment"),
        feat_keys=("coord", "energy"),
    ),
]

eval_transform = [
    transform[0],
    transform[1],
    dict(type="Copy", keys_dict={"segment_motif": "origin_segment"}),
    dict(transform[2], mode="deterministic", return_inverse=True),
    transform[3],
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
    ignore_index=-1,
    names=["shower", "track", "michel", "delta", "led"],
    train=dict(
        type="PILArNetH5Dataset",
        data_root=None,
        revision="v3",
        split="train",
        transform=transform,
        min_points=0,
        max_len=80,
    ),
    val=dict(
        type="PILArNetH5Dataset",
        data_root=None,
        revision="v3",
        split="val",
        transform=eval_transform,
        min_points=0,
        max_len=20,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=0),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester", verbose=False)
