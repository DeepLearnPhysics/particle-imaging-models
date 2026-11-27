"""
Self-contained PoLAr-MAE pre-training on PILArNet.

Unlike the wrapper config, coordinate normalisation and rotation are applied
in the dataset transform pipeline (NormalizeCoord + RandomRotate) rather than
inside the model.
"""

_base_ = ["../_base_/default_runtime.py"]

# --------------------------------------------------------------------------
# Training settings
# --------------------------------------------------------------------------
batch_size = 64
num_worker = 12
batch_size_val = 32
enable_amp = False
amp_dtype = "bfloat16"
evaluate = True
clip_grad = 3.0
find_unused_parameters = False
seed = 0
num_events = 100_000

# Weights & Biases
use_wandb = True
wandb_project = "Pretraining-PoLArMAE-PILArNet"

warmup_ratio = 0.05

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
_shared_transformer_kwargs = dict(
    postnorm=True,
    add_pos_at_every_layer=True,
    drop_rate=0.0,
    attn_drop_rate=0.05,
    drop_path_rate=0.25,
)

model = dict(
    type="PoLAr-MAE",
    arch="vit_small",
    num_channels=4,
    voxel_size=5,
    masking_ratio=0.6,
    masking_type="rand",
    transformer_kwargs=_shared_transformer_kwargs,
    decoder_arch="vit_small",
    decoder_kwargs=dict(
        transformer_kwargs=dict(depth=4, **_shared_transformer_kwargs),
    ),
    mae_prediction="full",
    loss_weights=dict(chamfer=1.0, energy=1.0),
)

# --------------------------------------------------------------------------
# Optimizer & scheduler
# --------------------------------------------------------------------------
epoch = 100
optimizer = dict(type="AdamW", lr=0.0004, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.0004,
    pct_start=warmup_ratio,
    anneal_strategy="cos",
    div_factor=11.6,
    final_div_factor=11.6,
)

# --------------------------------------------------------------------------
# Dataset
#
# Transforms: NormalizeCoord centres + scales coords to unit-sphere, then
# RandomRotate applies random rotation around each axis (training only).
# --------------------------------------------------------------------------
_norm = dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=665.1076)  # 768*sqrt(3)/2
_log = dict(type="LogTransform", min_val=0.01, max_val=20.0, log=True, keys=("energy",))
_rot_x = dict(type="RandomRotate", angle=[-1, 1], axis="x", always_apply=True, center=[0, 0, 0])
_rot_y = dict(type="RandomRotate", angle=[-1, 1], axis="y", always_apply=True, center=[0, 0, 0])
_rot_z = dict(type="RandomRotate", angle=[-1, 1], axis="z", always_apply=True, center=[0, 0, 0])

data = dict(
    num_classes=5,
    names=["shower", "track", "michel", "delta", "led"],
    train=dict(
        type="PILArNetH5Dataset",
        split="train",
        transform=[
            _log, _norm, _rot_x, _rot_y, _rot_z,
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "energy"), feat_keys=("coord", "energy")),
        ],
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=num_events,
        remove_low_energy_scatters=True,
        loop=1,
    ),
    val=dict(
        type="PILArNetH5Dataset",
        split="val",
        transform=[
            _log, _norm,
            dict(type="Copy", keys_dict={"segment_motif": "segment"}),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "energy", "segment"), feat_keys=("coord", "energy")),
        ],
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=5000,
        remove_low_energy_scatters=True,
        loop=1,
    ),
    test=dict(
        type="PILArNetH5Dataset",
        split="val",
        transform=[
            _log, _norm,
            dict(type="Copy", keys_dict={"segment_motif": "segment"}),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "energy", "segment"), feat_keys=("coord", "energy")),
        ],
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1000,
        remove_low_energy_scatters=True,
        loop=1,
    ),
)

# --------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------
hooks = [
    dict(type="WandbNamer", keys=("model.type", "data.train.max_len")),
    dict(
        type="ParameterCounter",
        show_details=True, show_gradients=False,
        sort_by_params=True, min_params=1,
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=1000, save_iter_checkpoints=False),
    dict(
        type="WeightDecayExclusion",
        exclude_bias_from_wd=True, exclude_norm_from_wd=True,
        exclude_gamma_from_wd=True, exclude_token_from_wd=True,
        exclude_ndim_1_from_wd=True,
    ),
    dict(type="CheckpointLoader"),
]
