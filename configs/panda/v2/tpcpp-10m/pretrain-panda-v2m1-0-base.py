_base_ = ["../_base_/default_runtime.py"]

seed = 0
batch_size = 96
batch_size_val = None  # one validation event per GPU
num_worker = 32  # Total across four ranks: eight workers per rank.
gradient_accumulation_steps = 1
epoch = eval_epoch = 20
schedule_total_steps = 208334  # 1M events, 20 epochs, batch 96
# A smaller pool of N events keeps this schedule, as in the paper: data.train.max_len=N data.train.loop=1000000//N.
# H5 loaders run complete epochs; an exact step-limited stream is unavailable.
enable_amp = True
amp_dtype = "bfloat16"
matmul_precision = "high"
# Four ranks with 24 images each reproduce the paper's local statistical groups.
sync_bn = False
use_wandb = False
checkpoint_format = "standard"
resume_strict_state = True

model = dict(
    type="panda-ssl-v2m1",
    backbone=dict(
        type="panda-encoder-v2m1",
        in_channels=4,
        grid_size=0.003,
        drop_path=0.3,
    ),
    teacher_custom=dict(drop_path=0.0),
    up_cast_level=3,  # losses at the input points
    head_in_channels=1170,
    observable=True,
    mask_size=0.04,
    mask_ratio=0.6,
    match_max_r=0.006,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=8192,
    num_global_view=2,
    num_local_view=6,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.1,
    student_temp=0.1,
    mask_loss_weight=0.25,
    roll_mask_loss_weight=0.25,
    unmask_loss_weight=0.5,
    momentum_base=0.996,
    momentum_final=1.0,
    ema_power=4,
    split_backward=True,
    fused_sinkhorn=True,
    sync_sinkhorn=False,
    sinkhorn_iters=3,
)

clip_grad = 2.4
param_dicts = [dict(keyword="block", lr=0.000594)]
optimizer = dict(
    type="AdamW", fused=True, lr=0.00594, betas=(0.9, 0.996), weight_decay=0.05
)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.00594, 0.000594],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
)

# The model's detector settings, which SonataPreparation must share.
detector_model = dict(
    backbone=dict(in_channels=4, grid_size=0.003),
    up_cast_level=3,  # losses at the input points
    head_in_channels=1170,
    observable=True,
    mask_size=0.04,
    mask_ratio=0.6,
    match_max_r=0.006,
)

data = dict(
    num_classes=5,
    names=["other", "pion", "kaon", "proton", "electron"],
    ignore_index=-1,
    train=dict(
        type="PILArNetH5Dataset",
        revision="v2",
        split="train",
        data_root="data/tpcpp-10m-unlabeled",
        min_points=1024,
        energy_threshold=0.0,
        remove_low_energy_scatters=False,
        max_len=1000000,
        loop=1,
        transform=[
            dict(type="NormalizeCoord", center=[0.0, 0.0, 0.0], scale=180.0),
            dict(type="RandomScale", scale=[0.9, 1.2]),
            dict(
                type="GridSample",
                grid_size=0.003,
                hash_type="fnv",
                mode="train",
                sum_keys=["energy"],
            ),
            dict(type="Copy", keys_dict=dict(coord="origin_coord")),
            dict(
                type="Update",
                keys_dict=dict(index_valid_keys=["coord", "origin_coord", "energy"]),
            ),
            dict(
                type="MultiViewGenerator",
                view_keys=["coord", "origin_coord", "energy"],
                global_view_num=2,
                global_view_scale=[0.85, 1.0],
                local_view_num=6,
                local_view_scale=[0.3, 0.6],
                max_size=30000,
                global_shared_transform=[
                    dict(
                        type="MultiplicativeRandomJitter",
                        sigma=0.05,
                        clip=0.1,
                        keys=["energy"],
                        p=0.8,
                    ),
                ],
                global_transform=[
                    dict(type="CenterShift", axes=["x", "y", "z"]),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.8,
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="x",
                        center=[0, 0, 0],
                        p=0.8,
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="y",
                        center=[0, 0, 0],
                        p=0.8,
                    ),
                    dict(type="RandomFlip", p=0.5, axes=["x", "y", "z"]),
                    dict(
                        type="LogTransform",
                        min_val=20.0,
                        max_val=2000.0,
                        log=True,
                        keys=["energy"],
                    ),
                    dict(
                        type="RandomJitter", sigma=0.000375, clip=0.003, keys=["coord"]
                    ),
                ],
                local_transform=[
                    dict(type="CenterShift", axes=["x", "y", "z"]),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.8,
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="x",
                        center=[0, 0, 0],
                        p=0.8,
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="y",
                        center=[0, 0, 0],
                        p=0.8,
                    ),
                    dict(type="RandomFlip", p=0.5, axes=["x", "y", "z"]),
                    dict(
                        type="LogTransform",
                        min_val=20.0,
                        max_val=2000.0,
                        log=True,
                        keys=["energy"],
                    ),
                    dict(
                        type="RandomJitter", sigma=0.00075, clip=0.003, keys=["coord"]
                    ),
                ],
            ),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict=dict(grid_size=0.003)),
            dict(
                type="Collect",
                keys=[
                    "global_origin_coord",
                    "global_coord",
                    "global_energy",
                    "global_offset",
                    "local_origin_coord",
                    "local_coord",
                    "local_energy",
                    "local_offset",
                    "grid_size",
                    "name",
                ],
                global_feat_keys=["global_coord", "global_energy"],
                local_feat_keys=["local_coord", "local_energy"],
            ),
        ],
        subset_seed=0,
    ),
    val=dict(
        type="PILArNetH5Dataset",
        revision="v2",
        split="val",
        data_root="data/tpcpp-10m",
        min_points=1024,
        energy_threshold=0.0,
        remove_low_energy_scatters=False,
        max_len=10000,
        loop=1,
        test_mode=False,
        transform=[
            dict(type="NormalizeCoord", center=[0.0, 0.0, 0.0], scale=180.0),
            dict(
                type="GridSample",
                grid_size=0.003,
                hash_type="fnv",
                mode="deterministic",
                return_grid_coord=True,
                return_inverse=True,
                sum_keys=["energy"],
            ),
            dict(
                type="LogTransform",
                min_val=20.0,
                max_val=2000.0,
                log=True,
                keys=["energy"],
            ),
            dict(
                type="BinaryLabel",
                key="segment_motif",
                values=[4],
                out_key="segment_noise",
            ),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict=dict(grid_size=0.003)),
            dict(
                type="Collect",
                keys=[
                    "coord",
                    "grid_coord",
                    "energy",
                    "inverse",
                    "segment_pid",
                    "segment_noise",
                ],
                feat_keys=["coord", "energy"],
            ),
        ],
    ),
)

data["train"]["transform"].append(dict(type="SonataPreparation", model=detector_model))

hooks = [
    dict(type="WeightDecayExclusion"),
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="PretrainEvaluator",
        force_fp32_features=True,
        label="segment_pid",
        every_n_steps=5208,
        class_names=["other", "pion", "kaon", "proton", "electron"],
        train_config=dict(
            held_out_selection=True, criteria=[dict(type="CrossEntropyLoss")]
        ),
    ),
    dict(
        type="CheckpointSaver",
        save_freq=10417,
        evaluator_every_n_steps=5208,
        save_iter_checkpoints=True,
    ),
]

save_path = "exp/tpcpp-10m/pretrain-panda-v2m1-0-base"
# Set data.train.max_len to the size of the pretraining pool.
