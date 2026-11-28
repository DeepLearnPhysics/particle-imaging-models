# The model's per-detector settings, which SonataPreparation is also given.
detector_model = dict(
    backbone=dict(in_channels=4, grid_size=0.001),
    up_cast_level=3,  # losses at the input points
    head_in_channels=1170,
    observable=True,
    mask_size=0.04,
    mask_ratio=0.6,
    match_max_r=0.004,
)

data = dict(
    num_classes=5,
    names=["shower", "track", "michel", "delta", "led"],
    train=dict(
        type="PILArNetH5Dataset",
        revision="v2",
        split="train",
        data_root="data/lartpc_v2",
        min_points=1024,
        energy_threshold=0.13,
        remove_low_energy_scatters=False,
        old_pid_mapping=False,
        max_len=1000000,
        loop=1,
        transform=[
            dict(
                type="NormalizeCoord",
                center=[384.0, 384.0, 384.0],
                scale=665.1075101064489,
            ),
            dict(type="RandomScale", scale=[0.9, 1.2]),
            dict(
                type="GridSample",
                grid_size=0.001,
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
                global_view_scale=[0.4, 1.0],
                local_view_num=6,
                local_view_scale=[0.1, 0.4],
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
                        min_val=0.01,
                        max_val=20.0,
                        log=True,
                        keys=["energy"],
                    ),
                    dict(
                        type="RandomJitter", sigma=0.000125, clip=0.001, keys=["coord"]
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
                        min_val=0.01,
                        max_val=20.0,
                        log=True,
                        keys=["energy"],
                    ),
                    dict(
                        type="RandomJitter", sigma=0.00025, clip=0.001, keys=["coord"]
                    ),
                ],
            ),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict=dict(grid_size=0.001)),
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
        data_root="data/lartpc_v2",
        min_points=1024,
        energy_threshold=0.13,
        remove_low_energy_scatters=False,
        max_len=10000,
        loop=1,
        test_mode=False,
        transform=[
            dict(
                type="NormalizeCoord",
                center=[384.0, 384.0, 384.0],
                scale=665.1075101064489,
            ),
            dict(
                type="GridSample",
                grid_size=0.001,
                hash_type="fnv",
                mode="deterministic",
                return_grid_coord=True,
                return_inverse=True,
                sum_keys=["energy"],
            ),
            dict(
                type="LogTransform",
                min_val=0.01,
                max_val=20.0,
                log=True,
                keys=["energy"],
            ),
            dict(type="Copy", keys_dict=dict(segment_motif="segment")),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=[
                    "coord",
                    "grid_coord",
                    "energy",
                    "inverse",
                    "segment",
                    "segment_motif",
                    "segment_pid",
                ],
                feat_keys=["coord", "energy"],
            ),
        ],
    ),
)

data["train"]["transform"].append(dict(type="SonataPreparation", model=detector_model))

detector_hooks = [
    dict(type="WeightDecayExclusion"),
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="PretrainEvaluator",
        force_fp32_features=True,
        label=[["segment_motif", "segment_pid"]],
        every_n_steps=5208,
        class_names=dict(
            segment_motif=["shower", "track", "michel", "delta", "led"],
            segment_pid=["photon", "electron", "muon", "pion", "proton", "none"],
        ),
        exclude_combinations=[["track", "electron"], ["shower", "muon"]],
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
