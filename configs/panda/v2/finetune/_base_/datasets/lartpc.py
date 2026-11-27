_base_ = ["../decoder.py"]
weight = "weights/lartpc.pth"
encoder = dict(in_channels=4, grid_size=0.001)
model = dict(
    model=dict(
        labels=("interaction", "particle"),
        num_queries=(12, 32),
        num_classes=(2, 6),
        eval_label="particle",
        label_configs=dict(
            interaction=dict(
                instance_key="instance_interaction",
                segment_key="segment_interaction",
                stuff_classes=[0],
                stuff_class_label=0,
                overlap=False,
                query_heads=[],
            ),
            particle=dict(
                instance_key="instance_particle",
                segment_key="segment_pid",
                stuff_classes=[5],
                stuff_class_label=5,
                overlap=False,
                query_heads=[],
            ),
        ),
        use_stuff_head=(True, True),
        stuff_classes=dict(interaction=[0], particle=[5]),
        backbone=dict(in_channels=4, grid_size=0.001),
    ),
)

train_transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=665.1075101064489),
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
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=(
            "coord",
            "grid_coord",
            "segment_pid",
            "segment_interaction",
            "instance_particle",
            "instance_interaction",
            "name",
        ),
        feat_keys=("coord", "energy"),
    ),
]
eval_transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=665.1075101064489),
    dict(
        type="Copy",
        keys_dict=dict(
            instance_particle="origin_instance_particle",
            segment_pid="origin_segment_pid",
            instance_interaction="origin_instance_interaction",
            segment_interaction="origin_segment_interaction",
        ),
    ),
    dict(
        type="GridSample",
        grid_size=0.001,
        hash_type="fnv",
        mode="deterministic",
        return_grid_coord=True,
        return_inverse=True,
        sum_keys=("energy",),
    ),
    dict(type="LogTransform", min_val=0.01, max_val=20.0, keys=("energy",)),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=(
            "coord",
            "grid_coord",
            "segment_pid",
            "segment_interaction",
            "instance_particle",
            "instance_interaction",
            "name",
            "origin_instance_particle",
            "origin_segment_pid",
            "origin_instance_interaction",
            "origin_segment_interaction",
            "inverse",
        ),
        feat_keys=("coord", "energy"),
    ),
]

data = dict(
    num_classes=6,
    ignore_index=-1,
    names=["photon", "electron", "muon", "pion", "proton", "led"],
    interaction_names=["background", "interaction"],
    train=dict(
        type="PILArNetH5Dataset",
        revision="v3_extra",
        data_root="data/lartpc",
        energy_threshold=0.13,
        min_points=0,
        remove_low_energy_scatters=False,
        test_mode=False,
        loop=1,
        split="train",
        overlay_n_events=2,
        overlay_within_subset=True,
        overlay_prob=0.5,
        overlay_rotate_90=False,
        overlay_dedup=False,
        overlay_max_instances=dict(instance_particle=32, instance_interaction=12),
    ),
    val=dict(
        type="PILArNetH5Dataset",
        revision="v3_extra",
        data_root="data/lartpc",
        energy_threshold=0.13,
        min_points=0,
        remove_low_energy_scatters=False,
        test_mode=False,
        loop=1,
        split="val",
        max_len=1000,
    ),
    test=dict(
        type="PILArNetH5Dataset",
        revision="v3_extra",
        data_root="data/lartpc",
        energy_threshold=0.13,
        min_points=0,
        remove_low_energy_scatters=False,
        test_mode=False,
        loop=1,
        split="test",
        max_len=-1,
    ),
)
data["train"].update(max_len={{_base_.K}}, transform=train_transform)
data["val"]["transform"] = eval_transform
data["test"]["transform"] = eval_transform

evaluate = True
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
    dict(
        type="CheckpointLoader",
        source="weights/decoder_core_lartpc.pth",
        replacements={"decoder.": "model.decoder."},
        strict=False,
    ),
    dict(type="ParameterCounter", show_details=True, show_gradients=True),
    dict(type="GradientNormLogger", log_frequency=100),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="InstanceSegmentationEvaluator",
        ari_include_unassigned=True,
        pid_tables=False,
        every_n_steps={{_base_.check}},
        labels=("interaction", "particle"),
        primary_label="particle",
        stuff_threshold=0.5,
        mask_threshold=0.5,
        iou_thresh=0.5,
        require_class_for_match=False,
        class_names=dict(
            interaction=["background", "interaction"],
            particle=["photon", "electron", "muon", "pion", "proton"],
        ),
        stuff_classes=dict(interaction=[0], particle=[5]),
        panoptic_selection=True,
        set_current_metric=True,
    ),
    dict(
        type="CheckpointSaver",
        save_freq={{_base_.check}},
        evaluator_every_n_steps={{_base_.check}},
        save_iter_checkpoints=False,
    ),
]
