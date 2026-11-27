_base_ = ["../decoder.py"]
weight = "weights/sphenix.pth"
encoder = dict(in_channels=4, grid_size=0.003)
model = dict(
    model=dict(
        labels=("particle",),
        num_queries=(64,),
        num_classes=(5,),
        eval_label="particle",
        use_stuff_head=False,
        label_configs=dict(
            particle=dict(
                instance_key="instance_particle",
                segment_key="segment_pid",
                stuff_classes=[5],
                overlap=False,
                query_heads=[
                    dict(
                        name="noise",
                        kind="categorical",
                        num_classes=2,
                        target_key="segment_noise",
                        aggregation="mode",
                        loss_weight=1.0,
                    ),
                ],
            ),
        ),
        backbone=dict(in_channels=4, grid_size=0.003),
    ),
)

train_transform = [
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
    dict(
        type="JointLabel",
        keys=("segment_pid",),
        combinations=((0,), (1,), (2,), (3,), (4,)),
        out_key="segment_pid",
        ignore_index=-1,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=(
            "coord",
            "grid_coord",
            "segment_pid",
            "instance_particle",
            "segment_noise",
        ),
        feat_keys=("coord", "energy"),
    ),
]
eval_transform = [
    dict(type="NormalizeCoord", center=[0.0, 0.0, 0.0], scale=180.0),
    dict(
        type="Copy",
        keys_dict=dict(
            instance_particle="origin_instance_particle",
            segment_pid="origin_segment_pid",
            segment_motif="origin_segment_motif",
        ),
    ),
    dict(
        type="GridSample",
        grid_size=0.003,
        hash_type="fnv",
        mode="deterministic",
        return_grid_coord=True,
        return_inverse=True,
        sum_keys=("energy",),
    ),
    dict(type="LogTransform", min_val=20.0, max_val=2000.0, log=True, keys=("energy",)),
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
    dict(
        type="JointLabel",
        keys=("segment_pid",),
        combinations=((0,), (1,), (2,), (3,), (4,)),
        out_key="segment_pid",
        ignore_index=-1,
    ),
    dict(
        type="JointLabel",
        keys=("origin_segment_pid",),
        combinations=((0,), (1,), (2,), (3,), (4,)),
        out_key="origin_segment_pid",
        ignore_index=-1,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=(
            "coord",
            "grid_coord",
            "segment_pid",
            "instance_particle",
            "segment_noise",
            "origin_instance_particle",
            "origin_segment_pid",
            "origin_segment_noise",
            "inverse",
        ),
        feat_keys=("coord", "energy"),
    ),
]

data = dict(
    num_classes=5,
    ignore_index=-1,
    names=["other", "pion", "kaon", "proton", "electron"],
    train=dict(
        type="PILArNetH5Dataset",
        data_root="data/sphenix",
        revision="v2",
        test_mode=False,
        energy_threshold=0.0,
        min_points=0,
        remove_low_energy_scatters=False,
        loop=1,
        split="train",
        overlay_n_events=2,
        overlay_within_subset=True,
        overlay_prob=0.5,
        overlay_rotate_90=False,
        overlay_dedup=False,
        overlay_max_instances=dict(instance_particle=64),
    ),
    val=dict(
        type="PILArNetH5Dataset",
        data_root="data/sphenix",
        revision="v2",
        test_mode=False,
        energy_threshold=0.0,
        min_points=0,
        remove_low_energy_scatters=False,
        loop=1,
        split="val",
        max_len=1000,
        max_points=3200,
    ),
    test=dict(
        type="PILArNetH5Dataset",
        data_root="data/sphenix",
        revision="v2",
        test_mode=False,
        energy_threshold=0.0,
        min_points=0,
        remove_low_energy_scatters=False,
        loop=1,
        split="test",
        max_len=-1,
        max_points=3200,
    ),
)
data["train"].update(max_len={{_base_.K}}, transform=train_transform)
data["val"]["transform"] = eval_transform
data["test"]["transform"] = eval_transform

evaluate = False
# Use scripts/panda/eval/test_sphenix.py for validation checkpoint scoring.
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
        source="weights/decoder_core_sphenix.pth",
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
        labels=("particle",),
        primary_label="particle",
        stuff_threshold=0.5,
        mask_threshold=0.5,
        iou_thresh=0.5,
        require_class_for_match=False,
        class_names=dict(particle=["other", "pion", "kaon", "proton", "electron"]),
        panoptic_selection=True,
        set_current_metric=False,
    ),
    dict(
        type="CheckpointSaver",
        save_freq={{_base_.check}},
        evaluator_every_n_steps=None,
        save_iter_checkpoints=False,
    ),
]
