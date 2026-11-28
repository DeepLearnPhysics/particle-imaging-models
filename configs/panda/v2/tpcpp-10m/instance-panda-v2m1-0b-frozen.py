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
seed = 0
clip_grad = 1.0
enable_amp = True
amp_dtype = "float16"
matmul_precision = "high"
mix_prob = 0.0
use_wandb = False
# Task evaluators are configured through hooks.
test = None
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
save_path = f"exp/tpcpp-10m/instance-panda-v2m1-0b-frozen/k{num_train_events}"

weight = "weights/tpcpp-10m.pth"
encoder = dict(
    type="panda-encoder-v2m1",
    in_channels=4,
    drop_path=0.0,
    grid_size=0.003,
    depths=(3, 6, 12, 6),
    conv_stages=2,
)
instance_loss = dict(
    type="FastUnifiedInstanceLoss",
    cost_mask=1.0,
    cost_dice=1.0,
    cost_class=1.0,
    loss_weight_focal=2.0,
    loss_weight_dice=5.0,
    cls_weight_matched=2.0,
    cls_weight_noobj=0.5,
    focal_alpha=0.25,
    focal_gamma=2.0,
    aux_loss_weight=1.0,
    num_points=100000,
    truth_label="instance",
    noobj_mask_loss_weight=0.0,
)
model = dict(
    type="LoRAAdapter",
    rank=16,
    alpha=32,
    dropout=0.0,
    target_modules=[],
    trainable_keywords=("decoder",),
    freeze_batchnorm=True,
    model=dict(
        type="panda-detector-v2m1",
        backbone=encoder,
        mlp_point_proj=True,
        full_in_channels=1170,
        hidden_channels=256,
        num_heads=16,
        depth=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        upcast_attention=False,
        pos_emb=True,
        criteria=[instance_loss],
        decoding="panda_v2",
        stuff_threshold=0.5,
        mask_threshold=0.5,
        nms_kernel="gaussian",
        nms_sigma=2.0,
        nms_pre=-1,
        nms_max=-1,
        min_points=2,
        zero_init_positions=True,  # with the decoder core loaded below
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
        data_root="data/tpcpp-10m",
        revision="v2",
        test_mode=False,
        energy_threshold=0.0,
        min_points=0,
        remove_low_energy_scatters=False,
        loop=1,
        split="train",
        max_len=num_train_events,
        overlay_n_events=2,
        overlay_prob=0.5,
        overlay_rotate_90=False,
        overlay_dedup=False,
        overlay_max_instances=dict(instance_particle=64),
        transform=train_transform,
    ),
    val=dict(
        type="PILArNetH5Dataset",
        data_root="data/tpcpp-10m",
        revision="v2",
        test_mode=False,
        energy_threshold=0.0,
        min_points=0,
        remove_low_energy_scatters=False,
        loop=1,
        split="val",
        max_len=1000,
        max_points=3200,
        transform=eval_transform,
    ),
    test=dict(
        type="PILArNetH5Dataset",
        data_root="data/tpcpp-10m",
        revision="v2",
        test_mode=False,
        energy_threshold=0.0,
        min_points=0,
        remove_low_energy_scatters=False,
        loop=1,
        split="test",
        max_len=-1,
        max_points=3200,
        transform=eval_transform,
    ),
)

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
        source="hf://DeepLearnPhysics/Panda-V2-Decoder/decoder_core.pth",
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
        every_n_steps=check,
        labels=("particle",),
        primary_label="particle",
        stuff_threshold=0.5,
        mask_threshold=0.5,
        iou_thresh=0.5,
        require_class_for_match=False,
        class_names=dict(particle=["other", "pion", "kaon", "proton", "electron"]),
        panoptic_selection=True,
        set_current_metric=True,
    ),
    dict(
        type="CheckpointSaver",
        save_freq=check,
        evaluator_every_n_steps=check,
        save_iter_checkpoints=False,
    ),
]
