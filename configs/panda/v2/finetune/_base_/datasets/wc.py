_base_ = ["../decoder.py"]
from os import environ as __env

weight = "weights/wc.pth"
encoder = dict(in_channels=29, grid_size=0.03)
model = dict(
    model=dict(
        decoding="wc",
        zero_init_positions=True,  # with the decoder core loaded below
        labels=("particle",),
        num_queries=(32,),
        num_classes=(8,),
        eval_label="particle",
        label_configs=dict(
            particle=dict(
                instance_key="inst_pe",
                segment_key="ring_pid",
                stuff_classes=[],
                overlap=True,
                query_heads=[
                    dict(
                        name="momentum",
                        kind="continuous",
                        target_key="ring_momentum",
                        dim=1,
                        squeeze=True,
                        criterion=dict(
                            type="SmoothL1RegressionLoss", beta=0.1, reduction="mean"
                        ),
                        loss_weight=1.0,
                    ),
                    dict(
                        name="vertex",
                        kind="continuous",
                        target_key="ring_vertex",
                        valid_key="ring_vertex_valid",
                        dim=3,
                        criterion=dict(
                            type="SmoothL1RegressionLoss", beta=0.1, reduction="mean"
                        ),
                        loss_weight=2.0,
                    ),
                    dict(
                        name="direction",
                        kind="continuous",
                        target_key="ring_direction",
                        dim=3,
                        criterion=dict(
                            type="GeodesicDirectionLoss",
                            clamp_deg=0.1,
                            reduction="mean",
                        ),
                        loss_weight=1.0,
                    ),
                ],
            ),
        ),
        use_stuff_head=(False,),
        backbone=dict(in_channels=29, grid_size=0.03),
    ),
)
model["model"]["criteria"] = [
    dict(
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
        noobj_mask_loss_weight=1.0,
    ),
]
K = int(__env.get("K", "1000"))

train_transform = [
    dict(
        type="LogTransform", min_val=0.01, max_val=1000.0, clip=True, keys=("energy",)
    ),
    dict(
        type="FourierTime",
        periods_ns=[
            8.0,
            32.0,
            128.0,
            512.0,
            2048.0,
            8192.0,
            32768.0,
            131072.0,
            524288.0,
            2097152.0,
            8388608.0,
        ],
        keys=("time",),
        subtract_min=False,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=(
            "coord",
            "grid_coord",
            "inst_pe",
            "ring_pid",
            "ring_anc_pdg",
            "ring_anc_id",
            "ring_valid",
            "name",
            "ring_momentum",
            "ring_direction",
            "ring_vertex",
            "ring_vertex_valid",
        ),
        feat_keys=("coord", "energy", "time", "hit", "hidden"),
    ),
]
eval_transform = [
    dict(
        type="LogTransform", min_val=0.01, max_val=1000.0, clip=True, keys=("energy",)
    ),
    dict(
        type="FourierTime",
        periods_ns=[
            8.0,
            32.0,
            128.0,
            512.0,
            2048.0,
            8192.0,
            32768.0,
            131072.0,
            524288.0,
            2097152.0,
            8388608.0,
        ],
        keys=("time",),
        subtract_min=False,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=(
            "coord",
            "grid_coord",
            "inst_pe",
            "ring_pid",
            "ring_anc_pdg",
            "ring_anc_id",
            "ring_valid",
            "name",
            "ring_momentum",
            "ring_direction",
            "ring_vertex",
            "ring_vertex_valid",
        ),
        feat_keys=("coord", "energy", "time", "hit", "hidden"),
    ),
]

data = dict(
    num_classes=8,
    ignore_index=-1,
    names=["e", "gamma", "mu", "pi", "p", "n", "K", "other"],
    train=dict(
        type="WANDRingSitesDataset",
        data_root="data/wc",
        config="config_01",
        max_rings=32,
        min_ring_hits=3,
        coord_scale=18.1,
        grid_size=0.03,
        split="train",
        all_train_events=True,
        random_rotation=dict(z=0.8, x=0.8, y=0.8),
        random_flip=0.5,
    ),
    val=dict(
        type="WANDRingSitesDataset",
        data_root="data/wc",
        config="multi_primary",
        max_rings=32,
        min_ring_hits=3,
        coord_scale=18.1,
        grid_size=0.03,
        split="val",
        max_len=2000,
    ),
)
data["train"].update(max_len={{_base_.K}}, transform=train_transform)
data["val"]["transform"] = eval_transform

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
        type="RingPanopticEvaluator",
        every_n_steps={{_base_.check}},
        iou_thresh=0.5,
        conf_threshold=0.65,  # the paper's WC decoding rule (scripts/panda/eval/test_wc.py)
        mask_threshold=0.5,
        min_points=2,
        coord_scale=18.1,
        dedup_iou=1.0,
        set_current_metric=True,
        selection_metric="direction_mu_q68_deg",  # the paper's WC checkpoint rule
    ),
    dict(
        type="CheckpointSaver",
        save_freq={{_base_.check}},
        evaluator_every_n_steps={{_base_.check}},
        save_iter_checkpoints=False,
    ),
]
