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
param_dicts = None
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0003],
    pct_start=625,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
save_path = f"exp/wand-sk-like/instance-panda-v2m1-0c-scratch/k{num_train_events}"

weight = None
encoder = dict(
    type="panda-encoder-v2m1",
    in_channels=29,
    drop_path=0.0,
    grid_size=0.03,
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
    noobj_mask_loss_weight=1.0,
)
model = dict(
    type="LoRAAdapter",
    rank=16,
    alpha=32,
    dropout=0.0,
    target_modules=[],
    trainable_keywords=("decoder", "backbone"),
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
        decoding="wc",
        stuff_threshold=0.5,
        mask_threshold=0.5,
        nms_kernel="gaussian",
        nms_sigma=2.0,
        nms_pre=-1,
        nms_max=-1,
        min_points=2,
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
    ),
)

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
        max_len=num_train_events,
        all_train_events=True,
        random_rotation=dict(z=0.8, x=0.8, y=0.8),
        random_flip=0.5,
        transform=train_transform,
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
        type="RingPanopticEvaluator",
        every_n_steps=check,
        iou_thresh=0.5,
        conf_threshold=0.65,  # the paper's WC decoding rule
        mask_threshold=0.5,
        min_points=2,
        coord_scale=18.1,
        dedup_iou=1.0,
        set_current_metric=True,
        selection_metric="direction_mu_q68_deg",  # the paper's WC checkpoint rule
    ),
    dict(
        type="CheckpointSaver",
        save_freq=check,
        evaluator_every_n_steps=check,
        save_iter_checkpoints=False,
    ),
]
