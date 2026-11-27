_base_ = ["../_base_/default_runtime.py"]

seed = 0
save_path = "exp/decoder/synthetic"
num_worker = 8
batch_size = 8
batch_size_val = 8
batch_size_test = 8
epoch = 1
clip_grad = 3.0
enable_amp = True
amp_dtype = "float16"
matmul_precision = "high"
use_wandb = False

hooks = [
    dict(
        type="WeightDecayExclusion",
        exclude_bias_from_wd=True,
        exclude_norm_from_wd=True,
        exclude_ndim_1_from_wd=True,
    ),
    dict(type="CheckpointLoader"),
    dict(type="ParameterCounter", show_details=True, min_params=1),
    dict(type="IterationTimer", warmup_iter=5),
    dict(type="InformationWriter", log_frequency=20),
    dict(type="SpherePriorEvaluator", every_n_steps=5000, prefix="val"),
    dict(type="CheckpointSaver", save_freq=10000, evaluator_every_n_steps=5000),
]

model = dict(
    type="panda-detector-v2m1",
    backbone=None,
    full_in_channels=256,
    labels=("instance",),
    num_queries=64,
    num_classes=1,
    label_configs=dict(
        instance=dict(instance_key="instance", segment_key="inst_class", overlap=True)
    ),
    pos_emb=False,
    hidden_channels=256,
    num_heads=16,
    depth=3,
    mlp_ratio=4.0,
    qkv_bias=True,
    layer_scale=None,
    upcast_attention=False,
    criteria=[
        dict(
            type="FastUnifiedInstanceLoss",
            # Set consistent_matching=True to use the paper objective.
            consistent_matching=False,
            overlap=True,
            query_heads=[],
            membership_key="inst_pe",
            valid_key="ring_valid",
            segment_key="inst_class",
            mask_pe_thresh=0.0,
            cost_mask=2.0,
            cost_dice=2.0,
            cost_class=1.0,
            loss_weight_focal=2.0,
            loss_weight_dice=5.0,
            cls_weight_matched=2.0,
            cls_weight_noobj=0.5,
            focal_alpha=0.25,
            focal_gamma=2.0,
            aux_loss_weight=1.0,
        ),
    ],
)

optimizer = dict(type="AdamW", fused=True, lr=0.0003, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.0003,
    pct_start=2000,
    anneal_strategy="cos",
    div_factor=1000.0,
    final_div_factor=1000.0,
)

data = dict(
    train=dict(type="SpherePriorDataset", length=720000, split="train", seed=0),
    val=dict(type="SpherePriorDataset", length=512, split="val", seed=0),
    test=dict(type="SpherePriorDataset", length=1024, split="test", seed=0),
)
