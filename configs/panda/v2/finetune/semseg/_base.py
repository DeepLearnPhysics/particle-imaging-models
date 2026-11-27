_base_ = ["../_base_/lora.py"]

num_worker = 12
batch_size_val = batch_size_test = 8
matmul_precision = None
enable_amp, amp_dtype, clip_grad = (True, "bfloat16", 1.0)
decoder_lr, lora_lr = (0.0001, 0.0001)
optimizer = dict(type="AdamW", fused=True, lr=decoder_lr, weight_decay=0.01)
param_dicts = [dict(keyword="lora_", lr=lora_lr)]
scheduler = dict(max_lr=[decoder_lr, lora_lr])
evaluate = True
use_wandb = False
check = {{_base_.check}}

model = dict(
    _delete_=True,
    type="LoRAAdapter",
    rank=16,
    alpha=32,
    dropout=0.0,
    target_modules={{_base_.lora_targets}},
    freeze_batchnorm=True,
    trainable_keywords=("seg_head",),
    model=dict(
        type="DefaultSegmentorV2",
        backbone={{_base_.encoder}},
        backbone_out_channels=1170,
        mlp_head=True,
        mlp_hidden_channels=256,
        freeze_backbone=True,
    ),
)

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
    dict(type="ParameterCounter", show_details=True, show_gradients=True),
    dict(type="GradientNormLogger", log_frequency=100),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="PointClassificationEvaluator", every_n_steps=check, heads={}, names={}),
    dict(
        type="CheckpointSaver",
        save_freq=check,
        evaluator_every_n_steps=check,
        save_iter_checkpoints=False,
    ),
]
