_base_ = ["../_base_/datasets/next.py"]
from os import environ as __env

model = dict(
    _delete_=True,
    type="LoRAAdapter",
    rank=16,
    alpha=32,
    dropout=0.0,
    target_modules={{_base_.lora_targets}},
    freeze_batchnorm=True,
    trainable_keywords=("cls_head",),
    model=dict(
        type="EventClassifier",
        num_classes=2,
        backbone_out_channels=1170,
        backbone={{_base_.encoder}},
        pool="mean",
        freeze_backbone=False,
        criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0)],
    ),
)
amp_dtype = "float16"
matmul_precision = None
batch_size_test = None
save_path = "exp/finetune/next_lora" + "/k" + __env.get("K", "1000")
hooks = [
    dict(
        type="CheckpointLoader",
        replacements={
            "stem": "model.backbone.stem",
            "pools.": "model.backbone.pools.",
            "blocks.": "model.backbone.blocks.",
        },
        strict=False,
    ),
    dict(type="ParameterCounter", show_details=False, show_gradients=True),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="EventClassificationEvaluator",
        every_n_steps={{_base_.check}},
        metric="auroc",
    ),
    dict(
        type="CheckpointSaver",
        save_freq={{_base_.checkpoint_every}},
        evaluator_every_n_steps={{_base_.check}},
        save_iter_checkpoints=False,
    ),
]
