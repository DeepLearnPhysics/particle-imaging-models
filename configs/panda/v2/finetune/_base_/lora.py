_base_ = ["encoder.py"]
model = dict(
    type="LoRAAdapter",
    rank=16,
    alpha=32,
    dropout=0.0,
    target_modules={{_base_.lora_targets}},
    trainable_keywords=("decoder",),
    freeze_batchnorm=True,
    model=dict(backbone={{_base_.encoder}}),
)
