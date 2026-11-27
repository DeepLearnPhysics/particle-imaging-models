# SOURCE={lartpc,sphenix,wc,next} selects the pretrained encoder; K defaults to 1000.
_base_ = ["../image/next_lora.py"]
from os import environ as __env

source = __env.get("SOURCE", "next")
assert source in ("lartpc", "sphenix", "wc", "next")
weight = f"weights/{source}.pth"
save_path = f"exp/finetune/transfer/next_from_{source}" + "/k" + __env.get("K", "1000")
model = dict(freeze_batchnorm=False, trainable_keywords=("cls_head", "backbone.stem"))
amp_dtype = "bfloat16"
sync_bn = False
param_dicts = [
    dict(keyword="lora_", lr=0.0001),
    dict(keyword="backbone.stem", lr=0.0001),
]
scheduler = dict(max_lr=[0.0003, 0.0001, 0.0001])
hooks_override = dict(
    CheckpointLoader=dict(
        replacements={
            "stem": "model.backbone.stem",
            "pools.": "model.backbone.pools.",
            "blocks.": "model.backbone.blocks.",
            **(
                {"stem.weight": "unloaded_source_stem.weight"} if source == "wc" else {}
            ),
        },
    ),
)
