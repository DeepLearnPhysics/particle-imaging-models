_base_ = ["lartpc.py"]
from os import environ as __env

weight = None
seed = 0
amp_dtype = "float16"
model = dict(
    target_modules=[],
    trainable_keywords=("seg_head", "backbone"),
    model=dict(freeze_backbone=False),
)
param_dicts = [dict(keyword="backbone", lr=0.0003)]
scheduler = dict(max_lr=[0.0001, 0.0003])
save_path = "exp/finetune/semseg/lartpc_scratch" + "/k" + __env.get("K", "1000")
