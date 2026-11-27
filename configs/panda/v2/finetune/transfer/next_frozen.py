_base_ = ["next.py"]
from os import environ as __env

model = dict(target_modules=[])
param_dicts = [dict(keyword="backbone.stem", lr=0.0001)]
scheduler = dict(max_lr=[0.0003, 0.0001])
save_path = (
    "exp/finetune/transfer/next_frozen_from_"
    + __env.get("SOURCE", "next")
    + "/k"
    + __env.get("K", "1000")
)
