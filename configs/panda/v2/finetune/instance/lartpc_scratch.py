_base_ = ["lartpc.py"]
from os import environ as __env

K = int(__env.get("K", "1000"))
weight = None
model = dict(
    target_modules=[],
    trainable_keywords=("decoder", "backbone"),
)
param_dicts = None
scheduler = dict(max_lr=[0.0003])
save_path = "exp/finetune/instance_lartpc_scratch" + "/k" + __env.get("K", "1000")
