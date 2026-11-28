_base_ = ["lartpc.py"]
from os import environ as __env

K = int(__env.get("K", "1000"))
weight = None
amp_dtype = "bfloat16" if K == 1000 else "float16"
model = dict(
    target_modules=[],
    trainable_keywords=("decoder", "backbone"),
)
param_dicts = None
scheduler = dict(max_lr=[0.0003])
save_path = "exp/finetune/instance_lartpc_scratch" + "/k" + __env.get("K", "1000")
