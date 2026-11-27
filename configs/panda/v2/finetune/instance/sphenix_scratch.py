_base_ = ["sphenix.py"]
from os import environ as __env

K = int(__env.get("K", "1000"))
weight = None
amp_dtype = "float16" if K == 100 else "bfloat16"
model = dict(
    target_modules=[],
    trainable_keywords=("decoder", "backbone"),
    model=dict(center_pos_k=True),
)
param_dicts = None
scheduler = dict(max_lr=[0.0003])
save_path = "exp/finetune/instance_sphenix_scratch" + "/k" + __env.get("K", "1000")
