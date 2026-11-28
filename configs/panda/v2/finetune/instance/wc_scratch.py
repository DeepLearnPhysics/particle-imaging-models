_base_ = ["wc.py"]
from os import environ as __env

weight = None
amp_dtype = "float16"
model = dict(
    target_modules=[],
    trainable_keywords=("decoder", "backbone"),
)
param_dicts = None
scheduler = dict(max_lr=[0.0003])
save_path = "exp/finetune/instance_wc_scratch" + "/k" + __env.get("K", "1000")
