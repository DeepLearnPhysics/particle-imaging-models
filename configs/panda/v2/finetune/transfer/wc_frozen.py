_base_ = ["wc.py"]
from os import environ as __env

model = dict(target_modules=[])
save_path = (
    "exp/finetune/transfer/wc_frozen_from_"
    + __env.get("SOURCE", "wc")
    + "/k"
    + __env.get("K", "1000")
)
