_base_ = ["lartpc.py"]
from os import environ as __env

model = dict(target_modules=[])
save_path = (
    "exp/finetune/transfer/lartpc_frozen_from_"
    + __env.get("SOURCE", "lartpc")
    + "/k"
    + __env.get("K", "1000")
)
