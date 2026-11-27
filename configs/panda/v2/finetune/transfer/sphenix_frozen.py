_base_ = ["sphenix.py"]
from os import environ as __env

model = dict(target_modules=[])
save_path = (
    "exp/finetune/transfer/sphenix_frozen_from_"
    + __env.get("SOURCE", "sphenix")
    + "/k"
    + __env.get("K", "1000")
)
