_base_ = ["lartpc.py"]
from os import environ as __env

K = int(__env.get("K", "1000"))
model = dict(target_modules=[])
save_path = "exp/finetune/instance_lartpc_frozen" + "/k" + __env.get("K", "1000")
