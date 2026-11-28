_base_ = ["../_base_/datasets/lartpc.py"]
from os import environ as __env

K = int(__env.get("K", "1000"))
save_path = "exp/finetune/instance_lartpc" + "/k" + __env.get("K", "1000")
