_base_ = ["_lartpc.py"]
from os import environ as __env

save_path = "exp/finetune/semseg/lartpc" + "/k" + __env.get("K", "1000")
