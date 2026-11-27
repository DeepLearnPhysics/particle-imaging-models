_base_ = ["_sphenix.py"]
from os import environ as __env

seed = {100: 48035707, 1000: 26021118, 10000: 39861129, 70000: 53430084}.get(
    int(__env.get("K", "1000")), 26021118
)
save_path = "exp/finetune/semseg/sphenix" + "/k" + __env.get("K", "1000")
