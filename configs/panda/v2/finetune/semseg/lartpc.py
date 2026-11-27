_base_ = ["_lartpc.py"]
from os import environ as __env

seed = {100: 46255931, 1000: 20778032, 10000: 13898194, 100000: 56972379}.get(
    int(__env.get("K", "1000")), 20778032
)
save_path = "exp/finetune/semseg/lartpc" + "/k" + __env.get("K", "1000")
