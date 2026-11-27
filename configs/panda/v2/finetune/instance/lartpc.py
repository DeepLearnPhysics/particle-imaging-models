_base_ = ["../_base_/datasets/lartpc.py"]
from os import environ as __env

K = int(__env.get("K", "1000"))
amp_dtype = "float16" if K == 10000 else "bfloat16"
if K == 100000:
    model = dict(model=dict(center_pos_k=True))
save_path = "exp/finetune/instance_lartpc" + "/k" + __env.get("K", "1000")
