_base_ = ["../_base_/datasets/wc.py"]
from os import environ as __env

K = int(__env.get("K", "1000"))
amp_dtype = "float16" if K in (10000, 100000) else "bfloat16"
model = dict(model=dict(center_pos_k=K not in (100, 10000)))
save_path = "exp/finetune/instance_wc" + "/k" + __env.get("K", "1000")
