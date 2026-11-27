_base_ = ["../_base_/datasets/sphenix.py"]
from os import environ as __env

K = int(__env.get("K", "1000"))
amp_dtype = "float16" if K == 10000 else "bfloat16"
save_path = "exp/finetune/instance_sphenix" + "/k" + __env.get("K", "1000")
