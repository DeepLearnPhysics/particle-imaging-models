_base_ = ["lartpc.py"]
from os import environ as __env

K = int(__env.get("K", "1000"))
model = dict(target_modules=[])
amp_dtype = "bfloat16" if K == 1000000 else "float16"
if K == 100000:
    model["model"] = dict(center_pos_k=False)
save_path = "exp/finetune/instance_lartpc_frozen" + "/k" + __env.get("K", "1000")
