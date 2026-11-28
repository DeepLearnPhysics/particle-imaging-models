_base_ = ["wc.py"]
from os import environ as __env

K = int(__env.get("K", "1000"))
model = dict(target_modules=[])
amp_dtype = "bfloat16" if K == 981599 else "float16"
save_path = "exp/finetune/instance_wc_frozen" + "/k" + __env.get("K", "1000")
