_base_ = ["sphenix.py"]
from os import environ as __env

model = dict(target_modules=[])
amp_dtype = "float16"
save_path = "exp/finetune/instance_sphenix_frozen" + "/k" + __env.get("K", "1000")
