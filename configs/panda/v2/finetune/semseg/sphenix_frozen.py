_base_ = ["sphenix.py"]
from os import environ as __env

seed = {100: 46990708, 1000: 6272337, 10000: 7618066, 70000: 21526894}.get(
    int(__env.get("K", "1000")), 6272337
)
model = dict(target_modules=[], model=dict(heads={{_base_.linear_heads}}))
clip_grad = None
optimizer = dict(lr=0.0003)
scheduler = dict(max_lr=[0.0003, 0.0001])
amp_dtype = "float16"
hooks_override = dict(
    PointClassificationEvaluator=dict(
        heads=dict(
            linear_pid="segment_pid",
            linear_noise="segment_noise",
            mlp_pid="segment_pid",
            mlp_noise="segment_noise",
        ),
        groups=dict(
            linear=["linear_pid", "linear_noise"], mlp=["mlp_pid", "mlp_noise"]
        ),
    ),
)
save_path = "exp/finetune/semseg/sphenix_frozen" + "/k" + __env.get("K", "1000")
