# Set K=100/1000/10000/100000/<full split size> in the environment; PASSES overrides an extended cap.
_base_ = ["../../_base_/default_runtime.py"]
from os import environ as __env

K = int(__env.get("K", "1000"))
assert K >= 16
schedule_total_steps = 625000
epoch = eval_epoch = int(
    __env.get(
        "PASSES",
        str(
            {100: 2000, 1000: 1000, 10000: 300}.get(
                K, schedule_total_steps // (K // 16)
            )
        ),
    )
)
check = {100: 625, 1000: 1250, 10000: 3750}.get(K, 6250 if K <= 100000 else 12500)
batch_size = 16
batch_size_val = batch_size_test = 8
num_worker = 12
seed = 0
clip_grad = 1.0
enable_amp = True
amp_dtype = "bfloat16"
matmul_precision = "high"
mix_prob = 0.0
use_wandb = False
# Task evaluators are configured through hooks.
test = None
optimizer = dict(type="AdamW", fused=True, lr=0.0003, weight_decay=0.01)
param_dicts = [dict(keyword="lora_", lr=0.0001)]
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0003, 0.0001],
    pct_start=625,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
