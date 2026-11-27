_base_ = ["../_base_/default_runtime.py"]

seed = 0
batch_size = 96
batch_size_val = None  # one validation event per GPU
num_worker = 32  # Total across four ranks: eight workers per rank.
gradient_accumulation_steps = 1
epoch = eval_epoch = 20
schedule_total_steps = 208334
# H5 loaders run complete epochs; an exact step-limited stream is unavailable.
enable_amp = True
amp_dtype = "bfloat16"
matmul_precision = "high"
# Four ranks with 24 images each reproduce the paper's local statistical groups.
sync_bn = False
use_wandb = False
checkpoint_format = "standard"
resume_strict_state = True

model = dict(
    type="panda-ssl-v2m1",
    backbone=dict(
        type="panda-encoder-v2m1",
        in_channels=4,
        grid_size=0.001,
        drop_path=0.3,
    ),
    teacher_custom=dict(drop_path=0.0),
    # Masking, matching and the loss level are set per detector in _data_/*.py.
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=8192,
    num_global_view=2,
    num_local_view=6,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.1,
    student_temp=0.1,
    mask_loss_weight=0.25,
    roll_mask_loss_weight=0.25,
    unmask_loss_weight=0.5,
    momentum_base=0.996,
    momentum_final=1.0,
    ema_power=4,
    split_backward=True,
    fused_sinkhorn=True,
    sync_sinkhorn=False,
    sinkhorn_iters=3,
)

clip_grad = 2.4
param_dicts = [dict(keyword="block", lr=0.000594)]
optimizer = dict(
    type="AdamW", fused=True, lr=0.00594, betas=(0.9, 0.996), weight_decay=0.05
)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.00594, 0.000594],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
)
