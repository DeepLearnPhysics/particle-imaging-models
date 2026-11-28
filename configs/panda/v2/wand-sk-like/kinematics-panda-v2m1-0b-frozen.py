_base_ = ["../_base_/default_runtime.py"]

# Labeled single-particle events per species: 100, 1000 or 10000; PID heads are excluded.
num_train_events = 1000
check = {100: 625, 1000: 2500, 10000: 7500}[num_train_events]

schedule_total_steps = 625000
epoch = eval_epoch = {100: 2000, 1000: 1000, 10000: 300}[num_train_events]
batch_size = 16
batch_size_val = 16
batch_size_test = None
num_worker = 8
seed = 0
enable_amp, amp_dtype, clip_grad = (True, "float16", None)
matmul_precision = None
mix_prob = 0.0
use_wandb = False
# Task evaluators are configured through hooks.
test = None
evaluate = True
head_lr = 0.0003
optimizer = dict(type="AdamW", fused=True, lr=head_lr, weight_decay=0.01)
param_dicts = None
scheduler = dict(
    type="OneCycleLR",
    max_lr=head_lr,
    pct_start=625,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
save_path = f"exp/wand-sk-like/kinematics-panda-v2m1-0b-frozen/k{num_train_events}"

weight = "weights/wand-sk-like.pth"
encoder = dict(
    type="panda-encoder-v2m1",
    in_channels=29,
    drop_path=0.0,
    grid_size=0.03,
    depths=(3, 6, 12, 6),
    conv_stages=2,
)
tasks = [
    dict(name=f"{kind}_{species}", kind=kind, species=(species,))
    for species in ("e", "mu")
    for kind in ("vertex", "direction", "momentum")
]
data_root = "data/wand-sk-like"
pools = "data/wand-sk-like-probes/pools.npz"
model = dict(
    type="WANDEventProbes",
    backbone=encoder,
    backbone_out_channels=1170,
    tasks=tasks,
    mlp_hidden_channels=256,
)
transform = [
    dict(
        type="LogTransform", min_val=0.01, max_val=1000.0, clip=True, keys=("energy",)
    ),
    dict(
        type="FourierTime",
        periods_ns=[8.0 * 4.0**k for k in range(11)],
        keys=("time",),
        subtract_min=False,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=(
            "coord",
            "grid_coord",
            "name",
            "task_mask",
            "probe_label",
            "vertex",
            "direction",
            "momentum",
        ),
        feat_keys=("coord", "energy", "time", "hit", "hidden"),
    ),
]
dataset = dict(
    type="WANDProbeDataset",
    data_root=data_root,
    pools=pools,
    tasks=tasks,
    transform=transform,
)
data = dict(train=dict(dataset, split="train", n=num_train_events), val=dict(dataset, split="val"))

hooks = [
    dict(
        type="WeightDecayExclusion",
        exclude_bias_from_wd=True,
        exclude_norm_from_wd=True,
        exclude_gamma_from_wd=True,
        exclude_token_from_wd=True,
        exclude_ndim_1_from_wd=True,
    ),
    dict(
        type="CheckpointLoader",
        replacements={
            "stem": "backbone.stem",
            "pools.": "backbone.pools.",
            "blocks.": "backbone.blocks.",
        },
        strict=False,
    ),
    dict(type="ParameterCounter", show_details=False, show_gradients=True),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="WANDProbeEvaluator",
        selection_head="linear_direction_mu",
        every_n_steps=check,
        tasks=tasks,
    ),
    dict(
        type="CheckpointSaver",
        save_freq=min(check, 500),
        evaluator_every_n_steps=check,
        save_iter_checkpoints=False,
    ),
]
