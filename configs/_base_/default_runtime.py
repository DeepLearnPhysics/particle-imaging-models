weight = None  # path to model weight
resume = False  # whether to resume training process
evaluate = True  # evaluate after each epoch training process
test_only = False  # test process

seed = None  # train process will init a random seed and record
save_path = "exp/default"
num_worker = 16  # total worker in all gpu
batch_size = 16  # total batch size in all gpu
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = None  # auto adapt to bs 1 for each gpu
epoch = 800  # total epoch, data loop = epoch // eval_epoch
eval_epoch = 800  # sche total eval & checkpoint epoch
clip_grad = None  # disable with None, enable with a float

sync_bn = False
enable_amp = False
amp_dtype = "bfloat16"
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
matmul_precision = None
prefetch_factor = None
detect_anomaly = False
mix_prob = 0
param_dicts = None  # example: param_dicts = [dict(keyword="block", lr_scale=0.1)]
deterministic = False

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="FinalEvaluator", test_last=False),
]

# Trainer
train = dict(type="DefaultTrainer")

# Tester
test = dict(type="SemSegTester", verbose=True)
