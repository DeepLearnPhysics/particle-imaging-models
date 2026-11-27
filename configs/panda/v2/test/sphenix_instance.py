# Panda V2 decoding; paper hit-level scoring is provided by scripts/panda/eval/.
_base_ = ["../finetune/instance/sphenix_frozen.py"]
weight = "weights/test_sphenix_instance.pth"
save_path = "exp/test/sphenix_instance"
resume = False
seed = 0
use_wandb = False
batch_size_test = 1
hooks = []
test = dict(
    _delete_=True,
    type="InstanceSegTester",
    ari_include_unassigned=True,
    pid_tables=False,
)
data = dict(test=dict(split="test", max_len=-1, test_mode=False, max_points=3200))
