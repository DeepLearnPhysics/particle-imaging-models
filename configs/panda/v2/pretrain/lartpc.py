_base_ = ["_base_.py", "_data_/lartpc.py"]

model = {{_base_.detector_model}}
hooks = {{_base_.detector_hooks}}
save_path = "exp/pretrain/lartpc"
# Set data.train.max_len to the size of the pretraining pool.
