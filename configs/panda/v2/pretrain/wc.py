_base_ = ["_base_.py", "_data_/wc.py"]

model = {{_base_.detector_model}}
hooks = {{_base_.detector_hooks}}
save_path = "exp/pretrain/wc"
evaluate = False
# Set data.train.max_len to the size of the pretraining pool.
