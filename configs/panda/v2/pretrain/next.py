_base_ = ["_base_.py", "_data_/next.py"]

model = {{_base_.detector_model}}
hooks = {{_base_.detector_hooks}}
save_path = "exp/pretrain/next"
# Set data.train.subset=(0,0,K) to choose a pretraining pool of K images.
