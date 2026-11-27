_base_ = ["encoder.py"]
weight = "weights/wc.pth"
encoder = dict(in_channels=29, grid_size=0.03)
