_base_ = ["./pt-v3m2-ft-pid-fft.py"]
hooks_override = {"WandbNamer": {"extra": "scratch"}}
