_base_ = ["./detector-v5-pt-v3m2-ft-vtx-fft.py"]
hooks_override = {"WandbNamer": {"extra": "scratch"}}
