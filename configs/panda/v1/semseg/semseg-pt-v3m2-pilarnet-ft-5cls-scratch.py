_base_ = ["./semseg-pt-v3m2-pilarnet-ft-5cls-fft.py"]
hooks_override = {"WandbNamer": {"extra": "scratch"}}