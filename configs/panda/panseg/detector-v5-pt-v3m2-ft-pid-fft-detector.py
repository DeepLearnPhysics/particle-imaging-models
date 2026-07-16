"""Full fine-tuning from the published Panda Particle detector."""

_base_ = ["./detector-v5-pt-v3m2-ft-pid-fft.py"]

weight = "hf://DeepLearnPhysics/panda-particle@bd90792dfe83cd05b437b719564b311f0a0b785a"
hooks_override = {
    "WandbNamer": {"extra": "fft-detector"},
    # Load the complete backbone and detector decoder without key rewriting.
    "CheckpointLoader": {"replacements": {}, "strict": True},
}
