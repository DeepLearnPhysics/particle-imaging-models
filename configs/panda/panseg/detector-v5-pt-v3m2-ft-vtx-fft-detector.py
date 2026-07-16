"""Full fine-tuning from the published Panda Interaction detector."""

_base_ = ["./detector-v5-pt-v3m2-ft-vtx-fft.py"]

weight = (
    "hf://DeepLearnPhysics/panda-interaction@fa8bd4c1937c39a48e02bde5bf27e188d0be4ac4"
)
hooks_override = {
    "WandbNamer": {"extra": "fft-detector"},
    # Load the complete backbone and detector decoder without key rewriting.
    "CheckpointLoader": {"replacements": {}, "strict": True},
}
