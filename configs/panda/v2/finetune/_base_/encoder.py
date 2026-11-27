_base_ = ["schedule.py"]
weight = "weights/lartpc.pth"
encoder = dict(
    type="panda-encoder-v2m1",
    in_channels=4,
    drop_path=0.0,
    grid_size=0.001,
    depths=(3, 6, 12, 6),
    conv_stages=2,
)
# The encoder's linear layers, by stage type, and its pooling projections.
lora_targets = [
    f"backbone.blocks.{s}.{b}.{m}"
    for s, depth in enumerate(encoder["depths"])
    for b in range(depth)
    for m in (
        ("linear",)
        if s < encoder["conv_stages"]
        else ("attn.qkv", "attn.proj", "mlp.0", "mlp.2")
    )
] + [f"backbone.pools.{i}.proj" for i in range(len(encoder["depths"]) - 1)]
