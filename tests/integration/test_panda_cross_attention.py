import pytest
import torch

import pimm.models.panda_detector.layers as panda_layers

pytestmark = pytest.mark.gpu


def test_cross_attention_flash_varlen_matches_isolated_sdpa(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if panda_layers.flash_attn_varlen_func is None:
        pytest.skip("FlashAttention is unavailable")

    torch.manual_seed(0)
    device = torch.device("cuda")
    flash_layer = (
        panda_layers.CrossAttentionLayer(
            channels=32,
            num_heads=4,
            qkv_bias=True,
            enable_flash=True,
            upcast_attention=False,
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    sdpa_layer = (
        panda_layers.CrossAttentionLayer(
            channels=32,
            num_heads=4,
            qkv_bias=True,
            enable_flash=False,
            upcast_attention=False,
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    sdpa_layer.load_state_dict(flash_layer.state_dict())

    q = torch.randn(3, 32, device=device, dtype=torch.bfloat16)
    k = torch.randn(5, 32, device=device, dtype=torch.bfloat16)
    v = torch.randn(5, 32, device=device, dtype=torch.bfloat16)
    cu_seqlens_q = torch.tensor([0, 2, 3], device=device, dtype=torch.int32)
    cu_seqlens_kv = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)

    called = False
    flash_attn_varlen_func = panda_layers.flash_attn_varlen_func

    def tracked_flash_attention(*args, **kwargs):
        nonlocal called
        called = True
        return flash_attn_varlen_func(*args, **kwargs)

    monkeypatch.setattr(
        panda_layers,
        "flash_attn_varlen_func",
        tracked_flash_attention,
    )

    with torch.no_grad():
        flash_output = flash_layer(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q=2,
            max_seqlen_kv=3,
        )
        sdpa_output = sdpa_layer(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q=2,
            max_seqlen_kv=3,
        )

    assert called
    torch.testing.assert_close(flash_output, sdpa_output, rtol=5e-2, atol=5e-2)
