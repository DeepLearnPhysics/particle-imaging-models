import pytest
import torch

from pimm.models.panda_detector.layers import Block, CrossAttentionLayer


@pytest.mark.parametrize("enable_flash", [False, True])
def test_cross_attention_without_mask_isolates_packed_events(
    enable_flash,
):
    torch.manual_seed(0)
    layer = CrossAttentionLayer(
        channels=8,
        num_heads=2,
        qkv_bias=True,
        enable_flash=enable_flash,
        upcast_attention=True,
    ).eval()

    q = torch.randn(3, 8, requires_grad=True)
    k = torch.randn(5, 8, requires_grad=True)
    v = torch.randn(5, 8, requires_grad=True)
    cu_seqlens_q = torch.tensor([0, 2, 3], dtype=torch.int32)
    cu_seqlens_kv = torch.tensor([0, 3, 5], dtype=torch.int32)

    packed = layer(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q=2,
        max_seqlen_kv=3,
    )
    with torch.no_grad():
        first_event = layer(
            q[:2],
            k[:3],
            v[:3],
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([0, 3], dtype=torch.int32),
            max_seqlen_q=2,
            max_seqlen_kv=3,
        )

    torch.testing.assert_close(packed[:2], first_event)

    packed[:2].sum().backward()
    torch.testing.assert_close(k.grad[3:], torch.zeros_like(k.grad[3:]))
    torch.testing.assert_close(v.grad[3:], torch.zeros_like(v.grad[3:]))


@pytest.mark.parametrize("is_last_block", [False, True])
def test_unsupervised_mask_blocks_isolate_events(is_last_block):
    torch.manual_seed(1)
    block = Block(
        channels=8,
        num_heads=2,
        qkv_bias=True,
        enable_flash=False,
        use_attn_mask=True,
        supervise_attn_mask=False,
        is_last_block=is_last_block,
    ).eval()

    q = torch.randn(4, 8)
    kv = torch.randn(5, 8)
    changed_kv = kv.clone()
    changed_kv[3:] = torch.randn_like(changed_kv[3:]) * 100
    cu_seqlens_q = torch.tensor([0, 2, 4], dtype=torch.int32)
    cu_seqlens_kv = torch.tensor([0, 3, 5], dtype=torch.int32)

    with torch.no_grad():
        output, *_ = block(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q=2,
            max_seqlen_kv=3,
        )
        changed_output, *_ = block(
            q,
            changed_kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q=2,
            max_seqlen_kv=3,
        )

    torch.testing.assert_close(output[:2], changed_output[:2])
