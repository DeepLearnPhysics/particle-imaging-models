import pytest
import torch

from pimm.models.panda_v2.panda_encoder_v2m1 import window_indices


@pytest.mark.parametrize("size", [1024, 512, 700])
@pytest.mark.parametrize(
    "counts", [(1,), (1024,), (1025,), (2047, 17), (0, 2049, 1023)]
)
def test_windows_cover_every_point_once_as_a_query(counts, size):
    batch = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    gather, offsets, scatter, query_offsets = window_indices(batch, size)
    assert int(offsets.diff().max()) <= size
    assert torch.equal(gather[scatter], torch.arange(len(batch)))
    assert int(query_offsets[-1]) == len(batch)
    assert int(offsets[-1]) == len(gather)
    for begin, end, qbegin, qend in zip(
        offsets[:-1], offsets[1:], query_offsets[:-1], query_offsets[1:]
    ):
        primary = scatter[gather[begin:end]] == torch.arange(begin, end)
        assert int(primary.sum()) == qend - qbegin
        assert primary[-int(qend - qbegin) :].all()


@pytest.mark.gpu
@pytest.mark.parametrize("size", [1024, 512, 700])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("counts", [(17,), (1025,), (2047, 31)])
def test_fused_windows_match_gather_rope_and_attention(monkeypatch, dtype, counts, size):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("The fused producer requires Ampere or newer")
    flash_attn_interface = pytest.importorskip("flash_attn.flash_attn_interface")
    from pimm.ops import serialized_attention, varlen_attention
    from pimm.ops.rope import apply_rope, rope_tables

    forward = flash_attn_interface.flash_attn_varlen_func
    backward = flash_attn_interface._flash_attn_varlen_backward

    def deterministic_forward(*args, **kwargs):
        return forward(*args, **kwargs, deterministic=True)

    def deterministic_backward(*args, **kwargs):
        args = list(args)
        if len(args) > 20:
            args[20] = True
        else:
            kwargs["deterministic"] = True
        return backward(*args, **kwargs)

    monkeypatch.setattr(
        flash_attn_interface, "flash_attn_varlen_func", deterministic_forward
    )
    monkeypatch.setattr(
        flash_attn_interface, "_flash_attn_varlen_backward", deterministic_backward
    )

    torch.manual_seed(23)
    batch = torch.repeat_interleave(
        torch.arange(len(counts), device="cuda"), torch.tensor(counts, device="cuda")
    )
    gather, offsets, scatter, query_offsets = window_indices(batch, size)
    order = torch.randperm(len(batch), device="cuda")
    gather, scatter = order[gather], scatter[order.argsort()]
    cos, sin = rope_tables(
        torch.randint(0, 512, (len(batch), 3), device="cuda")[gather], 18
    )
    qkv = torch.randn(len(batch), 3 * 4 * 18, device="cuda", dtype=dtype) * 0.1
    reference = qkv.clone().requires_grad_()
    candidate = qkv.clone().requires_grad_()
    packed = reference[gather].unflatten(1, (3, 4, 18))
    expected = (
        varlen_attention(
            apply_rope(packed[:, 0].float(), cos, sin).half(),
            apply_rope(packed[:, 1].float(), cos, sin).half(),
            packed[:, 2].half(),
            offsets,
            offsets,
            size,
            size,
            fused=True,
        )
        .flatten(1)[scatter]
        .to(dtype)
    )
    actual = serialized_attention(
        candidate, gather, scatter, cos, sin, query_offsets, offsets, size, 4
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(
        candidate.grad,
        reference.grad,
        rtol=0.02 if dtype == torch.bfloat16 else 0.002,
        atol=2e-7,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("size", [1024, 512, 700])
def test_pytorch_fallback_matches_flash_attention(size):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pytest.importorskip(
        "flash_attn_turing" if torch.cuda.get_device_capability() == (7, 5) else "flash_attn"
    )
    from pimm.ops import serialized_attention
    from pimm.ops.rope import rope_tables

    torch.manual_seed(5)
    batch = torch.repeat_interleave(
        torch.arange(2, device="cuda"), torch.tensor([1500, 40], device="cuda")
    )
    gather, offsets, scatter, query_offsets = window_indices(batch, size)
    cos, sin = rope_tables(torch.randint(0, 512, (len(batch), 3), device="cuda")[gather], 18)
    qkv = torch.randn(len(batch), 3 * 4 * 18, device="cuda") * 0.1
    outputs = []
    for fused in (True, False):
        x = qkv.clone().requires_grad_()
        out = serialized_attention(x, gather, scatter, cos, sin, query_offsets, offsets, size, 4, fused=fused)
        out.square().mean().backward()
        outputs.append((out, x.grad))
    torch.testing.assert_close(outputs[1][0], outputs[0][0], rtol=0.01, atol=1e-3)
    torch.testing.assert_close(outputs[1][1], outputs[0][1], rtol=0.02, atol=1e-6)
