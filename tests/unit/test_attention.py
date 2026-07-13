import importlib.util
from pathlib import Path

import pytest
import torch


def load_attention_module():
    path = Path(__file__).parents[2] / "pimm/models/utils/attention.py"
    spec = importlib.util.spec_from_file_location("pimm_test_attention", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_varlen_adapter_uses_torch_210_signature_and_preserves_scale(monkeypatch):
    attention = load_attention_module()
    captured = {}

    def fake_varlen(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        captured.update(
            q=q,
            k=k,
            v=v,
            cu_q=cu_q,
            cu_k=cu_k,
            max_q=max_q,
            max_k=max_k,
            kwargs=kwargs,
        )
        return v

    monkeypatch.setattr(attention, "varlen_attn", fake_varlen)
    q = torch.ones(3, 2, 10)
    k = torch.ones_like(q)
    v = torch.arange(q.numel(), dtype=torch.float32).reshape_as(q)
    cu = torch.tensor([0, 3], dtype=torch.int32)

    result = attention.flash_attn_varlen_func(q, k, v, cu, cu, 3, 3, causal=True)

    assert captured["q"].shape[-1] == 16
    assert captured["k"].shape[-1] == 16
    assert captured["v"].shape[-1] == 16
    expected_q_scale = (10**-0.5) / (16**-0.5)
    torch.testing.assert_close(captured["q"][..., :10], q * expected_q_scale)
    assert captured["kwargs"] == {"is_causal": True}
    torch.testing.assert_close(result, v)


def test_varlen_adapter_rejects_dropout(monkeypatch):
    attention = load_attention_module()
    q = torch.ones(1, 1, 8)
    cu = torch.tensor([0, 1], dtype=torch.int32)

    with pytest.raises(NotImplementedError, match="does not support dropout"):
        attention.flash_attn_varlen_func(q, q, q, cu, cu, 1, 1, dropout_p=0.1)


def test_qkvpacked_adapter_preserves_unpadded_default_scale(monkeypatch):
    attention = load_attention_module()
    captured = {}

    def fake_varlen(q, k, v, *args, **kwargs):
        captured["q"] = q
        return v

    monkeypatch.setattr(attention, "varlen_attn", fake_varlen)
    qkv = torch.ones(3, 3, 2, 10)
    cu = torch.tensor([0, 3], dtype=torch.int32)

    result = attention.flash_attn_varlen_qkvpacked_func(qkv, cu, 3)

    expected_q_scale = (10**-0.5) / (16**-0.5)
    torch.testing.assert_close(captured["q"][..., :10], qkv[:, 0] * expected_q_scale)
    assert result.shape == (3, 2, 10)
