import pytest
import torch
import spconv.pytorch as spconv
from spconv.core import ConvAlgo

from pimm.ops import SubMConv3d


def test_native_fusion_requires_the_verified_input_width():
    assert SubMConv3d(4, 54, 5, algo=ConvAlgo.Native, fused=True).fused
    with pytest.raises(RuntimeError, match="four input channels"):
        SubMConv3d(29, 54, 5, algo=ConvAlgo.Native, fused=True)


def test_spconv_keeps_other_input_widths():
    conv = SubMConv3d(29, 54, 5, algo=ConvAlgo.Native)
    assert not conv.fused and conv.weight.shape[-1] == 29


@pytest.mark.parametrize("deterministic", [False, True])
def test_encoder_keeps_fallback_hooks(monkeypatch, deterministic):
    from types import SimpleNamespace
    from pimm.models.panda_v2.panda_encoder_v2m1 import SparseConv

    conv = SparseConv(4, 54, 5, algo=ConvAlgo.Native, fused=True)
    sparse = SimpleNamespace(features=torch.randn(3, 4))
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: deterministic)
    monkeypatch.setattr(spconv.SubMConv3d, "forward", lambda self, value: value)
    hooks = []
    conv.register_forward_hook(lambda *args: hooks.append(True))
    assert conv(sparse) is sparse
    assert hooks == [True]
