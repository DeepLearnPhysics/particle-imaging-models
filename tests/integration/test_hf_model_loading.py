from pathlib import Path

import pytest

from pimm.utils.config import Config

pytestmark = pytest.mark.external_model

ROOT = Path(__file__).resolve().parents[2]


def _assert_bare_uri_loads(uri, monkeypatch):
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")

    import pimm

    model = pimm.from_pretrained(uri, device="cpu")
    assert not model.training
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_bare_semantic_uri(monkeypatch):
    _assert_bare_uri_loads(
        "hf://DeepLearnPhysics/panda-semantic",
        monkeypatch,
    )


def test_bare_particle_uri(monkeypatch):
    _assert_bare_uri_loads(
        "hf://DeepLearnPhysics/panda-particle",
        monkeypatch,
    )


@pytest.mark.parametrize(
    ("config_path", "label"),
    (
        (
            "configs/panda/panseg/detector-v5-pt-v3m2-ft-pid-fft-detector.py",
            "particle",
        ),
        (
            "configs/panda/panseg/detector-v5-pt-v3m2-ft-vtx-fft-detector.py",
            "interaction",
        ),
    ),
)
def test_detector_v5_config_strictly_loads_published_weights(
    config_path, label, monkeypatch
):
    """Published v4 state dicts must exactly match the v5 architecture."""
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")

    import pimm
    from pimm.engines._train_utils import _apply_hook_overrides_from_dict

    cfg = Config.fromfile(str(ROOT / config_path))
    _apply_hook_overrides_from_dict(cfg, cfg.hooks_override)
    loader = next(hook for hook in cfg.hooks if hook.type == "CheckpointLoader")
    assert loader.replacements == {}
    assert loader.strict is True
    model = pimm.from_pretrained(
        cfg.weight,
        model_config=cfg.model,
        device="cpu",
        strict=True,
    )

    assert model.labels == (label,)
    assert type(model).__module__.endswith("detector_v5")
