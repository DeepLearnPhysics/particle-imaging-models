import json
from pathlib import Path

import pytest

from pimm.utils.config import Config

pytestmark = pytest.mark.external_model

ROOT = Path(__file__).resolve().parents[2]


def _assert_bare_uri_loads(uri, monkeypatch, config_path=None):
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")

    import pimm

    model = pimm.from_pretrained(uri, config_path=config_path, device="cpu")
    assert not model.training
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_bare_semantic_uri(monkeypatch):
    _assert_bare_uri_loads(
        "hf://DeepLearnPhysics/panda-semantic",
        monkeypatch,
    )


def _export_v1_config(config_path, tmp_path):
    cfg = Config.fromfile(str(ROOT / config_path))
    model = cfg.model.to_dict()
    model["enable_flash"] = False  # Match the released detector's attention setting.
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": model}))
    return path


def test_bare_particle_uri(monkeypatch, tmp_path):
    _assert_bare_uri_loads(
        "hf://DeepLearnPhysics/panda-particle",
        monkeypatch,
        _export_v1_config(
            "configs/panda/v1/panseg/pt-v3m2-ft-pid-fft-detector.py", tmp_path
        ),
    )


@pytest.mark.parametrize(
    ("config_path", "label"),
    (
        (
            "configs/panda/v1/panseg/pt-v3m2-ft-pid-fft-detector.py",
            "particle",
        ),
        (
            "configs/panda/v1/panseg/pt-v3m2-ft-vtx-fft-detector.py",
            "interaction",
        ),
    ),
)
def test_detector_v1_config_strictly_loads_published_weights(
    config_path, label, monkeypatch, tmp_path
):
    """Released state dictionaries strictly load with the local V1 export config."""
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
        config_path=_export_v1_config(config_path, tmp_path),
        device="cpu",
        strict=True,
    )

    assert model.labels == (label,)
    assert type(model).__module__.endswith("panda_detector_v1m1")
