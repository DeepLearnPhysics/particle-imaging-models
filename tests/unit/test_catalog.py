import pytest

from pimm.datasets import catalog


def test_resolve_data_root_order(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "CACHE", tmp_path / "cache")
    monkeypatch.setenv("PIMM_DATA_ROOT", str(tmp_path / "env"))
    config, env, cache = tmp_path / "data" / "x", tmp_path / "env" / "x", tmp_path / "cache" / "x"
    with pytest.raises(FileNotFoundError, match="pimm datasets download x"):
        catalog.resolve_data_root("x", str(config))
    for path in (cache, env, config):
        path.mkdir(parents=True)
        assert catalog.resolve_data_root("x", str(config)) == str(path)
