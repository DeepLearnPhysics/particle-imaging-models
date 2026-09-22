"""Completed continuations retain their checkpoint without changing its contents."""

import os

import pytest

from pimm.utils.checkpoints import publish_full_state_snapshot


@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("copy_only", [False, True])
def test_retained_checkpoint_survives_replacing_source(tmp_path, monkeypatch, directory, copy_only):
    source, destination = tmp_path / "input", tmp_path / "output"
    if directory:
        source.mkdir()
        (source / ".complete").touch()
    payload = source / "weights.pth" if directory else source
    payload.write_bytes(b"checkpoint")
    if copy_only:
        def no_links(*args, **kwargs):
            raise OSError("different filesystem")

        monkeypatch.setattr(os, "link", no_links)

    assert publish_full_state_snapshot(source, destination) == destination
    retained = destination / "weights.pth" if directory else destination
    assert retained.read_bytes() == b"checkpoint"
    if directory:
        assert (destination / ".complete").is_file()
    assert not (tmp_path / "output.tmp").exists()

    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"next checkpoint")
    os.replace(replacement, payload)
    assert retained.read_bytes() == b"checkpoint"
