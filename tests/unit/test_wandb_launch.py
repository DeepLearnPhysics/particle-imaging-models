"""History selection survives submission and clean container environments."""

import pytest

from pimm.launch.config import finalize_config, load_config
from pimm.launch.local import build_container_command


@pytest.mark.parametrize(
    "site,runtime",
    [("local", "none"), ("s3df-container", "singularity"), ("nersc-container", "shifter")],
)
def test_history_env_is_captured_and_forwarded_inside_container(monkeypatch, site, runtime):
    monkeypatch.setenv("PIMM_WANDB_HISTORY", "append")
    timestamp = "2026-01-02_03-04-05"
    cfg = load_config(site=site, recipe=None, launch_timestamp=timestamp)
    cfg["resources"].update(nnodes=1, nproc_per_node=1)
    cfg["paths"] = {"repo_root": "/work/pimm", "exp_root": "/work/exp"}
    cfg["train"]["config"] = "tests/tiny_semseg"
    cfg["container"]["runtime"] = runtime
    cfg = finalize_config(cfg, launch_timestamp=timestamp, require_config=True)
    assert cfg["env"]["PIMM_WANDB_HISTORY"] == "append"
    command = build_container_command(cfg, "echo worker")
    assert "export PIMM_WANDB_HISTORY=append" in command

    cfg["env"]["PIMM_WANDB_HISTORY"] = "fork"
    cfg = finalize_config(cfg, launch_timestamp=timestamp, require_config=True)
    assert cfg["env"]["PIMM_WANDB_HISTORY"] == "fork"
