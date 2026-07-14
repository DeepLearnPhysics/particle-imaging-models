import yaml

import pimm.launch.local as local_launch
from pimm.launch.config import finalize_config, load_config
from pimm.launch.submit import render_manifest


TIMESTAMP = "2026-01-02_03-04-05"


def _config(site, *, executor, python=None):
    cfg = load_config(site=site, recipe=None, launch_timestamp=TIMESTAMP)
    cfg["executor"] = executor
    cfg["paths"] = {"repo_root": "/work/pimm", "exp_root": "/work/exp"}
    cfg["resources"]["nnodes"] = 1
    cfg["resources"]["nproc_per_node"] = 1
    cfg["run"] = {"name": f"{site}-render", "timestamp": False}
    cfg["train"]["config"] = "tests/tiny_semseg"
    cfg["train"]["python"] = python
    return finalize_config(
        cfg,
        launch_timestamp=TIMESTAMP,
        require_config=True,
    )


def _manifest(cfg):
    return yaml.safe_load(render_manifest(cfg, cfg["run"]["name"]))


def test_local_uv_rendering(monkeypatch):
    monkeypatch.setattr(local_launch.sys, "executable", "/work/.venv/bin/python")
    cfg = _config("local", executor="local")

    _, script = local_launch.render_launch_script(cfg, TIMESTAMP)

    assert "sh /work/pimm/scripts/train.sh -m 1 -g 1" in script
    assert "-p /work/.venv/bin/python" in script
    assert "singularity" not in script
    assert "shifter" not in script


def test_generic_slurm_uv_rendering():
    cfg = _config(
        "slurm",
        executor="batch",
        python="/work/.venv/bin/python",
    )
    cfg["slurm"].update(account="science", partition="gpu")

    manifest = _manifest(cfg)
    params = manifest["parameters"]
    script = manifest["attempts"][0]["script"]

    assert params["slurm_account"] == "science"
    assert params["slurm_partition"] == "gpu"
    assert params["slurm_gres"] == "gpu:1"
    assert "-p /work/.venv/bin/python" in script
    assert "singularity" not in script
    assert "shifter" not in script


def test_s3df_apptainer_rendering():
    cfg = _config("s3df-container", executor="batch")

    manifest = _manifest(cfg)
    params = manifest["parameters"]
    script = manifest["attempts"][0]["script"]

    assert params["slurm_account"] == "mli:nu-ml-dev"
    assert params["slurm_partition"] == "ampere"
    assert params["slurm_gres"] == "gpu:1"
    assert "singularity run --nv" in script
    assert cfg["container"]["image"] in script
    assert "/work/pimm:/opt/pimm/src" in script
    assert "-p /opt/pimm/.venv/bin/python" in script


def test_nersc_shifter_rendering():
    cfg = _config("nersc-container", executor="batch")

    manifest = _manifest(cfg)
    params = manifest["parameters"]
    script = manifest["attempts"][0]["script"]

    assert params["slurm_account"] == "m5238_g"
    assert params["gpus_per_node"] == 1
    assert params["slurm_additional_parameters"]["image"] == "youngsm/pimm-nersc:main"
    assert "--image=youngsm/pimm-nersc:main" in script
    assert "--module=gpu,nccl-plugin" in script
    assert "--volume=/work/pimm:/opt/pimm/src" in script
    assert "-p /opt/pimm/.venv/bin/python" in script


def test_chain_attempts_share_one_wandb_run_name():
    cfg = _config(
        "slurm",
        executor="batch",
        python="/work/.venv/bin/python",
    )
    cfg.setdefault("chain", {})["jobs"] = 2

    attempts = _manifest(cfg)["attempts"]

    assert attempts[0]["resume"] is False
    assert attempts[1]["resume"] is True
    assert attempts[0]["wandb_name"] == "slurm-render"
    assert attempts[1]["wandb_name"] == "slurm-render"
    assert "wandb_job_index=1" in attempts[0]["script"]
    assert "wandb_job_index=2" in attempts[1]["script"]
