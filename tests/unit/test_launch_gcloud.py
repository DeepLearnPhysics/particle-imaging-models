import json
import re

import pytest

from pimm.launch.config import finalize_config, load_config, validate_launch_config
from pimm.launch.gcloud import (
    STAGE_EXCLUDE,
    build_batch_job,
    parse_gs_uri,
    sanitize_job_name,
)


TIMESTAMP = "2026-01-02_03-04-05"


def _config(**overrides):
    cfg = load_config(site="gcloud", recipe=None, launch_timestamp=TIMESTAMP)
    cfg["executor"] = "batch"
    cfg["resources"]["scheduler_options"].update(
        project="my-proj", location="us-central1", machine_type="a2-highgpu-1g"
    )
    cfg["paths"]["exp_root"] = "gs://my-bucket/pimm_exp"
    cfg["resources"]["time"] = "24:00:00"  # pin: independent of the site yaml
    cfg["run"] = {"name": "gcloud-render", "timestamp": False}
    cfg["train"]["config"] = "tests/tiny_semseg"
    # A gcloud run requires a real W&B key (the VM does not see the local .env).
    cfg.setdefault("env", {})["WANDB_API_KEY"] = "test-key"
    for key, value in overrides.items():
        cfg[key] = value
    return finalize_config(cfg, launch_timestamp=TIMESTAMP, require_config=True)


def test_stage_exclude_matches_nested_paths():
    # `gsutil rsync -x` anchors the pattern at the start of the relative path
    # (re.match), so the exclude must catch caches/artifacts at any depth, not
    # just the repo root.
    pat = re.compile(STAGE_EXCLUDE)

    excluded = [
        "pimm/__pycache__/utils.cpython-311.pyc",
        "__pycache__/foo.pyc",
        "pimm/models/foo.pyc",
        ".git/config",
        "pimm/sub/.git/config",
        ".venv/lib/x.py",
        ".env",
        "conf/.env",
        "exp/run1/ckpt.pth",
        "data/x.h5",
        "pimm/weights/model.pth",
    ]
    kept = [
        "pimm/launch/gcloud.py",
        "pimm/__init__.py",
        "README.md",
        "configs/base.py",
    ]
    for p in excluded:
        assert pat.match(p), f"expected {p!r} to be excluded"
    for p in kept:
        assert not pat.match(p), f"expected {p!r} to be kept"


def test_parse_gs_uri():
    assert parse_gs_uri("gs://b/p/q") == ("b", "p/q")
    assert parse_gs_uri("gs://b") == ("b", "")
    assert parse_gs_uri("gs://b/") == ("b", "")
    with pytest.raises(SystemExit):
        parse_gs_uri("/local/path")


def test_sanitize_job_name():
    raw = "detector-v5_FT.joint-2026-07-20_15-16-57"
    name = sanitize_job_name(raw)
    assert name == "detector-v5-ft-joint-2026-07-20-15-16-57"
    assert len(name) <= 63
    assert name[0].isalpha()
    # a leading digit gets a letter prefix
    assert sanitize_job_name("2026-run").startswith("pimm-")


def test_build_batch_job_structure():
    cfg = _config()
    cfg["resources"]["scheduler_options"]["stage_code"] = False  # baked-image path
    job, script, stage_plan = build_batch_job(cfg, cfg["run"]["name"])

    task_spec = job["taskGroups"][0]["taskSpec"]
    container = task_spec["runnables"][0]["container"]
    instance = job["allocationPolicy"]["instances"][0]

    # gs:// bucket is mounted; EXP_ROOT points at the mount, not the URI.
    assert task_spec["volumes"][0]["gcs"]["remotePath"] == "my-bucket"
    assert task_spec["volumes"][0]["mountPath"] == "/mnt/disks/gcs"
    assert "export EXP_ROOT=/mnt/disks/gcs/pimm_exp" in script
    assert "gs://" not in script

    # Cloud Batch runs the image directly: no nested docker, train.sh invoked.
    assert container["imageUri"] == "ghcr.io/deeplearnphysics/pimm:main"
    assert container["entrypoint"] == "/bin/bash"
    assert "docker run" not in script
    # Staging off: run from the baked-in source, no copy step.
    assert stage_plan is None
    assert "sh /opt/pimm/src/scripts/train.sh -m 1 -g 1" in script
    assert "gcsfuse mount" not in script

    # A100 VM with drivers installed, and the time budget becomes a duration.
    assert instance["policy"]["machineType"] == "a2-highgpu-1g"
    assert instance["installGpuDrivers"] is True
    assert task_spec["maxRunDuration"] == "86400s"

    # spec is JSON-serializable
    json.dumps(job)


def test_code_staging_default():
    cfg = _config()  # stage_code defaults to True
    job, script, stage_plan = build_batch_job(cfg, "gcloud-render")

    # Staged to a sibling prefix of exp_root in the SAME bucket.
    assert stage_plan is not None
    local_root, gs_dest = stage_plan
    assert gs_dest == "gs://my-bucket/_pimm_code/gcloud-render"
    assert local_root  # the submit-host checkout root

    # The job copies the source off the mount and runs from local disk, with
    # PYTHONPATH shadowing the baked-in install.
    assert "cp -a /mnt/disks/gcs/_pimm_code/gcloud-render/. /tmp/pimm_src/" in script
    assert "export PYTHONPATH=/tmp/pimm_src" in script
    assert "cd /tmp/pimm_src" in script
    assert "sh /tmp/pimm_src/scripts/train.sh -m 1 -g 1" in script


def test_container_shm_options():
    # Default: share host IPC so DataLoader workers aren't capped at 64 MB shm,
    # and raise the open-file hard limit so gcsfuse checkpoint writes don't hit
    # EMFILE ("Too many open files") when publishing the DCP directory.
    cfg = _config()
    job, _, _ = build_batch_job(cfg, cfg["run"]["name"])
    container = job["taskGroups"][0]["taskSpec"]["runnables"][0]["container"]
    assert container["options"] == "--ipc=host --ulimit nofile=1048576:1048576"

    # An explicit shm_size sizes an isolated /dev/shm instead.
    cfg = _config()
    cfg["resources"]["scheduler_options"]["shm_size"] = "16g"
    job, _, _ = build_batch_job(cfg, cfg["run"]["name"])
    container = job["taskGroups"][0]["taskSpec"]["runnables"][0]["container"]
    assert container["options"] == "--shm-size=16g --ulimit nofile=1048576:1048576"


def test_gcs_file_cache_mount_options():
    # On by default: the bucket is mounted with the gcsfuse file cache enabled
    # (a `--cache-dir` turns it on) so the dataset's random reads hit local disk
    # after first touch.
    cfg = _config()
    job, _, _ = build_batch_job(cfg, cfg["run"]["name"])
    volume = job["taskGroups"][0]["taskSpec"]["volumes"][0]
    assert volume["mountOptions"] == [
        "--cache-dir /mnt/disks/gcsfuse-cache",
        "--file-cache-max-size-mb -1",
        "--max-conns-per-host 100",
    ]

    # Overridable dir/size.
    cfg = _config()
    cfg["resources"]["scheduler_options"].update(
        gcs_cache_dir="/mnt/disks/cache", gcs_cache_max_size_mb=100000
    )
    job, _, _ = build_batch_job(cfg, cfg["run"]["name"])
    volume = job["taskGroups"][0]["taskSpec"]["volumes"][0]
    assert volume["mountOptions"] == [
        "--cache-dir /mnt/disks/cache",
        "--file-cache-max-size-mb 100000",
        "--max-conns-per-host 100",
    ]

    # Disable the file cache: only the connection cap remains.
    cfg = _config()
    cfg["resources"]["scheduler_options"]["gcs_file_cache"] = False
    job, _, _ = build_batch_job(cfg, cfg["run"]["name"])
    volume = job["taskGroups"][0]["taskSpec"]["volumes"][0]
    assert volume["mountOptions"] == ["--max-conns-per-host 100"]

    # Disable both: no mountOptions on the volume.
    cfg = _config()
    cfg["resources"]["scheduler_options"].update(
        gcs_file_cache=False, gcs_max_conns_per_host=0
    )
    job, _, _ = build_batch_job(cfg, cfg["run"]["name"])
    volume = job["taskGroups"][0]["taskSpec"]["volumes"][0]
    assert "mountOptions" not in volume


def test_explicit_accelerator_for_non_a2():
    cfg = _config()
    cfg["resources"]["scheduler_options"].update(
        machine_type="n1-standard-8",
        accelerator_type="nvidia-tesla-a100",
        accelerator_count=1,
    )
    job, _, _ = build_batch_job(cfg, cfg["run"]["name"])
    accel = job["allocationPolicy"]["instances"][0]["policy"]["accelerators"][0]
    assert accel == {"type": "nvidia-tesla-a100", "count": 1}


def _runnable(job):
    return job["taskGroups"][0]["taskSpec"]["runnables"][0]


def test_wandb_key_exported_in_script():
    # The W&B key (finalize_config moves --run.wandb-api-key into env) is
    # exported in the rendered script; no Secret Manager secretVariables exist.
    cfg = _config()
    cfg["env"]["WANDB_API_KEY"] = "abc123"
    job, script, _ = build_batch_job(cfg, cfg["run"]["name"])
    assert "environment" not in _runnable(job)
    assert "export WANDB_API_KEY=abc123" in script


@pytest.mark.parametrize("value", ["", "   ", None])
def test_missing_wandb_key_rejected(value):
    # A gcloud run cannot authenticate to W&B without a real key, and the VM
    # never sees the local .env -- so an empty/unset key fails fast at build.
    cfg = _config()
    if value is None:
        cfg["env"].pop("WANDB_API_KEY", None)
    else:
        cfg["env"]["WANDB_API_KEY"] = value
    with pytest.raises(SystemExit, match="WANDB_API_KEY"):
        build_batch_job(cfg, cfg["run"]["name"])


def test_non_gs_exp_root_rejected():
    cfg = _config()
    cfg["paths"]["exp_root"] = "/local/exp"
    with pytest.raises(SystemExit, match="gs://"):
        validate_launch_config(cfg)


def test_missing_project_rejected():
    cfg = _config()
    del cfg["resources"]["scheduler_options"]["project"]
    with pytest.raises(SystemExit, match="project"):
        validate_launch_config(cfg)
