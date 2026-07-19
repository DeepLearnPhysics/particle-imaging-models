import subprocess
import sys

import yaml

from pimm.launch import submit as submit_mod
from pimm.launch.config import load_config

CFG = "panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask"


def run_cli(command, *args):
    result = subprocess.run(
        [sys.executable, "-m", "pimm.cli", command, *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


def test_launch_no_args_shows_help():
    stdout = run_cli("launch")
    assert "usage: pimm launch" in stdout
    assert "--train.config" in stdout
    assert stdout.count("Use a blocking Slurm allocation") == 1
    assert "--slurm." not in stdout


def test_submit_no_args_and_help_show_help():
    for args in [(), ("--help",)]:
        stdout = run_cli("submit", *args)
        assert "usage: pimm submit" in stdout
        assert "--site STR" in stdout
        assert "--train.config" in stdout
        assert stdout.count("Use a blocking Slurm allocation") == 1
        assert "--slurm." not in stdout


def test_slurm_site_base_inheritance():
    s3df = load_config(site="s3df", recipe=None, launch_timestamp="ts")
    assert s3df["resources"]["scheduler"] == "slurm"
    assert s3df["resources"]["nproc_per_node"] == 1
    assert s3df["resources"]["cpus_per_proc"] == 12
    assert s3df["resources"]["gpu_directive"] == "gres"
    assert s3df["env"]["PYTHONFAULTHANDLER"] == "1"
    assert s3df["resources"]["mem"] == "512G"
    assert s3df["container"]["runtime"] == "none"
    assert s3df["container"]["repo_mount"] == "/opt/pimm/src"

    s3df_container = load_config(
        site="s3df-container", recipe=None, launch_timestamp="ts"
    )
    assert s3df_container["container"]["runtime"] == "singularity"

    nersc = load_config(site="nersc", recipe=None, launch_timestamp="ts")
    assert nersc["resources"]["nproc_per_node"] == 4
    assert nersc["resources"]["cpus_per_proc"] == 6
    assert nersc["resources"]["gpu_directive"] == "gpus-per-node"
    assert nersc["env"]["PYTHONFAULTHANDLER"] == "1"
    assert nersc["container"]["runtime"] == "none"

    nersc_container = load_config(
        site="nersc-container", recipe=None, launch_timestamp="ts"
    )
    assert nersc_container["container"]["runtime"] == "shifter"


def test_launch_defaults_to_local_dry_run():
    script = run_cli(
        "launch",
        "--dry-run",
        "--train.config",
        CFG,
        "--run.name",
        "implicit-local-smoke",
        "--run.no-timestamp",
    )
    assert "#SBATCH" not in script
    assert "scontrol" not in script
    assert "cd ." in script
    assert "export EXP_ROOT=./exp" in script
    assert "scripts/train.sh" in script
    assert "pimm.launch.run_train" not in script
    assert "pimm.launch.run_job" not in script
    assert sys.executable in script
    assert "hooks.CheckpointSaverIteration.backend=dcp" not in script
    assert f"-c {CFG}" in script
    assert "-n implicit-local-smoke" in script
    assert "-m 1" in script
    # local site defaults to nproc_per_node: auto -> -g is omitted so train.sh
    # auto-detects all visible GPUs (Pointcept behavior).
    assert "-g " not in script


def test_launch_forwards_resources_and_train_overrides():
    script = run_cli(
        "launch",
        "--dry-run",
        "--resources.nproc-per-node",
        "4",
        "--train.config",
        CFG,
        "--run.name",
        "local4-smoke",
        "--run.no-timestamp",
        "--",
        "epoch=1",
        "batch_size=4",
    )
    assert "scripts/train.sh" in script
    assert "-g 4" in script
    assert "--options" in script
    assert "epoch=1" in script
    assert "batch_size=4" in script
    assert "hooks.CheckpointSaverIteration.backend=dcp" in script


def test_launch_forwards_wandb_cli_options():
    script = run_cli(
        "launch",
        "--dry-run",
        "--train.config",
        CFG,
        "--run.name",
        "wandb-cli-smoke",
        "--run.wandb-name",
        "wandb-display-name",
        "--run.wandb-project",
        "wandb-project-name",
        "--run.wandb-api-key",
        "test-api-key",
        "--run.no-timestamp",
    )

    assert "test-api-key" not in script
    assert "export WANDB_API_KEY='<redacted>'" in script
    assert "-a wandb-display-name" in script
    assert "wandb_project=wandb-project-name" in script


def test_launch_dry_run_validates_config_before_rendering():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pimm.cli",
            "launch",
            "--dry-run",
            "--train.config",
            "panda/pretrain/not-a-real-config",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert "Training config not found" in result.stderr


def test_launch_accepts_grouped_launcher_overrides():
    script = run_cli(
        "launch",
        "--dry-run",
        "--train.config",
        CFG,
        "--run.name",
        "grouped-launch-smoke",
        "--run.no-timestamp",
        "--resources.nproc-per-node",
        "2",
        "--resources.cpus-per-proc",
        "3",
        "--",
        "epoch=1",
    )
    assert "OMP_NUM_THREADS=3" in script
    assert "-g 2" in script
    assert "epoch=1" in script
    assert "hooks.CheckpointSaverIteration.backend=dcp" in script


def test_submit_s3df_renders_submitit_manifest_with_one_task_per_node():
    stdout = run_cli(
        "submit",
        "--dry-run",
        "--site",
        "s3df-container",
        "--train.config",
        CFG,
        "--run.name",
        "submitit-smoke",
        "--run.no-timestamp",
        "--resources.nnodes",
        "2",
        "--resources.nproc-per-node",
        "4",
        "--resources.cpus-per-proc",
        "12",
    )
    manifest = yaml.safe_load(stdout)
    params = manifest["parameters"]
    script = manifest["attempts"][0]["script"]
    assert manifest["backend"] == "submitit"
    assert params["nodes"] == 2
    assert params["tasks_per_node"] == 1
    assert params["slurm_use_srun"] is True
    assert params["cpus_per_task"] == 48
    assert params["slurm_gres"] == "gpu:4"
    assert params["slurm_srun_args"] == [
        "--output",
        "slurm_logs/slurm-%j.out",
        "--error",
        "slurm_logs/slurm-%j.out",
    ]
    assert params["slurm_additional_parameters"]["output"] == (
        "slurm_logs/slurm-%j.out"
    )
    assert params["slurm_additional_parameters"]["error"] == ("slurm_logs/slurm-%j.out")
    assert "scripts/train.sh" in script
    # repo_root is env-relative now (de-personalized); just assert the mount target.
    assert ":/opt/pimm/src" in script
    assert "sh /opt/pimm/src/scripts/train.sh" in script
    assert "pimm.launch.run_train" not in script
    assert "pimm.launch.run_job" not in script
    assert "-m 2" in script
    assert "-g 4" in script
    assert script.index("scontrol show hostnames") < script.index("apptainer run")
    assert "hooks.CheckpointSaverIteration.backend=dcp" in script


def test_submit_rendezvous_is_resolved_before_nersc_container():
    stdout = run_cli(
        "submit",
        "--dry-run",
        "--site",
        "nersc-container",
        "--train.config",
        CFG,
        "--run.name",
        "nersc-rdzv-smoke",
        "--run.no-timestamp",
        "--resources.nnodes",
        "2",
        "--resources.nproc-per-node",
        "4",
        "--resources.time",
        "00:30:00",
    )
    manifest = yaml.safe_load(stdout)
    script = manifest["attempts"][0]["script"]
    assert script.index("scontrol show hostnames") < script.index("shifter")
    assert "scontrol show hostnames" not in script.split("shifter", 1)[1]


def test_submit_jobs_render_requeue_attempts_and_resume():
    stdout = run_cli(
        "submit",
        "--dry-run",
        "--site",
        "s3df",
        "--train.config",
        CFG,
        "--run.name",
        "chain-smoke",
        "--chain.jobs",
        "2",
        "--resources.time",
        "00:30:00",
    )
    manifest = yaml.safe_load(stdout)
    attempts = manifest["attempts"]
    assert manifest["max_timeouts"] == 1
    assert len(attempts) == 2
    assert attempts[0]["resume"] is False
    assert attempts[1]["resume"] is True
    assert attempts[0]["wandb_name"] == "chain-smoke"
    assert attempts[1]["wandb_name"] == "chain-smoke"
    assert "-r true" not in attempts[0]["script"]
    assert "-r true" in attempts[1]["script"]
    assert "wandb_job_index=1" in attempts[0]["script"]
    assert "wandb_job_index=2" in attempts[1]["script"]


def test_interactive_chain_dry_run_renders_watchdog():
    stdout = run_cli(
        "submit",
        "--dry-run",
        "--site",
        "nersc",
        "--interactive",
        "--chain.jobs",
        "2",
        "--train.config",
        CFG,
        "--run.name",
        "interactive-chain-smoke",
        "--run.no-timestamp",
    )

    assert "# watchdog wrapper ->" in stdout
    assert "PIMM_WATCHDOG_CHILD=1" in stdout
    assert "#SCRON -q cron" in stdout
    assert "#SCRON -A m5238_g" in stdout
    assert "pimm submit" in stdout


def test_submit_nersc_uses_container_python_path():
    stdout = run_cli(
        "submit",
        "--dry-run",
        "--site",
        "nersc-container",
        "--train.config",
        CFG,
        "--run.name",
        "nersc-python-smoke",
        "--run.no-timestamp",
        "--resources.nproc-per-node",
        "4",
        "--resources.time",
        "00:30:00",
    )
    manifest = yaml.safe_load(stdout)
    params = manifest["parameters"]
    script = manifest["attempts"][0]["script"]
    assert params["gpus_per_node"] == 4
    assert params["slurm_additional_parameters"]["image"]
    assert "scripts/train.sh" in script
    # repo_root is env-relative now (de-personalized); just assert the shifter mount.
    assert "--volume=" in script
    assert ":/opt/pimm/src" in script
    assert "sh /opt/pimm/src/scripts/train.sh" in script
    assert "pimm.launch.run_train" not in script
    # uses the in-image interpreter, not the launcher's host python
    assert "/opt/pimm/.venv/bin/python" in script
    assert sys.executable not in script


def test_submit_no_container_runs_directly_on_node_env():
    stdout = run_cli(
        "submit",
        "--dry-run",
        "--site",
        "s3df",
        "--train.config",
        CFG,
        "--run.name",
        "no-container-smoke",
        "--run.no-timestamp",
        "--container.runtime",
        "none",
        "--resources.nproc-per-node",
        "1",
        "--",
        "epoch=1",
        "batch_size=2",
    )
    manifest = yaml.safe_load(stdout)
    params = manifest["parameters"]
    script = manifest["attempts"][0]["script"]
    assert params["nodes"] == 1
    assert params["tasks_per_node"] == 1
    assert params["slurm_use_srun"] is True
    assert "apptainer run" not in script
    # conda activation removed from sites: with runtime=none the job runs train.sh
    # directly in the node's active environment (no mamba activate).
    assert "mamba activate" not in script
    assert "scripts/train.sh" in script
    assert "epoch=1" in script
    assert "batch_size=2" in script


def test_remote_submit_reinvokes_pimm_submit_on_host(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

        class Result:
            returncode = 0
            stdout = "Submitted batch job 123\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(submit_mod.subprocess, "run", fake_run)
    cfg = {
        "site": "s3df",
        "paths": {"repo_root": str(tmp_path)},
        "submit": {"host": "iana"},
        "setup": ["source ~/.bashrc"],
    }

    output = submit_mod.remote_submit(
        ["--site", "s3df", "--train.config", "x/y", "--", "epoch=1"],
        cfg,
    )

    assert output == "Submitted batch job 123\n"
    assert calls[0][0][0] == "ssh"
    assert calls[0][0][1] == "iana"
    assert "source ~/.bashrc" in calls[0][0][2]
    assert (
        "pimm submit --site s3df --train.config x/y --no-remote -- epoch=1"
        in calls[0][0][2]
    )


def test_submit_uses_submitit_executor(monkeypatch, tmp_path):
    submitted = {}

    class FakeExecutor:
        def __init__(self, folder, cluster, slurm_max_num_timeout):
            submitted["folder"] = folder
            submitted["cluster"] = cluster
            submitted["max_timeouts"] = slurm_max_num_timeout

        def update_parameters(self, **kwargs):
            submitted["params"] = kwargs

        def submit(self, job):
            submitted["job"] = job

            class Job:
                job_id = "456"

            return Job()

    monkeypatch.setattr(submit_mod.submitit, "AutoExecutor", FakeExecutor)
    cfg = {
        "site": "s3df",
        "paths": {"repo_root": str(tmp_path)},
        "resources": {
            "scheduler": "slurm",
            "nnodes": 1,
            "nproc_per_node": 4,
            "cpus_per_proc": 12,
            "account": "acct",
            "partition": "ampere",
            "gpu_directive": "gres",
            "time": "00:30:00",
            "mem": "512G",
        },
        "container": {"runtime": "none"},
        "train": {"config": CFG},
        "run": {},
        "submit": {},
    }

    output = submit_mod.submit(cfg, "submitit-test")

    assert "Job successfully submitted" in output
    assert "job id:       456" in output
    assert "Helpful commands:" in output
    assert "Next commands:" not in output
    assert (
        f"config.py:    {tmp_path}/exp/panda/pretrain/submitit-test/config.py" in output
    )
    assert "slurm log:    slurm-456.out" in output
    assert "tail -f slurm-456.out" in output
    assert "manifest.yaml" not in output
    assert (submitted["folder"] / "manifest.yaml").exists()
    assert submitted["cluster"] == "slurm"
    assert submitted["max_timeouts"] == 0
    assert submitted["params"]["slurm_gres"] == "gpu:4"
    assert isinstance(submitted["job"], submit_mod.SubmititTrainingJob)


def test_launch_site_s3df_container_runs_locally_in_container():
    """`pimm launch --site s3df-container` uses its container on the current node."""
    script = run_cli(
        "launch",
        "--dry-run",
        "--site",
        "s3df-container",
        "--train.config",
        CFG,
        "--run.name",
        "s3df-local",
        "--run.no-timestamp",
    )
    # local executor: no Slurm rendezvous / batch directives
    assert "#SBATCH" not in script
    assert "scontrol" not in script
    # but it DOES wrap in the s3df container, with a hermetic shell
    assert "apptainer run" in script
    assert "bash --noprofile --norc -c" in script
    # in-image interpreter, no host conda activation
    assert "-p /opt/pimm/.venv/bin/python" in script
    assert "mamba activate" not in script
    assert ":/opt/pimm/src" in script


def test_local_rdzv_port_is_hashed_not_constant():
    a = run_cli(
        "launch",
        "--dry-run",
        "--train.config",
        CFG,
        "--run.name",
        "run-aaaa",
        "--run.no-timestamp",
    )
    b = run_cli(
        "launch",
        "--dry-run",
        "--train.config",
        CFG,
        "--run.name",
        "run-bbbb",
        "--run.no-timestamp",
    )

    def port(script):
        line = next(l for l in script.splitlines() if "MASTER_PORT=" in l)
        return line.split("MASTER_PORT:-", 1)[1].split("}", 1)[0]

    assert port(a) != "29500"
    assert port(a) != port(b)  # derived from the run name


def test_docker_runtime_renders_docker_run():
    script = run_cli(
        "launch",
        "--dry-run",
        "--container.runtime",
        "docker",
        "--container.image",
        "youngsm/pimm:pytorch2.10.0-cuda12.6",
        "--train.config",
        CFG,
        "--run.name",
        "docker-smoke",
        "--run.no-timestamp",
    )
    assert "docker run --rm --gpus all" in script
    assert "youngsm/pimm:pytorch2.10.0-cuda12.6" in script
    assert "bash --noprofile --norc -c" in script
    assert ":/opt/pimm/src" in script
    assert "sh /opt/pimm/src/scripts/train.sh" in script


def test_config_discovery_command():
    listing = run_cli("ls", "panda/pretrain")
    assert CFG in listing
