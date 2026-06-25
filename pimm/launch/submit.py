"""Submitit managed Slurm submission for `pimm submit`."""

from __future__ import annotations

import copy
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import submitit
import yaml

from .config import build_run_name, validate_launch_config, validate_training_config
from .local import build_train_sh_command, redact_config, render_script
from .utils import ROOT, as_bool, chain_jobs, resources, scheduler, slurm_time_to_minutes, write_text


@dataclass
class Attempt:
    """One submitit execution attempt for a possibly requeued training run."""

    job_index: int
    run_name: str
    resume: bool
    wandb_name: str | None
    script: str


class SubmititTrainingJob:
    """Run one pimm training attempt and describe the next requeued attempt."""

    def __init__(self, attempts: list[Attempt], attempt_index: int, cwd: str):
        self.attempts = attempts
        self.attempt_index = attempt_index
        self.cwd = cwd

    def __call__(self) -> int:
        attempt = self.attempts[self.attempt_index]
        print(
            "pimm submitit attempt "
            f"{attempt.job_index}/{len(self.attempts)}: {attempt.run_name}",
            flush=True,
        )
        env = os.environ.copy()
        env["PIMM_SUBMITIT_ATTEMPT"] = str(attempt.job_index)
        result = subprocess.run(
            ["bash", "-lc", attempt.script],
            cwd=self.cwd,
            env=env,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, "pimm submit job")
        return result.returncode

    def checkpoint(self):
        next_index = min(self.attempt_index + 1, len(self.attempts) - 1)
        next_attempt = self.attempts[next_index]
        print(
            "Requeueing pimm submitit attempt "
            f"{next_attempt.job_index}/{len(self.attempts)}",
            flush=True,
        )
        return submitit.helpers.DelayedSubmission(
            type(self)(self.attempts, next_index, self.cwd)
        )


def build_slurm_job_name(cfg: dict[str, Any], run_name: str) -> str:
    """Sanitize the run name for Slurm's job-name field."""
    name = str(cfg.get("slurm", {}).get("job_name") or run_name)
    name = re.sub(r"[^A-Za-z0-9_.+-]+", "-", name).strip("-")
    return name[:128] or "pimm"


def submitit_folder(cfg: dict[str, Any], run_name: str) -> Path:
    """Return the folder where submitit stores launcher pickles and logs."""
    repo_root = Path(str(cfg.get("paths", {}).get("repo_root", ROOT)))
    folder = cfg.get("submit", {}).get("folder")
    if folder is None:
        folder = Path("slurm_logs") / "submitit" / run_name
    folder_path = Path(str(folder))
    if not folder_path.is_absolute():
        folder_path = repo_root / folder_path
    return folder_path


def user_slurm_log_path(cfg: dict[str, Any], run_name: str) -> str:
    """Return the Slurm log path users should tail for training output."""
    slurm = cfg.get("slurm", {})
    output = str(slurm.get("output") or "slurm-%j.out")
    # sbatch expands %x to the Slurm job name, but submitit's generated srun
    # command writes its own output file. Resolve %x here so both land in the
    # same user-facing path.
    return output.replace("%x", build_slurm_job_name(cfg, run_name))


def experiment_config_path(cfg: dict[str, Any], run_name: str) -> Path:
    """Return where training will write the resolved run config."""
    paths = cfg.get("paths", {})
    repo_root = Path(str(paths.get("repo_root", ROOT)))
    exp_root = Path(str(paths.get("exp_root") or "exp"))
    if not exp_root.is_absolute():
        exp_root = repo_root / exp_root

    config_ref = Path(str(cfg.get("train", {}).get("config") or ""))
    if config_ref.suffix == ".py":
        config_ref = config_ref.with_suffix("")
    if config_ref.parts and config_ref.parts[0] == "configs":
        config_ref = Path(*config_ref.parts[1:])
    config_group = config_ref.parent
    exp_dir = (
        exp_root / run_name
        if str(config_group) in {"", "."}
        else exp_root / config_group / run_name
    )
    return exp_dir / "config.py"


def submitit_parameters(cfg: dict[str, Any], run_name: str) -> dict[str, Any]:
    """Map pimm launch resources into submitit's Slurm parameter names."""
    res = resources(cfg)
    slurm = cfg.get("slurm", {})
    log_path = user_slurm_log_path(cfg, run_name)
    params: dict[str, Any] = {
        "name": build_slurm_job_name(cfg, run_name),
        "nodes": res["nnodes"],
        "tasks_per_node": 1,
        "cpus_per_task": res["cpus_per_proc"] * res["nproc_per_node"],
        "timeout_min": slurm_time_to_minutes(res["time"]),
        "slurm_signal_delay_s": int(slurm.get("signal_delay_s", 120)),
        "slurm_use_srun": True,
        "slurm_srun_args": ["--output", log_path, "--error", log_path],
    }
    for cfg_key, submitit_key in (
        ("account", "slurm_account"),
        ("partition", "slurm_partition"),
        ("qos", "slurm_qos"),
        ("constraint", "slurm_constraint"),
        ("dependency", "slurm_dependency"),
    ):
        value = slurm.get(cfg_key)
        if value is not None and value != "":
            params[submitit_key] = value
    if res.get("mem"):
        params["slurm_mem"] = res["mem"]

    gpu_directive = slurm.get("gpu_directive", "gres")
    if gpu_directive == "gres":
        params["slurm_gres"] = f"gpu:{res['nproc_per_node']}"
    elif gpu_directive == "gpus-per-node":
        params["gpus_per_node"] = res["nproc_per_node"]
    else:
        raise SystemExit(f"Unsupported slurm.gpu_directive: {gpu_directive}")

    additional = {
        "output": log_path,
        "error": str(slurm.get("error") or log_path),
    }
    # A chained run requeues itself on timeout/preemption: submitit's checkpoint()
    # calls `scontrol requeue`, which SLURM rejects ("Requested operation is
    # presently disabled") unless the job was submitted requeuable. Enable it
    # automatically whenever more than one attempt exists. Renders as a bare
    # `#SBATCH --requeue` flag (submitit maps the bool True to a value-less flag).
    if chain_jobs(cfg) > 1:
        additional["requeue"] = True
    additional.update(dict(slurm.get("additional_parameters") or {}))
    for key in ("image", "module"):
        value = slurm.get(key)
        if value is not None and value != "":
            additional[key] = value
    params["slurm_additional_parameters"] = additional
    return params


def build_attempts(cfg: dict[str, Any], run_name: str) -> list[Attempt]:
    """Pre-render every possible submitit timeout/requeue attempt."""
    total_jobs = chain_jobs(cfg)
    chain_cfg = cfg.get("chain", {})
    base_wandb_name = str(cfg.get("run", {}).get("wandb_name") or run_name)
    resume_first = as_bool(chain_cfg.get("resume_first", False))
    existing_resume = as_bool(cfg.get("train", {}).get("resume", False))

    attempts: list[Attempt] = []
    for job_index in range(1, total_jobs + 1):
        job_cfg = copy.deepcopy(cfg)
        if total_jobs > 1:
            job_cfg.setdefault("train", {})["resume"] = (
                existing_resume or resume_first or job_index > 1
            )
            wandb_name = f"{base_wandb_name}-job{job_index:04d}"
            job_cfg.setdefault("run", {})["wandb_name"] = wandb_name
            options = job_cfg.setdefault("train", {}).setdefault("options", {})
            options["wandb_group"] = chain_cfg.get("wandb_group") or run_name
            options["wandb_job_type"] = chain_cfg.get("wandb_job_type", "train-job")
            options["wandb_job_index"] = job_index
            options["chain_jobs"] = total_jobs
        train_cmd = build_train_sh_command(job_cfg, run_name)
        script = render_script(job_cfg, train_cmd)
        attempts.append(
            Attempt(
                job_index=job_index,
                run_name=run_name,
                resume=as_bool(job_cfg.get("train", {}).get("resume", False)),
                wandb_name=job_cfg.get("run", {}).get("wandb_name"),
                script=script,
            )
        )
    return attempts


def render_manifest(cfg: dict[str, Any], run_name: str, *, redact: bool = False) -> str:
    """Render the submitit launch shape for dry-runs and exports."""
    manifest_cfg = redact_config(cfg) if redact else cfg
    attempts = build_attempts(manifest_cfg, run_name)
    manifest = {
        "backend": "submitit",
        "run_name": run_name,
        "folder": str(submitit_folder(manifest_cfg, run_name)),
        "max_timeouts": max(chain_jobs(cfg) - 1, 0),
        "parameters": submitit_parameters(manifest_cfg, run_name),
        "attempts": [asdict(attempt) for attempt in attempts],
    }
    return yaml.safe_dump(manifest, sort_keys=False)


def _run_submit_command(command: list[str]) -> str:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        if output:
            print(output, end="" if output.endswith("\n") else "\n")
        raise SystemExit(result.returncode)
    return output


def _insert_before_training_tail(argv: list[str], *items: str) -> list[str]:
    """Insert launcher args before the `--` training-override separator."""
    if "--" not in argv:
        return [*argv, *items]
    index = argv.index("--")
    return [*argv[:index], *items, *argv[index:]]


def remote_submit(argv: list[str], cfg: dict[str, Any]) -> str:
    """Re-run `pimm submit` on a remote login/submit host."""
    repo_root = Path(str(cfg.get("paths", {}).get("repo_root", ROOT)))
    submit_cfg = cfg.get("submit") or {}
    host = submit_cfg.get("host")
    if not host:
        raise SystemExit("remote_submit requires submit.host")

    remote_argv = [*argv]
    if "--no-remote" not in remote_argv:
        remote_argv = _insert_before_training_tail(remote_argv, "--no-remote")
    remote_parts = [f"cd {shlex.quote(str(repo_root))}"]
    remote_parts.extend(str(cmd) for cmd in submit_cfg.get("setup") or [])
    remote_parts.append("mkdir -p slurm_logs")
    remote_parts.append(f"pimm submit {shlex.join(remote_argv)}")
    remote_inner = " && ".join(remote_parts)
    remote_cmd = f"bash -lc {shlex.quote(remote_inner)}"
    return _run_submit_command(["ssh", str(host), remote_cmd])


def submit(cfg: dict[str, Any], run_name: str) -> str:
    """Submit one Slurm job through submitit."""
    if scheduler(cfg) != "slurm":
        raise SystemExit("pimm submit requires a Slurm site, e.g. --site s3df")
    repo_root = Path(str(cfg.get("paths", {}).get("repo_root", ROOT)))
    cwd = repo_root if repo_root.exists() else ROOT
    folder = submitit_folder(cfg, run_name)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.yaml").write_text(
        render_manifest(cfg, run_name, redact=True), encoding="utf-8"
    )
    attempts = build_attempts(cfg, run_name)
    executor = submitit.AutoExecutor(
        folder=folder,
        cluster="slurm",
        slurm_max_num_timeout=max(chain_jobs(cfg) - 1, 0),
    )
    executor.update_parameters(**submitit_parameters(cfg, run_name))
    job = executor.submit(SubmititTrainingJob(attempts, 0, str(cwd)))
    # Save the exact Slurm submission script submitit generated into the
    # experiment folder, next to config.py / run_metadata.json.
    try:
        submission_file = Path(job.paths.submission_file)
        if submission_file.is_file():
            exp_dir = experiment_config_path(cfg, run_name).parent
            exp_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(submission_file, exp_dir / "launch.sbatch")
    except OSError:
        pass
    log_path = user_slurm_log_path(cfg, run_name)
    config_path = experiment_config_path(cfg, run_name)
    log_glob = (
        log_path.replace("%j", str(job.job_id))
        .replace("%A", str(job.job_id))
        .replace("%a", "*")
        .replace("%n", "*")
    )
    return (
        f"{'='*80}\n"
        "Job successfully submitted\n"
        f"  job id:       {job.job_id}\n"
        f"  run:          {run_name}\n"
        f"  config.py:    {config_path}\n"
        f"  slurm log:    {log_glob}\n"
        f"  submitit dir: {folder}\n"
        f"{'='*80}\n"
        "Helpful commands:\n"
        f"  scontrol show job {job.job_id}\n"
        f"  squeue -j {job.job_id}\n"
        f"  tail -f {log_glob}\n"
    )


def build_interactive_argv(cfg: dict[str, Any], run_name: str) -> list[str]:
    """Build a blocking `salloc ... srun ... bash -lc <script>` command.

    Reuses the exact rendered launch script (container + train.sh + torchrun) that
    a batch job runs, so interactive and batch behave identically; `salloc` just
    grabs the allocation live instead of queuing. The slurm flags mirror the batch
    parameters. The QOS is ``slurm.qos`` -- for NERSC's interactive queue pass
    ``--slurm.qos interactive`` (or ``shared_interactive``).
    """
    res = resources(cfg)
    slurm = cfg.get("slurm", {})
    nodes = int(res["nnodes"])
    gpus = int(res["nproc_per_node"])
    cpus = int(res["cpus_per_proc"])
    qos = slurm.get("qos")

    alloc = [
        "salloc",
        "--job-name", build_slurm_job_name(cfg, run_name),
        "--nodes", str(nodes),
        "--ntasks-per-node", "1",
        "--cpus-per-task", str(cpus * gpus),
    ]
    if slurm.get("account"):
        alloc += ["--account", str(slurm["account"])]
    if slurm.get("partition"):
        alloc += ["--partition", str(slurm["partition"])]
    if qos:
        alloc += ["--qos", str(qos)]
    if slurm.get("constraint"):
        alloc += ["--constraint", str(slurm["constraint"])]
    if res.get("time"):
        alloc += ["--time", str(res["time"])]
    if res.get("mem"):
        alloc += ["--mem", str(res["mem"])]

    gpu_directive = slurm.get("gpu_directive", "gres")
    if gpu_directive == "gres":
        alloc += ["--gres", f"gpu:{gpus}"]
    elif gpu_directive == "gpus-per-node":
        alloc += ["--gpus-per-node", str(gpus)]
    else:
        raise SystemExit(f"Unsupported slurm.gpu_directive: {gpu_directive}")

    # Shifter image/module directives, mirroring the batch path's #SBATCH --image/
    # --module
    for key in ("image", "module"):
        value = slurm.get(key)
        if value:
            alloc += [f"--{key}", str(value)]
    for key, value in (slurm.get("additional_parameters") or {}).items():
        if value is not None and value != "":
            alloc += [f"--{key}", str(value)]

    # srun launches the rendered script one task per node (mirrors batch); the
    # script's rdzv derives MASTER_ADDR from $SLURM_JOB_NODELIST.
    script = render_script(cfg, build_train_sh_command(cfg, run_name))
    return [*alloc, "srun", "--ntasks-per-node", "1", "bash", "-lc", script]


def run_interactive(cfg: dict[str, Any], run_name: str, argv: list[str]) -> int:
    """Run a blocking interactive allocation, streaming output to the terminal."""
    repo_root = Path(str(cfg.get("paths", {}).get("repo_root", ROOT)))
    cwd = repo_root if repo_root.exists() else ROOT
    print(
        f"{'=' * 80}\n"
        f"Interactive allocation (salloc) for run: {run_name}\n"
        "  Runs live in this terminal and ends if you disconnect.\n"
        f"  cd: {cwd}\n"
        f"{'=' * 80}",
        flush=True,
    )
    return subprocess.run(argv, cwd=str(cwd)).returncode


def run_submit(
    cfg: dict[str, Any],
    *,
    launch_timestamp: str,
    dry_run: bool,
    output: str | None,
    remote_argv: list[str],
    no_remote: bool,
) -> int:
    """Validate, dry-run, or submit a managed Slurm launch."""
    validate_launch_config(cfg)
    validate_training_config(cfg)
    run_name = build_run_name(cfg, launch_timestamp)
    if not run_name:
        raise SystemExit("Could not determine run name")

    if as_bool(cfg.get("interactive", False)):
        if scheduler(cfg) != "slurm":
            raise SystemExit("--interactive requires a Slurm site (e.g. --site nersc).")
        if chain_jobs(cfg) > 1:
            raise SystemExit(
                "--interactive is single-shot and does not support chaining "
                "(chain.jobs>1). Drop --interactive for a chained batch run."
            )
        argv = build_interactive_argv(cfg, run_name)
        if output:
            path = write_text(output, shlex.join(argv) + "\n")
            print(f"# wrote interactive salloc command: {path}")
        if dry_run:
            print(shlex.join(argv))
            return 0
        return run_interactive(cfg, run_name, argv)

    manifest = render_manifest(cfg, run_name, redact=True)
    if output:
        path = write_text(output, manifest)
        print(f"# wrote submitit manifest: {path}")
    if dry_run:
        print(manifest)
        return 0

    submit_cfg = cfg.get("submit") or {}
    result = remote_submit(remote_argv, cfg) if submit_cfg.get("host") and not no_remote else submit(cfg, run_name)
    if result:
        print(result, end="" if result.endswith("\n") else "\n")
    return 0
