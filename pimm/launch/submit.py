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
from .utils import (
    ROOT,
    as_bool,
    chain_jobs,
    resources,
    scheduler,
    slurm_time_to_minutes,
    write_text,
)


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
    name = str(cfg.get("resources", {}).get("job_name") or run_name)
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
    resource_cfg = cfg.get("resources", {})
    output = str(resource_cfg.get("output") or "slurm-%j.out")
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
    resource_cfg = cfg.get("resources", {})
    log_path = user_slurm_log_path(cfg, run_name)
    params: dict[str, Any] = {
        "name": build_slurm_job_name(cfg, run_name),
        "nodes": res["nnodes"],
        "tasks_per_node": 1,
        "cpus_per_task": res["cpus_per_proc"] * res["nproc_per_node"],
        "timeout_min": slurm_time_to_minutes(resource_cfg.get("time")),
        "slurm_signal_delay_s": int(resource_cfg.get("signal_delay_s", 120)),
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
        value = resource_cfg.get(cfg_key)
        if value is not None and value != "":
            params[submitit_key] = value
    if resource_cfg.get("mem"):
        params["slurm_mem"] = resource_cfg["mem"]

    gpu_directive = resource_cfg.get("gpu_directive", "gres")
    if gpu_directive == "gres":
        params["slurm_gres"] = f"gpu:{res['nproc_per_node']}"
    elif gpu_directive == "gpus-per-node":
        params["gpus_per_node"] = res["nproc_per_node"]
    else:
        raise SystemExit(f"Unsupported resources.gpu_directive: {gpu_directive}")

    additional = {
        "output": log_path,
        "error": str(resource_cfg.get("error") or log_path),
    }
    # A chained run requeues itself on timeout/preemption: submitit's checkpoint()
    # calls `scontrol requeue`, which SLURM rejects ("Requested operation is
    # presently disabled") unless the job was submitted requeuable. Enable it
    # automatically whenever more than one attempt exists. Renders as a bare
    # `#SBATCH --requeue` flag (submitit maps the bool True to a value-less flag).
    if chain_jobs(cfg) > 1:
        additional["requeue"] = True
    scheduler_options = resource_cfg.get("scheduler_options") or {}
    if not isinstance(scheduler_options, dict):
        raise SystemExit("resources.scheduler_options must be a mapping")
    additional.update(scheduler_options)
    container = cfg.get("container", {})
    if container.get("runtime") == "shifter":
        for key in ("image", "module"):
            value = container.get(key)
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
            job_cfg.setdefault("run", {})["wandb_name"] = base_wandb_name
            options = job_cfg.setdefault("train", {}).setdefault("options", {})
            options["wandb_group"] = chain_cfg.get("wandb_group") or run_name
            options["wandb_job_type"] = chain_cfg.get("wandb_job_type", "train-job")
            options["wandb_job_index"] = job_index
            options["chain_jobs"] = total_jobs
        train_cmd = build_train_sh_command(job_cfg, run_name)
        script = render_script(job_cfg, train_cmd, run_name)
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
    remote_parts.extend(str(cmd) for cmd in cfg.get("setup") or [])
    remote_parts.append("mkdir -p slurm_logs")
    remote_parts.append(f"pimm submit {shlex.join(remote_argv)}")
    remote_inner = " && ".join(remote_parts)
    remote_cmd = f"bash -lc {shlex.quote(remote_inner)}"
    return _run_submit_command(["ssh", str(host), remote_cmd])


def submit(cfg: dict[str, Any], run_name: str) -> str:
    """Submit one Slurm job through submitit."""
    if scheduler(cfg) != "slurm":
        raise SystemExit(
            "pimm submit requires resources.scheduler='slurm', e.g. --site s3df"
        )
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
    except (AttributeError, OSError):
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


def build_interactive_argv(
    cfg: dict[str, Any], run_name: str, script: str
) -> list[str]:
    """Build a blocking `salloc ... srun ... bash -lc <script>` command.

    `script` is a standalone rendered launch segment (container + train.sh +
    torchrun) produced by `build_attempts`, identical to what a batch job runs;
    `salloc` just grabs the allocation live instead of queuing. For a chained
    interactive run each attempt passes its own pre-rendered script (the later
    ones carry `-r true`), so the salloc flags below are shared while the inner
    training segment differs per attempt. The Slurm flags mirror the batch
    parameters. The QOS is ``resources.qos`` -- for NERSC's interactive queue
    pass ``--resources.qos interactive`` (or ``shared_interactive``).
    """
    res = resources(cfg)
    resource_cfg = cfg.get("resources", {})
    nodes = int(res["nnodes"])
    gpus = int(res["nproc_per_node"])
    cpus = int(res["cpus_per_proc"])
    qos = resource_cfg.get("qos")

    alloc = [
        "salloc",
        "--job-name",
        build_slurm_job_name(cfg, run_name),
        "--nodes",
        str(nodes),
        "--ntasks-per-node",
        "1",
        "--cpus-per-task",
        str(cpus * gpus),
    ]
    if resource_cfg.get("account"):
        alloc += ["--account", str(resource_cfg["account"])]
    if resource_cfg.get("partition"):
        alloc += ["--partition", str(resource_cfg["partition"])]
    if qos:
        alloc += ["--qos", str(qos)]
    if resource_cfg.get("constraint"):
        alloc += ["--constraint", str(resource_cfg["constraint"])]
    if resource_cfg.get("time"):
        alloc += ["--time", str(resource_cfg["time"])]
    if resource_cfg.get("mem"):
        alloc += ["--mem", str(resource_cfg["mem"])]

    gpu_directive = resource_cfg.get("gpu_directive", "gres")
    if gpu_directive == "gres":
        alloc += ["--gres", f"gpu:{gpus}"]
    elif gpu_directive == "gpus-per-node":
        alloc += ["--gpus-per-node", str(gpus)]
    else:
        raise SystemExit(f"Unsupported resources.gpu_directive: {gpu_directive}")

    # Shifter image/module directives, mirroring the batch path's #SBATCH --image/
    # --module
    container = cfg.get("container", {})
    if container.get("runtime") == "shifter":
        for key in ("image", "module"):
            value = container.get(key)
            if value:
                alloc += [f"--{key}", str(value)]
    scheduler_options = resource_cfg.get("scheduler_options") or {}
    if not isinstance(scheduler_options, dict):
        raise SystemExit("resources.scheduler_options must be a mapping")
    for key, value in scheduler_options.items():
        if value is not None and value != "":
            alloc += [f"--{key}", str(value)]

    # srun launches the rendered script one task per node (mirrors batch); the
    # script's rdzv derives MASTER_ADDR from $SLURM_JOB_NODELIST.
    return [*alloc, "srun", "--ntasks-per-node", "1", "bash", "-lc", script]


def render_interactive_commands(commands: list[tuple[Attempt, list[str]]]) -> str:
    """Render the salloc command(s) for --dry-run / --output.

    A single allocation renders as the bare command (unchanged). A chained run
    renders one annotated block per attempt so the resume/wandb shape is visible.
    """
    if len(commands) == 1:
        return shlex.join(commands[0][1]) + "\n"
    total = len(commands)
    blocks = [
        f"# interactive attempt {index}/{total} "
        f"(resume={attempt.resume}, wandb={attempt.wandb_name})\n"
        f"{shlex.join(argv)}"
        for index, (attempt, argv) in enumerate(commands, start=1)
    ]
    return "\n\n".join(blocks) + "\n"


def _latest_checkpoint_mtime(cfg: dict[str, Any], run_name: str) -> float:
    """Return the mtime of the run's newest complete checkpoint, or 0.0 if none.

    Used to tell a walltime/preemption kill (training advanced the checkpoint
    before dying -> resume the next allocation) apart from a startup crash (no
    new checkpoint -> resuming would just repeat it). Mirrors the resume lookup
    in scripts/train.sh, which reads `<exp_dir>/model`.
    """
    from pimm.utils.path import latest_complete_checkpoint

    model_dir = experiment_config_path(cfg, run_name).parent / "model"
    checkpoint = latest_complete_checkpoint(model_dir)
    if checkpoint is None:
        return 0.0
    try:
        return checkpoint.stat().st_mtime
    except OSError:
        return 0.0


def run_interactive(
    cfg: dict[str, Any],
    run_name: str,
    commands: list[tuple[Attempt, list[str]]],
) -> int:
    """Run one or more blocking interactive allocations in sequence.

    A single allocation (chain.jobs == 1) behaves exactly like a plain `salloc`.
    With chain.jobs > 1 this is the interactive analogue of batch requeue
    chaining: each attempt grabs its own short `salloc` slot and resumes from the
    previous attempt's newest checkpoint, so a long run is reached through
    several small interactive slots instead of one oversized allocation.

    Interactive `salloc` installs no preemption signal and pimm has no timeout
    handler, so a walltime kill and a crash share the same nonzero exit code. We
    disambiguate with checkpoint progress: if the model dir gained a newer
    complete checkpoint during the attempt, the kill was a walltime/preemption
    and we resume; if not, it was a crash and we stop rather than spin re-running
    the same failure.
    """
    repo_root = Path(str(cfg.get("paths", {}).get("repo_root", ROOT)))
    cwd = repo_root if repo_root.exists() else ROOT
    total = len(commands)
    chained = total > 1
    print(
        f"{'=' * 80}\n"
        f"Interactive allocation (salloc) for run: {run_name}\n"
        + (
            f"  Chained: up to {total} sequential salloc slots, each resuming the last.\n"
            if chained
            else ""
        )
        + "  Runs live in this terminal and ends if you disconnect.\n"
        f"  cd: {cwd}\n"
        f"{'=' * 80}",
        flush=True,
    )
    for index, (attempt, argv) in enumerate(commands, start=1):
        if chained:
            print(
                f"\n{'-' * 80}\n"
                f"Interactive attempt {index}/{total} "
                f"(resume={attempt.resume}, wandb={attempt.wandb_name})\n"
                f"{'-' * 80}",
                flush=True,
            )
        before = _latest_checkpoint_mtime(cfg, run_name)
        try:
            returncode = subprocess.run(argv, cwd=str(cwd)).returncode
        except KeyboardInterrupt:
            print("\nInterrupted; stopping interactive run.", flush=True)
            return 130
        if returncode == 0:
            if chained and index < total:
                print(
                    f"Attempt {index}/{total} finished cleanly -- training "
                    "completed before exhausting the chain. Stopping.",
                    flush=True,
                )
            return 0
        # Ctrl-C / SIGINT: the user asked to stop, not a slot timing out.
        if returncode in (130, -2):
            return returncode
        if index >= total:
            return returncode
        if _latest_checkpoint_mtime(cfg, run_name) <= before:
            print(
                f"Attempt {index}/{total} exited {returncode} without writing a "
                "newer checkpoint -- treating this as a crash, not a walltime "
                "timeout (resuming would just repeat it). Stopping.",
                flush=True,
            )
            return returncode
        print(
            f"Attempt {index}/{total} exited {returncode} after advancing the "
            "checkpoint -- assuming a walltime/preemption kill. Requesting the "
            "next allocation to resume...",
            flush=True,
        )
    return 0


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
    if scheduler(cfg) != "slurm":
        raise SystemExit("pimm submit requires resources.scheduler='slurm'")
    run_name = build_run_name(cfg, launch_timestamp)
    if not run_name:
        raise SystemExit("Could not determine run name")

    if as_bool(cfg.get("interactive", False)):
        # A chained interactive run needs its foreground driver to survive the
        # login session/node; host it under a scron watchdog instead (unless we
        # already ARE that watchdog's child, which runs the chain below). See
        # pimm/launch/watchdog.py. Single-slot interactive stays foreground.
        from .watchdog import CHILD_ENV, install_watchdog

        if chain_jobs(cfg) > 1 and not os.environ.get(CHILD_ENV):
            return install_watchdog(
                cfg, run_name, remote_argv, dry_run=dry_run, output=output
            )
        # chain.jobs > 1 chains sequential salloc slots, each resuming the last:
        # the interactive analogue of batch requeue chaining (see run_interactive).
        attempts = build_attempts(cfg, run_name)
        commands = [
            (attempt, build_interactive_argv(cfg, run_name, attempt.script))
            for attempt in attempts
        ]
        if output:
            path = write_text(output, render_interactive_commands(commands))
            label = "command" if len(commands) == 1 else f"{len(commands)}-slot chain"
            print(f"# wrote interactive salloc {label}: {path}")
        if dry_run:
            rendered = render_interactive_commands(commands)
            print(rendered, end="" if rendered.endswith("\n") else "\n")
            return 0
        return run_interactive(cfg, run_name, commands)

    manifest = render_manifest(cfg, run_name, redact=True)
    if output:
        path = write_text(output, manifest)
        print(f"# wrote submitit manifest: {path}")
    if dry_run:
        print(manifest)
        return 0

    submit_cfg = cfg.get("submit") or {}
    result = (
        remote_submit(remote_argv, cfg)
        if submit_cfg.get("host") and not no_remote
        else submit(cfg, run_name)
    )
    if result:
        print(result, end="" if result.endswith("\n") else "\n")
    return 0
