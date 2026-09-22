"""Experimental CLI adapter: translate pimm settings into ordinary exex jobs."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from .config import build_run_name
from .local import interpreter_args, redact_config, rendezvous_setup_lines
from .utils import (
    ROOT,
    chain_jobs,
    option_value,
    resources,
    scheduler,
    slurm_time_to_minutes,
)


def enabled() -> bool:
    value = os.environ.get("PIMM_USE_EXEX", "0")
    if value not in {"0", "1"}:
        raise SystemExit("PIMM_USE_EXEX must be 0 or 1")
    return value == "1"


def _executor(cfg, xc, options, staging):
    if scheduler(cfg) == "local":
        return xc.Local(container_options=options, workdir_root=str(staging / "work"))
    raw = cfg["resources"]
    res = resources(cfg)
    native = {
        key.replace("_", "-"): raw[key]
        for key in (
            "account",
            "partition",
            "qos",
            "constraint",
            "dependency",
            "mem",
            "time_min",
            "job_name",
        )
        if raw.get(key) is not None
    }
    native.update(
        {
            key.replace("_", "-"): value
            for key, value in raw.get("scheduler_options", {}).items()
        }
    )
    nodes = res["nnodes"]
    native.update(nodes=nodes, ntasks=nodes)
    native["cpus-per-task"] = res["cpus_per_proc"] * res["nproc_per_node"]
    gpu_key = raw["gpu_directive"]
    native[gpu_key] = (
        f"gpu:{res['nproc_per_node']}" if gpu_key == "gres" else res["nproc_per_node"]
    )
    return xc.Slurm(
        cluster=cfg["site"],
        mode="salloc" if cfg.get("interactive") else "sbatch",
        resources=native,
        walltime=slurm_time_to_minutes(raw["time"]) * 60,
        workdir_root=str(staging / "work"),
        container_options=options,
        srun_options=[
            f"--nodes={nodes}",
            f"--ntasks={nodes}",
            "--ntasks-per-node=1",
            "--kill-on-bad-exit=1",
            "--overlap",
            f"--cpus-per-task={native['cpus-per-task']}",
            f"--{gpu_key}={native[gpu_key]}",
        ]
        if nodes > 1
        else None,
    )


def _container(cfg, source, xc):
    container = cfg["container"]
    runtime = container["runtime"]
    bind = {}
    for item in container.get("binds", []):
        src, _, dst = item.partition(":")
        bind[src] = dst or src
    if runtime in {"singularity", "apptainer"}:
        return xc.SingularityContainer(
            source, container["image"]
        ), xc.SingularityOptions(bind=bind)
    if runtime == "shifter":
        modules = (container.get("module") or "").split(",")
        return xc.ShifterContainer(source, container["image"]), xc.ShifterOptions(
            bind=bind, modules=list(filter(None, modules)), extra_options=["--clearenv"]
        )
    if runtime == "docker":
        return xc.DockerContainer(source, container["image"]), xc.DockerOptions(
            volumes=bind
        )
    return source, None


def _plan(cfg):
    from exex import xm_cluster as xc

    res = resources(cfg)
    count = chain_jobs(cfg)
    if (
        scheduler(cfg) not in {"local", "slurm"}
        or (scheduler(cfg) == "local" and res["nnodes"] != 1)
        or (scheduler(cfg) == "slurm" and res["nproc_per_node"] == "auto")
    ):
        raise SystemExit(
            "PIMM_USE_EXEX supports single-node Local or Slurm with an explicit GPU count"
        )
    if count > 1 and scheduler(cfg) != "slurm":
        raise SystemExit("Exex continuation requires Slurm; local jobs run once")
    if cfg.get("submit", {}).get("secret_env"):
        raise SystemExit(
            "Exex does not translate submit.secret_env; configure credentials in the execution environment"
        )
    if cfg.get("watchdog", {}) not in (
        {},
        {"qos": "cron", "interval_min": 5, "time": "1-00:00:00"},
    ):
        raise SystemExit(
            "The exex path has no watchdog; salloc requires a live launcher"
        )

    name = build_run_name(cfg, cfg["timestamp"])
    train = cfg["train"]
    config_path = Path(train["config"])
    config_path = (
        config_path if config_path.is_absolute() else ROOT / "configs" / config_path
    )
    config_path = config_path.with_suffix(".py")
    relative_config = config_path.resolve().relative_to(ROOT.resolve())
    if not config_path.is_file():
        raise SystemExit(f"Training config not found: {config_path}")
    submit = cfg.get("submit", {})
    staging = Path(submit.get("folder") or (Path(cfg["paths"]["exp_root"]) / ".exex"))
    host = submit.get("host") if scheduler(cfg) == "slurm" else None
    if host and not staging.is_absolute():
        raise SystemExit(
            "Remote exex submission requires an absolute --paths.exp-root or --submit.folder on the target"
        )
    local_storage = ROOT / ".exex"
    config = xc.Config(
        {
            "local": {"storage": {"staging": str(local_storage)}},
            "clusters": [
                {
                    "name": cfg["site"],
                    "server": host,
                    "storage": {"staging": str(staging)},
                }
            ],
        },
        project="pimm",
    )
    python_args = interpreter_args(cfg)
    python = python_args[1] if python_args else "python3"
    # A relative interpreter belongs to the prepared runtime, not the unpacked source.
    if "/" in python and not Path(python).is_absolute():
        if host:
            raise SystemExit(
                "Remote train.python must be absolute on the execution host"
            )
        python = str(Path(python).resolve())
    probe = shlex.join(
        [
            python,
            "-c",
            f"from pimm.launch.local import local_master_port; print(local_master_port({name!r}))",
        ]
    )
    rendezvous = cfg.get("rdzv", {})
    nproc = "gpu" if res["nproc_per_node"] == "auto" else res["nproc_per_node"]
    # Slurm's worker launch originates on the first allocated compute node.
    # Its address is available inside containers without installing scontrol.
    rendezvous_commands = (
        rendezvous_setup_lines(cfg, name, slurm_master="${SLURM_LAUNCH_NODE_IPADDR:?}")
        if res["nnodes"] > 1
        else [
            'export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"',
            f'MASTER_PORT="${{MASTER_PORT:-$({probe})}}"',
            "export MASTER_PORT",
        ]
    )
    commands = [
        "set -eo pipefail",
        *cfg.get("setup", []),
        'export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"',
        *rendezvous_commands,
        shlex.join(
            [
                "export",
                f"PIMM_RDZV_BACKEND={rendezvous.get('backend') or 'static'}",
                f"PIMM_RDZV_ID={rendezvous.get('id') or name}",
            ]
        ),
        'set -- "$@" "save_path=$EXEX_OUTPUT_DIR/checkpoint"',
        "exec "
        + shlex.join(
            [
                "sh",
                "pimm/launch/torchrun.sh",
                python,
                str(res["nnodes"]),
                str(nproc),
                "--module",
                "pimm.train",
            ]
        )
        + ' "$@"',
    ]
    listing = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "pimm",
            "configs",
        ]
    )
    files = sorted(
        {
            os.fsdecode(item)
            for item in listing.split(b"\0")
            if item and (ROOT / os.fsdecode(item)).is_file()
        }
    )
    source = xc.SourceTree(xc.CommandList(commands), path=ROOT, files=files)
    package, options = _container(cfg, source, xc)
    executor = _executor(cfg, xc, options, staging)
    overrides = {
        "wandb_run_name": cfg["run"].get("wandb_name") or name,
        **train.get("options", {}),
    }
    if train.get("weight"):
        overrides["weight"] = train["weight"]
    if train.get("resume") or cfg.get("chain", {}).get("resume_first"):
        overrides["resume"] = True
    if overrides.get("resume") and not overrides.get("weight"):
        raise SystemExit("Exex requires --train.weight for the initial resume")
    overrides.pop("save_path", None)  # Owned by the attempt, not by a reused run name.
    args = [
        "--config-file",
        str(relative_config),
        "--options",
        *[f"{key}={option_value(value)}" for key, value in overrides.items()],
    ]
    env = {
        key: str(value)
        for key, value in cfg.get("env", {}).items()
        if value is not None
    }
    env.update(PIMM_USE_EXEX="1", OMP_NUM_THREADS=str(res["cpus_per_proc"]))
    continuation = (
        xc.Continuation(
            checkpoint="checkpoint",
            max_attempts=count,
            pause_before=cfg["resources"]["signal_delay_s"],
        )
        if count > 1
        else None
    )
    if continuation and continuation.pause_before >= executor.walltime.total_seconds():
        raise SystemExit("resources.signal_delay_s must be shorter than resources.time")
    return name, config, package, executor, args, env, continuation


def run(cfg: dict, *, dry_run: bool = False, output: str | None = None) -> int:
    """Run the opt-in path; dry-run prints intent without staging or submission."""
    import attr
    from exex import xm
    from exex import xm_cluster as xc

    name, config, package, executor, args, env, continuation = _plan(cfg)
    source = package if isinstance(package, xc.SourceTree) else package.entrypoint
    preview = json.dumps(
        redact_config(
            {
                "name": name,
                "executor": type(executor).__name__,
                "settings": attr.asdict(executor),
                "package": {
                    "kind": type(package).__name__,
                    "source": source.path,
                    "file_count": len(source.files),
                    "commands": source.entrypoint.commands,
                    "image": getattr(
                        package, "image", getattr(package, "image_path", None)
                    ),
                },
                "host": cfg.get("submit", {}).get("host")
                if scheduler(cfg) == "slurm"
                else None,
                "args": args,
                "env": env,
                "continuation": vars(continuation) if continuation else None,
                "outputs": {"checkpoint": "checkpoint"},
            }
        ),
        indent=2,
        default=str,
    )
    if output:
        Path(output).write_text(preview + "\n")
    if dry_run:
        print(preview)
        return 0
    with xc.create_experiment(name, config=config, project="pimm") as experiment:
        [executable] = experiment.package([xm.Packageable(package, executor.Spec())])
        experiment.add(
            xm.Job(executable, executor, args=args, env_vars=env),
            outputs={"checkpoint": "checkpoint"},
            continuation=continuation,
        )
        print(
            f"exex experiment {experiment.experiment_id}; catalog: {ROOT / '.exex'}",
            flush=True,
        )
    return 0
