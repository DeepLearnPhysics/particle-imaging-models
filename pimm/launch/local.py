"""Local execution for `pimm launch`."""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import build_run_name, validate_launch_config, validate_training_config
from .utils import ROOT, as_bool, option_value, resources, scheduler, shell_join, write_text


REDACTED = "<redacted>"
SECRET_KEY_RE = re.compile(r"(api[_-]?key|token|secret|password|passwd|credential)", re.I)


def is_secret_key(key: str) -> bool:
    """Return whether a config/env key should not be rendered with its value."""
    return bool(SECRET_KEY_RE.search(str(key)))


def redact_config(data: Any) -> Any:
    """Return a copy with secret-looking mapping values replaced."""
    if isinstance(data, dict):
        return {
            key: REDACTED if is_secret_key(str(key)) else redact_config(value)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [redact_config(item) for item in data]
    if isinstance(data, tuple):
        return tuple(redact_config(item) for item in data)
    return data


def redact_script(script: str) -> str:
    """Redact secret-looking shell exports in rendered scripts."""
    redacted_lines = []
    for line in script.splitlines():
        match = re.match(r"^(export\s+([A-Za-z_][A-Za-z0-9_]*)=).*$", line)
        if match and is_secret_key(match.group(2)):
            redacted_lines.append(f"{match.group(1)}{shlex.quote(REDACTED)}")
        else:
            redacted_lines.append(line)
    return "\n".join(redacted_lines)


def host_repo_root(cfg: dict[str, Any]) -> str:
    """Return the repository path as seen by the host scheduler shell."""
    return str(cfg.get("paths", {}).get("repo_root", ROOT))


def container_repo_root(cfg: dict[str, Any]) -> str:
    """Return the repository path that commands should use inside a container."""
    container = cfg.get("container", {})
    runtime = container.get("runtime")
    repo_mount = container.get("repo_mount")
    if runtime in {"singularity", "shifter"} and repo_mount:
        return str(repo_mount)
    return host_repo_root(cfg)


def repo_bind(cfg: dict[str, Any]) -> str | None:
    """Return the host-to-container repo bind spec for editable pimm installs."""
    container = cfg.get("container", {})
    runtime = container.get("runtime")
    repo_mount = container.get("repo_mount")
    if runtime not in {"singularity", "shifter"} or not repo_mount:
        return None
    host_root_path = Path(host_repo_root(cfg)).expanduser()
    host_root = str(
        host_root_path if host_root_path.is_absolute() else host_root_path.resolve()
    )
    if host_root == str(repo_mount):
        return None
    return f"{host_root}:{repo_mount}"


def bind_specs(cfg: dict[str, Any]) -> list[str]:
    """Return user configured binds plus the editable-install repo bind."""
    specs = [str(bind) for bind in cfg.get("container", {}).get("binds") or []]
    repo_spec = repo_bind(cfg)
    if repo_spec and repo_spec not in specs:
        specs.append(repo_spec)
    return specs


def shifter_volume_spec(bind: str) -> str:
    """Convert a bind entry into Shifter's source:target volume syntax."""
    if ":" in bind:
        return bind
    return f"{bind}:{bind}"


def build_train_sh_command(cfg: dict[str, Any], run_name: str) -> str:
    """Build the `scripts/train.sh` invocation for this resolved launch config."""
    train_cfg = cfg.get("train", {})
    config = train_cfg.get("config")
    if not config:
        raise SystemExit("Need --train.config")

    repo_root = container_repo_root(cfg)
    res = resources(cfg)
    parts: list[Any] = [
        "sh",
        f"{repo_root}/scripts/train.sh",
        "-m",
        res["nnodes"],
        "-g",
        res["nproc_per_node"],
        "-c",
        config,
        "-n",
        run_name,
    ]

    python = train_cfg.get("python")
    if python is None and scheduler(cfg) == "local":
        python = sys.executable
    if python:
        parts += ["-p", python]

    wandb_name = cfg.get("run", {}).get("wandb_name") or run_name
    if wandb_name:
        parts += ["-a", wandb_name]
    if train_cfg.get("weight"):
        parts += ["-w", train_cfg["weight"]]
    if as_bool(train_cfg.get("resume", False)):
        parts += ["-r", "true"]
    if as_bool(train_cfg.get("no_code_copy", False)):
        parts += ["-C"]

    options = train_cfg.get("options") or {}
    if options:
        parts += ["--", "--options"]
        for key, value in options.items():
            if value is not None:
                parts.append(f"{key}={option_value(value)}")

    return shell_join(parts)


def build_container_command(cfg: dict[str, Any], train_cmd: str) -> str:
    """Wrap the train.sh command with site setup and the configured container."""
    container = cfg.get("container", {})
    runtime = container.get("runtime")
    setup = list(container.get("setup") or [])
    setup.append(
        "export MASTER_ADDR MASTER_PORT "
        f"OMP_NUM_THREADS={resources(cfg)['cpus_per_proc']}"
    )
    inner_cmd = "\n".join([*setup, train_cmd])

    if runtime == "singularity":
        parts: list[Any] = ["singularity", "run", "--nv"]
        binds = bind_specs(cfg)
        if binds:
            parts += ["-B", ",".join(str(bind) for bind in binds)]
        parts += [container["image"], "bash", "-lc", inner_cmd]
        return shell_join(parts)

    if runtime == "shifter":
        parts = ["env"]
        for var in container.get("unset_env") or []:
            parts += ["-u", var]
        parts.append("shifter")
        if container.get("module"):
            parts.append(f"--module={container['module']}")
        if container.get("image"):
            parts.append(f"--image={container['image']}")
        volumes = [shifter_volume_spec(bind) for bind in bind_specs(cfg)]
        if volumes:
            parts.append(f"--volume={';'.join(volumes)}")
        parts += ["/bin/bash", "-lc", inner_cmd]
        return shell_join(parts)

    if runtime in {None, "none"}:
        return inner_cmd

    raise SystemExit(f"Unsupported container.runtime: {runtime}")


def rendezvous_setup_lines(cfg: dict[str, Any]) -> list[str]:
    """Return outer-shell rendezvous setup before entering any container."""
    if scheduler(cfg) == "slurm":
        return [
            'MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:-${SLURM_NODELIST:?missing Slurm nodelist}}" | head -n 1)}',
            "MASTER_PORT=${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 10000))}",
            "export MASTER_ADDR MASTER_PORT",
        ]
    return [
        "MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}",
        "MASTER_PORT=${MASTER_PORT:-29500}",
        "export MASTER_ADDR MASTER_PORT",
    ]


def render_script(cfg: dict[str, Any], train_cmd: str) -> str:
    """Render the small shell wrapper around `scripts/train.sh`."""
    repo_root = host_repo_root(cfg)
    env = cfg.get("env") or {}
    exports = [
        f"export {key}={shlex.quote(str(value))}"
        for key, value in env.items()
        if value is not None
    ]
    if cfg.get("paths", {}).get("exp_root"):
        exports.append(f"export EXP_ROOT={shlex.quote(str(cfg['paths']['exp_root']))}")

    lines = [
        "#!/bin/bash",
        "",
        "set -euo pipefail",
        "",
    ]
    if scheduler(cfg) == "slurm":
        lines.append("mkdir -p slurm_logs")
    lines.extend(
        [
            *exports,
            *rendezvous_setup_lines(cfg),
            f"cd {shlex.quote(str(repo_root))}",
            "",
            build_container_command(cfg, train_cmd),
            "",
        ]
    )
    return "\n".join(lines)


def render_launch_script(cfg: dict[str, Any], launch_timestamp: str) -> tuple[str, str]:
    """Return `(run_name, script)` for local execution."""
    run_name = build_run_name(cfg, launch_timestamp)
    if not run_name:
        raise SystemExit("Could not determine run name")
    train_cmd = build_train_sh_command(cfg, run_name)
    return run_name, render_script(cfg, train_cmd)


def run_script(script: str, cfg: dict[str, Any]) -> int:
    """Execute a rendered local launch script."""
    if scheduler(cfg) != "local":
        raise SystemExit("pimm launch runs locally; use `pimm submit` for Slurm.")
    repo_root = Path(str(cfg.get("paths", {}).get("repo_root", ROOT)))
    cwd = repo_root if repo_root.exists() else ROOT
    return subprocess.run(["bash", "-lc", script], cwd=cwd).returncode


def launch(
    cfg: dict[str, Any],
    *,
    launch_timestamp: str,
    dry_run: bool,
    output: str | None,
) -> int:
    """Render and optionally execute a local launch."""
    validate_launch_config(cfg)
    validate_training_config(cfg)
    _, script = render_launch_script(cfg, launch_timestamp)
    display_script = redact_script(script)

    if output:
        path = write_text(output, display_script)
        print(f"# wrote script: {path}")
    if dry_run:
        print(display_script)
        return 0
    return run_script(script, cfg)
