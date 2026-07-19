"""Shared launch utilities and lightweight resource normalization."""

from __future__ import annotations

import datetime as dt
import os
import re
import shlex
from pathlib import Path
from typing import Any

import yaml


def find_repo_root() -> Path:
    """Locate the checkout root from cwd or this module path."""
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "launch" / "defaults.yaml").is_file():
            return candidate
    return Path(__file__).resolve().parents[2]


ROOT = find_repo_root()
LAUNCH_DIR = ROOT / "launch"
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.]*)\}")


def timestamp() -> str:
    """Return the launch timestamp format used for experiment names."""
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def as_bool(value: Any) -> bool:
    """Parse bool-like launcher values from YAML or CLI input."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def parse_value(raw: str) -> Any:
    """Parse CLI override values using YAML scalar/list/map syntax when valid."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def option_value(value: Any) -> str:
    """Format a training option for the shell `--options` interface."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "None"
    return str(value)


def shell_join(parts: list[Any]) -> str:
    """Quote command parts after dropping empty optional arguments."""
    return shlex.join(str(part) for part in parts if part is not None and part != "")


def scheduler(cfg: dict[str, Any]) -> str:
    """Return the scheduler used by the current launcher command."""
    configured = cfg.get("resources", {}).get("scheduler")
    if configured not in {"local", "slurm"}:
        raise SystemExit(
            "resources.scheduler must be 'local' or 'slurm', "
            f"got {configured!r}"
        )
    executor = cfg.get("executor")
    if executor is None:
        return configured
    if executor not in {"local", "batch", "interactive"}:
        raise SystemExit(
            "executor must be 'local', 'batch', or 'interactive', "
            f"got {executor!r}"
        )
    return "local" if executor == "local" else configured


def nproc_is_auto(cfg: dict[str, Any]) -> bool:
    """Return whether GPUs/node should be auto-detected (local only)."""
    return str(cfg.get("resources", {}).get("nproc_per_node")).lower() == "auto"


def chain_jobs(cfg: dict[str, Any]) -> int:
    """Return the validated number of submitit attempts."""
    jobs = int(cfg.get("chain", {}).get("jobs", 1) or 1)
    if jobs < 1:
        raise SystemExit("jobs must be >= 1")
    return jobs


def resources(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return normalized torchrun-style resource values.

    `nproc_per_node` may be the string "auto" (local only) meaning "let train.sh
    detect all visible GPUs"; it is preserved as "auto" rather than coerced.
    """
    raw = cfg.get("resources", {})
    if not isinstance(raw, dict):
        raise SystemExit("resources must be a mapping")

    def positive_int(key: str) -> int:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SystemExit(f"resources.{key} must be a positive integer")
        return value

    nproc = raw.get("nproc_per_node")
    if nproc == "auto":
        nproc_value: Any = "auto"
    else:
        nproc_value = positive_int("nproc_per_node")
    return {
        "nnodes": positive_int("nnodes"),
        "nproc_per_node": nproc_value,
        "cpus_per_proc": positive_int("cpus_per_proc"),
    }


def distributed_world_size(cfg: dict[str, Any]) -> int:
    """Return the number of ranks implied by the launcher resources.

    With `nproc_per_node: auto` (local) the count is unknown at render time, so
    GPUs/node is treated as 1 for launcher-side math (DCP defaults, batch-size
    divisibility); the engine validates the real device count at runtime.
    """
    res = resources(cfg)
    nproc = res["nproc_per_node"]
    nproc = 1 if nproc == "auto" else int(nproc)
    return res["nnodes"] * nproc


def slurm_time_to_minutes(value: Any) -> int:
    """Convert Slurm time strings into submitit's timeout_min integer."""
    if isinstance(value, int):
        return value
    text = str(value)
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 1:
        hours = 0
        minutes = parts[0]
        seconds = 0
    else:
        raise SystemExit(f"Invalid Slurm time format: {value}")
    total = days * 24 * 60 + hours * 60 + minutes
    if seconds:
        total += 1
    return max(total, 1)


def write_text(path: str, text: str) -> Path:
    """Write rendered launch output and create parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path
