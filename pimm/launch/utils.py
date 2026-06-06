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
    """Return local or slurm from the resolved site."""
    return "local" if str(cfg.get("site", "local")) == "local" else "slurm"


def chain_jobs(cfg: dict[str, Any]) -> int:
    """Return the validated number of submitit attempts."""
    jobs = int(cfg.get("chain", {}).get("jobs", 1) or 1)
    if jobs < 1:
        raise SystemExit("jobs must be >= 1")
    return jobs


def resources(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return normalized torchrun-style resource values."""
    raw = cfg.get("resources", {})
    return {
        "nnodes": int(raw.get("nnodes") or 1),
        "nproc_per_node": int(raw.get("nproc_per_node") or 1),
        "cpus_per_proc": int(raw.get("cpus_per_proc") or 1),
        "time": raw.get("time"),
        "mem": raw.get("mem"),
    }


def distributed_world_size(cfg: dict[str, Any]) -> int:
    """Return the number of ranks implied by the launcher resources."""
    res = resources(cfg)
    return res["nnodes"] * res["nproc_per_node"]


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

