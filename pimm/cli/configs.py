"""Filesystem-based config discovery commands for the pimm CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pimm.launch.utils import ROOT


CONFIG_ROOT = ROOT / "configs"


def iter_config_files(prefix: str | None = None) -> list[Path]:
    """Return sorted config files under `configs/`, optionally filtered by prefix."""
    root = CONFIG_ROOT
    if prefix:
        prefix_path = CONFIG_ROOT / prefix.strip("/")
        root = prefix_path
        if root.is_file():
            root = root.parent
        if not root.exists():
            raise SystemExit(f"No config directory: {root}")
    return sorted(
        path
        for path in root.rglob("*.py")
        if "_base_" not in path.parts and not path.name.startswith("__")
    )


def rel_config(path: Path) -> str:
    """Return a config reference relative to `configs/` without `.py`."""
    return path.relative_to(CONFIG_ROOT).with_suffix("").as_posix()


def main_ls(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pimm ls", description="List pimm configs.")
    parser.add_argument("prefix", nargs="?")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    for path in iter_config_files(args.prefix):
        print(rel_config(path))
    return 0
