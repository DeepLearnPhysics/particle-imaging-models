"""Top-level dispatcher for the small `pimm` command namespace."""

from __future__ import annotations

import sys
from pathlib import Path


def list_configs(args: list[str]) -> int:
    """List training configs below an optional directory."""
    from pimm.launch.utils import ROOT

    configs = ROOT / "configs"
    path = (configs / Path(*args)).resolve()
    if not path.is_relative_to(configs.resolve()) or not path.is_dir():
        raise SystemExit(f"Unknown config directory: {'/'.join(args)}")
    for config in sorted(path.rglob("*.py")):
        if not config.name.startswith("_"):
            print(config.relative_to(configs).with_suffix("").as_posix())
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch `pimm <command>` while keeping command imports lazy."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: pimm <command> [args]\n\n"
            "commands:\n"
            "  ls       list training configs\n"
            "  launch   run training locally or inside an allocation\n"
            "  submit   submit training to Slurm through submitit\n"
            "  watchdog manage supervisors for chained interactive runs\n"
            "  export   export model weights (optionally push to the HF Hub)"
        )
        return 0

    command = args.pop(0)
    if command == "ls":
        return list_configs(args)
    if command == "launch":
        from .launch import main as launch_main

        return launch_main(args)
    if command == "submit":
        from .submit import main as submit_main

        return submit_main(args)
    if command == "watchdog":
        from .watchdog import main as watchdog_main

        return watchdog_main(args)
    if command == "export":
        from .export import main as export_main

        return export_main(args)

    raise SystemExit(f"Unknown pimm command: {command}")
