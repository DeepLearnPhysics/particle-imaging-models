"""Top-level dispatcher for the small `pimm` command namespace."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Dispatch `pimm <command>` while keeping command imports lazy."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: pimm <command> [args]\n\n"
            "commands:\n"
            "  launch   run training locally or inside an allocation\n"
            "  submit   submit training to Slurm through submitit\n"
            "  ls       list configs\n"
            "  export   export pretrained artifacts"
        )
        return 0

    command = args.pop(0)
    if command == "launch":
        from .launch import main as launch_main

        return launch_main(args)
    if command == "submit":
        from .submit import main as submit_main

        return submit_main(args)
    if command == "ls":
        from .configs import main_ls

        return main_ls(args)
    if command == "export":
        from .export import main as export_main

        return export_main(args)

    raise SystemExit(f"Unknown pimm command: {command}")
