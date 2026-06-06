#!/usr/bin/env python3
"""Submit pimm training to Slurm through submitit."""

from __future__ import annotations

import sys

from pimm.cli.common import parse_typed_command
from pimm.launch.config import finalize_config
from pimm.launch.schema import SubmitCommand
from pimm.launch.utils import timestamp
from pimm.launch.submit import run_submit


def parse_command(argv: list[str], launch_timestamp: str) -> tuple[SubmitCommand, list[str]]:
    """Parse typed `pimm submit` args with defaults from site and recipe YAML."""
    return parse_typed_command(
        SubmitCommand,
        argv,
        launch_timestamp=launch_timestamp,
        prog="pimm submit",
        require_site=True,
        default_site="s3df",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `pimm submit`."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    launch_timestamp = timestamp()
    command, training_overrides = parse_command(raw_argv, launch_timestamp)
    cfg = finalize_config(
        command.launch_config_dict(),
        launch_timestamp=launch_timestamp,
        require_config=True,
        training_overrides=training_overrides,
    )
    return run_submit(
        cfg,
        launch_timestamp=launch_timestamp,
        dry_run=command.dry_run,
        output=command.output,
        remote_argv=raw_argv,
        no_remote=command.no_remote,
    )


if __name__ == "__main__":
    raise SystemExit(main())
