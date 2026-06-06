#!/usr/bin/env python3
"""Run pimm training locally or inside an existing allocation."""

from __future__ import annotations

import sys

from pimm.cli.common import parse_typed_command
from pimm.launch.config import finalize_config
from pimm.launch.local import launch
from pimm.launch.schema import LaunchCommand
from pimm.launch.utils import timestamp


def parse_command(argv: list[str], launch_timestamp: str) -> tuple[LaunchCommand, list[str]]:
    """Parse typed `pimm launch` args with defaults from site and recipe YAML."""
    return parse_typed_command(
        LaunchCommand,
        argv,
        launch_timestamp=launch_timestamp,
        prog="pimm launch",
        require_site=False,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `pimm launch`."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    launch_timestamp = timestamp()
    command, training_overrides = parse_command(raw_argv, launch_timestamp)
    cfg = finalize_config(
        command.launch_config_dict(),
        launch_timestamp=launch_timestamp,
        require_config=True,
        training_overrides=training_overrides,
    )
    return launch(
        cfg,
        launch_timestamp=launch_timestamp,
        dry_run=command.dry_run,
        output=command.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
