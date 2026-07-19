"""Shared helpers for Tyro CLI launch commands."""

from __future__ import annotations

from typing import TypeVar

import tyro

from pimm.launch.compat import normalize_legacy_cli
from pimm.launch.config import load_config
from pimm.launch.schema import LaunchCommand, LaunchConfig

CommandT = TypeVar("CommandT", bound=LaunchCommand)
SECRET_OPTION_PARTS = ("api-key", "token", "secret", "password", "passwd", "credential")


def redact_cli_argv(argv: list[str]) -> list[str]:
    """Replace secret CLI values before recording a reproducible command."""
    result = []
    redact_next = False
    for arg in argv:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        option, separator, _ = arg.partition("=")
        normalized = option.lower().replace("_", "-")
        if option.startswith("--") and any(
            part in normalized for part in SECRET_OPTION_PARTS
        ):
            result.append(f"{option}=<redacted>" if separator else option)
            redact_next = not separator
        else:
            result.append(arg)
    return result


def split_training_tail(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split Tyro launcher args from schema-free training config overrides."""
    if "--" not in argv:
        return argv, []
    index = argv.index("--")
    return argv[:index], argv[index + 1 :]


def bootstrap_value(argv: list[str], name: str) -> str | None:
    """Read a simple `--name value` or `--name=value` selector before Tyro."""
    prefix = f"--{name}="
    for index, arg in enumerate(argv):
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
        if arg == f"--{name}" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def wants_help(argv: list[str]) -> bool:
    """Return whether this invocation should show Tyro help before validation."""
    return not argv or any(arg in {"-h", "--help"} for arg in argv)


def parse_typed_command(
    command_type: type[CommandT],
    argv: list[str],
    *,
    launch_timestamp: str,
    prog: str,
    require_site: bool,
    default_site: str = "local",
) -> tuple[CommandT, list[str]]:
    """Parse a Tyro command after loading site/recipe defaults."""
    tyro_args, training_overrides = split_training_tail(argv)
    tyro_args = normalize_legacy_cli(tyro_args)
    if not tyro_args:
        tyro_args = ["--help"]
    site = bootstrap_value(tyro_args, "site") or default_site
    has_explicit_site = bootstrap_value(tyro_args, "site") is not None
    recipe = bootstrap_value(tyro_args, "recipe")
    base_cfg = load_config(site=site, recipe=recipe, launch_timestamp=launch_timestamp)
    default = command_type.from_config(LaunchConfig.from_dict(base_cfg), recipe=recipe)
    command = tyro.cli(
        command_type,
        args=tyro_args,
        default=default,
        prog=prog,
    )
    if require_site and not has_explicit_site and not wants_help(tyro_args):
        raise SystemExit(f"{prog} requires --site, e.g. --site s3df")
    return command, training_overrides
