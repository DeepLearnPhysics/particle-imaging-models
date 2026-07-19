"""Temporary compatibility for the pre-0.6 launcher resource schema."""

from __future__ import annotations

import copy
from typing import Any

from pimm.utils.warnings import (
    PimmDeprecationWarning,
    deprecation_message,
    warn_deprecated,
    warn_once,
)


_REMOVE_IN = "0.6.0"
_LEGACY_RESOURCE_KEYS = {
    "account",
    "partition",
    "qos",
    "constraint",
    "dependency",
    "time",
    "mem",
    "output",
    "error",
    "gpu_directive",
    "job_name",
    "signal_delay_s",
}
_LEGACY_CONTAINER_KEYS = {"image", "module"}
_LEGACY_KEYS = _LEGACY_RESOURCE_KEYS | _LEGACY_CONTAINER_KEYS | {
    "additional_parameters"
}


def _mapping(value: Any, path: str, source: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must be a mapping in {source}")
    return dict(value)


def normalize_legacy_config(data: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Translate one legacy YAML/programmatic config layer to the canonical schema."""
    config = copy.deepcopy(data)
    if "slurm" not in config:
        return config

    legacy = _mapping(config.pop("slurm"), "slurm", source)
    unknown = sorted(set(legacy) - _LEGACY_KEYS)
    if unknown:
        raise SystemExit(
            f"Unknown legacy slurm setting(s) in {source}: {', '.join(unknown)}"
        )

    resources = _mapping(config.get("resources"), "resources", source)
    scheduler = resources.get("scheduler")
    if scheduler is None:
        resources["scheduler"] = "slurm"
    elif scheduler != "slurm":
        raise SystemExit(
            f"Conflicting launch settings in {source}: legacy slurm requires "
            f"resources.scheduler='slurm', got {scheduler!r}"
        )

    for key in sorted(_LEGACY_RESOURCE_KEYS):
        if key not in legacy:
            continue
        if key in resources:
            raise SystemExit(
                f"Conflicting launch settings in {source}: both slurm.{key} "
                f"and resources.{key} are set"
            )
        resources[key] = legacy[key]

    if "additional_parameters" in legacy:
        if "scheduler_options" in resources:
            raise SystemExit(
                f"Conflicting launch settings in {source}: both "
                "slurm.additional_parameters and resources.scheduler_options are set"
            )
        resources["scheduler_options"] = legacy["additional_parameters"]
    config["resources"] = resources

    if _LEGACY_CONTAINER_KEYS & legacy.keys():
        container = _mapping(config.get("container"), "container", source)
        for key in sorted(_LEGACY_CONTAINER_KEYS & legacy.keys()):
            if key in container and container[key] != legacy[key]:
                raise SystemExit(
                    f"Conflicting launch settings in {source}: slurm.{key} and "
                    f"container.{key} differ"
                )
            container[key] = legacy[key]
        config["container"] = container

    message = deprecation_message(
        f"The `slurm` launch configuration group in {source}",
        "the unified `resources` group",
        _REMOVE_IN,
    )
    warn_once(
        message,
        PimmDeprecationWarning,
        key=("legacy-launch-slurm", source),
        stacklevel=3,
    )
    return config


_LEGACY_CLI_TARGETS = {
    **{key.replace("_", "-"): f"resources.{key.replace('_', '-')}" for key in _LEGACY_RESOURCE_KEYS},
    "image": "container.image",
    "module": "container.module",
}


def normalize_legacy_cli(argv: list[str]) -> list[str]:
    """Rewrite supported ``--slurm.*`` options before Tyro parses the command."""
    separator = argv.index("--") if "--" in argv else len(argv)
    launcher_args = argv[:separator]
    training_args = argv[separator:]
    canonical_options = {
        arg.partition("=")[0][2:]
        for arg in launcher_args
        if arg.startswith("--") and not arg.startswith("--slurm.")
    }
    rewritten: list[str] = []
    legacy_targets: set[str] = set()
    used_legacy = False

    for arg in launcher_args:
        option, separator, value = arg.partition("=")
        if not option.startswith("--slurm."):
            rewritten.append(arg)
            continue

        legacy_name = option.removeprefix("--slurm.")
        target = _LEGACY_CLI_TARGETS.get(legacy_name)
        if target is None:
            raise SystemExit(f"Unknown legacy launcher option: {option}")
        if target in canonical_options or target in legacy_targets:
            raise SystemExit(
                f"Conflicting launcher options: {option} and --{target} "
                "refer to the same setting"
            )

        legacy_targets.add(target)
        used_legacy = True
        replacement = f"--{target}"
        rewritten.append(f"{replacement}={value}" if separator else replacement)

    if used_legacy:
        warn_deprecated(
            "`--slurm.*` launcher options",
            "`--resources.*` (or `--container.*` for image/module)",
            _REMOVE_IN,
            stacklevel=3,
        )
    return [*rewritten, *training_args]
