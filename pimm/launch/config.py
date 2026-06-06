"""Launch config loading, placeholder resolution, and CLI override merging."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .utils import (
    LAUNCH_DIR,
    PLACEHOLDER_RE,
    ROOT,
    as_bool,
    chain_jobs,
    distributed_world_size,
    parse_value,
    scheduler,
)


def load_yaml(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML mapping from disk, recursively applying optional `_base_` files."""
    path = path.resolve()
    seen = set(_seen or set())
    if path in seen:
        raise SystemExit(f"Cycle in launch config _base_ chain at: {path}")
    seen.add(path)
    if not path.exists():
        raise SystemExit(f"Missing launch config: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Launch config must be a mapping: {path}")

    base_spec = data.pop("_base_", None)
    if base_spec is None:
        return data

    base_items = base_spec if isinstance(base_spec, list) else [base_spec]
    merged: dict[str, Any] = {}
    for item in base_items:
        base_path = Path(str(item))
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = merge_dicts(merged, load_yaml(base_path, seen))
    return merge_dicts(merged, data)


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay mappings without mutating either input."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def set_path(cfg: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a dotted launcher config path."""
    cur: dict[str, Any] = cfg
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        next_value = cur.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise SystemExit(f"Cannot set {dotted_path}: {part} is not a mapping")
        cur = next_value
    cur[parts[-1]] = value


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings into dotted placeholder keys."""
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out[path] = value
            out.update(flatten(value, path))
    return out


def format_string(value: str, context: dict[str, Any]) -> str:
    """Expand `{path.to.value}` placeholders against a flattened context."""

    def repl(match):
        key = match.group(1)
        if key not in context:
            raise SystemExit(f"Unknown placeholder {{{key}}} in launch config")
        return str(context[key])

    return PLACEHOLDER_RE.sub(repl, value)


def resolve_placeholders(data: Any, context: dict[str, Any]) -> Any:
    """Resolve placeholders recursively in strings, lists, and mappings."""
    if isinstance(data, str):
        return format_string(data, context)
    if isinstance(data, list):
        return [resolve_placeholders(item, context) for item in data]
    if isinstance(data, dict):
        return {key: resolve_placeholders(value, context) for key, value in data.items()}
    return data


def resolve_all(cfg: dict[str, Any], launch_timestamp: str) -> dict[str, Any]:
    """Resolve timestamp and nested placeholders with bounded fixed-point passes."""
    cfg = copy.deepcopy(cfg)
    cfg["timestamp"] = launch_timestamp
    for _ in range(6):
        context = flatten(cfg)
        for key, value in cfg.get("paths", {}).items():
            context[key] = value
        context["repo_root"] = cfg.get("paths", {}).get("repo_root", "")
        new_cfg = resolve_placeholders(cfg, context)
        if new_cfg == cfg:
            return new_cfg
        cfg = new_cfg
    return cfg


def normalize_config_path(config: str | None) -> str | None:
    """Convert config file references into paths relative to `configs/`."""
    if not config:
        return config
    path = Path(str(config))
    if path.suffix == ".py":
        path = path.with_suffix("")
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT / "configs")
        except ValueError:
            return path.as_posix()
    else:
        parts = path.parts
        if parts and parts[0] == "configs":
            path = Path(*parts[1:])
    return path.as_posix()


def normalize_train_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Normalize the training config path while preserving the input mapping."""
    cfg = copy.deepcopy(cfg)
    train_cfg = cfg.setdefault("train", {})
    config = normalize_config_path(train_cfg.get("config"))
    if config is not None:
        train_cfg["config"] = config
    return cfg


def build_run_name(cfg: dict[str, Any], launch_timestamp: str) -> str | None:
    """Derive the experiment/run name and append the launch timestamp when enabled."""
    run_cfg = cfg.get("run", {})
    train_cfg = cfg.get("train", {})
    name = run_cfg.get("name") or train_cfg.get("config")
    if not name:
        return None
    name = Path(str(name)).name
    if as_bool(run_cfg.get("timestamp", True)) and launch_timestamp not in name:
        name = f"{name}-{launch_timestamp}"
    return name


def uses_fsdp2(cfg: dict[str, Any]) -> bool:
    """Return whether launch-time training options request FSDP2."""
    options = cfg.get("train", {}).get("options") or {}
    strategy = options.get("parallel.strategy", options.get("distributed.strategy"))
    return str(strategy).lower() == "fsdp2"


def has_explicit_checkpoint_backend(cfg: dict[str, Any]) -> bool:
    """Return whether the user/recipe explicitly chose a checkpoint backend."""
    options = cfg.get("train", {}).get("options") or {}
    for key in options:
        if not key.startswith("hooks."):
            continue
        parts = key.split(".")
        if len(parts) >= 3 and parts[-1] == "backend":
            return True
    return False


def apply_checkpoint_backend_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Default serious launch/resume paths to DCP unless explicitly overridden."""
    cfg = copy.deepcopy(cfg)
    should_use_dcp = distributed_world_size(cfg) > 1 or chain_jobs(cfg) > 1 or uses_fsdp2(cfg)
    if should_use_dcp and not has_explicit_checkpoint_backend(cfg):
        cfg.setdefault("train", {}).setdefault("options", {})[
            "hooks.CheckpointSaverIteration.backend"
        ] = "dcp"
    return cfg


def parse_training_overrides(overrides: list[str] | None) -> dict[str, Any]:
    """Parse training `key=value` overrides from the CLI tail or `--train`."""
    parsed: dict[str, Any] = {}
    for item in overrides or []:
        if item.startswith("--") or "=" not in item:
            raise SystemExit(
                "Training overrides must be KEY=VALUE arguments after `--` "
                "or passed with --train; "
                f"got: {item}"
            )
        key, raw_value = item.split("=", 1)
        parsed[key] = parse_value(raw_value)
    return parsed


def load_config(
    *,
    site: str,
    recipe: str | None,
    launch_timestamp: str,
) -> dict[str, Any]:
    """Load defaults, site overlay, and optional recipe YAML."""
    cfg = load_yaml(LAUNCH_DIR / "defaults.yaml")
    recipe_cfg: dict[str, Any] = {}
    if recipe:
        recipe_path = Path(recipe)
        if not recipe_path.is_absolute():
            recipe_path = ROOT / recipe_path
        recipe_cfg = load_yaml(recipe_path)

    resolved_site = site or recipe_cfg.get("site") or "local"
    site_cfg = load_yaml(LAUNCH_DIR / "sites" / f"{resolved_site}.yaml")
    cfg = merge_dicts(cfg, site_cfg)
    cfg = merge_dicts(cfg, recipe_cfg)
    cfg["site"] = resolved_site
    return cfg


def finalize_config(
    cfg: dict[str, Any],
    *,
    launch_timestamp: str,
    require_config: bool,
    training_overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Normalize a Tyro-parsed launch config and apply dynamic train overrides."""
    cfg = copy.deepcopy(cfg)
    train_cfg = cfg.setdefault("train", {})
    train_cfg.setdefault("options", {}).update(parse_training_overrides(training_overrides))

    run_cfg = cfg.setdefault("run", {})
    if run_cfg.get("wandb_project"):
        train_cfg.setdefault("options", {})["wandb_project"] = run_cfg["wandb_project"]
    if run_cfg.get("wandb_api_key"):
        cfg.setdefault("env", {})["WANDB_API_KEY"] = run_cfg["wandb_api_key"]

    rdzv_cfg = cfg.get("rdzv") or {}
    env = cfg.setdefault("env", {})
    endpoint = rdzv_cfg.get("endpoint")
    if endpoint:
        host_part, sep, port_part = str(endpoint).partition(":")
        if not sep or not host_part or not port_part:
            raise SystemExit("--rdzv.endpoint must be HOST:PORT")
        env["MASTER_ADDR"] = host_part
        env["MASTER_PORT"] = port_part
    if rdzv_cfg.get("id"):
        env["PIMM_RDZV_ID"] = rdzv_cfg["id"]
    if rdzv_cfg.get("backend"):
        env["PIMM_RDZV_BACKEND"] = rdzv_cfg["backend"]

    if cfg.get("chain", {}).get("jobs", 1) > 1 and run_cfg.get("name"):
        run_cfg["timestamp"] = False

    cfg = normalize_train_config(cfg)
    cfg = resolve_all(cfg, launch_timestamp)
    cfg = normalize_train_config(cfg)
    cfg = apply_checkpoint_backend_defaults(cfg)

    if require_config and not cfg.get("train", {}).get("config"):
        raise SystemExit("Need --train.config")
    return cfg


def require_path(cfg: dict[str, Any], dotted_path: str) -> Any:
    """Return a required dotted config value or exit with a launcher error."""
    cur: Any = cfg
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise SystemExit(f"Missing required launch setting: {dotted_path}")
        cur = cur[part]
    if cur is None or cur == "":
        raise SystemExit(f"Missing required launch setting: {dotted_path}")
    return cur


def validate_launch_config(cfg: dict[str, Any]) -> None:
    """Check the minimum resolved launch settings before rendering commands."""
    # Validate only cheap, already-resolved launcher fields here. Anything that
    # would require scheduler access, filesystem staging, or cluster state stays
    # out of the mandatory preflight.
    required = [
        "paths.repo_root",
        "paths.exp_root",
        "resources.nnodes",
        "resources.nproc_per_node",
        "resources.cpus_per_proc",
        "container.runtime",
    ]
    if scheduler(cfg) == "slurm":
        required.extend(["resources.time", "slurm.gpu_directive"])
    for dotted_path in required:
        require_path(cfg, dotted_path)

    runtime = cfg.get("container", {}).get("runtime")
    if runtime in {"singularity", "shifter"}:
        require_path(cfg, "container.image")


def validate_training_config(cfg: dict[str, Any]) -> None:
    """Validate the requested training config without building datasets/models."""
    config = cfg.get("train", {}).get("config")
    if not config:
        return

    # The launch layer stores configs relative to `configs/`, while users may
    # still pass absolute paths or paths that already include `configs/`.
    path = Path(str(config))
    candidates = []
    if path.is_absolute():
        candidates.extend([path, path.with_suffix(".py")])
    else:
        candidates.extend([ROOT / "configs" / f"{config}.py", ROOT / "configs" / str(config)])
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not path.is_file():
        # Missing config paths are one of the most common launcher mistakes; show
        # nearby config names so a typo can be fixed without another repository
        # search.
        suggestions = []
        parent = path.parent
        if parent.is_dir():
            for candidate in sorted(parent.glob("*.py")):
                if candidate.name.startswith("__"):
                    continue
                try:
                    display = candidate.relative_to(ROOT / "configs").with_suffix("").as_posix()
                except ValueError:
                    display = str(candidate)
                suggestions.append(display)
                if len(suggestions) >= 20:
                    break
        message = f"Training config not found: {path}"
        if suggestions:
            message += "\nAvailable configs in this directory:\n  " + "\n  ".join(suggestions)
        raise SystemExit(message)

    try:
        from pimm.engines.defaults import _split_hook_type_options
        from pimm.utils.config import Config

        # Load and merge only scalar/config-dict CLI overrides. Hook type
        # overrides are applied later by the training parser, and list-index hook
        # mutations are not needed for these cheap launch-time checks.
        training_cfg = Config.fromfile(str(path))
        options = cfg.get("train", {}).get("options") or {}
        _, merge_options = _split_hook_type_options(options)
        if merge_options:
            training_cfg.merge_from_dict(merge_options)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"Could not load training config {path}: {exc}") from exc

    # Batch sizes are divided per rank by the training parser. Catching
    # non-divisible values before torchrun/submitit starts avoids a slow failure
    # after job allocation.
    world_size = distributed_world_size(cfg)
    for key in ("batch_size", "batch_size_val", "batch_size_test"):
        value = getattr(training_cfg, key, None)
        if value is None:
            continue
        try:
            numeric_value = int(value)
        except (TypeError, ValueError):
            continue
        if numeric_value % world_size != 0:
            raise SystemExit(
                f"{key}={numeric_value} must be divisible by launcher world size "
                f"{world_size} (nnodes * nproc_per_node)."
            )
