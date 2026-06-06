"""Central loading and export helpers for pimm models."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Type, Union

import torch

from pimm.export.checkpoint import (
    clean_state_dict,
    filter_state_dict_by_prefix,
    load_state_dict_from_checkpoint,
    remap_state_dict_keys,
)
from pimm.utils.config import Config
from pimm.utils.path import split_checkpoint_weight_file

try:
    import safetensors.torch as safe_torch
    _HAS_SAFETENSORS = True
except ImportError:  # pragma: no cover
    safe_torch = None
    _HAS_SAFETENSORS = False

PathLike = Union[str, Path]


def _to_plain_data(value: Any) -> Any:
    """Convert config-like objects into JSON-serializable Python containers."""
    if isinstance(value, Config):
        return _to_plain_data(value._cfg_dict)
    if hasattr(value, "to_dict"):
        try:
            return _to_plain_data(value.to_dict())
        except TypeError:
            pass
    if isinstance(value, Mapping):
        return {str(k): _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(v) for v in value]
    return value


def _read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object from ``path``."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as stable, newline-terminated JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


def _load_config_payload(path: Path) -> Dict[str, Any]:
    """Load a Python or JSON config file into plain Python data."""
    if path.suffix == ".py":
        return _to_plain_data(Config.fromfile(str(path)))
    if path.suffix == ".json":
        return _read_json(path)
    raise ValueError(f"Unsupported config file for hub export: {path}")


def _extract_model_config(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the nested model config when present, otherwise the payload."""
    if "model" in payload and isinstance(payload["model"], Mapping):
        return dict(payload["model"])
    return dict(payload)


def _find_model_config_by_type(
    payload: Any,
    model_type: str,
) -> Optional[Dict[str, Any]]:
    """Return the first nested model config whose ``type`` matches."""
    if isinstance(payload, Mapping):
        if payload.get("type") == model_type:
            return dict(payload)
        for value in payload.values():
            found = _find_model_config_by_type(value, model_type)
            if found is not None:
                return found
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            found = _find_model_config_by_type(value, model_type)
            if found is not None:
                return found
    return None


def _select_target_model_config(
    source_model_cfg: Mapping[str, Any],
    model_type: Optional[str],
) -> Dict[str, Any]:
    """Choose the construction config for the requested output model."""
    model_cfg = dict(source_model_cfg)
    if model_type is None:
        return model_cfg
    if model_cfg.get("type") == model_type:
        return model_cfg
    nested_cfg = _find_model_config_by_type(model_cfg, model_type)
    if nested_cfg is not None:
        return nested_cfg
    model_cfg["type"] = model_type
    return model_cfg


def _find_run_root(checkpoint_path: Path) -> Optional[Path]:
    """Infer an experiment root from a checkpoint under a ``model`` directory."""
    checkpoint_path = checkpoint_path.resolve()
    if (
        checkpoint_path.name == "weights.pth"
        and checkpoint_path.parent.parent.name == "model"
    ):
        return checkpoint_path.parent.parent.parent
    if checkpoint_path.parent.name == "model":
        return checkpoint_path.parent.parent
    return None


def _is_dcp_checkpoint_dir(path: Path) -> bool:
    """Return whether ``path`` looks like a torch distributed checkpoint."""
    return path.is_dir() and (path / ".metadata").is_file()


def _infer_model_config(
    *,
    model_config: Optional[Mapping[str, Any]],
    cfg: Any,
    config_path: Optional[PathLike],
    checkpoint_path: Optional[Path],
) -> Dict[str, Any]:
    """Resolve the model construction config from explicit args or run files."""
    if model_config is not None:
        return _to_plain_data(model_config)
    if cfg is not None:
        cfg_payload = _to_plain_data(cfg)
        return _extract_model_config(cfg_payload)
    if config_path is not None:
        return _extract_model_config(_load_config_payload(Path(config_path)))

    candidate_dirs = []
    if checkpoint_path is not None:
        run_root = _find_run_root(checkpoint_path)
        if run_root is not None:
            candidate_dirs.append(run_root)
        candidate_dirs.append(checkpoint_path.parent)

    for root in candidate_dirs:
        for name in ("model_config.json", "resolved_config.json", "training_config.json"):
            candidate = root / name
            if candidate.is_file():
                return _extract_model_config(_read_json(candidate))
        candidate = root / "config.py"
        if candidate.is_file():
            return _extract_model_config(_load_config_payload(candidate))

    raise ValueError(
        "model_config, cfg, or config_path is required when it cannot be inferred "
        "from an experiment directory."
    )


def _state_dict_from_model_or_checkpoint(
    model_or_checkpoint: Any,
    device: str = "cpu",
    *,
    model_config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[PathLike] = None,
    model_cls: Optional[Type[torch.nn.Module]] = None,
) -> Dict[str, Any]:
    """Extract a cleaned state dict from a model, mapping, or checkpoint path."""
    if isinstance(model_or_checkpoint, torch.nn.Module):
        module = model_or_checkpoint.module if hasattr(model_or_checkpoint, "module") else model_or_checkpoint
        return clean_state_dict(module.state_dict())
    if isinstance(model_or_checkpoint, Mapping):
        if "state_dict" in model_or_checkpoint:
            return clean_state_dict(model_or_checkpoint["state_dict"])
        if "model" in model_or_checkpoint and isinstance(model_or_checkpoint["model"], Mapping):
            model_state = model_or_checkpoint["model"]
            if "state_dict" in model_state:
                return clean_state_dict(model_state["state_dict"])
            return clean_state_dict(model_state)
        return clean_state_dict(dict(model_or_checkpoint))
    checkpoint_path = Path(model_or_checkpoint)
    if Path(split_checkpoint_weight_file(checkpoint_path)).is_file():
        return load_state_dict_from_checkpoint(
            split_checkpoint_weight_file(checkpoint_path),
            device=device,
        )
    if _is_dcp_checkpoint_dir(checkpoint_path):
        raise ValueError(
            f"{checkpoint_path} is a raw DCP trainer-state directory. "
            "Use the split checkpoint directory containing weights.pth."
        )
    return load_state_dict_from_checkpoint(str(checkpoint_path), device=device)


def _build_model_from_config(
    model_cfg: Mapping[str, Any],
    *,
    model_cls: Optional[Type[torch.nn.Module]] = None,
) -> torch.nn.Module:
    """Construct a model from registry config or an explicit class."""
    if model_cls is not None:
        kwargs = dict(model_cfg)
        kwargs.pop("type", None)
        return model_cls(**kwargs)

    import pimm.models  # noqa: F401 - populate import-all registries
    from pimm.models.builder import build_model

    return build_model(dict(model_cfg))


def _copy_if_present(src: Optional[Path], dst: Path) -> None:
    """Copy ``src`` to ``dst`` when it exists and is not already that file."""
    if src is not None and src.is_file() and src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)


def save_pretrained(
    model_or_checkpoint: Any,
    save_directory: PathLike,
    *,
    cfg: Any = None,
    model_config: Optional[Mapping[str, Any]] = None,
    model_cls: Optional[Type[torch.nn.Module]] = None,
    config_path: Optional[PathLike] = None,
    training_config: Optional[Mapping[str, Any]] = None,
    safe_serialization: bool = True,
    device: str = "cpu",
    model_card: Optional[str] = None,
) -> Path:
    """Save model weights and metadata in a Hugging Face-style directory.

    The export contains ``config.json``, ``model_config.json``, serialized
    weights, and optionally the full training config, original pimm config, and
    model card. ``model_or_checkpoint`` may be an nn.Module, checkpoint mapping,
    or checkpoint path.
    """
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = None
    if isinstance(model_or_checkpoint, (str, Path)):
        checkpoint_path = Path(model_or_checkpoint)

    model_cfg = _infer_model_config(
        model_config=model_config,
        cfg=cfg,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    state_dict = _state_dict_from_model_or_checkpoint(
        model_or_checkpoint,
        device=device,
        model_config=model_cfg,
        config_path=config_path,
        model_cls=model_cls,
    )

    if safe_serialization:
        if not _HAS_SAFETENSORS:
            raise ImportError("safetensors is required for safe_serialization=True")
        weights_name = "model.safetensors"
        safe_torch.save_file(state_dict, str(save_dir / weights_name))
    else:
        weights_name = "pytorch_model.bin"
        torch.save(state_dict, save_dir / weights_name)

    _write_json(save_dir / "model_config.json", model_cfg)
    if training_config is None and cfg is not None:
        training_config = _to_plain_data(cfg)
    if training_config is not None:
        _write_json(save_dir / "training_config.json", _to_plain_data(training_config))

    config_src = Path(config_path) if config_path is not None else None
    if config_src is None and checkpoint_path is not None:
        run_root = _find_run_root(checkpoint_path)
        if run_root is not None:
            config_src = run_root / "config.py"
    _copy_if_present(config_src, save_dir / "pimm_config.py")

    manifest = {
        "library_name": "pimm",
        "format_version": 1,
        "model_type": model_cfg.get("type"),
        "weights": weights_name,
        "model_config": "model_config.json",
    }
    _write_json(save_dir / "config.json", manifest)

    if model_card is not None:
        (save_dir / "README.md").write_text(model_card, encoding="utf-8")

    return save_dir


def _resolve_pretrained_path(
    pretrained_model_name_or_path: PathLike,
    *,
    cache_dir: Optional[PathLike] = None,
    revision: Optional[str] = None,
) -> Path:
    """Resolve a local path or download a model snapshot from the Hub."""
    path = Path(pretrained_model_name_or_path)
    if path.exists():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("huggingface_hub is required to load pimm models from the Hub") from exc
    downloaded = snapshot_download(
        repo_id=str(pretrained_model_name_or_path),
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        revision=revision,
        allow_patterns=[
            "config.json",
            "model_config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "pimm_config.py",
            "README.md",
        ],
    )
    return Path(downloaded)


def load_model(
    pretrained_model_name_or_path: PathLike,
    *,
    model_config: Optional[Mapping[str, Any]] = None,
    model_type: Optional[str] = None,
    model_cls: Optional[Type[torch.nn.Module]] = None,
    config_path: Optional[PathLike] = None,
    cache_dir: Optional[PathLike] = None,
    revision: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = "cpu",
    strict: bool = True,
    prefix: Optional[str] = None,
    remove_prefix: bool = True,
    key_mapping: Optional[Dict[str, str]] = None,
    keep_unmapped_keys: bool = False,
    filter_fn: Optional[Callable[[Dict[str, Any], torch.nn.Module], Dict[str, Any]]] = None,
    return_metadata: bool = False,
    **model_kwargs: Any,
):
    """Load a pimm model from a local export, Hub repo, or raw checkpoint.

    The model config comes from ``model_config``, an exported directory, a
    supplied config path, or an experiment directory next to the checkpoint.
    ``model_type`` can override the registry ``type`` for the constructed
    output model. ``model_cls`` can construct an explicit Python class instead
    of using the registry. Extra ``model_kwargs`` override config fields before
    construction.

    ``prefix``/``remove_prefix`` and ``key_mapping`` mirror ``load_pretrained``
    so exported pretraining checkpoints can be loaded into a different model
    shape, for example ``key_mapping={"student.backbone.": "backbone."}``.
    By default, ``load_model`` drops keys that do not match ``key_mapping``;
    pass ``keep_unmapped_keys=True`` to preserve them.
    """
    resolved = _resolve_pretrained_path(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
    )

    if Path(split_checkpoint_weight_file(resolved)).is_file():
        manifest = {}
        weights_path = Path(split_checkpoint_weight_file(resolved))
        model_cfg = _infer_model_config(
            model_config=model_config,
            cfg=None,
            config_path=config_path,
            checkpoint_path=weights_path,
        )
    elif resolved.is_dir() and not _is_dcp_checkpoint_dir(resolved):
        manifest_path = resolved / "config.json"
        manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
        model_cfg = model_config or _read_json(resolved / "model_config.json")
        weights_name = manifest.get("weights")
        if weights_name is None:
            if (resolved / "model.safetensors").is_file():
                weights_name = "model.safetensors"
            elif (resolved / "pytorch_model.bin").is_file():
                weights_name = "pytorch_model.bin"
            else:
                raise FileNotFoundError(f"No model weights found in {resolved}")
        weights_path = resolved / weights_name
    else:
        manifest = {}
        weights_path = resolved
        model_cfg = _infer_model_config(
            model_config=model_config,
            cfg=None,
            config_path=config_path,
            checkpoint_path=resolved,
        )

    source_model_cfg = _to_plain_data(model_cfg)
    model_cfg = _select_target_model_config(source_model_cfg, model_type)
    model_cfg.update(model_kwargs)
    model = _build_model_from_config(model_cfg, model_cls=model_cls)
    if _is_dcp_checkpoint_dir(weights_path):
        raise ValueError(
            f"{weights_path} is a raw DCP trainer-state directory. "
            "Use the split checkpoint directory containing weights.pth."
        )
    if str(weights_path).endswith(".safetensors"):
        if not _HAS_SAFETENSORS:
            raise ImportError("safetensors is required to load model.safetensors")
        state_dict = safe_torch.load_file(str(weights_path), device=str(device or "cpu"))
    else:
        state_dict = load_state_dict_from_checkpoint(str(weights_path), device=device or "cpu")

    if prefix is not None:
        state_dict = filter_state_dict_by_prefix(
            state_dict,
            prefix=prefix,
            remove_prefix=remove_prefix,
        )
        if not state_dict:
            raise ValueError(f"No keys found with prefix '{prefix}' in {weights_path}")
    if key_mapping is not None:
        state_dict = remap_state_dict_keys(
            state_dict,
            key_mapping,
            keep_unmapped=keep_unmapped_keys,
        )
    if filter_fn is not None:
        state_dict = filter_fn(state_dict, model)

    if not state_dict:
        raise ValueError(f"No checkpoint keys selected for loading from {weights_path}")

    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    if device is not None:
        model.to(device)
    model.eval()

    metadata = {
        "config": manifest,
        "model_config": model_cfg,
        "source_model_config": source_model_cfg,
        "path": str(resolved),
        "weights": str(weights_path),
    }
    if not strict:
        metadata["incompatible_keys"] = {
            "missing_keys": list(incompatible_keys.missing_keys),
            "unexpected_keys": list(incompatible_keys.unexpected_keys),
        }
    if return_metadata:
        return model, metadata
    return model


def push_to_hub(
    model_or_directory: Any,
    repo_id: str,
    *,
    save_directory: Optional[PathLike] = None,
    private: Optional[bool] = None,
    token: Optional[str] = None,
    commit_message: str = "Upload pimm model",
    **save_kwargs: Any,
):
    """Upload an exported pimm model directory to the Hugging Face Hub.

    ``model_or_directory`` can already be an exported directory or any input
    accepted by ``save_pretrained``. Non-directory inputs are exported to a
    temporary or provided save directory before upload.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("huggingface_hub is required for push_to_hub") from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    temp_dir = None
    path = Path(model_or_directory) if isinstance(model_or_directory, (str, Path)) else None
    if path is not None and path.is_dir() and (path / "config.json").is_file():
        export_dir = path
    else:
        if save_directory is None:
            temp_dir = tempfile.TemporaryDirectory()
            export_dir = Path(temp_dir.name)
        else:
            export_dir = Path(save_directory)
        save_pretrained(model_or_directory, export_dir, **save_kwargs)

    try:
        return api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(export_dir),
            commit_message=commit_message,
            allow_patterns=[
                "config.json",
                "model_config.json",
                "model.safetensors",
                "pytorch_model.bin",
                "pimm_config.py",
                "training_config.json",
                "README.md",
            ],
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
