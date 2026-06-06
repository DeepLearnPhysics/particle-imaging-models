"""Checkpointable training-state helpers for exact resume."""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from pimm.utils.config import Config
from pimm.utils.env import set_seed


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = None if seed is None else num_workers * rank + worker_id + seed
    set_seed(worker_seed)


def _set_nested(d, key_path, value):
    """Set nested dict value: 'a.b.c' -> d['a']['b']['c'] = value."""
    keys = key_path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _apply_hook_overrides_from_dict(cfg, override_dict):
    """Apply hook parameter overrides from a dict.

    Args:
        cfg: Config object
        override_dict: Dict mapping hook type to parameter dict.
                      Example: {"WandbNamer": {"extra": "fft"}, "InstanceSegmentationEvaluator": {"every_n_steps": 500}}
    """
    if not hasattr(cfg, "hooks") or not override_dict:
        return

    for hook_type, params in override_dict.items():
        if not isinstance(params, dict):
            continue

        # find and update matching hook(s)
        for hook_cfg in cfg.hooks:
            if hook_cfg.get("type") == hook_type:
                for param_path, val in params.items():
                    _set_nested(hook_cfg, param_path, val)


def _apply_hook_overrides(cfg, options):
    """Process --options hooks.HookType.param=value entries.

    Example: --options hooks.PretrainEvaluator.every_n_steps=500
    """
    if not hasattr(cfg, "hooks") or not options:
        return

    for key, val in options.items():
        if not key.startswith("hooks."):
            continue

        # parse: hooks.HookType.param.subparam -> (HookType, param.subparam)
        parts = key.split(".", 2)
        if len(parts) < 3:
            continue

        hook_type = parts[1]
        param_path = parts[2]

        # find and update matching hook(s)
        for hook_cfg in cfg.hooks:
            if hook_cfg.get("type") == hook_type:
                _set_nested(hook_cfg, param_path, val)


def _is_hook_type_override(key):
    """Return whether an option targets hooks by hook type, not list index."""
    if not key.startswith("hooks."):
        return False
    parts = key.split(".", 2)
    if len(parts) < 3:
        return False
    return not parts[1].isdigit()


def _split_hook_type_options(options):
    """Separate hook-type overrides from generic config merge options."""
    if not options:
        return {}, {}
    hook_options = {}
    merge_options = {}
    for key, value in options.items():
        if _is_hook_type_override(key):
            hook_options[key] = value
        else:
            merge_options[key] = value
    return hook_options, merge_options


def _to_plain_data(value):
    """Convert config objects into JSON-serializable Python containers."""
    if isinstance(value, Config):
        return _to_plain_data(value._cfg_dict)
    if hasattr(value, "to_dict"):
        try:
            return _to_plain_data(value.to_dict())
        except TypeError:
            pass
    if isinstance(value, dict):
        return {str(k): _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(v) for v in value]
    return value


def _write_json(path, payload):
    """Write an indented JSON file with a trailing newline."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


def _git_metadata():
    """Return best-effort Git metadata for run provenance."""

    def run_git(args):
        """Run a git command in the current repo and return stdout."""
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=os.getcwd(),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    commit = run_git(["rev-parse", "HEAD"])
    if commit is None:
        return {}
    status = run_git(["status", "--short", "--untracked-files=no"])
    return {
        "commit": commit,
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "is_dirty": bool(status),
        "dirty_check": "tracked_files_only",
    }


def _save_config_artifacts(cfg, file_path, options):
    """Persist resolved config, model config, and run metadata artifacts."""
    cfg.dump(os.path.join(cfg.save_path, "config.py"))

    cfg_dict = _to_plain_data(cfg)
    _write_json(os.path.join(cfg.save_path, "resolved_config.json"), cfg_dict)

    if isinstance(cfg_dict, dict) and "model" in cfg_dict:
        _write_json(os.path.join(cfg.save_path, "model_config.json"), cfg_dict["model"])

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "config_file": file_path,
        "config_file_abs": os.path.abspath(file_path) if file_path else None,
        "cli_options": _to_plain_data(options or {}),
        "cli_option_keys": sorted((options or {}).keys()),
        "save_path": cfg.save_path,
        "resume": bool(cfg.resume),
        "git": _git_metadata(),
    }
    _write_json(os.path.join(cfg.save_path, "run_metadata.json"), metadata)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and Torch RNG state for exact resume."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore Python, NumPy, and Torch RNG state captured earlier."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def capture_distributed_rng_state() -> dict[str, Any]:
    """Capture per-rank RNG state in a rank-indexed container."""
    local_state = capture_rng_state()
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        states = [None for _ in range(world_size)]
        dist.all_gather_object(states, local_state)
        return {
            "world_size": world_size,
            "states": states,
        }
    return {
        "world_size": 1,
        "states": [local_state],
    }


def restore_distributed_rng_state(state: dict[str, Any], *, strict: bool = True) -> None:
    """Restore the RNG state for the current rank from a gathered state."""
    if not state:
        return
    if "states" not in state:
        restore_rng_state(state)
        return
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    saved_world_size = int(state.get("world_size", len(state["states"])))
    if strict and saved_world_size != world_size:
        raise ValueError(
            f"RNG state was saved with world_size={saved_world_size}, "
            f"but current world_size={world_size}."
        )
    states = state["states"]
    if rank >= len(states):
        if strict:
            raise ValueError(f"No RNG state available for rank {rank}.")
        rank = 0
    restore_rng_state(states[rank])


@dataclass
class TrainState:
    """Serializable resume point for trainer, sampler, dataloader, and RNG state."""

    schema_version: int = 1
    epoch: int = 0
    iter_in_epoch: int = 0
    global_step: int = 0
    samples_seen: int = 0
    world_size: int = 1
    batch_size_per_rank: int | None = None
    grad_accum_step: int = 0
    best_metric_value: float | None = None
    rng_state: dict[str, Any] | None = None
    sampler_state: dict[str, Any] | None = None
    dataloader_state: dict[str, Any] | None = None
    extra_state: dict[str, Any] = field(default_factory=dict)

    @property
    def iteration(self) -> int:
        """Backward-compatible alias for iter_in_epoch."""
        return self.iter_in_epoch

    @iteration.setter
    def iteration(self, value: int) -> None:
        """Set iter_in_epoch through the legacy iteration name."""
        self.iter_in_epoch = int(value)

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpoint-friendly dictionary representation."""
        return {
            "schema_version": int(self.schema_version),
            "epoch": int(self.epoch),
            "iter_in_epoch": int(self.iter_in_epoch),
            "iteration": int(self.iter_in_epoch),
            "global_step": int(self.global_step),
            "samples_seen": int(self.samples_seen),
            "world_size": int(self.world_size),
            "batch_size_per_rank": self.batch_size_per_rank,
            "grad_accum_step": int(self.grad_accum_step),
            "best_metric_value": self.best_metric_value,
            "rng_state": self.rng_state,
            "sampler_state": self.sampler_state,
            "dataloader_state": self.dataloader_state,
            "extra_state": dict(self.extra_state),
        }

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "TrainState":
        """Create TrainState from checkpoint data."""
        state = cls()
        state.load_state_dict(state_dict)
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state with compatibility for older iteration key names."""
        self.schema_version = int(state_dict.get("schema_version", 0) or 0)
        self.epoch = int(state_dict.get("epoch", 0))
        self.iter_in_epoch = int(
            state_dict.get("iter_in_epoch", state_dict.get("iteration", state_dict.get("iter", 0)))
        )
        self.global_step = int(state_dict.get("global_step", 0))
        self.samples_seen = int(state_dict.get("samples_seen", 0))
        self.world_size = int(state_dict.get("world_size", 1))
        self.batch_size_per_rank = state_dict.get("batch_size_per_rank")
        self.grad_accum_step = int(state_dict.get("grad_accum_step", 0))
        self.best_metric_value = state_dict.get("best_metric_value")
        self.rng_state = state_dict.get("rng_state")
        self.sampler_state = state_dict.get("sampler_state")
        self.dataloader_state = state_dict.get("dataloader_state")
        self.extra_state = dict(state_dict.get("extra_state", {}))

    @classmethod
    def from_trainer(cls, trainer: Any) -> "TrainState":
        """Capture the next resume position from a live trainer."""
        comm_info = getattr(trainer, "comm_info", {})
        iter_per_epoch = int(comm_info.get("iter_per_epoch", 0) or 0)
        raw_iter = comm_info.get("iter", -1)
        current_iter = -1 if raw_iter is None else int(raw_iter)
        next_iter = current_iter + 1
        epoch = int(getattr(trainer, "epoch", 0))
        # Store the next batch to consume, rolling to the next epoch at the end.
        if iter_per_epoch > 0 and next_iter >= iter_per_epoch:
            next_epoch = epoch + 1
            iter_in_epoch = 0
        else:
            next_epoch = epoch
            iter_in_epoch = max(0, next_iter)

        global_step = int(
            getattr(trainer, "global_step", 0)
            or (epoch * iter_per_epoch + max(0, next_iter))
        )
        batch_size_per_rank = getattr(trainer.cfg, "batch_size_per_gpu", None)
        world_size = 1
        try:
            import pimm.utils.comm as comm

            world_size = comm.get_world_size()
        except Exception:
            pass
        samples_seen = int(getattr(trainer, "samples_seen", 0) or 0)
        if not samples_seen and batch_size_per_rank is not None:
            samples_seen = global_step * int(batch_size_per_rank) * world_size

        best_metric_value = getattr(trainer, "best_metric_value", None)
        if isinstance(best_metric_value, torch.Tensor):
            best_metric_value = best_metric_value.item()

        return cls(
            epoch=next_epoch,
            iter_in_epoch=iter_in_epoch,
            global_step=global_step,
            samples_seen=samples_seen,
            world_size=world_size,
            batch_size_per_rank=batch_size_per_rank,
            best_metric_value=best_metric_value,
        )


def apply_train_state_to_trainer(trainer: Any, train_state: TrainState) -> None:
    """Apply a restored TrainState to the mutable trainer counters."""
    trainer.train_state = train_state
    trainer.start_epoch = int(train_state.epoch)
    trainer.start_iter = int(train_state.iter_in_epoch)
    trainer.global_step = int(train_state.global_step)
    trainer.samples_seen = int(train_state.samples_seen)
    if train_state.best_metric_value is not None:
        trainer.best_metric_value = train_state.best_metric_value
