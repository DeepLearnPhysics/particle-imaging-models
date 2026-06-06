"""Parallel-context and model-device helpers for pimm training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import pimm.utils.comm as comm


@dataclass(frozen=True)
class ParallelContext:
    """Resolved process, device, and strategy information for one rank."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    process_group: Any = None
    local_group: Any = None
    strategy: str = "ddp"
    device_mesh: Any = None


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a config key from dict-like or attribute-style objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def get_parallel_cfg(cfg: Any) -> Any:
    """Return the parallel/distributed config block."""
    return _cfg_get(cfg, "parallel", _cfg_get(cfg, "distributed", {}))


def create_parallel_context(cfg: Any = None) -> ParallelContext:
    """Resolve this rank's device and distributed strategy."""
    parallel_cfg = get_parallel_cfg(cfg)
    strategy = str(_cfg_get(parallel_cfg, "strategy", "ddp"))
    world_size = comm.get_world_size()
    rank = comm.get_rank()
    local_rank = 0
    try:
        local_rank = comm.get_local_rank()
    except AssertionError:
        local_rank = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        # Respect CUDA_VISIBLE_DEVICES while still mapping local ranks to devices.
        num_visible = torch.cuda.device_count()
        device_index = local_rank if num_visible > 1 else 0
        if device_index >= num_visible:
            device_index = device_index % num_visible
        device = torch.device("cuda", device_index)
    else:
        device = torch.device("cpu")
    return ParallelContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        process_group=dist.group.WORLD if dist.is_available() and dist.is_initialized() else None,
        strategy=strategy,
    )


def move_batch_to_device(batch: Any, device: torch.device, *, non_blocking: bool = True) -> Any:
    """Recursively move tensors in a nested batch structure to device."""
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device, non_blocking=non_blocking) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device, non_blocking=non_blocking) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device, non_blocking=non_blocking) for value in batch)
    return batch


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the wrapped module when using DDP-like containers."""
    if hasattr(model, "module"):
        return model.module
    return model


def _fsdp2_wrap(model: nn.Module, parallel_cfg: Any, context: ParallelContext) -> nn.Module:
    """Wrap modules with PyTorch composable FSDP2."""
    try:
        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.device_mesh import init_device_mesh
    except Exception as exc:
        raise RuntimeError("parallel.strategy='fsdp2' requires PyTorch FSDP2 fully_shard support") from exc

    if context.device.type != "cuda":
        raise RuntimeError("parallel.strategy='fsdp2' currently requires CUDA")
    if context.world_size < 1:
        raise RuntimeError("Invalid world_size for FSDP2")

    mesh = init_device_mesh("cuda", (context.world_size,), mesh_dim_names=("fsdp",))
    wrap_classes = set(_cfg_get(parallel_cfg, "wrap_classes", []) or [])
    if wrap_classes:
        for module in model.modules():
            if module is model:
                continue
            if module.__class__.__name__ in wrap_classes:
                fully_shard(module, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def prepare_model(model: nn.Module, cfg: Any, context: ParallelContext) -> nn.Module:
    """Move model to the resolved device and apply the parallel strategy."""
    parallel_cfg = get_parallel_cfg(cfg)
    strategy = str(_cfg_get(parallel_cfg, "strategy", context.strategy))
    sync_bn = bool(_cfg_get(cfg, "sync_bn", False))
    if sync_bn and context.world_size > 1 and context.device.type == "cuda":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if strategy == "none" or context.world_size == 1:
        return model.to(context.device)
    if strategy == "ddp":
        model = model.to(context.device)
        ddp_kwargs = {
            "broadcast_buffers": False,
            "find_unused_parameters": bool(_cfg_get(cfg, "find_unused_parameters", False)),
        }
        if context.device.type == "cuda":
            ddp_kwargs["device_ids"] = [context.device.index]
            ddp_kwargs["output_device"] = context.device.index
        try:
            return DistributedDataParallel(model, static_graph=True, **ddp_kwargs)
        except TypeError:
            return DistributedDataParallel(model, **ddp_kwargs)
    if strategy == "fsdp2":
        return _fsdp2_wrap(model, parallel_cfg, context)
    raise ValueError(f"Unsupported parallel.strategy: {strategy}")
