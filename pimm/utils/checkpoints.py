"""Checkpoint format, IO, and resume management for pimm training runtimes."""

from __future__ import annotations

import os
import shutil
from collections import OrderedDict

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

import pimm.utils.comm as comm
from pimm.datasets.stateful import (
    assert_exact_dataloader_state_available,
    dataloader_state_dict,
    load_dataloader_state_dict,
)
from pimm.engines._train_utils import (
    TrainState,
    apply_train_state_to_trainer,
    capture_distributed_rng_state,
    capture_rng_state,
    restore_distributed_rng_state,
)
from pimm.utils.comm import is_main_process, synchronize
from pimm.utils.path import (
    checkpoint_success_file as _dcp_success_file,
    is_complete_dcp_checkpoint,
    is_complete_split_checkpoint,
    latest_complete_checkpoint,
    resolve_model_weight_file,
    split_checkpoint_trainer_dir,
    split_checkpoint_weight_file,
)


def _distributed_object_state(local_state):
    """Gather one Python state object per rank into a checkpointable wrapper."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        states = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(states, local_state)
        return {
            "_pimm_distributed_state": True,
            "world_size": world_size,
            "states": states,
        }
    return local_state


def local_object_state(state, *, strict=True):
    """Return the current rank's state from a distributed object wrapper."""
    if not isinstance(state, dict) or not state.get("_pimm_distributed_state"):
        return state
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = 1
        rank = 0
    saved_world_size = int(state.get("world_size", len(state.get("states", []))))
    if strict and saved_world_size != world_size:
        raise ValueError(
            f"State was saved with world_size={saved_world_size}, "
            f"but current world_size={world_size}."
        )
    states = state.get("states", [])
    if rank >= len(states):
        if strict:
            raise ValueError(f"No distributed state available for rank {rank}.")
        rank = 0
    return states[rank]


def build_checkpoint_payload(trainer, *, distributed_rng=False):
    """Build the structured checkpoint payload consumed by checkpoint loads."""
    train_state = TrainState.from_trainer(trainer)
    local_dataloader_state = dataloader_state_dict(trainer.train_loader)
    assert_exact_dataloader_state_available(
        local_dataloader_state,
        loader=trainer.train_loader,
        iter_in_epoch=train_state.iter_in_epoch,
    )
    train_state.dataloader_state = (
        _distributed_object_state(local_dataloader_state)
        if distributed_rng
        else local_dataloader_state
    )
    rng_state = (
        capture_distributed_rng_state()
        if distributed_rng
        else capture_rng_state()
    )
    train_state.rng_state = rng_state
    model_state = trainer.model.state_dict()
    optimizer_state = get_optimizer_state_dict(
        trainer.model,
        trainer.optimizer,
        options=StateDictOptions(),
    )
    scheduler_state = trainer.scheduler.state_dict()
    scaler_state = (
        trainer.scaler.state_dict()
        if getattr(trainer, "scaler", None) is not None
        else None
    )
    world_size = comm.get_world_size()
    distributed_backend = (
        torch.distributed.get_backend()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else None
    )
    logger_state = {
        "backend": "wandb" if getattr(trainer.cfg, "use_wandb", False) else "tensorboard",
        "wandb": {
            "group": trainer.cfg.get("wandb_group", None),
            "run_name": trainer.cfg.get("wandb_run_name", None),
            "run_id": trainer.cfg.get("wandb_run_id", None),
            "job_type": trainer.cfg.get("wandb_job_type", None),
            "resume": trainer.cfg.get("wandb_resume", None),
            "step_offset": trainer.cfg.get("log_step_offset", 0),
            "checkpoint_global_step": train_state.global_step,
        },
    }
    return {
        "schema": "pimm.trainer_checkpoint",
        "version": 3,
        "checkpoint_version": 3,
        "model": {"state_dict": model_state},
        "optimizer": {
            "state_dict": optimizer_state,
            "class": trainer.optimizer.__class__.__name__,
            "format": "torch.distributed.checkpoint.state_dict",
        },
        "scheduler": {
            "state_dict": scheduler_state,
            "class": trainer.scheduler.__class__.__name__,
            "total_steps": getattr(trainer.scheduler, "total_steps", None),
        },
        "scaler": {
            "enabled": bool(getattr(trainer.cfg, "enable_amp", False)),
            "state_dict": scaler_state,
        },
        "dataloader": {
            "backend": trainer.train_loader.__class__.__name__,
            "state": train_state.dataloader_state,
            "world_size": world_size,
            "batch_size_per_rank": getattr(trainer.cfg, "batch_size_per_gpu", None),
            "num_workers": getattr(trainer.cfg, "num_worker_per_gpu", None),
            "drop_last": getattr(trainer.train_loader, "drop_last", None),
        },
        "rng": {
            "world_size": world_size,
            "state": rng_state,
        },
        "trainer": {
            "epoch": train_state.epoch,
            "iter_in_epoch": train_state.iter_in_epoch,
            "global_step": train_state.global_step,
            "samples_seen": train_state.samples_seen,
            "best_metric_value": train_state.best_metric_value,
        },
        "logger": logger_state,
        "distributed": {
            "world_size": world_size,
            "backend": distributed_backend,
            "rank_order": list(range(world_size)),
        },
    }


def empty_checkpoint_payload(trainer):
    """Build an empty typed payload for DCP load to fill in place."""
    payload = build_checkpoint_payload(trainer, distributed_rng=True)
    payload["trainer"]["best_metric_value"] = -float("inf")
    payload["trainer"].update(
        {"epoch": 0, "iter_in_epoch": 0, "global_step": 0, "samples_seen": 0}
    )
    return payload


def checkpoint_model_state_dict(checkpoint):
    """Extract model weights from structured or legacy checkpoint formats."""
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("model"), dict) and "state_dict" in checkpoint["model"]:
            return checkpoint["model"]["state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def checkpoint_optimizer_state_dict(checkpoint):
    """Extract optimizer state from structured or legacy checkpoints."""
    optimizer = checkpoint.get("optimizer", None)
    if isinstance(optimizer, dict) and "state_dict" in optimizer:
        return optimizer["state_dict"]
    return optimizer


def checkpoint_scheduler_state_dict(checkpoint):
    """Extract scheduler state from structured or legacy checkpoints."""
    scheduler = checkpoint.get("scheduler", None)
    if isinstance(scheduler, dict) and "state_dict" in scheduler:
        return scheduler["state_dict"]
    return scheduler


def checkpoint_scaler_state_dict(checkpoint):
    """Extract AMP scaler state from structured or legacy checkpoints."""
    scaler = checkpoint.get("scaler", None)
    if isinstance(scaler, dict) and "state_dict" in scaler:
        return scaler["state_dict"]
    return scaler


def checkpoint_dataloader_state(checkpoint, train_state=None):
    """Extract dataloader resume state, preferring parsed TrainState."""
    if train_state is not None and train_state.dataloader_state is not None:
        return train_state.dataloader_state
    dataloader = checkpoint.get("dataloader", None)
    if isinstance(dataloader, dict) and "state" in dataloader:
        return dataloader["state"]
    return dataloader


def checkpoint_rng_state(checkpoint, train_state=None):
    """Extract RNG resume state, preferring parsed TrainState."""
    if train_state is not None and train_state.rng_state is not None:
        return train_state.rng_state
    rng = checkpoint.get("rng", None)
    if isinstance(rng, dict) and "state" in rng:
        return rng["state"]
    return checkpoint.get("rng_state", None)


def checkpoint_train_state(checkpoint):
    """Parse structured trainer state, returning None for legacy checkpoints."""
    if checkpoint.get("train_state", None) is not None:
        return TrainState.from_state_dict(checkpoint["train_state"])
    trainer_state = checkpoint.get("trainer", None)
    if not isinstance(trainer_state, dict):
        return None
    dataloader_state = checkpoint_dataloader_state(checkpoint)
    rng_state = checkpoint_rng_state(checkpoint)
    dataloader = checkpoint.get("dataloader", {})
    world_size = (
        int(dataloader.get("world_size", comm.get_world_size()))
        if isinstance(dataloader, dict)
        else comm.get_world_size()
    )
    return TrainState(
        schema_version=int(checkpoint.get("checkpoint_version", checkpoint.get("version", 0)) or 0),
        epoch=int(trainer_state.get("epoch", 0)),
        iter_in_epoch=int(trainer_state.get("iter_in_epoch", trainer_state.get("iteration", 0))),
        global_step=int(trainer_state.get("global_step", 0)),
        samples_seen=int(trainer_state.get("samples_seen", 0)),
        world_size=world_size,
        batch_size_per_rank=(
            dataloader.get("batch_size_per_rank") if isinstance(dataloader, dict) else None
        ),
        best_metric_value=trainer_state.get("best_metric_value"),
        rng_state=rng_state,
        dataloader_state=dataloader_state,
    )


def build_trainer_state_payload(checkpoint):
    """Return checkpoint state with model weights removed."""
    return {key: value for key, value in checkpoint.items() if key != "model"}


def empty_trainer_state_payload(trainer):
    """Build an empty typed payload for split-checkpoint trainer state loading."""
    return build_trainer_state_payload(empty_checkpoint_payload(trainer))


def atomic_torch_save(payload, filename):
    """Save a torch checkpoint via a temp file and one-level backup."""
    tmp = filename + ".tmp"
    prev = filename + ".prev"
    torch.save(payload, tmp)
    with open(tmp, "rb") as handle:
        os.fsync(handle.fileno())
    if os.path.exists(prev):
        os.remove(prev)
    if os.path.exists(filename):
        os.replace(filename, prev)
    os.replace(tmp, filename)


def save_dcp_checkpoint(payload, checkpoint_dir):
    """Save a distributed checkpoint directory with atomic publish semantics."""
    import torch.distributed.checkpoint as dcp

    tmp_dir = checkpoint_dir + ".tmp"
    prev_dir = checkpoint_dir + ".prev"
    if is_main_process():
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
    synchronize()

    dcp.save(payload, checkpoint_id=tmp_dir)
    if is_main_process():
        with open(_dcp_success_file(tmp_dir), "w", encoding="utf-8") as handle:
            handle.write("ok\n")
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)
        if os.path.exists(checkpoint_dir):
            os.replace(checkpoint_dir, prev_dir)
        os.replace(tmp_dir, checkpoint_dir)
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)
    synchronize()


def save_split_checkpoint(payload, checkpoint_dir):
    """Save model weights plus DCP trainer state without duplicating tensors."""
    tmp_dir = checkpoint_dir + ".tmp"
    prev_dir = checkpoint_dir + ".prev"
    if is_main_process():
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        atomic_torch_save(
            {"state_dict": checkpoint_model_state_dict(payload)},
            split_checkpoint_weight_file(tmp_dir),
        )
    synchronize()

    save_dcp_checkpoint(
        build_trainer_state_payload(payload),
        split_checkpoint_trainer_dir(tmp_dir),
    )
    if is_main_process():
        with open(_dcp_success_file(tmp_dir), "w", encoding="utf-8") as handle:
            handle.write("ok\n")
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)
        if os.path.exists(checkpoint_dir):
            os.replace(checkpoint_dir, prev_dir)
        os.replace(tmp_dir, checkpoint_dir)
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)
    synchronize()


def load_dcp_trainer_state(checkpoint_dir, trainer):
    """Load a complete trainer-state DCP into a typed placeholder payload."""
    import torch.distributed.checkpoint as dcp

    if not is_complete_dcp_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"Incomplete DCP checkpoint: {checkpoint_dir}")
    payload = empty_trainer_state_payload(trainer)
    dcp.load(payload, checkpoint_id=checkpoint_dir)
    return payload


def load_dcp_checkpoint(checkpoint_dir, trainer):
    """Load a complete full DCP checkpoint into a typed placeholder payload."""
    import torch.distributed.checkpoint as dcp

    if not is_complete_dcp_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"Incomplete DCP checkpoint: {checkpoint_dir}")
    payload = empty_checkpoint_payload(trainer)
    dcp.load(payload, checkpoint_id=checkpoint_dir)
    return payload


def load_split_checkpoint(checkpoint_dir, trainer, map_location):
    """Load a split checkpoint for exact resume."""
    if not is_complete_split_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"Incomplete split checkpoint: {checkpoint_dir}")
    checkpoint = load_dcp_trainer_state(split_checkpoint_trainer_dir(checkpoint_dir), trainer)
    weight_checkpoint = torch.load(
        split_checkpoint_weight_file(checkpoint_dir),
        map_location=map_location,
        weights_only=False,
    )
    checkpoint["model"] = {"state_dict": checkpoint_model_state_dict(weight_checkpoint)}
    return checkpoint


def _cfg_get(cfg, key, default=None):
    """Read config values from dict-like or attribute-style config objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _parallel_strategy(cfg):
    """Return the configured parallel strategy name."""
    parallel_cfg = _cfg_get(cfg, "parallel", _cfg_get(cfg, "distributed", {}))
    strategy = _cfg_get(parallel_cfg, "strategy", "ddp")
    return str(strategy).lower()


def _needs_dcp_checkpointing(trainer):
    """Return whether this run should use DCP for durable training resume."""
    cfg = getattr(trainer, "cfg", None)
    chain_jobs = int(_cfg_get(cfg, "chain_jobs", 1) or 1)
    return (
        comm.get_world_size() > 1
        or chain_jobs > 1
        or _parallel_strategy(cfg) == "fsdp2"
    )


class CheckpointManager:
    """Own checkpoint format, save/load backends, and trainer resume semantics."""

    def __init__(self, trainer):
        self.trainer = trainer

    def warn_non_dcp(self, backend_description):
        """Warn when a serious distributed/resume run is not using DCP."""
        if not _needs_dcp_checkpointing(self.trainer):
            return
        cfg = getattr(self.trainer, "cfg", None)
        self.trainer.logger.warning(
            "%s is not using DCP. This remains supported for legacy, local, and "
            "export-style checkpoints, but DCP is the recommended training "
            "checkpoint backend for multi-rank, chained, and FSDP2 runs. "
            "Set hooks.CheckpointSaverIteration.backend=dcp for robust distributed "
            "resume. world_size=%s, chain_jobs=%s, parallel.strategy=%s",
            backend_description,
            comm.get_world_size(),
            _cfg_get(cfg, "chain_jobs", 1),
            _parallel_strategy(cfg),
        )

    def save_epoch_checkpoint(self, *, is_best=False, step_count=0, save_freq=None):
        """Save the legacy epoch/metric-oriented torch checkpoint."""
        filename = os.path.join(self.trainer.cfg.save_path, "model", "model_last.pth")
        self.trainer.logger.info("Saving checkpoint to: " + filename)
        atomic_torch_save(build_checkpoint_payload(self.trainer), filename)
        if is_best:
            shutil.copyfile(
                filename,
                os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
            )
        if save_freq and step_count % save_freq == 0:
            shutil.copyfile(
                filename,
                os.path.join(
                    self.trainer.cfg.save_path,
                    "model",
                    f"iter_{step_count}.pth",
                ),
            )

    def save_iteration_checkpoint(
        self,
        *,
        backend="torch",
        is_best=False,
        step_count=0,
        save_freq=None,
        save_iter_checkpoints=False,
    ):
        """Save the current structured checkpoint with the selected backend."""
        payload = build_checkpoint_payload(self.trainer, distributed_rng=True)
        if backend == "dcp":
            checkpoint_dir = os.path.join(self.trainer.cfg.save_path, "model", "last")
            if is_main_process():
                self.trainer.logger.info(
                    "Saving model weights to: " + split_checkpoint_weight_file(checkpoint_dir)
                )
            save_split_checkpoint(payload, checkpoint_dir)
            return

        filename = os.path.join(self.trainer.cfg.save_path, "model", "model_last.pth")
        if is_main_process():
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            atomic_torch_save(payload, filename)
            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )
            if save_iter_checkpoints and save_freq and step_count % save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"iter_{step_count}.pth",
                    ),
                )

    def load_weight_and_resume(self, *, keywords="", replacement=None, strict=False):
        """Load configured weights and restore training state when cfg.resume is true."""
        replacement = replacement if replacement is not None else keywords
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        weight_path = self.trainer.cfg.weight
        if weight_path and (os.path.isfile(weight_path) or os.path.isdir(weight_path)):
            self.trainer.logger.info(f"Loading weight at: {weight_path}")
            checkpoint = self._load_checkpoint(weight_path)
            self._load_model_weights(
                checkpoint,
                keywords=keywords,
                replacement=replacement,
                strict=strict,
            )
            if self.trainer.cfg.resume:
                self.resume_training_state(checkpoint)
            return

        message = f"No weight found at: {weight_path}"
        if self.trainer.cfg.resume:
            raise FileNotFoundError(message)
        self.trainer.logger.info(message)

    def _load_checkpoint(self, weight_path):
        """Load a direct, split, or directory checkpoint reference."""
        map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else "cpu"
        if os.path.isdir(weight_path):
            if is_complete_split_checkpoint(weight_path):
                if self.trainer.cfg.resume:
                    return load_split_checkpoint(weight_path, self.trainer, map_location)
                weight_file = resolve_model_weight_file(weight_path)
                return torch.load(weight_file, map_location=map_location, weights_only=False)
            if is_complete_dcp_checkpoint(weight_path):
                return load_dcp_checkpoint(weight_path, self.trainer)
            if self.trainer.cfg.resume:
                raise FileNotFoundError(f"Incomplete checkpoint directory: {weight_path}")
            weight_file = resolve_model_weight_file(weight_path)
            return torch.load(weight_file, map_location=map_location, weights_only=False)
        return torch.load(weight_path, map_location=map_location, weights_only=False)

    def _load_model_weights(self, checkpoint, *, keywords="", replacement="", strict=False):
        """Load checkpoint model weights with the existing keyword rewrite rules."""
        self.trainer.logger.info(
            f"Loading layer weights with keyword: {keywords}, "
            f"replace keyword with: {replacement}"
        )
        weight = OrderedDict()
        for key, value in checkpoint_model_state_dict(checkpoint).items():
            if not key.startswith("module."):
                key = "module." + key
            if keywords in key:
                key = key.replace(keywords, replacement)
            if comm.get_world_size() == 1:
                key = key[7:]
            weight[key] = value
        load_state_info = self.trainer.model.load_state_dict(weight, strict=strict)
        self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")

    def resume_training_state(self, checkpoint):
        """Restore structured or legacy optimizer, scheduler, RNG, and cursor state."""
        strict_state = self.trainer.cfg.get("resume_strict_state", True)
        iter_per_epoch = len(self.trainer.train_loader)

        train_state = checkpoint_train_state(checkpoint)
        if train_state is not None:
            dataloader_state = checkpoint_dataloader_state(checkpoint, train_state)
            train_state.dataloader_state = dataloader_state
            local_dataloader_state = local_object_state(
                dataloader_state,
                strict=strict_state,
            )
            apply_train_state_to_trainer(self.trainer, train_state)
            if train_state.iter_in_epoch > 0:
                if not local_dataloader_state:
                    self.trainer.logger.warning(
                        "Checkpoint is mid-epoch but has no dataloader state; "
                        "resuming from the beginning of the saved epoch and "
                        "replaying already-completed batches."
                    )
                    self.trainer.start_iter = 0
                    self.trainer.global_step = self.trainer.start_epoch * iter_per_epoch
                else:
                    load_dataloader_state_dict(
                        self.trainer.train_loader,
                        local_dataloader_state,
                        strict=strict_state,
                    )
            rng_state = checkpoint_rng_state(checkpoint, train_state)
            restore_distributed_rng_state(rng_state, strict=strict_state)
            self.trainer.logger.info(
                "Resuming train from structured state: "
                f"epoch={self.trainer.start_epoch}, "
                f"iter={self.trainer.start_iter}, "
                f"global_step={self.trainer.global_step}"
            )
        else:
            self._resume_legacy_training_state(checkpoint, iter_per_epoch)

        checkpoint_trainer_state = checkpoint.get("trainer", {})
        if (
            isinstance(checkpoint_trainer_state, dict)
            and "best_metric_value" in checkpoint_trainer_state
        ):
            self.trainer.best_metric_value = checkpoint_trainer_state["best_metric_value"]
        else:
            self.trainer.best_metric_value = checkpoint.get(
                "best_metric_value", self.trainer.best_metric_value
            )

        optimizer_state = checkpoint_optimizer_state_dict(checkpoint)
        if optimizer_state is not None:
            self.load_optimizer_state(optimizer_state)
        else:
            self.trainer.logger.info("No optimizer state found in checkpoint.")
        scheduler_state = checkpoint_scheduler_state_dict(checkpoint)
        if scheduler_state is not None:
            self.trainer.scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint_scaler_state_dict(checkpoint)
        if self.trainer.cfg.enable_amp and scaler_state is not None:
            self.trainer.scaler.load_state_dict(scaler_state)

    def _resume_legacy_training_state(self, checkpoint, iter_per_epoch):
        """Translate legacy epoch/iter fields into current trainer cursors."""
        checkpoint_epoch = int(checkpoint["epoch"])
        checkpoint_iter = int(checkpoint.get("iter", 0) or 0)
        self.trainer.logger.info(
            f"Resuming train at saved epoch: {checkpoint_epoch}, saved iteration: {checkpoint_iter}"
        )
        if 0 < checkpoint_iter < iter_per_epoch:
            self.trainer.logger.warning(
                "Legacy checkpoint is mid-epoch and has no dataloader state; "
                "resuming from the beginning of the saved epoch and replaying "
                "already-completed batches."
            )
            self.trainer.start_epoch = max(0, checkpoint_epoch - 1)
            self.trainer.start_iter = 0
        elif checkpoint_iter >= iter_per_epoch:
            self.trainer.start_epoch = checkpoint_epoch
            self.trainer.start_iter = 0
        else:
            self.trainer.start_epoch = checkpoint_epoch
            self.trainer.start_iter = 0
        self.trainer.global_step = (
            self.trainer.start_epoch * iter_per_epoch
            + self.trainer.start_iter
        )
        self.trainer.logger.info(
            "Resuming train at epoch index: "
            f"{self.trainer.start_epoch}, iteration: {self.trainer.start_iter}"
        )

    def load_optimizer_state(self, optimizer_state):
        """Load canonical optimizer state and fail if moments are not restored."""
        set_optimizer_state_dict(
            self.trainer.model,
            self.trainer.optimizer,
            optimizer_state,
            options=StateDictOptions(),
        )
        if optimizer_state.get("state") and not self.trainer.optimizer.state_dict().get("state"):
            raise RuntimeError(
                "Optimizer checkpoint contained state tensors, but optimizer resume "
                "left no optimizer state. Exact resume would restart optimizer moments."
            )
