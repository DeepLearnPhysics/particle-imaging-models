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


HF_URI_PREFIX = "hf://"

# Serialized-weights filenames a pimm export may contain (one of them), in
# preference order. Centralized so download/upload/probe sites stay in sync.
EXPORT_WEIGHT_NAMES = ("model.safetensors", "model.bin")

# Config filename a pimm export writes (HF-idiomatic `config.json`) and the
# names it will read, in preference order. `training_config.json` is the legacy
# name kept for backward compatibility with already-published exports; the run
# dir may also carry resolved_config.json / model_config.json.
EXPORT_CONFIG_NAME = "config.json"
EXPORT_CONFIG_READ_NAMES = (
    "config.json",
    "training_config.json",
    "resolved_config.json",
    "model_config.json",
)


def hf_cache_dir():
    """Resolve pimm's preferred HF download cache directory.

    Priority: ``PIMM_HF_CACHE`` > ``$MODEL_DIR/hub`` (keep Hub downloads next to
    checkpoints) > ``None`` (HF's own ``HF_HOME``/``HF_HUB_CACHE`` defaults).
    """
    explicit = os.environ.get("PIMM_HF_CACHE")
    if explicit:
        return explicit
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        return os.path.join(model_dir, "hub")
    return None


def configure_hf_cache():
    """Point Hugging Face's own cache env at pimm's cache, once, so EVERY hub
    download in this process shares one location -- pimm's `hf://` warm-start and
    `from_pretrained`, the `PushToHub` `HfApi`, and any direct `huggingface_hub`
    use. Sets ``HF_HUB_CACHE`` rather than threading ``cache_dir=`` per call.

    Precedence: ``PIMM_HF_CACHE`` (always wins) > an existing
    ``HF_HUB_CACHE``/``HF_HOME`` (respected, already shared) > ``$MODEL_DIR/hub``
    > HF's default. Returns the active cache dir (or ``None``).
    """
    explicit = os.environ.get("PIMM_HF_CACHE")
    if explicit:
        os.environ["HF_HUB_CACHE"] = explicit
        return explicit
    if os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME"):
        return os.environ.get("HF_HUB_CACHE")
    target = hf_cache_dir()  # PIMM_HF_CACHE is empty here, so this is $MODEL_DIR/hub or None
    if target:
        os.environ["HF_HUB_CACHE"] = target
    return target


def is_remote_weight(uri):
    """Return True if ``uri`` is a remote weight reference (currently ``hf://``)."""
    return isinstance(uri, str) and uri.startswith(HF_URI_PREFIX)


def parse_hf_uri(uri):
    """Parse an ``hf://`` URI into ``(repo_id, revision, filename)``.

    A Hub repo id is ``namespace/name`` (exactly one slash); an optional
    ``@revision`` attaches to it, and anything after is the in-repo file path.
    ``revision``/``filename`` are ``None``/``""`` when absent.
    """
    if not is_remote_weight(uri):
        raise ValueError(f"Not an hf:// reference: {uri}")
    spec = uri[len(HF_URI_PREFIX):]
    revision = None
    if "@" in spec:
        repo_id, _, rest = spec.partition("@")
        revision, _, filename = rest.partition("/")
        revision = revision or None
    else:
        parts = spec.split("/", 2)
        if len(parts) >= 2 and parts[1]:
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = parts[2] if len(parts) == 3 else ""
        else:
            repo_id = parts[0]
            filename = ""
    if not repo_id:
        raise ValueError(f"Malformed hf:// reference (missing repo id): {uri}")
    return repo_id, revision, filename


def resolve_remote_weight(uri):
    """Resolve an ``hf://`` weight reference to a local path, downloading on first use.

    Accepted forms (non-``hf://`` strings are returned unchanged)::

        hf://<repo_id>                     -> the repo's single weights file
        hf://<repo_id>/<path/to/file>      -> a single file (e.g. model_best.pth)
        hf://<repo_id>@<revision>/<file>   -> a file at a branch/tag/commit

    Downloads land in the Hugging Face cache (``PIMM_HF_CACHE`` overrides, else
    ``HF_HOME``); subsequent runs hit the cache. In a multi-rank job only each
    node's local rank 0 fetches; the others read the warm cache after a barrier,
    so the Hub is hit once per node (correct for node-local caches).
    """
    if not is_remote_weight(uri):
        return uri
    repo_id, revision, filename = parse_hf_uri(uri)

    def _download():
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "huggingface_hub is required to load hf:// weights"
            ) from exc
        # Export pimm's cache to HF's env (process-wide sharing) AND pass it
        # explicitly -- HF reads HF_HUB_CACHE into a constant at import time, so
        # the explicit cache_dir keeps our own call correct regardless of timing.
        cache = configure_hf_cache()
        target = filename
        if not target:
            # Repo form (no file): pick the single weights file from the repo
            # listing -- prefer a consolidated export, else a raw checkpoint --
            # and fetch only that, so a repo that also holds a large raw .pth is
            # not pulled in full and a raw-only repo still resolves.
            target = _pick_repo_weight_file(
                HfApi().list_repo_files(repo_id=repo_id, revision=revision), repo_id
            )

        # Report progress via the global logger: a start line (ref + cache dir), a
        # heartbeat every 15s with bytes-so-far (so a multi-minute fetch never
        # looks hung), and a final line with the saved path, size, and elapsed.
        import glob
        import threading
        import time

        from huggingface_hub.constants import HF_HUB_CACHE
        from pimm.utils.logger import get_root_logger

        log = get_root_logger()
        ref = f"hf://{repo_id}/{target}" + (f"@{revision}" if revision else "")
        cache_dir = cache or HF_HUB_CACHE
        blob_dir = os.path.join(cache_dir, f"models--{repo_id.replace('/', '--')}", "blobs")

        # Total size up front (one cheap HEAD) so the heartbeat can show %/ETA.
        total_mb = None
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url

            meta = get_hf_file_metadata(hf_hub_url(repo_id, target, revision=revision))
            if meta.size:
                total_mb = meta.size / 1e6
        except Exception:  # pragma: no cover - best-effort metadata
            pass

        total_str = f" ({total_mb:.0f} MB)" if total_mb else ""
        log.info(f"Fetching weight {ref}{total_str}  (cache dir: {cache_dir}) ...")

        def _partial_mb():
            """Best-effort size of the in-flight `.incomplete` blob, in MB."""
            try:
                parts = glob.glob(os.path.join(blob_dir, "*.incomplete"))
                if parts:
                    return max(os.path.getsize(p) for p in parts) / 1e6
            except OSError:
                pass
            return 0.0

        start = time.time()
        done = threading.Event()
        last = {"t": start, "mb": 0.0}

        def _heartbeat():
            # hf's own tqdm goes to stderr and is invisible in captured/non-TTY
            # job logs, so surface real progress (MB / % / MB-s / ETA) here.
            while not done.wait(10):
                now = time.time()
                mb = _partial_mb()
                speed = (mb - last["mb"]) / max(now - last["t"], 1e-9)
                last["t"], last["mb"] = now, mb
                if total_mb and total_mb > 0:
                    pct = 100.0 * mb / total_mb
                    eta = (total_mb - mb) / speed if speed > 1e-6 else None
                    eta_str = f", ETA {eta:.0f}s" if eta is not None else ""
                    log.info(
                        f"  ... {target}: {mb:.0f}/{total_mb:.0f} MB "
                        f"({pct:.0f}%), {speed:.0f} MB/s{eta_str}"
                    )
                else:
                    log.info(
                        f"  ... downloading {target}: {mb:.0f} MB, "
                        f"{speed:.0f} MB/s ({int(now - start)}s)"
                    )

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()
        try:
            path = hf_hub_download(
                repo_id=repo_id, filename=target, revision=revision, cache_dir=cache
            )
        finally:
            done.set()
        elapsed = time.time() - start
        try:
            size_mb = os.path.getsize(path) / 1e6
        except OSError:
            size_mb = float("nan")
        cached = " (from cache)" if elapsed < 1.0 else ""
        log.info(
            f"Resolved weight {ref}{cached}\n"
            f"  saved to: {path}\n"
            f"  size:     {size_mb:.1f} MB    elapsed: {elapsed:.0f}s"
        )
        return path

    if comm.get_world_size() > 1:
        # Only each node's local lead hits the Hub (one download per node-local
        # cache). Every other rank then receives the resolved local path via
        # all_gather and reads the warm cache directly -- it never calls the Hub
        # (no list_repo_files, no HEAD), so there is exactly one set of network
        # requests per node instead of one per rank. all_gather also acts as the
        # barrier: it returns only once the lead has finished downloading.
        local_path = _download() if comm.get_local_rank() == 0 else None
        gathered: list = [None] * comm.get_world_size()
        # Gather over GLOO (CPU), not NCCL: the non-lead ranks block here for the
        # whole download, and an NCCL collective would trip its watchdog timeout
        # on a multi-minute fetch (the classic "hang"). GLOO has no such watchdog.
        torch.distributed.all_gather_object(
            gathered, local_path, group=comm._get_global_gloo_group()
        )
        # The cache path is identical across nodes (same HF_HUB_CACHE + repo +
        # commit), and each node's lead populated it locally, so any lead's path
        # resolves on every rank.
        resolved = next((p for p in gathered if p), None)
        if resolved is None:
            raise RuntimeError(f"hf:// download produced no path on any rank: {uri}")
        return resolved
    return _download()


def _pick_repo_weight_file(files, repo_id):
    """Choose the one weights file to load from a Hub repo's file listing."""
    for name in EXPORT_WEIGHT_NAMES:
        if name in files:
            return name
    pths = [f for f in files if f.endswith(".pth") and "/" not in f]
    for name in ("model_best.pth", "model_last.pth"):
        if name in pths:
            return name
    if len(pths) == 1:
        return pths[0]
    raise FileNotFoundError(
        f"No loadable weights file in hf://{repo_id} (saw: {sorted(files)}). "
        "Use the explicit file form hf://<repo>/<file> to disambiguate."
    )


def exported_weights_file(path):
    """Return the weights file inside a pimm-export directory, or None.

    A pimm export (``save_pretrained``) writes the serialized tensors as
    ``model.safetensors`` (default) or ``model.bin``; this probes for them.
    """
    for name in EXPORT_WEIGHT_NAMES:
        candidate = os.path.join(str(path), name)
        if os.path.isfile(candidate):
            return candidate
    return None


def load_weight_state(path, map_location):
    """Load a raw checkpoint or a ``.safetensors`` file into a state mapping.

    ``safetensors`` needs a device string; honor an explicit string
    ``map_location`` (e.g. ``"cpu"``) and otherwise fall back to GPU-if-available
    (mirroring the torch ``map_location`` lambda used for resume loads).
    """
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        if isinstance(map_location, str):
            device = map_location
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return load_file(str(path), device=device)
    return torch.load(path, map_location=map_location, weights_only=False)


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


def build_logger_state(
    trainer,
    *,
    checkpoint_global_step=None,
    initialize_wandb=False,
):
    """Build checkpointable logging state, using the active W&B run when present."""
    use_wandb = bool(_cfg_get(trainer.cfg, "use_wandb", False))
    wandb_state = {
        "group": _cfg_get(trainer.cfg, "wandb_group", None),
        "run_name": _cfg_get(trainer.cfg, "wandb_run_name", None),
        "run_id": _cfg_get(trainer.cfg, "wandb_run_id", None),
        "job_type": _cfg_get(trainer.cfg, "wandb_job_type", None),
        "resume": _cfg_get(trainer.cfg, "wandb_resume", None),
        "resume_step": None,
        "step_metric": "train/global_step",
        "step_offset": _cfg_get(trainer.cfg, "log_step_offset", 0),
        "checkpoint_global_step": checkpoint_global_step,
    }
    if use_wandb and initialize_wandb:
        writer = getattr(trainer, "writer", None)
        local_state = None
        if is_main_process() and writer is not None:
            checkpoint_state = getattr(writer, "checkpoint_state", None)
            if checkpoint_state is not None:
                local_state = checkpoint_state()
        gathered = comm.all_gather(local_state)
        active_state = next((state for state in gathered if state is not None), None)
        if active_state is None:
            raise RuntimeError(
                "W&B checkpointing requires the main process writer to expose "
                "checkpoint_state()."
            )
        wandb_state.update(active_state)
    return {
        "backend": "wandb" if use_wandb else "tensorboard",
        "wandb": wandb_state,
    }


def configure_logger_from_checkpoint(trainer, checkpoint):
    """Configure the lazy W&B writer to rewind to checkpoint history state."""
    if not _cfg_get(trainer.cfg, "use_wandb", False):
        return
    writer = getattr(trainer, "writer", None)
    configure_resume = getattr(writer, "resume_from_checkpoint", None)
    if configure_resume is None:
        return

    logger_state = checkpoint.get("logger", {})
    wandb_state = logger_state.get("wandb", {}) if isinstance(logger_state, dict) else {}
    if not isinstance(wandb_state, dict):
        wandb_state = {}
    run_id = wandb_state.get("run_id", checkpoint.get("wandb_run_id"))
    resume_step = wandb_state.get(
        "resume_step",
        checkpoint.get("wandb_resume_step"),
    )
    if run_id and resume_step is not None:
        configure_resume({"run_id": run_id, "resume_step": resume_step})
        return
    if resume_step is not None:
        raise ValueError(
            "Checkpoint contains a W&B resume_step without a W&B run_id."
        )
    trainer.logger.warning(
        "Checkpoint has no W&B rewind state; falling back to configured W&B "
        "resume behavior. Optimizer steps replayed after rollback may appear "
        "more than once."
    )


def build_checkpoint_payload(
    trainer,
    *,
    distributed_rng=False,
    initialize_logger=False,
):
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
    logger_state = build_logger_state(
        trainer,
        checkpoint_global_step=train_state.global_step,
        initialize_wandb=initialize_logger,
    )
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


def save_model_weights_file(payload, filename):
    """Save just the model weights as a portable single-file checkpoint.

    Produces the same ``{"state_dict": ...}`` layout as the split-checkpoint
    weight file, so it loads identically via ``checkpoint_model_state_dict`` and
    a plain ``torch.load`` (used for ``model_best.pth`` and iter snapshots).
    """
    atomic_torch_save({"state_dict": checkpoint_model_state_dict(payload)}, filename)


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
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    dcp.load(payload, checkpoint_id=checkpoint_dir,
             planner=DefaultLoadPlanner(allow_partial_load=True))
    return payload


def load_dcp_checkpoint(checkpoint_dir, trainer):
    """Load a complete full DCP checkpoint into a typed placeholder payload."""
    import torch.distributed.checkpoint as dcp

    if not is_complete_dcp_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"Incomplete DCP checkpoint: {checkpoint_dir}")
    payload = empty_checkpoint_payload(trainer)
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    dcp.load(payload, checkpoint_id=checkpoint_dir,
             planner=DefaultLoadPlanner(allow_partial_load=True))
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


def _summarize_keys(keys, *, depth=3, examples=2, max_groups=12):
    """Collapse a list of dotted state-dict keys into a few grouped, counted lines.

    A 500-key mismatch on a deep model is unreadable when dumped in full and
    near-identical across ranks. Grouping by a shallow prefix (e.g.
    ``model.backbone.enc``) turns it into a handful of "(N): example, ..." lines
    that still name the offending subtree.
    """
    strip_module = lambda s: s[7:] if s.startswith("module.") else s  # noqa: E731
    groups: "OrderedDict[str, list]" = OrderedDict()
    for key in keys:
        bare = strip_module(key)
        prefix = ".".join(bare.split(".")[:depth])
        groups.setdefault(prefix, []).append(bare)
    lines = []
    for prefix in sorted(groups)[:max_groups]:
        members = groups[prefix]
        ex = ", ".join(members[:examples])
        more = f", +{len(members) - examples} more" if len(members) > examples else ""
        lines.append(f"    {prefix}.* ({len(members)}): {ex}{more}")
    if len(groups) > max_groups:
        lines.append(f"    ... and {len(groups) - max_groups} more group(s)")
    return "\n".join(lines)


class CheckpointManager:
    """Own checkpoint format, save/load backends, and trainer resume semantics."""

    def __init__(self, trainer):
        self.trainer = trainer

    def _checkpoint_format(self, hook_backend=None):
        """Resolve the on-disk checkpoint format."""
        aliases = {"dcp": "standard", "torch": "legacy"}
        fmt = _cfg_get(self.trainer.cfg, "checkpoint_format", None)
        if fmt is None:
            fmt = hook_backend
        fmt = str(fmt or "standard").lower()
        fmt = aliases.get(fmt, fmt)
        if fmt not in ("standard", "legacy"):
            raise ValueError(
                "checkpoint_format must be 'standard' or 'legacy' "
                f"(or the deprecated 'dcp'/'torch'), got {fmt!r}"
            )
        return fmt

    def _write_checkpoint(
        self, payload, *, fmt, is_best, step_count, save_freq, save_iter_checkpoints
    ):
        """Write a built payload in the resolved format.

        Must be called on ALL ranks: the ``standard`` format performs a
        collective DCP save. Rank-0-only side artifacts (the best/iter weight
        files and the legacy single file) are guarded internally.
        """
        model_dir = os.path.join(self.trainer.cfg.save_path, "model")
        best_file = os.path.join(model_dir, "model_best.pth")
        do_iter_snapshot = bool(
            save_iter_checkpoints and save_freq and step_count and step_count % save_freq == 0
        )
        if fmt == "standard":
            last_dir = os.path.join(model_dir, "last")
            if is_main_process():
                self.trainer.logger.info(
                    f"Saving checkpoint to: {last_dir} (weights.pth + trainer/ DCP)"
                )
            save_split_checkpoint(payload, last_dir)  # collective: all ranks
            if is_main_process():
                if is_best:
                    save_model_weights_file(payload, best_file)
                if do_iter_snapshot:
                    save_model_weights_file(
                        payload, os.path.join(model_dir, f"iter_{step_count}.pth")
                    )
            return
        # legacy: single monolithic file, written by rank 0 only
        if is_main_process():
            filename = os.path.join(model_dir, "model_last.pth")
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            atomic_torch_save(payload, filename)
            if is_best:
                shutil.copyfile(filename, best_file)
            if do_iter_snapshot:
                shutil.copyfile(
                    filename, os.path.join(model_dir, f"iter_{step_count}.pth")
                )

    def save_epoch_checkpoint(self, *, is_best=False, step_count=0, save_freq=None):
        """Save an epoch/metric-oriented checkpoint. Must run on all ranks."""
        fmt = self._checkpoint_format()
        payload = build_checkpoint_payload(
            self.trainer,
            distributed_rng=True,
            initialize_logger=True,
        )
        self._write_checkpoint(
            payload,
            fmt=fmt,
            is_best=is_best,
            step_count=step_count,
            save_freq=save_freq,
            save_iter_checkpoints=bool(save_freq),
        )

    def save_iteration_checkpoint(
        self,
        *,
        backend=None,
        is_best=False,
        step_count=0,
        save_freq=None,
        save_iter_checkpoints=False,
    ):
        """Save an iteration-oriented checkpoint. Must run on all ranks."""
        fmt = self._checkpoint_format(backend)
        payload = build_checkpoint_payload(
            self.trainer,
            distributed_rng=True,
            initialize_logger=True,
        )
        self._write_checkpoint(
            payload,
            fmt=fmt,
            is_best=is_best,
            step_count=step_count,
            save_freq=save_freq,
            save_iter_checkpoints=save_iter_checkpoints,
        )

    def load_weight_and_resume(self, *, keywords="", replacement=None, rules=None, strict=False):
        """Load configured weights and restore training state when cfg.resume is true.

        Pass ``rules`` (a list of ``(keywords, replacement)`` pairs) to apply several
        key rewrites in a single load; the scalar ``keywords``/``replacement`` form is
        kept for back-compat and is treated as a one-rule list.
        """
        if rules is None:
            rules = [(keywords, replacement if replacement is not None else keywords)]
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        weight_path = self.trainer.cfg.weight
        if is_remote_weight(weight_path):
            if self.trainer.cfg.resume:
                raise ValueError(
                    f"resume=True is not supported with an hf:// weight ({weight_path}). "
                    "The Hub holds model weights only, not trainer state "
                    "(optimizer/scheduler/step/dataloader — the DCP 'trainer.dcp/' is "
                    "never uploaded). Set resume=False to warm-start a new run from these "
                    "weights, or point `weight` at a local checkpoint dir (.../model/last) "
                    "to resume the original run."
                )
            self.trainer.logger.info(f"Resolving remote weight: {weight_path}")
            weight_path = resolve_remote_weight(weight_path)
        if weight_path and (os.path.isfile(weight_path) or os.path.isdir(weight_path)):
            self.trainer.logger.info(f"Loading weight at: {weight_path}")
            checkpoint = self._load_checkpoint(weight_path)
            self._load_model_weights(checkpoint, rules=rules, strict=strict)
            if self.trainer.cfg.resume:
                self.resume_training_state(checkpoint)
            return

        message = f"No weight found at: {weight_path}"
        # A non-empty weight path that does not resolve is always an error: the
        # user asked to load weights, so silently training from random init would
        # hide a typo'd/moved checkpoint. Only the genuinely-unset case is a no-op
        # (unless resuming, which requires a checkpoint).
        if weight_path or self.trainer.cfg.resume:
            raise FileNotFoundError(message)
        self.trainer.logger.info(message)

    def _load_checkpoint(self, weight_path):
        """Load a direct, split, or directory checkpoint reference."""
        map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else "cpu"
        if os.path.isdir(weight_path):
            exported = exported_weights_file(weight_path)
            if exported is not None and not self.trainer.cfg.resume:
                return load_weight_state(exported, map_location)
            if is_complete_split_checkpoint(weight_path):
                if self.trainer.cfg.resume:
                    return load_split_checkpoint(weight_path, self.trainer, map_location)
                weight_file = resolve_model_weight_file(weight_path)
                return load_weight_state(weight_file, map_location)
            if is_complete_dcp_checkpoint(weight_path):
                return load_dcp_checkpoint(weight_path, self.trainer)
            if self.trainer.cfg.resume:
                raise FileNotFoundError(f"Incomplete checkpoint directory: {weight_path}")
            weight_file = resolve_model_weight_file(weight_path)
            return load_weight_state(weight_file, map_location)
        return load_weight_state(weight_path, map_location)

    def _load_model_weights(self, checkpoint, *, rules=None, strict=False):
        """Load checkpoint model weights, applying all keyword-rewrite rules in one pass.

        ``rules`` is a list of ``(keywords, replacement)`` pairs. Each source key is
        rewritten by the *most specific* (longest-keyword) rule whose (module-stripped)
        keyword it starts with, and every key lands in a single state dict. Because
        there is one ``load_state_dict``, the reported missing/unexpected keys are the
        truth about the final mapping -- unlike stacking one loader per rule, where
        each rule's ``load_state_dict`` flags the keys another rule owns as "missing".

        Matching by longest keyword (not input order) means rules can be passed as a
        plain ``{keyword: replacement}`` dict without order-dependent surprises when
        two keywords overlap (e.g. ``decoder`` vs ``decoder.cls_pred``).
        """
        rules = rules or [("", "")]
        strip_module = lambda s: s[7:] if s.startswith("module.") else s  # noqa: E731
        norm_rules = sorted(
            ((strip_module(kw), strip_module(repl)) for kw, repl in rules),
            key=lambda r: len(r[0]),
            reverse=True,
        )
        if is_main_process():
            for kw, repl in rules:
                self.trainer.logger.info(
                    f"Weight key rule: {kw or '<all>'!r} -> {repl or '<unchanged>'!r}"
                )

        weight = OrderedDict()
        ddp = comm.get_world_size() > 1
        for key, value in checkpoint_model_state_dict(checkpoint).items():
            bare = strip_module(key)
            for kw, repl in norm_rules:
                if kw and bare.startswith(kw):
                    bare = repl + bare[len(kw):]
                    break
            weight["module." + bare if ddp else bare] = value
        # Skip shape-mismatched keys (load_state_dict(strict=False) still raises on
        # these). Lets a checkpoint warm-start a model whose architecture changed
        # shape -- e.g. different num_classes (class head) or num_queries (query
        # embeddings); the mismatched tensors stay at their init.
        mismatched = []
        if not strict:
            model_sd = self.trainer.model.state_dict()
            mismatched = [
                k for k, v in weight.items()
                if k in model_sd and tuple(model_sd[k].shape) != tuple(v.shape)
            ]
            for k in mismatched:
                del weight[k]
            if mismatched and is_main_process():
                self.trainer.logger.info(
                    f"Skipped {len(mismatched)} shape-mismatched key(s), kept at init:\n"
                    f"{_summarize_keys(mismatched)}"
                )
        missing, unexpected = self.trainer.model.load_state_dict(weight, strict=strict)
        n_model = len(self.trainer.model.state_dict())
        n_loaded = n_model - len(missing)
        # `missing` from load_state_dict includes the shape-skipped keys above; those
        # are reported separately, so don't double-count them as a problem.
        real_missing = [k for k in missing if k not in set(mismatched)]
        if is_main_process():
            self.trainer.logger.info(
                f"Loaded {n_loaded}/{n_model} model params "
                f"(missing: {len(missing)}, unexpected: {len(unexpected)})"
            )
            # One combined pass -> these are genuinely absent from the checkpoint.
            # Surface at WARNING with a grouped summary instead of a per-key wall.
            if real_missing:
                self.trainer.logger.warning(
                    f"{len(real_missing)} model param(s) not in checkpoint, left at "
                    f"init:\n{_summarize_keys(real_missing)}"
                )
            if unexpected:
                self.trainer.logger.info(
                    f"{len(unexpected)} checkpoint key(s) unused by the model:\n"
                    f"{_summarize_keys(unexpected)}"
                )
        # Guard against rules that matched nothing: with strict=False this would
        # otherwise leave the whole model randomly initialized while training
        # proceeds, with only an INFO line to distinguish it from a real load.
        if n_model and n_loaded == 0:
            raise RuntimeError(
                f"Checkpoint load matched 0 of {n_model} model parameters "
                f"(rules={rules!r}). The model would train from random init. "
                f"Fix the keyword/replacement rules, or set strict=True to find the "
                f"mismatch."
            )

    def resume_training_state(self, checkpoint):
        """Restore structured or legacy optimizer, scheduler, RNG, and cursor state."""
        configure_logger_from_checkpoint(self.trainer, checkpoint)
        strict_state = self.trainer.cfg.get("resume_strict_state", True)
        iter_per_epoch = len(self.trainer.train_loader)

        train_state = checkpoint_train_state(checkpoint)
        if train_state is not None:
            dataloader_state = checkpoint_dataloader_state(checkpoint, train_state)
            train_state.dataloader_state = dataloader_state
            # Decide whether to drop the torchdata StatefulDataLoader cursor BEFORE
            # extracting it. That cursor asserts (lazily, on the first __iter__) on
            # ANY change to the world_size or num_workers it was saved with, and
            # local_object_state() below would itself raise on a world_size change
            # under strict resume. So when the world_size or num_workers changed
            # (or the resume is explicitly non-strict) we drop the cursor and
            # restart the resumed epoch from its first batch -- model / optimizer /
            # scheduler / global_step still restore below, so at most a sub-epoch of
            # data order is replayed. This makes resharding across GPU/worker counts
            # automatic without requiring resume_strict_state=False.
            skip_cursor, reason = False, ""
            if not strict_state:
                skip_cursor, reason = True, "resume_strict_state=False"
            elif isinstance(dataloader_state, dict) and dataloader_state.get(
                "_pimm_distributed_state"
            ):
                saved_ws = int(
                    dataloader_state.get(
                        "world_size", len(dataloader_state.get("states", []))
                    )
                )
                cur_ws = comm.get_world_size()
                if saved_ws != cur_ws:
                    skip_cursor, reason = True, f"world_size {saved_ws}->{cur_ws}"
            if not skip_cursor:
                saved_dl = checkpoint.get("dataloader", {})
                saved_workers = saved_dl.get("num_workers") if isinstance(saved_dl, dict) else None
                cur_workers = self.trainer.cfg.get("num_worker_per_gpu", None)
                if (
                    saved_workers is not None
                    and cur_workers is not None
                    and int(saved_workers) != int(cur_workers)
                ):
                    skip_cursor, reason = True, f"num_workers {saved_workers}->{cur_workers}"
            # Extract this rank's cursor. When we are going to drop it anyway, load
            # non-strictly so a world_size change does not raise here.
            local_dataloader_state = local_object_state(
                dataloader_state,
                strict=strict_state and not skip_cursor,
            )
            if skip_cursor and local_dataloader_state:
                self.trainer.logger.warning(
                    f"Skipping dataloader-cursor restore ({reason}); restarting the "
                    "resumed epoch from its first batch."
                )
                local_dataloader_state = None
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
