"""Internal exex bridge. Importing this module does not initialize training."""

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pimm.utils.config import Config


def configure_training(cfg: "Config") -> bool:
    """Attach checkpoint safe points and report whether this job is managed."""
    if os.environ.get("PIMM_USE_EXEX") != "1" or "EXEX_OUTPUT_DIR" not in os.environ:
        return False
    from exex import execution

    from pimm.engines.hooks.builder import HOOKS
    from pimm.engines.hooks.default import HookBase
    from pimm.utils import comm
    from pimm.utils.checkpoints import CheckpointManager, publish_full_state_snapshot
    from pimm.utils.path import latest_complete_checkpoint

    cfg.save_path = str(Path(os.environ["EXEX_OUTPUT_DIR"]) / "checkpoint")
    (Path(cfg.save_path) / "model").mkdir(parents=True, exist_ok=True)
    previous = Path(os.environ.get("EXEX_INPUT_DIR", "/nonexistent")) / "checkpoint"
    if previous.exists():
        checkpoint = latest_complete_checkpoint(previous / "model")
        if checkpoint is None:
            raise FileNotFoundError(f"No complete pimm checkpoint in {previous}")
        cfg.weight, cfg.resume = str(checkpoint), True
    if cfg.resume:
        metadata = Path(cfg.weight).parent.parent / "resolved_config.json"
        if metadata.is_file():
            cfg.seed = json.loads(metadata.read_text())["seed"]
    # A managed continuation must restore full state, even for custom hook lists.
    if not any(hook["type"] == "CheckpointLoader" for hook in cfg.hooks):
        cfg.hooks.insert(0, {"type": "CheckpointLoader"})

    class _ExecutionCheckpoint(HookBase):
        def before_train(self):
            owner = self.trainer
            if not owner.cfg.resume or not comm.is_main_process():
                return
            source = Path(owner.cfg.weight)
            best = source.parent / "model_best.pth"
            if best.is_file():
                # Legacy saves overwrite this file in place; do not hard-link it.
                shutil.copy2(best, Path(owner.cfg.save_path) / "model/model_best.pth")
            # A completed resume exits without after_train. Retain its input
            # unchanged so the new attempt still has a resumable output.
            if owner._training_already_complete():
                destination = "last" if source.is_dir() else "model_last.pth"
                publish_full_state_snapshot(
                    source, Path(owner.cfg.save_path) / "model" / destination
                )

        def after_step(self):
            owner = self.trainer
            if owner.comm_info["iter"] + 1 < owner.comm_info["iter_per_epoch"]:
                self._pause()

        def after_epoch(self):
            if self.trainer.train_state.epoch >= self.trainer.max_epoch:
                # Final evaluation may load best weights into the training model.
                self._save()
            else:
                self._pause()

        def _pause(self):
            owner = self.trainer
            if (
                "EXEX_PAUSE_REQUEST" not in os.environ
                or owner.train_state.epoch >= owner.max_epoch
            ):
                return
            requested = comm.all_gather(
                execution.pause_requested() if comm.is_main_process() else False
            )[0]
            if requested:
                self._save()
                comm.synchronize()
                if comm.is_main_process():
                    execution.mark_paused()
                owner.request_pause()

        def _save(self):
            CheckpointManager(self.trainer).save_epoch_checkpoint(
                step_count=self.trainer.global_step,
            )

    HOOKS.register_module(
        name="_ExecutionCheckpoint", module=_ExecutionCheckpoint, force=True
    )
    cfg.hooks.append({"type": "_ExecutionCheckpoint"})
    return True
