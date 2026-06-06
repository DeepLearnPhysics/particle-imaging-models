"""Checkpoint hooks that delegate checkpoint semantics to CheckpointManager."""

from __future__ import annotations

from pimm.utils.checkpoints import CheckpointManager
from pimm.utils.comm import is_main_process

from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    """Save epoch/metric-oriented torch checkpoints on the main process."""

    def __init__(self, save_freq=None, evaluator_every_n_steps=None):
        """Configure periodic saves and optional metric-driven best snapshots."""
        self.save_freq = save_freq
        self.evaluator_every_n_steps = evaluator_every_n_steps
        self.step_count = 0

    def before_train(self):
        """Seed internal step count from resumed trainer progress."""
        self.checkpoint_manager = CheckpointManager(self.trainer)
        self.checkpoint_manager.warn_non_dcp("CheckpointSaver")
        self.step_count = int(getattr(self.trainer, "global_step", 0) or (
            self.trainer.start_epoch * len(self.trainer.train_loader)
            + self.trainer.start_iter
        ))

    def after_step(self):
        """Save after scheduled steps or after evaluator metric updates."""
        self.step_count += 1
        is_eval_step = (
            self.trainer.cfg.evaluate
            and self.evaluator_every_n_steps
            and self.step_count % self.evaluator_every_n_steps == 0
            and "current_metric_value" in self.trainer.comm_info
        )
        is_save_step = bool(self.save_freq and self.step_count % self.save_freq == 0)
        if not is_eval_step and not is_save_step:
            return
        if is_main_process():
            is_best = False
            if is_eval_step:
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )
            self.checkpoint_manager.save_epoch_checkpoint(
                is_best=is_best,
                step_count=self.step_count,
                save_freq=self.save_freq,
            )

    def after_train(self):
        """Persist a final checkpoint when training finishes."""
        if is_main_process():
            self.trainer.logger.info("Saving final checkpoint")
            self.checkpoint_manager.save_epoch_checkpoint(
                is_best=False,
                step_count=self.step_count,
                save_freq=self.save_freq,
            )


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    """Load model weights and, when requested, resume optimizer/train state."""

    def __init__(self, keywords="", replacement=None, strict=False):
        """Configure key replacement and model weight strictness."""
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_train(self):
        """Load configured weights before the trainer enters the train loop."""
        CheckpointManager(self.trainer).load_weight_and_resume(
            keywords=self.keywords,
            replacement=self.replacement,
            strict=self.strict,
        )


@HOOKS.register_module()
class CheckpointSaverIteration(HookBase):
    """Save iteration-oriented checkpoints, optionally using torch DCP."""

    def __init__(self, save_freq=None, save_iter_checkpoints=False, backend="torch"):
        """Configure save cadence, iteration copies, and checkpoint backend."""
        self.save_freq = save_freq
        self.save_iter_checkpoints = save_iter_checkpoints
        self.backend = backend
        self.step_count = 0

    def before_train(self):
        """Seed internal step count from resumed trainer progress."""
        self.checkpoint_manager = CheckpointManager(self.trainer)
        if self.backend != "dcp":
            self.checkpoint_manager.warn_non_dcp(
                f"CheckpointSaverIteration backend={self.backend!r}"
            )
        self.step_count = int(getattr(self.trainer, "global_step", 0) or (
            self.trainer.start_epoch * len(self.trainer.train_loader)
            + self.trainer.start_iter
        ))

    def after_step(self):
        """Save whenever the configured global-step cadence is reached."""
        self.step_count += 1
        if not self.save_freq or self.step_count % self.save_freq != 0:
            return
        if is_main_process():
            self.trainer.logger.info(f"Saving checkpoint at global step {self.step_count}")
        is_best = False
        if (
            is_main_process()
            and self.trainer.cfg.evaluate
            and "current_metric_value" in self.trainer.comm_info.keys()
        ):
            current_metric_value = self.trainer.comm_info["current_metric_value"]
            current_metric_name = self.trainer.comm_info["current_metric_name"]
            if current_metric_value > self.trainer.best_metric_value:
                self.trainer.best_metric_value = current_metric_value
                is_best = True
                self.trainer.logger.info(
                    "Best validation {} updated to: {:.4f}".format(
                        current_metric_name, current_metric_value
                    )
                )
            self.trainer.logger.info(
                "Currently Best {}: {:.4f}".format(
                    current_metric_name, self.trainer.best_metric_value
                )
            )
        self.checkpoint_manager.save_iteration_checkpoint(
            backend=self.backend,
            is_best=is_best,
            step_count=self.step_count,
            save_freq=self.save_freq,
            save_iter_checkpoints=self.save_iter_checkpoints,
        )

    def after_train(self):
        """Persist a final checkpoint after the last training step."""
        if is_main_process():
            self.trainer.logger.info("Saving final checkpoint")
        self.checkpoint_manager.save_iteration_checkpoint(
            backend=self.backend,
            is_best=False,
            step_count=self.step_count,
            save_freq=self.save_freq,
            save_iter_checkpoints=self.save_iter_checkpoints,
        )
