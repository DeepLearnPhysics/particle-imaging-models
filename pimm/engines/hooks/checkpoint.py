"""Checkpoint hooks that delegate checkpoint semantics to CheckpointManager."""

from __future__ import annotations

from pimm.utils.checkpoints import CheckpointManager
from pimm.utils.comm import is_main_process

from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    """Save epoch/metric-oriented checkpoints (called on all ranks)."""

    def __init__(self, save_freq=None, evaluator_every_n_steps=None):
        """Configure periodic saves and optional metric-driven best snapshots."""
        self.save_freq = save_freq
        self.evaluator_every_n_steps = evaluator_every_n_steps
        self.step_count = 0

    def before_train(self):
        """Seed internal step count from resumed trainer progress."""
        self.checkpoint_manager = CheckpointManager(self.trainer)
        self.step_count = int(getattr(self.trainer, "global_step", 0) or (
            self.trainer.start_epoch * len(self.trainer.train_loader)
            + self.trainer.start_iter
        ))

    def _update_best(self, is_eval_step):
        """Return is_best from the metric, when this rank has one.

        Some evaluators publish ``current_metric_value`` on rank 0 only (they
        return early on non-main ranks), so the key may be absent here — in
        which case there is no "best" decision to make on this rank. The
        collective save itself is gated on rank-consistent state in
        ``after_step``, never on this metric, so ranks cannot diverge there.
        """
        if not is_eval_step or "current_metric_value" not in self.trainer.comm_info:
            return False
        current_metric_value = self.trainer.comm_info["current_metric_value"]
        current_metric_name = self.trainer.comm_info["current_metric_name"]
        is_best = False
        if current_metric_value > self.trainer.best_metric_value:
            self.trainer.best_metric_value = current_metric_value
            is_best = True
            if is_main_process():
                self.trainer.logger.info(
                    "Best validation {} updated to: {:.4f}".format(
                        current_metric_name, current_metric_value
                    )
                )
        if is_main_process():
            self.trainer.logger.info(
                "Currently Best {}: {:.4f}".format(
                    current_metric_name, self.trainer.best_metric_value
                )
            )
        return is_best

    def after_step(self):
        """Save after scheduled steps or after evaluator metric updates."""
        self.step_count += 1
        # The save decision MUST be identical on every rank: a standard-format
        # save is collective, so if some ranks save and others don't the run
        # deadlocks. Gate only on rank-consistent state (step counters) — never
        # on `"current_metric_value" in comm_info`, which gather-to-rank-0
        # evaluators publish on rank 0 only.
        is_eval_step = bool(
            self.trainer.cfg.evaluate
            and self.evaluator_every_n_steps
            and self.step_count % self.evaluator_every_n_steps == 0
        )
        is_save_step = bool(self.save_freq and self.step_count % self.save_freq == 0)
        if not is_eval_step and not is_save_step:
            return
        is_best = self._update_best(is_eval_step)
        # Called on ALL ranks: a standard-format save is a collective op.
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

    def __init__(self, keywords="", replacement=None, replacements=None, strict=False):
        """Configure key replacement and model weight strictness.

        Pass ``replacements`` as a ``{keyword: replacement}`` dict to apply several
        key rewrites in a single load (one ``load_state_dict``, so missing/unexpected
        keys are reported truthfully), e.g.::

            dict(type="CheckpointLoader", replacements={
                "module.backbone": "module.model.backbone",
                "module.decoder":  "module.model.decoder",
            })

        Overlapping keywords are resolved by longest match, so dict order does not
        matter. The scalar ``keywords``/``replacement`` form remains for the common
        single-rule case.
        """
        if replacements is not None:
            if not isinstance(replacements, dict):
                raise TypeError(
                    "CheckpointLoader.replacements must be a {keyword: replacement} "
                    f"dict, got {type(replacements).__name__}"
                )
            self.rules = list(replacements.items())
        else:
            self.rules = [(keywords, replacement if replacement is not None else keywords)]
        self.strict = strict

    def before_train(self):
        """Load configured weights before the trainer enters the train loop."""
        CheckpointManager(self.trainer).load_weight_and_resume(
            rules=self.rules,
            strict=self.strict,
        )


@HOOKS.register_module()
class CheckpointSaverIteration(HookBase):
    """Save iteration-oriented checkpoints (called on all ranks).

    ``backend`` is deprecated in favor of the top-level ``checkpoint_format``
    config key. ``backend="dcp"`` maps to the standard (hybrid) format and
    ``backend="torch"`` to legacy; the default (``None``) resolves to standard.
    """

    def __init__(self, save_freq=None, save_iter_checkpoints=False, backend=None):
        """Configure save cadence, iteration copies, and checkpoint format."""
        self.save_freq = save_freq
        self.save_iter_checkpoints = save_iter_checkpoints
        self.backend = backend
        self.step_count = 0

    def before_train(self):
        """Seed internal step count from resumed trainer progress."""
        self.checkpoint_manager = CheckpointManager(self.trainer)
        self.step_count = int(getattr(self.trainer, "global_step", 0) or (
            self.trainer.start_epoch * len(self.trainer.train_loader)
            + self.trainer.start_iter
        ))

    def _update_best(self):
        """Return is_best, updating best_metric_value identically on every rank."""
        if not (
            self.trainer.cfg.evaluate
            and "current_metric_value" in self.trainer.comm_info
        ):
            return False
        current_metric_value = self.trainer.comm_info["current_metric_value"]
        current_metric_name = self.trainer.comm_info["current_metric_name"]
        is_best = False
        if current_metric_value > self.trainer.best_metric_value:
            self.trainer.best_metric_value = current_metric_value
            is_best = True
            if is_main_process():
                self.trainer.logger.info(
                    "Best validation {} updated to: {:.4f}".format(
                        current_metric_name, current_metric_value
                    )
                )
        if is_main_process():
            self.trainer.logger.info(
                "Currently Best {}: {:.4f}".format(
                    current_metric_name, self.trainer.best_metric_value
                )
            )
        return is_best

    def after_step(self):
        """Save whenever the configured global-step cadence is reached."""
        self.step_count += 1
        if not self.save_freq or self.step_count % self.save_freq != 0:
            return
        if is_main_process():
            self.trainer.logger.info(f"Saving checkpoint at global step {self.step_count}")
        is_best = self._update_best()
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
