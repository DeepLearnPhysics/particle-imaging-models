"""Checkpoint hooks that delegate checkpoint semantics to CheckpointManager."""

from __future__ import annotations

from pimm.utils.checkpoints import CheckpointManager
from pimm.utils.comm import is_main_process

from .builder import HOOKS
from .default import HookBase


def _is_last_epoch_step(trainer):
    """Return whether the current optimizer step finishes its epoch."""
    current_iter = int(trainer.comm_info.get("iter", -1)) + 1
    iter_per_epoch = trainer.comm_info.get("iter_per_epoch")
    if iter_per_epoch is None:
        try:
            iter_per_epoch = len(trainer.train_loader)
        except TypeError:
            return False
    return current_iter >= int(iter_per_epoch)


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    """Save epoch/metric-oriented checkpoints during and after training.

    Maintains a running global-step counter (seeded from resumed trainer
    progress in ``before_train``) and, in ``after_step``, saves a checkpoint
    whenever either cadence fires: a periodic save every ``save_freq`` steps, or
    an evaluator step every ``evaluator_every_n_steps`` steps (only when
    ``cfg.evaluate`` is set). On an evaluator step it also reads the evaluator's
    ``current_metric_value`` from ``trainer.comm_info`` and, if it improves on
    ``trainer.best_metric_value``, writes a ``model_best`` snapshot. A final
    checkpoint is written in ``after_train``. All actual save logic is delegated
    to :class:`~pimm.utils.checkpoints.CheckpointManager`. Registered as
    ``CheckpointSaver``.

    Args:
        save_freq (int, optional): Save a rolling checkpoint every this many
            global steps. ``None`` disables periodic saving. Defaults to ``None``.
        evaluator_every_n_steps (int, optional): Treat every this many steps as
            an evaluation step, consulting the metric and updating ``model_best``
            on improvement. Requires ``cfg.evaluate``. ``None`` disables.
            Defaults to ``None``.

    Note:
        Runs on **all** ranks because a standard-format (DCP) save is a
        collective operation; the save decision is gated only on
        rank-consistent step counters, never on the metric (which some
        evaluators publish on rank 0 only), so ranks cannot diverge and
        deadlock. Place any evaluator hook **before** this saver in the
        ``hooks`` list so the metric is available when the saver runs.

    Example:
        Add to ``cfg.hooks`` after an evaluator; it saves a rolling checkpoint
        every ``save_freq`` steps and, on evaluator steps, promotes
        ``model_best`` whenever the evaluator's metric improves:

        .. code-block:: python

            hooks = [
                dict(type="SemSegEvaluator", every_n_steps=1000),
                dict(type="CheckpointSaver", save_freq=1000,
                     evaluator_every_n_steps=1000),
            ]
            # → every 1000 steps writes the rolling model/last/ checkpoint; on each
            #   eval step reads comm_info["current_metric_value"] and, if it beats
            #   trainer.best_metric_value, writes model/model_best.pth; writes a
            #   final checkpoint in after_train
    """

    def __init__(self, save_freq=None, evaluator_every_n_steps=None):
        """Configure periodic saves and optional metric-driven best snapshots."""
        self.save_freq = save_freq
        self.evaluator_every_n_steps = evaluator_every_n_steps
        self.step_count = 0

    def before_train(self):
        """Seed internal step count from resumed trainer progress."""
        self.checkpoint_manager = CheckpointManager(self.trainer)
        self._pending_epoch_save = None
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
        save_kwargs = dict(
            is_best=is_best,
            step_count=self.step_count,
            save_freq=self.save_freq,
        )
        if _is_last_epoch_step(self.trainer):
            # Let after-epoch loggers add their metrics to the same W&B row
            # before checkpoint_state() commits it and records the rewind cursor.
            self._pending_epoch_save = save_kwargs
            return
        # Called on ALL ranks: a standard-format save is a collective op.
        self.checkpoint_manager.save_epoch_checkpoint(**save_kwargs)

    def after_epoch(self):
        """Save a last-step checkpoint after epoch-end metrics are logged."""
        if self._pending_epoch_save is None:
            return
        self.checkpoint_manager.save_epoch_checkpoint(**self._pending_epoch_save)
        self._pending_epoch_save = None

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
    """Load model weights and, when requested, resume optimizer/train state.

    In ``before_train``, before the trainer enters the loop, delegates to
    :meth:`~pimm.utils.checkpoints.CheckpointManager.load_weight_and_resume`,
    which loads weights from ``cfg.weight`` (warm-start) or resumes a full run
    from ``cfg.resume``. State-dict keys are rewritten by the configured
    replacement rules before loading, which is how a checkpoint trained under
    one module layout is mapped onto another (e.g. wrapping a backbone under
    ``module.model.backbone``). Registered as ``CheckpointLoader``.

    Args:
        keywords (str): Single substring to match in checkpoint keys when using
            the scalar form. Defaults to ``""`` (matches everything / no-op).
        replacement (str, optional): String to substitute for ``keywords`` in
            matched keys. When ``None``, falls back to ``keywords`` (i.e. no
            rewrite). Defaults to ``None``.
        replacements (dict, optional): A ``{keyword: replacement}`` mapping to
            apply several key rewrites in a single ``load_state_dict`` call, so
            missing/unexpected keys are reported truthfully. Overlapping
            keywords are resolved by longest match, so dict order is irrelevant.
            Mutually exclusive in practice with the scalar form. Must be a dict
            or a ``TypeError`` is raised. Defaults to ``None``.
        strict (bool): Whether to require an exact key match when loading model
            weights (passed through to ``load_state_dict``). Defaults to
            ``False``.

    Note:
        Warm-starting (``cfg.weight``) loads model weights only; full resume
        (``cfg.resume``) additionally restores optimizer, scheduler, and step
        state. After loading, judge success by inspecting the reported missing /
        unexpected keys rather than assuming silence means success.

    Example:
        Add to ``cfg.hooks``; once in ``before_train`` it loads ``cfg.weight``
        (warm-start) or ``cfg.resume`` (full resume), rewriting state-dict keys
        by the given rules before ``load_state_dict``:

        .. code-block:: python

            hooks = [
                dict(type="CheckpointLoader", replacements={
                    "module.backbone": "module.model.backbone",
                    "module.decoder":  "module.model.decoder",
                }),
            ]
            # → before the loop, loads cfg.weight/cfg.resume after renaming
            #   "module.backbone.*" -> "module.model.backbone.*" (etc.); inspect the
            #   reported missing/unexpected keys to confirm the load took
    """

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
    """Save iteration-oriented checkpoints on a pure global-step cadence.

    Like :class:`CheckpointSaver`, but the save decision depends only on the
    global step: in ``after_step`` it saves whenever ``step_count % save_freq ==
    0``. When ``cfg.evaluate`` is set and an evaluator has published
    ``current_metric_value`` in ``trainer.comm_info``, an improving metric also
    updates ``trainer.best_metric_value`` and writes ``model_best``. A final
    checkpoint is written in ``after_train``. Save logic is delegated to
    :class:`~pimm.utils.checkpoints.CheckpointManager`. Registered as
    ``CheckpointSaverIteration``.

    Args:
        save_freq (int, optional): Save every this many global steps. ``None``
            disables periodic saving. Defaults to ``None``.
        save_iter_checkpoints (bool): If ``True``, keep a distinct per-step
            checkpoint copy at each save instead of only rolling the latest.
            Defaults to ``False``.
        backend (str, optional): Deprecated selector for the on-disk format,
            superseded by the top-level ``checkpoint_format`` config key.
            ``"dcp"`` maps to the standard (hybrid) format and ``"torch"`` to
            the legacy format; ``None`` resolves to standard. Defaults to
            ``None``.

    Note:
        Called on **all** ranks because a standard-format save is collective;
        because the cadence is purely step-based it is identical across ranks.
        Place any evaluator hook **before** this saver so the metric is
        available for the ``model_best`` decision.

    Example:
        Add to ``cfg.hooks``; in ``after_step`` it saves on a pure global-step
        cadence:

        .. code-block:: python

            hooks = [dict(type="CheckpointSaverIteration", save_freq=5000)]
            # → writes a rolling checkpoint every 5000 global steps (and a final
            #   one in after_train); with cfg.evaluate set and an evaluator's
            #   comm_info["current_metric_value"], an improving metric also updates
            #   trainer.best_metric_value and writes model_best
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
        self._pending_epoch_save = None
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
        save_kwargs = dict(
            backend=self.backend,
            is_best=is_best,
            step_count=self.step_count,
            save_freq=self.save_freq,
            save_iter_checkpoints=self.save_iter_checkpoints,
        )
        if _is_last_epoch_step(self.trainer):
            self._pending_epoch_save = save_kwargs
            return
        self.checkpoint_manager.save_iteration_checkpoint(**save_kwargs)

    def after_epoch(self):
        """Save a last-step checkpoint after epoch-end metrics are logged."""
        if self._pending_epoch_save is None:
            return
        self.checkpoint_manager.save_iteration_checkpoint(
            **self._pending_epoch_save,
        )
        self._pending_epoch_save = None

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
