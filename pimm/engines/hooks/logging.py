"""Training logging and W&B naming hooks."""

import re
import time

from pimm.utils.timer import Timer

from .default import HookBase
from .builder import HOOKS


LARGE_MODEL_OUTPUT_KEYS = {
    "seg_logits",
    "sem_logits",
    "instance_embedding",
    "vertex_embedding",
    "sigma",
    "point",
    "pred_logits",
    "pred_masks",
    "teacher_logits",
}


def model_output_scalar_keys(model_output_dict):
    """Return model-output keys that InformationWriter should log."""
    has_total_loss = "total_loss" in model_output_dict
    keys = [
        key
        for key in model_output_dict.keys()
        if (
            key not in LARGE_MODEL_OUTPUT_KEYS
            and "match_" not in key
            and not (has_total_loss and key == "loss")
            and key != "total_loss"
        )
    ]
    if has_total_loss:
        keys.append("loss")
    return keys


def model_output_scalar_value(model_output_dict, key):
    """Return a scalar float for a selected model-output key, or None."""
    val = model_output_dict["total_loss"] if key == "loss" and "total_loss" in model_output_dict else model_output_dict[key]
    try:
        if hasattr(val, "item") and callable(getattr(val, "item", None)):
            return float(val.item())
        if isinstance(val, (int, float)):
            return float(val)
        return float(val)
    except Exception:
        return None


@HOOKS.register_module()
class WandbNamer(HookBase):
    """Auto-generate ``cfg.wandb_run_name`` from selected config values.

    Joins the resolved values of the requested (possibly nested) config keys
    with a separator, optionally formatting large numbers with K/M/B suffixes,
    then appends any ``extra`` strings, to build a descriptive W&B run name.
    Runs once in ``modify_config`` (before the writer is built), so the name is
    in place before logging starts. Registered as ``WandbNamer``.

    Args:
        keys (tuple): Config keys to include in the name, in order. Supports
            dotted nested access such as ``"data.train.max_len"`` or
            ``"model.type"``; keys that resolve to ``None`` are skipped.
            Defaults to ``()``.
        template (str, optional): Name template containing dotted config
            placeholders such as
            ``"{experiment.study}__{experiment.variant}__s{seed}"``.
            When set, it takes precedence over ``keys``, ``sep``, and ``extra``.
            Defaults to ``None``.
        sep (str): Separator joining the parts. Defaults to ``"-"``.
        format_numbers (bool): If ``True``, render numeric values with K/M/B
            suffixes (e.g. ``1000000`` -> ``"1M"``). Defaults to ``True``.
        extra (str or tuple, optional): Extra string(s) appended after the
            keyed parts, e.g. ``"fft"`` or ``"scratch"``. A bare string is
            wrapped to a one-tuple. Defaults to ``None``.

    Note:
        If ``wandb_run_name`` was set on the command line (present in
        ``cfg._cli_options``), this hook leaves it untouched. So
        ``--options wandb_run_name=my-name`` overrides the auto-generated name.

    Example:
        Add to ``cfg.hooks``; once in ``modify_config`` (before the writer is
        built) it sets ``cfg.wandb_run_name`` from the joined config values:

        .. code-block:: python

            hooks = [
                dict(type="WandbNamer",
                     keys=("model.type", "data.train.max_len", "seed"),
                     extra="fft"),
            ]
            # → sets cfg.wandb_run_name = "Sonata-v1m1-1M-0-fft" (left untouched if
            #   wandb_run_name was passed on the CLI)
    """

    def __init__(
        self,
        keys=(),
        sep="-",
        format_numbers=True,
        extra=None,
        template=None,
    ):
        """Store config keys and formatting options for generated run names."""
        self.keys = keys
        self.sep = sep
        self.format_numbers = format_numbers
        self.template = template
        # normalize extra to tuple
        if extra is None:
            self.extra = ()
        elif isinstance(extra, str):
            self.extra = (extra,)
        else:
            self.extra = tuple(extra)
    
    def _get_nested(self, cfg, key_path):
        """Get nested config value: 'data.train.max_len' -> cfg.data['train']['max_len']"""
        parts = key_path.split('.')
        val = cfg
        for part in parts:
            if hasattr(val, part):
                val = getattr(val, part)
            elif isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return None
        return val
    
    def _format_value(self, val):
        """Format a value for the run name."""
        if self.format_numbers and isinstance(val, (int, float)):
            return self._format_number(val)
        return str(val)
    
    def _format_number(self, n):
        """Format large numbers with suffixes (K, M, B)."""
        n_val = float(n)
        if abs(n_val) >= 1_000_000_000:
            return f"{n_val / 1_000_000_000:.1f}B".rstrip('0').rstrip('.')
        elif abs(n_val) >= 1_000_000:
            return f"{n_val / 1_000_000:.1f}M".rstrip('0').rstrip('.')
        elif abs(n_val) >= 1_000:
            return f"{n_val / 1_000:.1f}K".rstrip('0').rstrip('.')
        return str(int(n_val) if n_val == int(n_val) else n_val)

    def _format_template(self, cfg):
        """Resolve dotted config placeholders in the configured template."""

        def replace(match):
            key = match.group(1)
            value = self._get_nested(cfg, key)
            if value is None:
                raise ValueError(
                    f"WandbNamer template placeholder {key!r} is missing."
                )
            return self._format_value(value)

        return re.sub(
            r"\{([A-Za-z_][A-Za-z0-9_.]*)\}",
            replace,
            self.template,
        )
    
    def modify_config(self, cfg):
        """Build wandb_run_name from specified keys."""
        # skip if wandb_run_name already set via CLI
        if 'wandb_run_name' in getattr(cfg, '_cli_options', set()):
            return

        if self.template is not None:
            cfg.wandb_run_name = self._format_template(cfg)
            return
        
        parts = []
        for key in self.keys:
            val = self._get_nested(cfg, key)
            if val is not None:
                parts.append(self._format_value(val))
        
        # append extra strings
        parts.extend(self.extra)
        
        if parts:
            cfg.wandb_run_name = self.sep.join(parts)

@HOOKS.register_module()
class IterationTimer(HookBase):
    """Measure data/batch latency and append timing and ETA to iteration logs.

    Times each training iteration and feeds the results into
    ``trainer.storage`` and the per-iteration log string. In ``before_step`` it
    records ``data_time`` (time spent waiting for the batch); in ``after_step``
    it records ``batch_time`` (full iteration), decrements the remaining-
    iteration counter, and appends ``Data``, ``Batch``, and ``Remain`` (ETA,
    computed from the running average batch time) to ``comm_info["iter_info"]``.
    The remaining-iteration budget is initialized in ``before_train`` and the
    per-iteration timer is reset in ``before_epoch``. Registered as
    ``IterationTimer``.

    Args:
        warmup_iter (int): Number of initial iterations whose timing is reset
            (excluded from the running averages) so startup/compile overhead
            does not skew ETA. Defaults to ``1``.

    Example:
        Add to ``cfg.hooks``; it times each iteration and feeds the timing into
        ``trainer.storage`` and the per-step log line:

        .. code-block:: python

            hooks = [dict(type="IterationTimer", warmup_iter=2)]
            # → puts scalars "data_time" (before_step) and "batch_time" (after_step)
            #   into trainer.storage and appends "Data … Batch … Remain HH:MM:SS"
            #   (ETA) to comm_info["iter_info"]; the first 2 iters reset the averages
    """

    def __init__(self, warmup_iter=1):
        """Configure how many initial iterations are excluded from averages."""
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        """Initialize remaining-iteration accounting at train start."""
        self._start_time = time.perf_counter()
        _remain_epoch = self.trainer.max_epoch - self.trainer.start_epoch
        self._remain_iter = _remain_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        """Reset the per-iteration timer at the start of each epoch."""
        self._iter_timer.reset()

    def before_step(self):
        """Record data loading time before the model consumes the batch."""
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        """Record batch time and append timing fields to iter_info."""
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()

@HOOKS.register_module()
class InformationWriter(HookBase):
    """Assemble per-step console logs and write scalar train metrics.

    The primary training logger. In ``before_step`` it seeds the shared
    ``comm_info["iter_info"]`` string with the epoch/iteration position. In
    ``after_step`` it filters the model output dict down to scalar-like entries
    (dropping large tensors such as logits/masks and ``match_*`` keys, and
    mapping ``total_loss`` to ``loss``), records them in ``trainer.storage``,
    appends them plus the current learning rate to the iteration log, emits that
    line to the logger every ``log_frequency`` global steps (and always on the
    last iteration of an epoch), and mirrors each scalar to the writer under
    ``train_batch/*`` (with ``params/lr``). In ``after_epoch`` it logs the
    epoch-average of each tracked key under ``train/*``. Registered as
    ``InformationWriter``.

    Args:
        log_frequency (int): Emit the assembled console line every this many
            global steps (coerced to at least ``1``). Writer scalars are logged
            every step regardless. Defaults to ``1``.
        step_offset (int): Constant added to the computed global step, e.g. to
            continue a warm-started run's W&B x-axis from where the parent
            checkpoint left off. Defaults to ``0``.

    Note:
        Writer scalars are only emitted when ``trainer.writer`` is not ``None``.
        The global step is computed to match the other logging/diagnostic hooks
        so all curves share an x-axis.

    Example:
        Add to ``cfg.hooks``; it is the main per-step logger — it records the
        scalar model outputs and learning rate and mirrors them to the writer:

        .. code-block:: python

            hooks = [dict(type="InformationWriter", log_frequency=20)]
            # → every step writes "train_batch/<key>" for each scalar loss/metric
            #   and "params/lr" to the writer; emits the assembled console log line
            #   every 20 steps; in after_epoch writes the "train/<key>" epoch averages
    """

    def __init__(self, log_frequency=1, step_offset=0):
        """Configure how often per-step training summaries are emitted.

        step_offset shifts the logged global step (e.g. to continue a warm-started
        run's wandb x-axis from where the parent checkpoint left off).
        """
        self.model_output_keys = []
        self.log_frequency = max(1, int(log_frequency))
        self.step_offset = int(step_offset)

    def before_train(self):
        """Initialize the shared iteration-info string used by logging hooks."""
        self.trainer.comm_info["iter_info"] = ""

    def _get_global_step(self):
        """Return a one-based global step aligned with other logging hooks."""
        # compute global step same way as GradientNormLogger for consistency
        current_epoch = self.trainer.comm_info["epoch"] + 1
        current_iter = self.trainer.comm_info["iter"]
        return (current_epoch - 1) * len(self.trainer.train_loader) + current_iter + 1 + self.step_offset

    def before_step(self):
        """Append epoch and batch position to the current iteration summary."""
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        """Filter scalar-like model outputs and write train metrics."""
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            self.model_output_keys = model_output_scalar_keys(model_output_dict)
            for key in self.model_output_keys:
                scalar = model_output_scalar_value(model_output_dict, key)
                if scalar is None:
                    continue
                self.trainer.storage.put_scalar(key, scalar)

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        global_step = self._get_global_step()
        is_last_iter = (
            self.trainer.comm_info["iter"] + 1 >= len(self.trainer.train_loader)
        )
        if global_step % self.log_frequency == 0 or is_last_iter:
            self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/lr", lr, global_step)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    global_step,
                )

    def after_epoch(self):
        """Write epoch-average training metrics for keys seen this epoch."""
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            global_step = (self.trainer.epoch + 1) * len(self.trainer.train_loader)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    global_step,
                )
