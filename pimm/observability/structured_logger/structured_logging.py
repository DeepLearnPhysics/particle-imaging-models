# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in this directory.
#
# Adapted from TorchTitan:
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/structured_logger/structured_logging.py#L1-L443

"""Structured-logging API: init_structured_logger, log_trace_span,
log_trace_instant, log_trace_scalar.

Emits structured JSONL events for phase timing, scalars, and diagnostics.
Handler factories (JSONL, custom, etc.) are loaded dynamically via the
``PIMM_STRUCT_LOGGER_HANDLERS`` env var. The default factory writes bounded,
rotating JSONL files under ``structured_logs/``.
"""

import asyncio
import enum
import functools
import importlib
import inspect
import logging
import os
import threading
from collections.abc import Callable
from timeit import default_timer as timer
from typing import Any, TypeVar, cast

import torch

from pimm.observability.structured_logger.step_state import (
    get_epoch,
    get_iteration,
    get_relative_step,
    get_step,
)

F = TypeVar("F", bound=Callable[..., Any])

console_logger: logging.Logger = logging.getLogger(__name__)

_structured_logger: logging.Logger = logging.getLogger("pimm.structured_logger")
_structured_logger.propagate = False

_is_initialized: bool = False
_disabled: bool = True
_registered_handlers: list[logging.Handler] = []
_state_lock = threading.RLock()
_logger_generation: int = 0

_DEFAULT_HANDLER_FACTORY = (
    "pimm.observability.structured_logger.jsonl_handler.register_jsonl_handler"
)
_HANDLERS_ENV = "PIMM_STRUCT_LOGGER_HANDLERS"


def _structured_logger_disabled() -> bool:
    """Whether structured logging is disabled.

    Driven by the ``enable`` flag passed to :func:`init_structured_logger`
    (sourced from pimm's ``structured_logging.enabled`` config value).
    """
    return _disabled


def _current_task_name() -> str | None:
    try:
        task = asyncio.current_task()
        return task.get_name() if task else None
    except RuntimeError:
        return None


class StrEnum(enum.Enum):
    """Stand-in for ``enum.StrEnum`` (added in Python 3.11).

    Mimics it for our use case: ``str(member)`` returns the value (e.g.
    ``"event"``), not ``"LogType.EVENT"``. Drop in favor of ``enum.StrEnum``
    once Python 3.10 support is no longer needed.

    From: https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/structured_logger/structured_logging.py#L61-L70
    """

    def __str__(self) -> str:
        return self.value


class LogType(StrEnum):
    """Record kind in the JSONL stream.

    - ``EVENT``: paired span record (``*_start`` / ``*_end`` from
      ``log_trace_span``).
    - ``INSTANT``: point-in-time record (``log_trace_instant``,
      ``log_trace_scalar``).
    - ``TEXT``: free-text log record (filtered out by
      ``TraceEventsOnlyFilter``).

    From: https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/structured_logger/structured_logging.py#L73-L83
    """

    EVENT = "event"
    INSTANT = "instant"
    TEXT = "text"


class ExtraFields(StrEnum):
    """Keys stored in the ``extra`` dictionary of a logging record.

    From: https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/structured_logger/structured_logging.py#L86-L95
    """

    LOG_TYPE = "log_type"
    LOG_TYPE_NAME = "log_type_name"
    EVENT_NAME = "event_name"
    STEP = "step"
    VALUE = "value"
    RELATIVE_STEP = "relative_step"
    EPOCH = "epoch"
    ITERATION = "iteration"
    TASK_NAME = "task_name"


def event_extra(
    event_type: str,
    event_name: str | None = None,
    step: int | None = None,
    relative_step: int | None = None,
    value: float | int | None = None,
    task_name: str | None = None,
    log_type: LogType = LogType.EVENT,
    epoch: int | None = None,
    iteration: int | None = None,
) -> dict[str, Any]:
    """Build the extra dictionary for one structured event."""
    return {
        str(ExtraFields.LOG_TYPE): str(log_type),
        str(ExtraFields.LOG_TYPE_NAME): str(event_type),
        str(ExtraFields.EVENT_NAME): event_name,
        str(ExtraFields.STEP): step,
        str(ExtraFields.RELATIVE_STEP): relative_step,
        str(ExtraFields.VALUE): value,
        str(ExtraFields.TASK_NAME): task_name,
        str(ExtraFields.EPOCH): epoch,
        str(ExtraFields.ITERATION): iteration,
    }


def _current_event_extra(
    event_type: str,
    *,
    event_name: str | None = None,
    value: float | int | None = None,
    task_name: str | None = None,
    log_type: LogType = LogType.EVENT,
) -> dict[str, Any]:
    return event_extra(
        event_type,
        event_name=event_name,
        step=get_step(),
        relative_step=get_relative_step(),
        value=value,
        task_name=task_name,
        log_type=log_type,
        epoch=get_epoch(),
        iteration=get_iteration(),
    )


class TraceEventsOnlyFilter(logging.Filter):
    """Defensive filter: drop any record on the structured logger that did not
    come through the ``log_trace_*`` API.

    How records get a ``log_type_name`` attribute:

    1. ``log_trace_span`` / ``log_trace_instant`` / ``log_trace_scalar`` all
       call ``_structured_logger.info(msg, extra=event_extra(...))``.
    2. ``event_extra`` always sets ``log_type_name`` in the ``extra`` dict.
    3. Python's logging attaches ``extra`` keys as attributes on the
       ``LogRecord``, so ``record.log_type_name`` is populated.

    So any record reaching this filter WITHOUT ``log_type_name`` is a plain
    ``.info("text")`` call made directly on the structured logger -- bypassing
    the API. That should not happen in this codebase. The filter exists as a
    safeguard against future accidents: if someone grabs the logger by name
    (``logging.getLogger("pimm.structured_logger")``) and writes free text, this
    filter keeps it out of the JSONL stream so the schema stays strict. The
    first drop emits a one-shot warning to make the trap discoverable.

    From: https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/structured_logger/structured_logging.py#L119-L155
    """

    def __init__(self) -> None:
        super().__init__()
        self._warned = False

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, str(ExtraFields.LOG_TYPE_NAME), None) is not None:
            return True
        if not self._warned:
            self._warned = True
            console_logger.warning(
                "Plain-text record on the structured logger was dropped. "
                "Use log_trace_span / log_trace_scalar / log_trace_instant."
            )
        return False


def _close_handlers(handlers: list[logging.Handler]) -> None:
    for handler in handlers:
        _structured_logger.removeHandler(handler)
        try:
            handler.flush()
        except Exception:
            console_logger.exception("Failed to flush structured log handler")
        try:
            handler.close()
        except Exception:
            console_logger.exception("Failed to close structured log handler")


def shutdown_structured_logger() -> None:
    """Flush, close, and detach handlers installed by this subsystem.

    The function is idempotent. Other handlers attached directly to the named
    logger are left untouched. A subsequent :func:`init_structured_logger` call
    starts a fresh generation and can enable logging again.
    """
    global _disabled, _is_initialized, _logger_generation

    with _state_lock:
        _disabled = True
        _is_initialized = False
        _logger_generation += 1
        handlers = list(_registered_handlers)
        _registered_handlers.clear()
        _close_handlers(handlers)


def init_structured_logger(
    source: str,
    output_dir: str,
    rank: int | None = None,
    enable: bool = True,
    *,
    max_file_size_mb: float = 128,
    backup_count: int = 3,
    world_size: int | None = None,
    attempt: int | None = None,
    job_id: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Attach handlers to the structured logger. Call once per process.

    Handler factories come from the ``PIMM_STRUCT_LOGGER_HANDLERS`` env var
    (comma-separated ``module.path.factory_name``). When unset, a default JSONL
    handler is registered; when set, ONLY the listed factories run.

    ``rank`` and ``world_size`` default to ``$RANK`` and ``$WORLD_SIZE`` as set
    by torchrun, so this can run before ``torch.distributed`` init. Pimm's
    Submitit attempt number, Slurm job ID, and torchrun ID are used when their
    corresponding explicit arguments are omitted. Repeated calls are no-ops
    until :func:`shutdown_structured_logger` is called.

    When ``enable=False``, all subsequent ``log_trace_*`` calls become no-ops
    and no handlers are attached. A later enabled call can initialize logging.

    Does not configure console output; use pimm's normal logging setup for that.

    Example::

        init_structured_logger(source="training", output_dir="./outputs")
        log_trace_instant("structured_logger_started")
    """
    global _disabled, _is_initialized, _logger_generation

    if not enable:
        shutdown_structured_logger()
        console_logger.info("Structured logging is disabled")
        return

    with _state_lock:
        if _is_initialized:
            return

        resolved_rank = int(rank if rank is not None else os.environ.get("RANK", 0))
        resolved_world_size = int(
            world_size if world_size is not None else os.environ.get("WORLD_SIZE", 1)
        )
        resolved_attempt = int(
            attempt
            if attempt is not None
            else os.environ.get("PIMM_SUBMITIT_ATTEMPT", 1)
        )
        resolved_job_id = (
            str(job_id) if job_id is not None else os.environ.get("SLURM_JOB_ID")
        )
        resolved_trace_id = (
            str(trace_id)
            if trace_id is not None
            else os.environ.get("TORCHELASTIC_RUN_ID")
        )

        configured_factories = os.environ.get(_HANDLERS_ENV, "")
        if configured_factories.strip():
            factory_paths = [
                path.strip() for path in configured_factories.split(",") if path.strip()
            ]
        else:
            factory_paths = [_DEFAULT_HANDLER_FACTORY]

        original_handler_ids = {id(handler) for handler in _structured_logger.handlers}
        try:
            for factory_path in factory_paths:
                module_path, function_name = factory_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                getattr(module, function_name)(
                    structured_logger=_structured_logger,
                    rank=resolved_rank,
                    world_size=resolved_world_size,
                    source=str(source),
                    output_dir=str(output_dir),
                    attempt=resolved_attempt,
                    job_id=resolved_job_id,
                    trace_id=resolved_trace_id,
                    max_file_size_mb=max_file_size_mb,
                    backup_count=backup_count,
                )
        except Exception:
            added = [
                handler
                for handler in _structured_logger.handlers
                if id(handler) not in original_handler_ids
            ]
            _close_handlers(added)
            _disabled = True
            _is_initialized = False
            raise

        _registered_handlers.extend(
            handler
            for handler in _structured_logger.handlers
            if id(handler) not in original_handler_ids
        )

        if (
            _structured_logger.level == logging.NOTSET
            or _structured_logger.level > logging.INFO
        ):
            _structured_logger.setLevel(logging.INFO)

        _disabled = False
        _is_initialized = True
        _logger_generation += 1


def log_trace_scalar(scalars: dict[str, float | int], *, stacklevel: int = 2) -> None:
    """Emit a record per (name, value) pair. Useful when adding more context
    to the trace for debugging, e.g. registering ``num_tokens_processed``.

    Step is read from ``set_step()``; non-numeric values are skipped
    with a warning. Bump ``stacklevel`` when wrapping in a helper so
    ``caller`` points at the real call site.

    Args:
        scalars: Mapping of scalar name to numeric value. Non-numeric
            values are skipped with a warning.
        stacklevel: Passed through to ``logger.info`` so the ``caller`` field
            in the emitted record points at the real call site. Increase from
            the default 2 if you wrap this function in a helper.

    Example::

        log_trace_scalar({"train.loss": 2.5, "train.tflops": 45.6})
    """
    if _structured_logger_disabled() or torch.compiler.is_compiling():
        return

    task_name = _current_task_name()
    bad_keys: list[str] = []
    for name, value in scalars.items():
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            bad_keys.append(name)
            continue
        step = get_step()
        _structured_logger.info(
            f"[step {step if step is not None else 'N/A'}] {name}={value}",
            extra=_current_event_extra(
                "metric_value",
                event_name=name,
                value=value,
                task_name=task_name,
                log_type=LogType.INSTANT,
            ),
            stacklevel=stacklevel,
        )
    if bad_keys:
        console_logger.warning(
            "log_trace_scalar skipped non-numeric values for keys: %s", bad_keys
        )


def log_trace_instant(event_type: str, *, stacklevel: int = 2) -> None:
    """Emit a zero-duration event or marker (e.g. ``"training_start"``).

    Use ``log_trace_span`` when you want start+end+duration.

    Args:
        event_type: Free-form string. Becomes ``log_type_name`` in the
            emitted record.
        stacklevel: Passed through to ``logger.info`` so the ``caller`` field
            in the emitted record points at the real call site. Increase from
            the default 2 if you wrap this function in a helper.

    Example::

        log_trace_instant("training_start")
    """
    if _structured_logger_disabled() or torch.compiler.is_compiling():
        return

    _structured_logger.info(
        str(event_type),
        extra=_current_event_extra(
            str(event_type),
            task_name=_current_task_name(),
            log_type=LogType.INSTANT,
        ),
        stacklevel=stacklevel,
    )


class log_trace_span:  # noqa: N801
    """Time a block of work; emits ``_start`` and ``_end`` records.

    Usable as a context manager or decorator. On entry, captures the
    enclosing :class:`asyncio.Task`'s name via
    ``asyncio.current_task().get_name()`` (``None`` outside any task)
    and stamps it on both records. Analysis pairs ``_start`` / ``_end``
    via a LIFO stack on ``(source, task_name)``, so nested spans in one
    task pair correctly. The ``_end`` record's ``value`` is the elapsed
    wall-time in ms.

    On exception in an active logger generation, emits an extra ``_error``
    record (with exception type and message), then the normal ``_end``.
    Shutdown can intentionally leave an incomplete ``_start``; process
    termination can do the same. Analysis tools retain either case.

    From: https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/structured_logger/structured_logging.py#L296-L443

    Example::

        # context manager
        with log_trace_span("fwd_bwd"):
            loss = model(batch)
            loss.backward()

        # decorator of sync or async function
        @log_trace_span("rl_rollout")
        async def rollout(self, prompts):
            return await self.engine.generate(prompts)

    Args:
        event_type: Becomes ``log_type_name`` in the records.
        description: Human-readable label in the log line; doesn't
            affect ``log_type_name`` or filtering.
        stacklevel: Bump when wrapping in a helper so ``caller`` points
            at the real call site.

    """

    def __init__(
        self,
        event_type: str,
        description: str | None = None,
        *,
        stacklevel: int = 2,
    ):
        self.base_name = str(event_type)
        self.description = description
        self.stacklevel = stacklevel
        self.start_time: float = 0.0
        self._task_name: str | None = None
        self._active = False
        self._generation = -1
        self.start_type_name = self.base_name + "_start"
        self.end_type_name = self.base_name + "_end"

    def __enter__(self):
        if _structured_logger_disabled() or torch.compiler.is_compiling():
            return self

        self._task_name = _current_task_name()
        self.start_time = timer()
        self._generation = _logger_generation
        self._active = True
        display_name = self.description or self.base_name
        step = get_step()
        _structured_logger.info(
            f"[step {step if step is not None else 'N/A'}] "
            f"{display_name} {self.start_type_name}",
            extra=_current_event_extra(
                self.start_type_name,
                task_name=self._task_name,
            ),
            stacklevel=self.stacklevel,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (
            not self._active
            or self._generation != _logger_generation
            or _structured_logger_disabled()
            or torch.compiler.is_compiling()
        ):
            return None

        self._active = False
        delta_ms = (timer() - self.start_time) * 1000
        step = get_step()

        if exc_type is not None:
            error_type_name = self.base_name + "_error"
            _structured_logger.info(
                f"[step {step if step is not None else 'N/A'}] "
                f"{error_type_name}: {exc_type.__name__}: {exc_val}",
                extra=_current_event_extra(
                    error_type_name,
                    task_name=self._task_name,
                ),
                stacklevel=self.stacklevel,
            )

        _structured_logger.info(
            f"[step {step if step is not None else 'N/A'}] "
            f"{self.end_type_name} took {delta_ms:.2f} ms",
            extra=_current_event_extra(
                self.end_type_name,
                value=delta_ms,
                task_name=self._task_name,
            ),
            stacklevel=self.stacklevel,
        )
        return None

    def __call__(self, func: F) -> F:
        base_name, description, stacklevel = (
            self.base_name,
            self.description,
            self.stacklevel,
        )

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with log_trace_span(base_name, description, stacklevel=stacklevel):
                    return await func(*args, **kwargs)

            return cast(F, async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with log_trace_span(base_name, description, stacklevel=stacklevel + 1):
                return func(*args, **kwargs)

        return cast(F, sync_wrapper)
