# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in this directory.
#
# Adapted from TorchTitan:
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/structured_logger/step_state.py#L1-L182

"""Per-step state for structured trace records.

Stores current step, relative step, epoch, iteration, and step tags. Every trace
call (``log_trace_span`` / ``log_trace_instant`` / ``log_trace_scalar``) stamps
records with these values at emit time.

Two runtime models share this state:

- **Synchronous trainer.** One process, one training thread, no asyncio. Call
  ``set_step(step)`` once per step; add tags like ``add_step_tag("gc")`` when GC
  ran. A module global suffices.
- **Async tasks.** Multiple tasks may run concurrently on the same process.
  Each task may add different tags for the same step, and those views must stay
  isolated -- one task's ``"gc"`` must not leak into another's ``"eval"``.

Design
------
Each value is stored in two places:

- **module-level global** -- synchronous path; also the fallback for plain
  threads (``threading.Thread``), which do not inherit ContextVar state.
- **ContextVar** -- per-task path; inherited when an asyncio task is spawned,
  isolated from sibling tasks.

``set_step`` writes both (the training position agrees across tasks).
``add_step_tag`` is task-aware: inside an asyncio task it writes the ContextVar
only; outside, the global only. Readers check the ContextVar first and fall back
to the global.

Example::

    # Synchronous code sets a global tag.
    add_step_tag("checkpoint_step")
    assert get_step_tags() == ("checkpoint_step",)

    # Inside async tasks, tags are task-scoped.
    set_step(42)
    async def task_gc():
        add_step_tag("gc")
        return get_step_tags()
    async def task_eval():
        add_step_tag("eval")
        return get_step_tags()

    gc_view, eval_view = await asyncio.gather(task_gc(), task_eval())
    assert gc_view == ("gc",)
    assert eval_view == ("eval",)

Inside a task, global tags are not visible when the task has its own tags. This
keeps concurrent task contexts intentionally isolated.
"""

import asyncio
from contextvars import ContextVar

# Synchronous / ordinary-thread state.
_STEP_GLOBAL: int | None = None
_RELATIVE_STEP_GLOBAL: int | None = None
_EPOCH_GLOBAL: int | None = None
_ITERATION_GLOBAL: int | None = None
_TAGS_GLOBAL: tuple[str, ...] = ()

# Per-async-task overrides.
_STEP_CV: ContextVar[int | None] = ContextVar("_STEP_CV", default=None)
_RELATIVE_STEP_CV: ContextVar[int | None] = ContextVar(
    "_RELATIVE_STEP_CV", default=None
)
_EPOCH_CV: ContextVar[int | None] = ContextVar("_EPOCH_CV", default=None)
_ITERATION_CV: ContextVar[int | None] = ContextVar("_ITERATION_CV", default=None)
_TAGS_CV: ContextVar[tuple[str, ...]] = ContextVar("_TAGS_CV", default=())


def _is_in_async_task() -> bool:
    """True iff called from inside a running asyncio task.

    On Python 3.12 ``asyncio.current_task()`` calls ``get_running_loop()``,
    which raises ``RuntimeError`` when no loop is running. On Python 3.14+ it is
    expected to return None directly. The try/except handles both.
    """
    try:
        return asyncio.current_task() is not None
    except RuntimeError:
        return False


def set_step(
    step: int,
    *,
    relative_step: int | None = None,
    epoch: int | None = None,
    iteration: int | None = None,
) -> None:
    """Set the current training position.

    All subsequent trace records include this context. Calling this function
    also clears step tags from the previous step.

    Args:
        step: Absolute training step, including progress restored on resume.
        relative_step: Steps since this process attempt started (1 on its first
            new step). When None, defaults to ``step`` -- correct for runs
            without checkpoint resume. Resumed trainers must pass it explicitly.
        epoch: Optional zero-based epoch.
        iteration: Optional zero-based iteration within the epoch.

    Omitted epoch and iteration values are explicitly reset so stale context
    cannot leak into a later step.

    Example::

        loaded_step = 100
        for step in range(loaded_step + 1, num_steps + 1):
            set_step(step, relative_step=step - loaded_step)
            train_step(...)
    """
    global _EPOCH_GLOBAL, _ITERATION_GLOBAL
    global _RELATIVE_STEP_GLOBAL, _STEP_GLOBAL

    if relative_step is None:
        relative_step = step

    _STEP_GLOBAL = step
    _STEP_CV.set(step)
    _RELATIVE_STEP_GLOBAL = relative_step
    _RELATIVE_STEP_CV.set(relative_step)
    _EPOCH_GLOBAL = epoch
    _EPOCH_CV.set(epoch)
    _ITERATION_GLOBAL = iteration
    _ITERATION_CV.set(iteration)
    clear_step_tags()


def get_step() -> int | None:
    """Return the current absolute step, or ``None`` if it has not been set."""
    value = _STEP_CV.get()
    return value if value is not None else _STEP_GLOBAL


def get_relative_step() -> int | None:
    """Return the current process-relative step, or ``None``."""
    value = _RELATIVE_STEP_CV.get()
    return value if value is not None else _RELATIVE_STEP_GLOBAL


def get_epoch() -> int | None:
    """Return the current epoch, or ``None``."""
    value = _EPOCH_CV.get()
    return value if value is not None else _EPOCH_GLOBAL


def get_iteration() -> int | None:
    """Return the current iteration within the epoch, or ``None``."""
    value = _ITERATION_CV.get()
    return value if value is not None else _ITERATION_GLOBAL


def get_step_tags() -> tuple[str, ...]:
    """Return task-local tags when present, otherwise synchronous tags."""
    value = _TAGS_CV.get()
    return value if value else _TAGS_GLOBAL


def add_step_tag(tag: str) -> None:
    """Annotate the current step. Tags appear in trace JSONL for filtering.

    Task-aware write: inside an asyncio task the tag is appended to the
    ContextVar only (so sibling tasks with different tags do not cross-pollinate
    through the shared global); outside any task the tag goes to the global only
    (the synchronous path).

    Example::

        if gc_happened:
            add_step_tag("gc")
        if is_validation:
            add_step_tag("eval")
    """
    global _TAGS_GLOBAL

    tag = str(tag)
    if _is_in_async_task():
        current = _TAGS_CV.get()
        if tag not in current:
            _TAGS_CV.set(current + (tag,))
    elif tag not in _TAGS_GLOBAL:
        _TAGS_GLOBAL = _TAGS_GLOBAL + (tag,)


def clear_step_tags() -> None:
    """Clear synchronous and task-local step tags."""
    global _TAGS_GLOBAL

    _TAGS_GLOBAL = ()
    _TAGS_CV.set(())
