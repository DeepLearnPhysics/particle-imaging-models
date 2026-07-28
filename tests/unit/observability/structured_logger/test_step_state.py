# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file is adapted from TorchTitan and is licensed under the BSD-style
# license found in pimm/observability/structured_logger/LICENSE.
#
# From:
# https://github.com/pytorch/torchtitan/blob/main/tests/unit_tests/observability/test_structured_logging.py#L91-L301

from __future__ import annotations

import asyncio
import threading

from pimm.observability.structured_logger import step_state
from pimm.observability.structured_logger.step_state import (
    add_step_tag,
    clear_step_tags,
    get_epoch,
    get_iteration,
    get_relative_step,
    get_step,
    get_step_tags,
    set_step,
)


def test_step_and_relative_step_round_trip():
    assert get_step() is None
    assert get_relative_step() is None

    set_step(101, relative_step=1)

    assert get_step() == 101
    assert get_relative_step() == 1
    assert step_state._STEP_GLOBAL == 101
    assert step_state._STEP_CV.get() == 101


def test_relative_step_defaults_to_absolute_step():
    set_step(101, relative_step=1)
    set_step(102)

    assert get_relative_step() == 102


def test_context_var_precedes_global_value():
    step_state._STEP_GLOBAL = 1
    step_state._STEP_CV.set(2)

    assert get_step() == 2


def test_plain_thread_falls_back_to_global_value():
    set_step(42)
    result: list[int | None] = []

    thread = threading.Thread(target=lambda: result.append(get_step()))
    thread.start()
    thread.join()

    assert result == [42]


def test_epoch_and_iteration_are_part_of_step_context():
    set_step(205, relative_step=5, epoch=4, iteration=17)

    assert get_step() == 205
    assert get_relative_step() == 5
    assert get_epoch() == 4
    assert get_iteration() == 17


def test_set_step_replaces_epoch_and_iteration_context():
    set_step(1, epoch=0, iteration=0)
    set_step(2, epoch=1, iteration=3)

    assert get_epoch() == 1
    assert get_iteration() == 3


def test_step_tags_are_ordered_deduplicated_tuples():
    add_step_tag("checkpoint")
    add_step_tag("gc")
    add_step_tag("checkpoint")

    assert get_step_tags() == ("checkpoint", "gc")
    assert isinstance(get_step_tags(), tuple)


def test_setting_a_new_step_clears_tags():
    add_step_tag("evaluation")
    set_step(2)

    assert get_step_tags() == ()


def test_clear_step_tags_resets_both_stores():
    step_state._TAGS_GLOBAL = ("checkpoint",)
    step_state._TAGS_CV.set(("evaluation",))

    clear_step_tags()

    assert step_state._TAGS_GLOBAL == ()
    assert step_state._TAGS_CV.get() == ()


def test_async_task_inherits_step_context():
    results: list[tuple[int | None, int | None, int | None]] = []

    async def worker():
        results.append((get_step(), get_epoch(), get_iteration()))

    async def main():
        set_step(42, epoch=3, iteration=9)
        await asyncio.create_task(worker())

    asyncio.run(main())

    assert results == [(42, 3, 9)]


def test_async_task_tags_are_isolated():
    add_step_tag("spmd-only")

    async def actor(tag: str, delay: float):
        await asyncio.sleep(delay)
        add_step_tag(tag)
        return get_step_tags()

    async def main():
        return await asyncio.gather(
            actor("gc", 0.001),
            actor("evaluation", 0.002),
        )

    gc_tags, evaluation_tags = asyncio.run(main())

    assert gc_tags == ("gc",)
    assert evaluation_tags == ("evaluation",)
    assert step_state._TAGS_GLOBAL == ("spmd-only",)
