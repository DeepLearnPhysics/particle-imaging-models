# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Portions of this file are adapted from TorchTitan and are licensed under the
# BSD-style license found in
# pimm/observability/structured_logger/LICENSE.
#
# From:
# https://github.com/pytorch/torchtitan/blob/main/tests/unit_tests/observability/test_structured_logging.py#L53-L89

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pimm.observability.structured_logger import shutdown_structured_logger
from pimm.observability.structured_logger import step_state


def _reset_step_state() -> None:
    """Reset both the SPMD and ContextVar stores used by the tracer."""
    defaults = {
        "_STEP_GLOBAL": None,
        "_RELATIVE_STEP_GLOBAL": None,
        "_EPOCH_GLOBAL": None,
        "_ITERATION_GLOBAL": None,
        "_TAGS_GLOBAL": (),
    }
    for name, value in defaults.items():
        if hasattr(step_state, name):
            setattr(step_state, name, value)

    context_defaults = {
        "_STEP_CV": None,
        "_RELATIVE_STEP_CV": None,
        "_EPOCH_CV": None,
        "_ITERATION_CV": None,
        "_TAGS_CV": (),
    }
    for name, value in context_defaults.items():
        context_var = getattr(step_state, name, None)
        if context_var is not None:
            context_var.set(value)


@pytest.fixture(autouse=True)
def clean_structured_logging_state():
    """Keep process-global logging state from leaking between unit tests."""
    shutdown_structured_logger()
    _reset_step_state()
    yield
    shutdown_structured_logger()
    _reset_step_state()


@pytest.fixture
def read_trace_records():
    """Read all trace records, including rotated JSONL segments."""

    def _read(output_dir: Path) -> list[dict]:
        trace_dir = output_dir / "structured_logs"
        records: list[dict] = []
        if not trace_dir.exists():
            return records
        for path in sorted(trace_dir.glob("*.jsonl*")):
            with path.open() as stream:
                records.extend(json.loads(line) for line in stream if line.strip())
        return records

    return _read
