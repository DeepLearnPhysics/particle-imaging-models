# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in this directory.
#
# Adapted from TorchTitan:
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/structured_logger/__init__.py#L1-L45

"""Structured logging: per-rank JSONL trace of training phases.

Typical use::

    from pimm.observability import structured_logger as sl

    sl.init_structured_logger(source="training", output_dir="./outputs")
    sl.log_trace_instant("structured_logger_started")
    with sl.log_trace_span("fwd_bwd"):
        ...
"""

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
from pimm.observability.structured_logger.structured_logging import (
    init_structured_logger,
    log_trace_instant,
    log_trace_scalar,
    log_trace_span,
    shutdown_structured_logger,
)

__all__ = [
    "add_step_tag",
    "clear_step_tags",
    "get_epoch",
    "get_iteration",
    "get_relative_step",
    "get_step",
    "get_step_tags",
    "init_structured_logger",
    "log_trace_instant",
    "log_trace_scalar",
    "log_trace_span",
    "set_step",
    "shutdown_structured_logger",
]
