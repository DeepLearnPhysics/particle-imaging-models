# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in this directory.
#
# Adapted from TorchTitan:
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/structured_logger/jsonl_handler.py#L1-L196

"""Default JSONL backend: formatter, rotating file handler, and factory.

``TraceJsonlFormatter`` can also be subclassed to enrich records for custom
backends.
"""

import datetime as dt
import itertools
import json
import logging
import os
import random
import re
import socket
import string
import threading
from logging.handlers import RotatingFileHandler
from timeit import default_timer as timer
from typing import Any

from pimm.observability.structured_logger.step_state import (
    get_epoch,
    get_iteration,
    get_relative_step,
    get_step,
    get_step_tags,
)
from pimm.observability.structured_logger.structured_logging import (
    ExtraFields,
    LogType,
    TraceEventsOnlyFilter,
)

console_logger: logging.Logger = logging.getLogger(__name__)

MAX_MESSAGE_SIZE: int = 1000
SCHEMA_VERSION: int = 1
DEFAULT_MAX_FILE_SIZE_MB: float = 128
DEFAULT_BACKUP_COUNT: int = 3


def _filename_component(value: str) -> str:
    """Make a source label safe to use as one filename component."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "trace"


class TraceJsonlFormatter(logging.Formatter):
    """Format trace records as one flat JSON object per line.

    Per-process fields (rank, world size, source, hostname, local rank, attempt,
    job ID, and trace ID) are captured in ``__init__``; per-step fields (step,
    relative step, epoch, iteration, and step tags) are pulled from
    :mod:`.step_state` at emit time.

    Subclass to enrich records with backend-specific fields.

    Example output (wrapped for readability)::

        {"rank": 0, "source": "training", "attempt": 1, "step": 5,
         "epoch": 0, "iteration": 4, "log_type_name": "fwd_bwd_end",
         "value": 12.5, "task_name": "Task-1", "step_tags": ["gc"],
         "time_us": 1709500000123456, "caller": "trainer.py:796:train_step",
         "seq_id": 42}

    From: https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/structured_logger/jsonl_handler.py#L41-L161
    """

    def __init__(
        self,
        rank: int,
        source: str,
        *,
        world_size: int | None = None,
        attempt: int | None = None,
        job_id: str | None = None,
        trace_id: str | None = None,
    ):
        super().__init__()
        self.rank = int(rank)
        self.source = str(source)
        self.world_size = int(world_size) if world_size is not None else 1
        self.attempt = int(attempt) if attempt is not None else 1
        self.job_id = str(job_id) if job_id is not None else None
        self.trace_id = str(trace_id) if trace_id is not None else None
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._host_name = socket.gethostname()
        self._seq_counter = itertools.count()
        self._thread_local = threading.local()

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(self._log_dict(record), separators=(",", ":"))

    def _log_dict(self, record: logging.LogRecord) -> dict[str, Any]:
        """Build the flat dict emitted as one JSONL line."""
        log_dict: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "delta_ms": self._refresh_event_delta(),
            "tid": threading.get_native_id(),
            "rank": self.rank,
            "global_rank": self.rank,
            "local_rank": self._local_rank,
            "world_size": self.world_size,
            "source": self.source,
            "host_name": self._host_name,
            "pid": os.getpid(),
            "attempt": self.attempt,
            "job_id": self.job_id,
            "trace_id": self.trace_id,
        }

        # Context values are sampled at emit time.
        step = get_step()
        if step is not None:
            log_dict["step"] = step
        relative_step = get_relative_step()
        if relative_step is not None:
            log_dict["relative_step"] = relative_step
        epoch = get_epoch()
        if epoch is not None:
            log_dict["epoch"] = epoch
        iteration = get_iteration()
        if iteration is not None:
            log_dict["iteration"] = iteration
        step_tags = get_step_tags()
        if step_tags:
            log_dict["step_tags"] = list(step_tags)

        log_dict["time"] = int(record.created)
        log_dict["time_ms"] = int(record.created * 1000)
        log_dict["time_us"] = int(record.created * 1_000_000)
        log_dict["log_type"] = getattr(
            record, str(ExtraFields.LOG_TYPE), str(LogType.TEXT)
        )
        log_dict["log_type_name"] = getattr(
            record, str(ExtraFields.LOG_TYPE_NAME), None
        )

        # Per-record values are authoritative. This keeps records stable even
        # if a custom handler formats them after the ambient context changes.
        for field, output_name in (
            (ExtraFields.STEP, "step"),
            (ExtraFields.RELATIVE_STEP, "relative_step"),
            (ExtraFields.EPOCH, "epoch"),
            (ExtraFields.ITERATION, "iteration"),
        ):
            value = getattr(record, str(field), None)
            if value is not None:
                log_dict[output_name] = value

        log_dict["event_name"] = getattr(record, str(ExtraFields.EVENT_NAME), None)
        value = getattr(record, str(ExtraFields.VALUE), None)
        if isinstance(value, (float, int)):
            log_dict["value"] = float(value)

        task_name = getattr(record, str(ExtraFields.TASK_NAME), None)
        if task_name is not None:
            log_dict["task_name"] = task_name

        try:
            pathname = os.path.relpath(record.pathname)
        except ValueError:
            pathname = record.pathname
        log_dict["caller"] = f"{pathname}:{record.lineno}:{record.funcName}"
        log_dict["log_file"] = record.filename
        log_dict["log_function"] = record.funcName
        log_dict["log_level"] = record.levelname
        log_dict["logger_name"] = record.name
        if record.stack_info:
            log_dict["stack_info"] = record.stack_info

        log_dict["seq_id"] = next(self._seq_counter)

        message = record.getMessage()
        if message is not None:
            if len(message) <= MAX_MESSAGE_SIZE:
                log_dict["message"] = message
            else:
                half = MAX_MESSAGE_SIZE // 2
                log_dict["message"] = message[:half] + "..." + message[-half:]

        return log_dict

    def _refresh_event_delta(self) -> float:
        now = timer()
        previous = getattr(self._thread_local, "last_event_time", now)
        self._thread_local.last_event_time = now
        return (now - previous) * 1000


class TraceJsonlHandler(RotatingFileHandler):
    """Bounded per-rank JSONL handler.

    Active files are named
    ``{source}.global_rank_{rank}.{timestamp}-{random}.jsonl``. Rotated
    segments use the standard ``.jsonl.1``, ``.jsonl.2``, ... suffixes.

    From: https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/structured_logger/jsonl_handler.py#L164-L182
    """

    def __init__(
        self,
        rank: int,
        source: str,
        output_dir: str,
        *,
        world_size: int | None = None,
        attempt: int | None = None,
        job_id: str | None = None,
        trace_id: str | None = None,
        max_file_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB,
        backup_count: int = DEFAULT_BACKUP_COUNT,
    ):
        max_file_size_mb = float(max_file_size_mb)
        backup_count = int(backup_count)
        if max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be greater than zero")
        if backup_count < 0:
            raise ValueError("backup_count must be non-negative")

        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        random_suffix = "".join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(6)
        )
        filename = (
            f"{_filename_component(str(source))}.global_rank_{int(rank)}."
            f"{timestamp}-{random_suffix}.jsonl"
        )
        filepath = os.path.join(str(output_dir), "structured_logs", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        super().__init__(
            filename=filepath,
            maxBytes=max(1, int(max_file_size_mb * 1024 * 1024)),
            backupCount=backup_count,
            encoding="utf-8",
        )
        self.setFormatter(
            TraceJsonlFormatter(
                rank=rank,
                source=source,
                world_size=world_size,
                attempt=attempt,
                job_id=job_id,
                trace_id=trace_id,
            )
        )
        self.addFilter(TraceEventsOnlyFilter())

    def doRollover(self) -> None:
        """Rotate while keeping the active file bounded with zero backups."""
        if self.backupCount:
            super().doRollover()
            return

        if self.stream:
            self.stream.close()
            self.stream = None
        # ``RotatingFileHandler`` does not rotate at all when backupCount is
        # zero. Truncate explicitly so ``backup_count=0`` means "retain only
        # the current segment" rather than silently becoming unbounded.
        with open(
            self.baseFilename,
            "w",
            encoding=self.encoding,
            errors=self.errors,
        ):
            pass
        if not self.delay:
            self.stream = self._open()


def register_jsonl_handler(
    *,
    structured_logger: logging.Logger,
    rank: int,
    source: str,
    output_dir: str,
    world_size: int | None = None,
    attempt: int | None = None,
    job_id: str | None = None,
    trace_id: str | None = None,
    max_file_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    **kw: Any,
) -> None:
    """Attach the default rotating JSONL handler."""
    del kw
    handler = TraceJsonlHandler(
        rank=rank,
        source=source,
        output_dir=output_dir,
        world_size=world_size,
        attempt=attempt,
        job_id=job_id,
        trace_id=trace_id,
        max_file_size_mb=max_file_size_mb,
        backup_count=backup_count,
    )
    structured_logger.addHandler(handler)
    console_logger.info("Structured logging -> JSONL: %s", handler.baseFilename)
