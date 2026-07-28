# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in this directory.
#
# Adapted from TorchTitan:
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/structured_logger/gantt_generator.py#L1-L484

"""Merge per-rank trace JSONL into a Chrome Trace JSON for Perfetto.

This module loads all records and builds the full event list in memory. It is
intended for debugging traces; very large runs may need a streaming converter.
"""

import heapq
import json
import logging
import os
import re
from collections import defaultdict
from glob import glob
from typing import Any

from pimm.observability.structured_logger.structured_logging import LogType

logger = logging.getLogger(__name__)

_TRACE_FILE_RE = re.compile(r"^(?P<source>.+)\.jsonl(?:\.(?P<rotation>\d+))?$")


def generate_gantt_trace(log_dir: str, output_path: str) -> dict[str, Any]:
    """Merge all active and rotated JSONL files into a Chrome Trace document.

    Open the result in https://ui.perfetto.dev or ``chrome://tracing``.

    Records are grouped by asyncio task so all spans that share a task,
    including nested spans, render on one track. Non-overlapping tasks reuse a
    track; truly concurrent tasks get separate tracks.

    Example::

        with log_trace_span("step"):
            with log_trace_span("forward"):
                model(batch)

        generate_gantt_trace(
            "outputs/structured_logs/", "outputs/structured_trace.json"
        )

    Incomplete ``*_start`` records are retained as explicitly marked spans.
    Their provisional end is the logical process's last observed timestamp,
    with a minimum one-microsecond duration so they remain visible in Perfetto.

    Args:
        log_dir: Directory containing per-rank active and rotated
            ``*.jsonl*`` trace files.
        output_path: Path to write the merged Chrome Trace JSON.

    Returns:
        The Chrome Trace dict (``{"traceEvents": [...], "metadata": {...}}``).
        It is also written to ``output_path``.
    """
    records = load_all_records(log_dir)
    if not records:
        trace: dict[str, Any] = {
            "traceEvents": [],
            "metadata": {
                "record_count": 0,
                "source_count": 0,
                "complete_spans": 0,
                "incomplete_spans": 0,
                "instant_events": 0,
            },
        }
        _write_trace(trace, output_path)
        logger.info("No structured trace records found in %s", log_dir)
        return trace

    sources = sorted({record["_source_file"] for record in records})
    source_to_pid = {source: index for index, source in enumerate(sources)}
    paired, instants = _collect_paired_and_instants(records, source_to_pid)
    tid_by_source_and_task = _assign_tids_per_source(paired)
    events = _emit_chrome_events(
        paired=paired,
        instants=instants,
        source_to_pid=source_to_pid,
        tid_by_source_and_task=tid_by_source_and_task,
    )

    incomplete_count = sum(bool(span.get("incomplete")) for span in paired)
    trace = {
        "traceEvents": events,
        "metadata": {
            "record_count": len(records),
            "source_count": len(sources),
            "complete_spans": len(paired) - incomplete_count,
            "incomplete_spans": incomplete_count,
            "instant_events": len(instants),
        },
    }
    _write_trace(trace, output_path)

    logger.info(
        "Chrome Trace: %s (%d events from %d sources)",
        output_path,
        len(events),
        len(sources),
    )
    if incomplete_count:
        logger.warning(
            "Retained %d incomplete structured trace span(s)", incomplete_count
        )
    return trace


def _write_trace(trace: dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(trace, output_file, indent=2)


def _trace_segments(log_dir: str) -> list[tuple[str, int, str]]:
    """Return ``(logical_source, rotation, path)`` in stream order.

    ``.jsonl.N`` uses the standard rotating-handler convention: larger N is
    older. Segments therefore sort ``N ... 2, 1, active`` for each logical
    source.
    """
    segments: list[tuple[str, int, str]] = []
    for path in glob(os.path.join(log_dir, "*.jsonl*")):
        match = _TRACE_FILE_RE.fullmatch(os.path.basename(path))
        if not match or not os.path.isfile(path):
            continue
        rotation = int(match.group("rotation") or 0)
        segments.append((match.group("source"), rotation, path))
    return sorted(segments, key=lambda item: (item[0], -item[1]))


def load_all_records(log_dir: str) -> list[dict[str, Any]]:
    """Load active and rotated trace segments, tolerating crash truncation.

    Rotated segments are canonicalized to the active filename without the
    ``.jsonl`` suffix. Malformed lines and valid non-object JSON values are
    warned about and skipped.

    Args:
        log_dir: Directory containing per-rank ``*.jsonl*`` files.

    Returns:
        Records across all readable files. Each record has a ``"_source_file"``
        key for grouping by logical process. For each process, retained rotated
        segments are read from oldest to newest and then the active file.
    """
    records: list[dict[str, Any]] = []
    for source, _rotation, path in _trace_segments(log_dir):
        try:
            with open(path, encoding="utf-8", errors="replace") as trace_file:
                for line_number, line in enumerate(trace_file, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as error:
                        logger.warning(
                            "Skipping malformed structured trace line %s:%d: %s",
                            path,
                            line_number,
                            error,
                        )
                        continue
                    if not isinstance(record, dict):
                        logger.warning(
                            "Skipping non-object structured trace line %s:%d",
                            path,
                            line_number,
                        )
                        continue
                    record["_source_file"] = source
                    records.append(record)
        except OSError as error:
            logger.warning(
                "Could not read structured trace segment %s: %s", path, error
            )
    return records


def _record_time_us(record: dict[str, Any]) -> int:
    value = record.get("time_us")
    if value is not None:
        return int(value)
    value = record.get("time_ms")
    if value is not None:
        return int(value) * 1000
    return int(record.get("time", 0)) * 1_000_000


def _trace_context(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "step",
            "relative_step",
            "epoch",
            "iteration",
            "step_tags",
            "rank",
            "attempt",
            "trace_id",
        )
        if record.get(key) is not None
    }


def _merge_span_context(
    start_context: dict[str, Any], end_context: dict[str, Any]
) -> dict[str, Any]:
    """Merge end context without replacing the start's step coordinates.

    A tag can be added from inside a span (checkpoint/evaluation are common
    examples), so using only the start record would silently lose it.
    """
    merged = dict(start_context)
    for key, value in end_context.items():
        if key != "step_tags" and key not in merged:
            merged[key] = value

    tags: list[Any] = []
    for context in (start_context, end_context):
        value = context.get("step_tags")
        values = value if isinstance(value, (list, tuple)) else [value]
        for tag in values:
            if tag is not None and tag not in tags:
                tags.append(tag)
    if tags:
        merged["step_tags"] = tags
    return merged


def _collect_paired_and_instants(
    records: list[dict[str, Any]], source_to_pid: dict[str, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair ``_start`` / ``_end`` via a per-(source, task_name) LIFO stack.

    Each asyncio task name gets its own stack, so nested spans within a task
    pair correctly regardless of what other tasks on the same process are
    doing. Synchronous code has ``task_name=None`` throughout and pairs on a
    single stack per source.

    Instants include ``log_trace_instant`` markers, ``log_trace_scalar`` metric
    values, and ``_error`` records.

    Example -- five records for one task and source::

        {"log_type_name": "step_start",    "time_us": 100, "task_name": "Task-1"}
        {"log_type_name": "forward_start", "time_us": 110, "task_name": "Task-1"}
        {"log_type_name": "forward_end",   "time_us": 500, "task_name": "Task-1", "value": 0.39}
        {"log_type_name": "metric_value",  "time_us": 520, "event_name": "loss", "value": 2.5}
        {"log_type_name": "step_end",      "time_us": 600, "task_name": "Task-1", "value": 0.5}

    The inner ``forward`` span is paired before the outer ``step`` span because
    the stack is LIFO. Orphan ends use their recorded duration; unmatched starts
    are retained as incomplete spans. Start/end context is merged so tags added
    inside a span are preserved.

    Args:
        records: Records across all logical sources, in segment/file order.
        source_to_pid: Map from ``_source_file`` to its Perfetto process ID.

    Returns:
        ``(paired, instants)`` -- paired spans in end order and instant events
        preserving record order.
    """
    paired: list[dict[str, Any]] = []
    instants: list[dict[str, Any]] = []
    pending: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    last_time_by_source: dict[str, int] = {}

    for record in records:
        event_type = str(record.get("log_type_name") or "")
        log_type = record.get("log_type", "")
        time_us = _record_time_us(record)
        source = record["_source_file"]
        last_time_by_source[source] = max(
            time_us, last_time_by_source.get(source, time_us)
        )
        pid = source_to_pid[source]
        task_name = record.get("task_name")
        caller = record.get("caller")
        key = (source, task_name)
        context = _trace_context(record)

        # Intent is authoritative: instant names may legitimately end in
        # ``_start``.
        if log_type == str(LogType.INSTANT):
            if event_type == "metric_value":
                event_name = record.get("event_name", "metric")
                value = record.get("value")
                display_name = (
                    f"{event_name}={float(value):.4f}"
                    if isinstance(value, (float, int))
                    else str(event_name)
                )
            else:
                display_name = event_type
            instants.append(
                {
                    "name": display_name,
                    "time_us": time_us,
                    "pid": pid,
                    "source": source,
                    "task_name": task_name,
                    "caller": caller,
                    "context": context,
                }
            )
        elif event_type.endswith("_start"):
            type_name = event_type.removesuffix("_start")
            pending[key].append(
                {
                    "ts": time_us,
                    "display_name": record.get("event_name") or type_name,
                    "pid": pid,
                    "task_name": task_name,
                    "source": source,
                    "caller": caller,
                    "context": context,
                }
            )
        elif event_type.endswith("_end"):
            type_name = event_type.removesuffix("_end")
            duration_ms = record.get("value", 0) or 0
            duration_us = max(0, int(float(duration_ms) * 1000))
            stack = pending.get(key)
            start = stack.pop() if stack else None
            if start is not None:
                start_ts = start["ts"]
                end_ts = start_ts + duration_us
                span_context = _merge_span_context(start["context"], context)
            else:
                # An active rotated segment can outlive deleted backups. Keep
                # an orphan end visible using its measured duration.
                end_ts = time_us
                start_ts = max(0, end_ts - duration_us)
                span_context = context
            paired.append(
                {
                    "pid": pid,
                    "task_name": (start or {}).get("task_name", task_name),
                    "source": (start or {}).get("source", source),
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "display_name": (start or {}).get("display_name", type_name),
                    "duration_ms": float(duration_ms),
                    "caller": (start or {}).get("caller", caller),
                    "context": span_context,
                    "incomplete": False,
                }
            )
        elif event_type.endswith("_error"):
            type_name = event_type.removesuffix("_error")
            instants.append(
                {
                    "name": f"ERROR: {type_name}",
                    "time_us": time_us,
                    "pid": pid,
                    "source": source,
                    "task_name": task_name,
                    "caller": caller,
                    "context": context,
                }
            )
        else:
            instants.append(
                {
                    "name": event_type,
                    "time_us": time_us,
                    "pid": pid,
                    "source": source,
                    "task_name": task_name,
                    "caller": caller,
                    "context": context,
                }
            )

    # A killed process can leave starts without ends. Preserve those spans
    # instead of silently dropping the most diagnostically useful records.
    for stack in pending.values():
        for start in stack:
            observed_end = last_time_by_source.get(start["source"], start["ts"])
            end_ts = max(start["ts"] + 1, observed_end)
            paired.append(
                {
                    "pid": start["pid"],
                    "task_name": start["task_name"],
                    "source": start["source"],
                    "start_ts": start["ts"],
                    "end_ts": end_ts,
                    "display_name": start["display_name"],
                    "duration_ms": (end_ts - start["ts"]) / 1000,
                    "caller": start["caller"],
                    "context": start["context"],
                    "incomplete": True,
                }
            )

    return paired, instants


def _assign_tids_per_source(
    paired: list[dict[str, Any]],
) -> dict[tuple[str, str | None], int]:
    """Assign a Perfetto ``tid`` to every paired span (mutates ``s["tid"]``).

    Task-level interval scheduling:

    1. Group spans by ``(source, task_name)``. Each asyncio task has one group;
       synchronous code groups under ``task_name=None``.
    2. Compute each group's time range ``[min(start_ts), max(end_ts)]``.
    3. Slot-pack task ranges within each source via min-heap interval
       scheduling. Non-overlapping task ranges reuse a slot.
    4. Every span inherits its task's slot.

    Nested spans within a task share a tid, so Perfetto renders them as a
    stacked flamegraph on one track. Concurrent tasks whose ranges overlap get
    different tids and render as parallel tracks. Sequential tasks reuse a tid.

    Example (one source, four spans across three tasks)::

        span         start  end    task
        outer        100    500    Task-1
        inner        150    200    Task-1
        weight_sync  600    800    Task-2
        log_shard    650    700    Task-3

        Task-1 -> tid 0
        Task-2 -> tid 0  # starts after Task-1 ends
        Task-3 -> tid 1  # overlaps Task-2

    Args:
        paired: Paired span dicts from ``_collect_paired_and_instants``.
            Mutated in place: each span gets a ``"tid"`` key.

    Returns:
        ``(source, task_name) -> tid`` mapping, also used to place instant
        events on their enclosing task's track.
    """
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in paired:
        by_source[span["source"]].append(span)

    tid_by_source_and_task: dict[tuple[str, str | None], int] = {}
    for source, spans in by_source.items():
        task_ranges: dict[str | None, list[int]] = {}
        for span in spans:
            task_name = span.get("task_name")
            start, end = span["start_ts"], span["end_ts"]
            current = task_ranges.get(task_name)
            if current is None:
                task_ranges[task_name] = [start, end]
            else:
                current[0] = min(current[0], start)
                current[1] = max(current[1], end)

        busy: list[tuple[int, int]] = []
        next_slot = 0
        for task_name, (start, end) in sorted(
            task_ranges.items(), key=lambda item: item[1][0]
        ):
            if busy and busy[0][0] <= start:
                _, slot = heapq.heappop(busy)
            else:
                slot = next_slot
                next_slot += 1
            heapq.heappush(busy, (end, slot))
            tid_by_source_and_task[(source, task_name)] = slot

        for span in spans:
            span["tid"] = tid_by_source_and_task[(source, span.get("task_name"))]

    return tid_by_source_and_task


def _resolve_instant_tid(
    *,
    source: str,
    task_name: str | None,
    tid_by_source_and_task: dict[tuple[str, str | None], int],
) -> int:
    """Pick the Perfetto tid for an instant event (no paired span).

    An instant lands on the same track as its enclosing task if that task has
    paired spans. Otherwise it falls back to tid 0, the per-process main track.

    Args:
        source: The instant's logical process source.
        task_name: Its asyncio task name, or ``None`` outside a task.
        tid_by_source_and_task: Map produced by ``_assign_tids_per_source``.

    Returns:
        The Perfetto ``tid`` for the instant.
    """
    return tid_by_source_and_task.get((source, task_name), 0)


def _emit_chrome_events(
    *,
    paired: list[dict[str, Any]],
    instants: list[dict[str, Any]],
    source_to_pid: dict[str, int],
    tid_by_source_and_task: dict[tuple[str, str | None], int],
) -> list[dict[str, Any]]:
    """Build the Chrome Trace event list (one Perfetto process per source).

    Emits three kinds of events:

    - ``"M"`` metadata ``process_name`` -- one per source, labels the process
      track in Perfetto.
    - ``"X"`` complete event -- one per paired span, with ``ts`` / ``dur`` for
      rendering as a bar. Incomplete spans are marked in ``args``.
    - ``"i"`` instant event -- rendered as a vertical marker on its track.

    Args:
        paired: Spans from ``_collect_paired_and_instants`` with ``tid`` already
            assigned by ``_assign_tids_per_source``.
        instants: Instant events from ``_collect_paired_and_instants``.
        source_to_pid: Map from source name to Perfetto pid.
        tid_by_source_and_task: Map used to place instants on their enclosing
            task's track.

    Returns:
        A flat list of Chrome Trace event dicts ready to serialize.
    """
    events: list[dict[str, Any]] = []

    for source, pid in source_to_pid.items():
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": source},
            }
        )

    for span in paired:
        args = dict(span["context"])
        args["duration_ms"] = f"{span['duration_ms']:.2f}"
        if span.get("caller"):
            args["caller"] = span["caller"]
        if span.get("incomplete"):
            args["incomplete"] = True
        events.append(
            {
                "name": span["display_name"],
                "ph": "X",
                "ts": span["start_ts"],
                "dur": span["end_ts"] - span["start_ts"],
                "pid": span["pid"],
                "tid": span.get("tid", 0),
                "args": args,
            }
        )

    for instant in instants:
        args = dict(instant["context"])
        if instant.get("caller"):
            args["caller"] = instant["caller"]
        events.append(
            {
                "name": instant["name"],
                "ph": "i",
                "ts": instant["time_us"],
                "pid": instant["pid"],
                "tid": _resolve_instant_tid(
                    source=instant["source"],
                    task_name=instant["task_name"],
                    tid_by_source_and_task=tid_by_source_and_task,
                ),
                "s": "t",
                "args": args,
            }
        )

    return events
