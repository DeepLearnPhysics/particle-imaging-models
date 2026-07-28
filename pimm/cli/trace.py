#!/usr/bin/env python3
"""Inspect and export pimm structured trace logs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceIdentity:
    """A single traced process within a run attempt."""

    source: str
    rank: str
    attempt: str
    trace_id: str


@dataclass(frozen=True)
class OpenSpan:
    """A span start for which no matching end record was found."""

    phase: str
    step: Any
    relative_step: Any
    time_us: float
    task_name: str | None


@dataclass(frozen=True)
class TraceStatus:
    """Last known state for one traced process."""

    identity: TraceIdentity
    last_step: Any
    last_relative_step: Any
    last_event: str
    last_time_us: float
    open_spans: tuple[OpenSpan, ...]


@dataclass(frozen=True)
class PhaseDuration:
    """Aggregate wall-time statistics for one traced phase."""

    phase: str
    count: int
    p50_ms: float
    p95_ms: float
    max_ms: float
    slowest_identity: TraceIdentity
    slowest_step: Any


@dataclass(frozen=True)
class TraceReadResult:
    """Records loaded from a structured log directory."""

    records: tuple[dict[str, Any], ...]
    files: tuple[Path, ...]
    malformed_lines: int


@dataclass(frozen=True)
class TraceSummary:
    """Post-hoc summary of a set of structured trace records."""

    statuses: tuple[TraceStatus, ...]
    phases: tuple[PhaseDuration, ...]
    record_count: int
    file_count: int
    malformed_lines: int


@dataclass(frozen=True)
class TraceLocation:
    """Resolved run and structured-log directories."""

    run_dir: Path
    log_dir: Path


def _trace_files(log_dir: Path) -> tuple[Path, ...]:
    """Return base and rotated JSONL files in deterministic order."""
    return tuple(sorted(path for path in log_dir.glob("*.jsonl*") if path.is_file()))


def resolve_trace_location(path: str | Path) -> TraceLocation:
    """Resolve either an experiment directory or its structured_logs directory."""
    candidate = Path(path).expanduser()
    if not candidate.is_dir():
        raise FileNotFoundError(f"Trace input directory does not exist: {candidate}")

    nested_logs = candidate / "structured_logs"
    if nested_logs.is_dir():
        run_dir, log_dir = candidate, nested_logs
    elif candidate.name == "structured_logs" or _trace_files(candidate):
        log_dir = candidate
        run_dir = candidate.parent
    else:
        raise FileNotFoundError(
            f"No structured_logs directory or *.jsonl* files found under: {candidate}"
        )

    if not _trace_files(log_dir):
        raise FileNotFoundError(f"No *.jsonl* trace files found under: {log_dir}")
    return TraceLocation(run_dir=run_dir, log_dir=log_dir)


def _source_name(path: Path) -> str:
    """Normalize a base or rotated filename to one process source name."""
    name = path.name
    marker = name.find(".jsonl")
    return name[:marker] if marker >= 0 else path.stem


def _numeric(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _record_time_us(record: dict[str, Any]) -> float:
    if record.get("time_us") is not None:
        return _numeric(record["time_us"])
    if record.get("time_ms") is not None:
        return _numeric(record["time_ms"]) * 1_000
    return _numeric(record.get("time")) * 1_000_000


def _record_sort_key(record: dict[str, Any]) -> tuple[float, float, str, int]:
    return (
        _record_time_us(record),
        _numeric(record.get("seq_id"), default=-1),
        str(record.get("_input_path", "")),
        int(record.get("_input_line", 0)),
    )


def load_trace_records(log_dir: str | Path) -> TraceReadResult:
    """Load base and rotated JSONL files, skipping damaged records.

    A process can be killed while its final JSON object is being written. Such a
    truncated line, or any other malformed/non-object line, is counted and
    ignored rather than making diagnosis of the surviving records impossible.
    """
    directory = Path(log_dir)
    files = _trace_files(directory)
    records: list[dict[str, Any]] = []
    malformed_lines = 0

    for path in files:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    malformed_lines += 1
                    continue
                if not isinstance(record, dict):
                    malformed_lines += 1
                    continue
                record["_source_file"] = _source_name(path)
                record["_input_path"] = str(path)
                record["_input_line"] = line_number
                records.append(record)

    records.sort(key=_record_sort_key)
    return TraceReadResult(
        records=tuple(records),
        files=files,
        malformed_lines=malformed_lines,
    )


def _label(value: Any, fallback: str = "-") -> str:
    if value is None:
        return fallback
    text = str(value)
    return text if text else fallback


def _identity(record: dict[str, Any]) -> TraceIdentity:
    source_file = _label(record.get("_source_file"), "unknown")
    rank = record.get("rank")
    if rank is None:
        rank = record.get("global_rank")
    return TraceIdentity(
        source=_label(record.get("source"), source_file),
        rank=_label(rank),
        attempt=_label(record.get("attempt")),
        # Old traces have no trace_id. The normalized filename keeps separate
        # process files distinct while still grouping their rotated segments.
        trace_id=_label(record.get("trace_id"), source_file),
    )


def _identity_sort_key(identity: TraceIdentity) -> tuple[Any, ...]:
    def sortable_number(value: str) -> tuple[int, int | str]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return (
        identity.source,
        sortable_number(identity.attempt),
        sortable_number(identity.rank),
        identity.trace_id,
    )


def _event_name(record: dict[str, Any]) -> str:
    event_type = _label(record.get("log_type_name"), "")
    event_name = _label(record.get("event_name"), "")
    if event_type == "metric_value" and event_name:
        return f"metric_value:{event_name}"
    return event_type or event_name or "<unknown>"


def _is_span_record(record: dict[str, Any]) -> bool:
    return record.get("log_type") != "instant"


def _phase_name(record: dict[str, Any], suffix: str) -> str:
    event_type = _label(record.get("log_type_name"), "")
    if event_type.endswith(suffix):
        return event_type[: -len(suffix)]
    return event_type


def _pairing_lane(record: dict[str, Any]) -> tuple[str | None, str]:
    task_name = record.get("task_name")
    return (
        str(task_name) if task_name is not None else None,
        _label(record.get("tid")),
    )


def _percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for already sorted values."""
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def summarize_records(
    records: Iterable[dict[str, Any]],
    *,
    file_count: int = 0,
    malformed_lines: int = 0,
) -> TraceSummary:
    """Compute process status, unmatched spans, and phase-duration statistics."""
    ordered = sorted(records, key=_record_sort_key)
    last_by_identity: dict[TraceIdentity, dict[str, Any]] = {}
    last_step_by_identity: dict[TraceIdentity, Any] = {}
    last_relative_step_by_identity: dict[TraceIdentity, Any] = {}
    pending: dict[
        tuple[TraceIdentity, tuple[str | None, str], str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    duration_samples: dict[str, list[tuple[float, TraceIdentity, Any]]] = defaultdict(
        list
    )

    for record in ordered:
        identity = _identity(record)
        last_by_identity[identity] = record
        if record.get("step") is not None:
            last_step_by_identity[identity] = record["step"]
        if record.get("relative_step") is not None:
            last_relative_step_by_identity[identity] = record["relative_step"]
        if not _is_span_record(record):
            continue

        event_type = _label(record.get("log_type_name"), "")
        lane = _pairing_lane(record)
        if event_type.endswith("_start"):
            phase = _phase_name(record, "_start")
            pending[(identity, lane, phase)].append(record)
        elif event_type.endswith("_end"):
            phase = _phase_name(record, "_end")
            starts = pending.get((identity, lane, phase))
            if starts:
                starts.pop()

            duration = record.get("value")
            duration_ms = _numeric(duration, default=math.nan)
            if math.isfinite(duration_ms) and duration_ms >= 0:
                duration_samples[phase].append(
                    (duration_ms, identity, record.get("step"))
                )

    open_by_identity: dict[TraceIdentity, list[OpenSpan]] = defaultdict(list)
    for (identity, lane, phase), starts in pending.items():
        for start in starts:
            open_by_identity[identity].append(
                OpenSpan(
                    phase=phase,
                    step=start.get("step"),
                    relative_step=start.get("relative_step"),
                    time_us=_record_time_us(start),
                    task_name=lane[0],
                )
            )

    statuses = []
    for identity in sorted(last_by_identity, key=_identity_sort_key):
        last = last_by_identity[identity]
        open_spans = tuple(
            sorted(open_by_identity.get(identity, []), key=lambda span: span.time_us)
        )
        statuses.append(
            TraceStatus(
                identity=identity,
                last_step=last_step_by_identity.get(identity),
                last_relative_step=last_relative_step_by_identity.get(identity),
                last_event=_event_name(last),
                last_time_us=_record_time_us(last),
                open_spans=open_spans,
            )
        )

    phases = []
    for phase in sorted(duration_samples):
        samples = duration_samples[phase]
        values = sorted(sample[0] for sample in samples)
        slowest_ms, slowest_identity, slowest_step = max(
            samples, key=lambda sample: sample[0]
        )
        phases.append(
            PhaseDuration(
                phase=phase,
                count=len(values),
                p50_ms=_percentile(values, 0.50),
                p95_ms=_percentile(values, 0.95),
                max_ms=slowest_ms,
                slowest_identity=slowest_identity,
                slowest_step=slowest_step,
            )
        )

    return TraceSummary(
        statuses=tuple(statuses),
        phases=tuple(phases),
        record_count=len(ordered),
        file_count=file_count,
        malformed_lines=malformed_lines,
    )


def _format_open_span(span: OpenSpan) -> str:
    fields = [f"step={_label(span.step)}"]
    if span.relative_step is not None:
        fields.append(f"relative_step={span.relative_step}")
    if span.task_name is not None:
        fields.append(f"task={span.task_name}")
    return f"{span.phase}({', '.join(fields)})"


def format_summary(summary: TraceSummary) -> str:
    """Render a stable, dependency-free text report."""
    lines = [
        (
            f"Structured trace: {summary.record_count} records from "
            f"{summary.file_count} files"
        ),
        "",
        "Process status:",
    ]
    if not summary.statuses:
        lines.append("  (no valid records)")
    for status in summary.statuses:
        identity = status.identity
        lines.append(
            "  "
            f"source={identity.source} rank={identity.rank} "
            f"attempt={identity.attempt} trace_id={identity.trace_id} "
            f"last_step={_label(status.last_step)} "
            f"relative_step={_label(status.last_relative_step)} "
            f"last_event={status.last_event} "
            f"open_spans={len(status.open_spans)}"
        )
        for span in status.open_spans:
            lines.append(f"    open: {_format_open_span(span)}")

    lines.extend(["", "Phase durations (ms):"])
    if not summary.phases:
        lines.append("  (no completed spans)")
    for phase in summary.phases:
        slowest = phase.slowest_identity
        lines.append(
            "  "
            f"{phase.phase}: count={phase.count} "
            f"p50={phase.p50_ms:.3f} p95={phase.p95_ms:.3f} "
            f"max={phase.max_ms:.3f} "
            f"slowest=source:{slowest.source}/rank:{slowest.rank}/"
            f"attempt:{slowest.attempt}/step:{_label(phase.slowest_step)}"
        )
    return "\n".join(lines)


def _generate_gantt_trace(log_dir: Path, output_path: Path) -> dict[str, Any]:
    """Import the torch-dependent trace exporter only when requested."""
    from pimm.observability.structured_logger.gantt_generator import (
        generate_gantt_trace,
    )

    return generate_gantt_trace(str(log_dir), str(output_path))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="trace_command", required=True)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize rank progress, open spans, and phase durations.",
    )
    summarize_parser.add_argument(
        "run_dir",
        help="Experiment directory or its structured_logs directory.",
    )

    export_parser = subparsers.add_parser(
        "export",
        help="Merge per-rank JSONL logs into a Perfetto-compatible trace.",
    )
    export_parser.add_argument(
        "run_dir",
        help="Experiment directory or its structured_logs directory.",
    )
    export_parser.add_argument(
        "-o",
        "--output",
        help="Output JSON path (default: RUN_DIR/analysis/structured_trace.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        location = resolve_trace_location(args.run_dir)
    except FileNotFoundError as error:
        print(f"pimm trace: {error}", file=sys.stderr)
        return 2

    if args.trace_command == "summarize":
        loaded = load_trace_records(location.log_dir)
        summary = summarize_records(
            loaded.records,
            file_count=len(loaded.files),
            malformed_lines=loaded.malformed_lines,
        )
        print(format_summary(summary))
        if loaded.malformed_lines:
            print(
                "pimm trace: skipped "
                f"{loaded.malformed_lines} malformed or truncated JSONL "
                f"line{'s' if loaded.malformed_lines != 1 else ''}",
                file=sys.stderr,
            )
        return 0 if loaded.records else 1

    output = (
        Path(args.output).expanduser()
        if args.output
        else location.run_dir / "analysis" / "structured_trace.json"
    )
    _generate_gantt_trace(location.log_dir, output)
    print(f"Exported structured trace to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
