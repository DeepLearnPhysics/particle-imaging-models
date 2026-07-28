# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file is adapted from TorchTitan and is licensed under the BSD-style
# license found in pimm/observability/structured_logger/LICENSE.
#
# From:
# https://github.com/pytorch/torchtitan/blob/main/tests/unit_tests/observability/test_structured_logging.py#L995-L1529

from __future__ import annotations

import json
from pathlib import Path

from pimm.observability.structured_logger.gantt_generator import (
    generate_gantt_trace,
    load_all_records,
)


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _start(
    name: str,
    time_us: int,
    *,
    task_name: str | None = None,
    step: int = 1,
) -> dict:
    return {
        "log_type": "event",
        "log_type_name": f"{name}_start",
        "time_us": time_us,
        "rank": 0,
        "step": step,
        "task_name": task_name,
    }


def _end(
    name: str,
    time_us: int,
    duration_ms: float,
    *,
    task_name: str | None = None,
    step: int = 1,
) -> dict:
    return {
        "log_type": "event",
        "log_type_name": f"{name}_end",
        "time_us": time_us,
        "rank": 0,
        "step": step,
        "task_name": task_name,
        "value": duration_ms,
    }


def _complete_events(trace: dict) -> list[dict]:
    return [event for event in trace["traceEvents"] if event.get("ph") == "X"]


def _instant_events(trace: dict) -> list[dict]:
    return [event for event in trace["traceEvents"] if event.get("ph") == "i"]


def test_pairs_start_and_end_into_complete_event(tmp_path):
    log_dir = tmp_path / "structured_logs"
    _write_records(
        log_dir / "training.global_rank_0.trace.jsonl",
        [_start("forward", 1_000), _end("forward", 2_000, 1.0)],
    )

    trace = generate_gantt_trace(str(log_dir), str(tmp_path / "trace.json"))

    events = _complete_events(trace)
    assert len(events) == 1
    assert events[0]["name"] == "forward"
    assert events[0]["ts"] == 1_000
    assert events[0]["dur"] == 1_000
    assert events[0]["args"]["step"] == 1


def test_span_keeps_tag_added_between_start_and_end(tmp_path):
    """Checkpoint/evaluation tags are commonly added from inside a span."""
    log_dir = tmp_path / "structured_logs"
    end = _end("checkpoint_save", 2_000, 1.0, step=42)
    end["step_tags"] = ["checkpoint"]
    _write_records(
        log_dir / "training.global_rank_0.trace.jsonl",
        [_start("checkpoint_save", 1_000, step=42), end],
    )

    trace = generate_gantt_trace(str(log_dir), str(tmp_path / "trace.json"))

    event = _complete_events(trace)[0]
    assert event["args"]["step_tags"] == ["checkpoint"]


def test_span_context_merges_tags_but_preserves_start_coordinates(tmp_path):
    log_dir = tmp_path / "structured_logs"
    start = _start("evaluation", 1_000, step=10)
    start.update(
        {
            "relative_step": 2,
            "epoch": 3,
            "iteration": 4,
            "step_tags": ["evaluation", "shared"],
        }
    )
    end = _end("evaluation", 2_000, 1.0, step=999)
    end.update(
        {
            "relative_step": 999,
            "epoch": 9,
            "iteration": 9,
            "step_tags": ["shared", "checkpoint"],
        }
    )

    missing_start_context = _start("data_fetch", 3_000)
    for key in ("step", "epoch", "iteration"):
        missing_start_context.pop(key, None)
    fallback_end = _end("data_fetch", 3_500, 0.5, step=11)
    fallback_end.update({"epoch": 4, "iteration": 0})

    _write_records(
        log_dir / "training.global_rank_0.trace.jsonl",
        [start, end, missing_start_context, fallback_end],
    )

    trace = generate_gantt_trace(str(log_dir), str(tmp_path / "trace.json"))

    by_name = {event["name"]: event for event in _complete_events(trace)}
    evaluation_args = by_name["evaluation"]["args"]
    assert evaluation_args["step"] == 10
    assert evaluation_args["relative_step"] == 2
    assert evaluation_args["epoch"] == 3
    assert evaluation_args["iteration"] == 4
    assert evaluation_args["step_tags"] == [
        "evaluation",
        "shared",
        "checkpoint",
    ]

    fallback_args = by_name["data_fetch"]["args"]
    assert fallback_args["step"] == 11
    assert fallback_args["epoch"] == 4
    assert fallback_args["iteration"] == 0


def test_empty_directory_has_no_events(tmp_path):
    log_dir = tmp_path / "structured_logs"
    log_dir.mkdir()

    trace = generate_gantt_trace(str(log_dir), str(tmp_path / "trace.json"))

    assert trace["traceEvents"] == []


def test_metric_error_and_unknown_records_become_instants(tmp_path):
    log_dir = tmp_path / "structured_logs"
    _write_records(
        log_dir / "training.global_rank_0.trace.jsonl",
        [
            {
                "log_type": "instant",
                "log_type_name": "metric_value",
                "event_name": "train.loss",
                "value": 2.5,
                "time_us": 1_000,
                "rank": 0,
                "step": 1,
            },
            {
                "log_type": "event",
                "log_type_name": "forward_error",
                "time_us": 1_500,
                "rank": 0,
                "step": 1,
            },
            {
                "log_type": "event",
                "log_type_name": "ad_hoc_marker",
                "time_us": 2_000,
                "rank": 0,
                "step": 1,
            },
        ],
    )

    trace = generate_gantt_trace(str(log_dir), str(tmp_path / "trace.json"))

    names = [event["name"] for event in _instant_events(trace)]
    assert "train.loss=2.5000" in names
    assert "ERROR: forward" in names
    assert "ad_hoc_marker" in names


def test_instant_name_ending_in_start_is_not_treated_as_span(tmp_path):
    log_dir = tmp_path / "structured_logs"
    _write_records(
        log_dir / "training.global_rank_0.trace.jsonl",
        [
            {
                "log_type": "instant",
                "log_type_name": "training_start",
                "time_us": 1_000,
                "rank": 0,
            }
        ],
    )

    trace = generate_gantt_trace(str(log_dir), str(tmp_path / "trace.json"))

    assert [event["name"] for event in _instant_events(trace)] == ["training_start"]
    assert _complete_events(trace) == []


def test_nested_spans_pair_lifo_on_one_task_track(tmp_path):
    log_dir = tmp_path / "structured_logs"
    _write_records(
        log_dir / "training.global_rank_0.trace.jsonl",
        [
            _start("step", 1_000, task_name="Task-1"),
            _start("forward", 1_200, task_name="Task-1"),
            _end("forward", 1_500, 0.3, task_name="Task-1"),
            _end("step", 2_000, 1.0, task_name="Task-1"),
        ],
    )

    trace = generate_gantt_trace(str(log_dir), str(tmp_path / "trace.json"))

    by_name = {event["name"]: event for event in _complete_events(trace)}
    assert by_name["step"]["ts"] == 1_000
    assert by_name["step"]["dur"] == 1_000
    assert by_name["forward"]["ts"] == 1_200
    assert by_name["forward"]["dur"] == 300
    assert by_name["step"]["tid"] == by_name["forward"]["tid"]


def test_concurrent_tasks_receive_separate_tracks(tmp_path):
    log_dir = tmp_path / "structured_logs"
    _write_records(
        log_dir / "training.global_rank_0.trace.jsonl",
        [
            _start("loader_a", 1_000, task_name="Task-1"),
            _start("loader_b", 1_500, task_name="Task-2"),
            _end("loader_a", 3_000, 2.0, task_name="Task-1"),
            _end("loader_b", 3_500, 2.0, task_name="Task-2"),
        ],
    )

    trace = generate_gantt_trace(str(log_dir), str(tmp_path / "trace.json"))

    by_name = {event["name"]: event for event in _complete_events(trace)}
    assert by_name["loader_a"]["tid"] != by_name["loader_b"]["tid"]


def test_sequential_tasks_reuse_a_track(tmp_path):
    log_dir = tmp_path / "structured_logs"
    _write_records(
        log_dir / "training.global_rank_0.trace.jsonl",
        [
            _start("first", 1_000, task_name="Task-1"),
            _end("first", 1_500, 0.5, task_name="Task-1"),
            _start("second", 2_000, task_name="Task-2"),
            _end("second", 2_500, 0.5, task_name="Task-2"),
        ],
    )

    trace = generate_gantt_trace(str(log_dir), str(tmp_path / "trace.json"))

    assert {event["tid"] for event in _complete_events(trace)} == {0}


def test_unclosed_start_is_exported_as_incomplete_span(tmp_path):
    """A killed or hung phase must remain visible in the exported trace."""
    log_dir = tmp_path / "structured_logs"
    _write_records(
        log_dir / "training.global_rank_3.trace.jsonl",
        [
            _start("data_fetch", 1_000, step=42),
            {
                "log_type": "instant",
                "log_type_name": "heartbeat",
                "time_us": 2_500,
                "rank": 3,
                "step": 42,
            },
        ],
    )

    trace = generate_gantt_trace(str(log_dir), str(tmp_path / "trace.json"))

    incomplete = [
        event
        for event in _complete_events(trace)
        if event.get("args", {}).get("incomplete") is True
    ]
    assert len(incomplete) == 1
    assert incomplete[0]["name"] == "data_fetch"
    assert incomplete[0]["ts"] == 1_000
    assert incomplete[0]["dur"] == 1_500
    assert incomplete[0]["args"]["step"] == 42
    assert trace["metadata"]["record_count"] == 2
    assert trace["metadata"]["complete_spans"] == 0
    assert trace["metadata"]["incomplete_spans"] == 1
    assert trace["metadata"]["instant_events"] == 1


def test_rotated_segments_load_oldest_to_newest_as_one_source(tmp_path):
    log_dir = tmp_path / "structured_logs"
    stem = "training.global_rank_0.20260727-120000-ABC123.jsonl"
    _write_records(log_dir / f"{stem}.2", [{"seq_id": 0, "time_us": 1}])
    _write_records(log_dir / f"{stem}.1", [{"seq_id": 1, "time_us": 2}])
    _write_records(log_dir / stem, [{"seq_id": 2, "time_us": 3}])

    records = load_all_records(str(log_dir))

    assert [record["seq_id"] for record in records] == [0, 1, 2]
    assert len({record["_source_file"] for record in records}) == 1


def test_loader_skips_malformed_and_non_object_lines(tmp_path, caplog):
    log_dir = tmp_path / "structured_logs"
    path = log_dir / "training.global_rank_0.trace.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"seq_id": 0, "time_us": 1}\n'
        '{"seq_id":\n'
        '["not", "an", "event"]\n'
        '{"seq_id": 1, "time_us": 2}\n'
    )

    records = load_all_records(str(log_dir))

    assert [record["seq_id"] for record in records] == [0, 1]
    assert "malformed" in caplog.text.lower()


def test_cross_source_pairing_is_isolated(tmp_path):
    log_dir = tmp_path / "structured_logs"
    _write_records(
        log_dir / "training.global_rank_0.trace.jsonl",
        [
            _start("step", 1_000, task_name="Task-1"),
            _end("step", 9_000, 8.0, task_name="Task-1"),
        ],
    )
    _write_records(
        log_dir / "training.global_rank_1.trace.jsonl",
        [
            _start("step", 2_000, task_name="Task-1"),
            _end("step", 3_000, 1.0, task_name="Task-1"),
        ],
    )

    trace = generate_gantt_trace(str(log_dir), str(tmp_path / "trace.json"))

    assert {event["dur"] for event in _complete_events(trace)} == {
        1_000,
        8_000,
    }
