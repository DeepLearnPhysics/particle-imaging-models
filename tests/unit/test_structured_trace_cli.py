from __future__ import annotations

import json
from pathlib import Path

import pytest

from pimm.cli import trace as trace_cli
from pimm.cli.main import main as pimm_main


def _record(
    event: str,
    time_us: int,
    *,
    rank: int = 0,
    attempt: int = 1,
    trace_id: str = "trace-0",
    source: str = "training",
    step: int | None = 7,
    relative_step: int | None = 2,
    value: float | None = None,
    log_type: str = "event",
) -> dict:
    record = {
        "source": source,
        "rank": rank,
        "attempt": attempt,
        "trace_id": trace_id,
        "step": step,
        "relative_step": relative_step,
        "time_us": time_us,
        "log_type": log_type,
        "log_type_name": event,
        "tid": 42,
    }
    if value is not None:
        record["value"] = value
    return record


def _write_jsonl(path: Path, records: list[dict], trailing: str = "") -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records) + trailing,
        encoding="utf-8",
    )


def test_reader_merges_rotated_files_and_skips_truncated_final_line(tmp_path):
    log_dir = tmp_path / "structured_logs"
    log_dir.mkdir()
    base = log_dir / "training.global_rank_0.run.jsonl"
    rotated = log_dir / "training.global_rank_0.run.jsonl.1"
    _write_jsonl(
        rotated,
        [_record("step_start", 100)],
    )
    _write_jsonl(
        base,
        [_record("step_end", 200, value=0.1)],
        trailing='{"time_us": 201, "log_type_name": ',
    )

    loaded = trace_cli.load_trace_records(log_dir)

    assert loaded.files == (base, rotated)
    assert [record["log_type_name"] for record in loaded.records] == [
        "step_start",
        "step_end",
    ]
    assert {record["_source_file"] for record in loaded.records} == {
        "training.global_rank_0.run"
    }
    assert loaded.malformed_lines == 1


def test_summary_reports_process_progress_open_spans_and_phase_percentiles():
    records = [
        _record("forward_start", 100, step=7),
        _record("forward_end", 110, step=7, value=10.0),
        _record("forward_start", 200, step=8),
        _record("forward_end", 220, step=8, value=20.0),
        _record(
            "process_shutdown",
            225,
            step=None,
            relative_step=None,
            log_type="instant",
        ),
        _record(
            "forward_start",
            300,
            rank=1,
            trace_id="trace-1",
            step=9,
            relative_step=4,
        ),
        _record(
            "waiting_for_collective",
            310,
            rank=1,
            trace_id="trace-1",
            step=9,
            relative_step=4,
            log_type="instant",
        ),
    ]

    summary = trace_cli.summarize_records(
        records,
        file_count=2,
        malformed_lines=1,
    )

    assert summary.record_count == 7
    assert len(summary.statuses) == 2
    rank_zero, rank_one = summary.statuses
    assert rank_zero.identity.rank == "0"
    assert rank_zero.last_step == 8
    assert rank_zero.last_event == "process_shutdown"
    assert rank_zero.open_spans == ()
    assert rank_one.identity.rank == "1"
    assert rank_one.last_event == "waiting_for_collective"
    assert [span.phase for span in rank_one.open_spans] == ["forward"]
    assert rank_one.open_spans[0].step == 9

    assert len(summary.phases) == 1
    forward = summary.phases[0]
    assert forward.phase == "forward"
    assert forward.count == 2
    assert forward.p50_ms == pytest.approx(15.0)
    assert forward.p95_ms == pytest.approx(19.5)
    assert forward.max_ms == pytest.approx(20.0)
    assert forward.slowest_identity.rank == "0"
    assert forward.slowest_step == 8

    rendered = trace_cli.format_summary(summary)
    assert "rank=1 attempt=1" in rendered
    assert "last_step=9" in rendered
    assert "open: forward(step=9, relative_step=4)" in rendered
    assert "forward: count=2 p50=15.000 p95=19.500 max=20.000" in rendered


def test_instant_name_ending_in_start_is_not_reported_as_open_span():
    summary = trace_cli.summarize_records(
        [_record("training_start", 100, log_type="instant")]
    )

    assert summary.statuses[0].last_event == "training_start"
    assert summary.statuses[0].open_spans == ()
    assert summary.phases == ()


def test_trace_summarize_is_dispatched_from_top_level_cli(tmp_path, capsys):
    run_dir = tmp_path / "run"
    log_dir = run_dir / "structured_logs"
    log_dir.mkdir(parents=True)
    _write_jsonl(
        log_dir / "training.global_rank_0.trace.jsonl",
        [_record("step_end", 100, value=12.5)],
        trailing='{"truncated":',
    )

    result = pimm_main(["trace", "summarize", str(run_dir)])

    assert result == 0
    captured = capsys.readouterr()
    assert "Structured trace: 1 records from 1 files" in captured.out
    assert "step: count=1" in captured.out
    assert "skipped 1 malformed or truncated JSONL line" in captured.err


@pytest.mark.parametrize("use_log_dir", [False, True])
def test_trace_export_resolves_default_output(
    tmp_path,
    monkeypatch,
    capsys,
    use_log_dir,
):
    run_dir = tmp_path / "run"
    log_dir = run_dir / "structured_logs"
    log_dir.mkdir(parents=True)
    _write_jsonl(
        log_dir / "training.global_rank_0.trace.jsonl",
        [_record("training_start", 100, log_type="instant")],
    )
    calls = []

    def fake_generate(input_dir: Path, output_path: Path):
        calls.append((input_dir, output_path))
        return {"traceEvents": []}

    monkeypatch.setattr(trace_cli, "_generate_gantt_trace", fake_generate)

    trace_input = log_dir if use_log_dir else run_dir
    result = trace_cli.main(["export", str(trace_input)])

    expected_output = run_dir / "analysis" / "structured_trace.json"
    assert result == 0
    assert calls == [(log_dir, expected_output)]
    assert f"Exported structured trace to: {expected_output}" in capsys.readouterr().out


def test_trace_export_accepts_custom_output(tmp_path, monkeypatch):
    log_dir = tmp_path / "run" / "structured_logs"
    log_dir.mkdir(parents=True)
    _write_jsonl(
        log_dir / "training.global_rank_0.trace.jsonl",
        [_record("training_start", 100, log_type="instant")],
    )
    output = tmp_path / "trace.json"
    calls = []
    monkeypatch.setattr(
        trace_cli,
        "_generate_gantt_trace",
        lambda input_dir, output_path: calls.append((input_dir, output_path)),
    )

    assert trace_cli.main(["export", str(log_dir), "-o", str(output)]) == 0
    assert calls == [(log_dir, output)]


def test_trace_command_reports_missing_logs(tmp_path, capsys):
    assert trace_cli.main(["summarize", str(tmp_path)]) == 2
    assert "No structured_logs directory or *.jsonl* files" in capsys.readouterr().err
