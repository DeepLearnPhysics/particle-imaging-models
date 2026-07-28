# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file is adapted from TorchTitan and is licensed under the BSD-style
# license found in pimm/observability/structured_logger/LICENSE.
#
# From:
# https://github.com/pytorch/torchtitan/blob/main/tests/unit_tests/observability/test_structured_logging.py#L303-L1553

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from unittest import mock

import pytest

from pimm.observability.structured_logger.jsonl_handler import (
    MAX_MESSAGE_SIZE,
    TraceJsonlFormatter,
    TraceJsonlHandler,
    register_jsonl_handler,
)
from pimm.observability.structured_logger.step_state import (
    add_step_tag,
    set_step,
)
from pimm.observability.structured_logger.structured_logging import (
    _structured_logger,
    _structured_logger_disabled,
    event_extra,
    ExtraFields,
    init_structured_logger,
    log_trace_instant,
    log_trace_scalar,
    log_trace_span,
    LogType,
    shutdown_structured_logger,
    TraceEventsOnlyFilter,
)


def _record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="tests/user_code.py",
        lineno=17,
        msg=extra.pop("message", "test"),
        args=None,
        exc_info=None,
        func="user_function",
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def _event_record(event_type: str, **kwargs) -> logging.LogRecord:
    message = kwargs.pop("message", "test")
    return _record(message=message, **event_extra(event_type, **kwargs))


class TestEventExtra:
    def test_basic_event_fields(self):
        extra = event_extra("forward")

        assert extra[str(ExtraFields.LOG_TYPE)] == str(LogType.EVENT)
        assert extra[str(ExtraFields.LOG_TYPE_NAME)] == "forward"

    def test_optional_event_fields(self):
        extra = event_extra(
            "metric_value",
            event_name="train.loss",
            step=10,
            relative_step=2,
            value=4.2,
            task_name="worker-0",
            log_type=LogType.INSTANT,
        )

        assert extra[str(ExtraFields.EVENT_NAME)] == "train.loss"
        assert extra[str(ExtraFields.STEP)] == 10
        assert extra[str(ExtraFields.RELATIVE_STEP)] == 2
        assert extra[str(ExtraFields.VALUE)] == 4.2
        assert extra[str(ExtraFields.TASK_NAME)] == "worker-0"


class TestTraceJsonlFormatter:
    def test_emits_flat_json_and_step_context(self):
        formatter = TraceJsonlFormatter(rank=3, source="training")
        set_step(101, relative_step=1, epoch=4, iteration=17)
        add_step_tag("checkpoint")

        parsed = json.loads(formatter.format(_event_record("step_start")))

        assert "normal" not in parsed
        assert "int" not in parsed
        assert parsed["rank"] == 3
        assert parsed["global_rank"] == 3
        assert parsed["source"] == "training"
        assert parsed["step"] == 101
        assert parsed["relative_step"] == 1
        assert parsed["epoch"] == 4
        assert parsed["iteration"] == 17
        assert parsed["step_tags"] == ["checkpoint"]

    def test_emits_process_and_restart_metadata(self):
        formatter = TraceJsonlFormatter(
            rank=5,
            source="training",
            world_size=16,
            attempt=2,
            job_id="12345",
            trace_id="trace-a",
        )

        parsed = json.loads(formatter.format(_event_record("training_start")))

        assert parsed["world_size"] == 16
        assert parsed["attempt"] == 2
        assert parsed["job_id"] == "12345"
        assert parsed["trace_id"] == "trace-a"
        assert parsed["pid"] == os.getpid()
        assert isinstance(parsed["local_rank"], int)
        assert parsed["host_name"]

    def test_formatter_does_not_repeat_launch_environment_inference(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "16")
        monkeypatch.setenv("PIMM_SUBMITIT_ATTEMPT", "4")
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "run-a")

        formatter = TraceJsonlFormatter(rank=0, source="training")
        parsed = json.loads(formatter.format(_event_record("training_start")))

        assert parsed["world_size"] == 1
        assert parsed["attempt"] == 1
        assert parsed["job_id"] is None
        assert parsed["trace_id"] is None

    def test_record_step_overrides_context_step(self):
        formatter = TraceJsonlFormatter(rank=0, source="training")
        set_step(10, relative_step=4)

        parsed = json.loads(
            formatter.format(_event_record("step", step=12, relative_step=6))
        )

        assert parsed["step"] == 12
        assert parsed["relative_step"] == 6

    def test_sequence_id_increments(self):
        formatter = TraceJsonlFormatter(rank=0, source="training")

        seq_ids = [
            json.loads(formatter.format(_event_record("step")))["seq_id"]
            for _ in range(3)
        ]

        assert seq_ids == [0, 1, 2]

    def test_has_microsecond_timestamp_and_caller(self):
        formatter = TraceJsonlFormatter(rank=0, source="training")

        parsed = json.loads(formatter.format(_event_record("step")))

        assert isinstance(parsed["time_us"], int)
        assert parsed["caller"].endswith("tests/user_code.py:17:user_function")

    def test_truncates_long_messages(self):
        formatter = TraceJsonlFormatter(rank=0, source="training")
        message = "x" * (MAX_MESSAGE_SIZE + 100)

        parsed = json.loads(formatter.format(_event_record("step", message=message)))

        assert "..." in parsed["message"]
        assert len(parsed["message"]) < len(message)


class TestTraceEventsOnlyFilter:
    def test_passes_structured_event(self):
        trace_filter = TraceEventsOnlyFilter()

        assert trace_filter.filter(_event_record("step")) is True

    def test_blocks_plain_log_record(self):
        trace_filter = TraceEventsOnlyFilter()

        assert trace_filter.filter(_record()) is False


class TestInitializationLifecycle:
    def test_trace_calls_are_noops_before_initialization(
        self, tmp_path, read_trace_records
    ):
        set_step(1, epoch=0, iteration=0)
        with log_trace_span("step"):
            pass
        log_trace_instant("training_start")
        log_trace_scalar({"loss": 1.0})

        assert read_trace_records(tmp_path) == []

    def test_default_handler_creates_per_rank_jsonl(self, tmp_path, read_trace_records):
        init_structured_logger(
            source="training",
            output_dir=str(tmp_path),
            rank=3,
        )
        log_trace_instant("training_start")

        files = list((tmp_path / "structured_logs").glob("*.jsonl"))
        assert len(files) == 1
        assert files[0].name.startswith("training.global_rank_3.")
        record = read_trace_records(tmp_path)[0]
        assert record["rank"] == 3
        assert record["source"] == "training"

    def test_process_metadata_is_inferred_before_distributed_init(
        self, tmp_path, read_trace_records, monkeypatch
    ):
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("PIMM_SUBMITIT_ATTEMPT", "2")
        monkeypatch.setenv("SLURM_JOB_ID", "9012")
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "run-9012")

        init_structured_logger(
            source="training",
            output_dir=str(tmp_path),
        )
        log_trace_instant("distributed_setup_start")

        record = read_trace_records(tmp_path)[0]
        assert record["rank"] == 3
        assert record["local_rank"] == 1
        assert record["world_size"] == 8
        assert record["attempt"] == 2
        assert record["job_id"] == "9012"
        assert record["trace_id"] == "run-9012"

    def test_unsupported_slurm_identity_fallbacks_are_not_used(
        self, tmp_path, read_trace_records, monkeypatch
    ):
        for name in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "PIMM_SUBMITIT_ATTEMPT",
            "TORCHELASTIC_RUN_ID",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("SLURM_PROCID", "7")
        monkeypatch.setenv("SLURM_LOCALID", "3")
        monkeypatch.setenv("SLURM_NTASKS", "8")
        monkeypatch.setenv("SLURM_RESTART_COUNT", "5")
        monkeypatch.setenv("PIMM_RDZV_ID", "legacy-rdzv")
        monkeypatch.setenv("PIMM_TRACE_ID", "legacy-trace")

        init_structured_logger(source="training", output_dir=str(tmp_path))
        log_trace_instant("training_start")

        record = read_trace_records(tmp_path)[0]
        assert record["rank"] == 0
        assert record["local_rank"] == 0
        assert record["world_size"] == 1
        assert record["attempt"] == 1
        assert record["trace_id"] is None

    def test_init_is_idempotent(self, tmp_path):
        init_structured_logger(
            source="training",
            output_dir=str(tmp_path),
            rank=0,
        )
        handlers_before = list(_structured_logger.handlers)

        init_structured_logger(
            source="other",
            output_dir=str(tmp_path),
            rank=0,
        )

        assert _structured_logger.handlers == handlers_before

    def test_enable_false_is_a_noop(self, tmp_path, read_trace_records):
        init_structured_logger(
            source="training",
            output_dir=str(tmp_path),
            rank=0,
            enable=False,
        )

        with log_trace_span("step"):
            pass
        log_trace_instant("training_start")
        log_trace_scalar({"loss": 1.0})

        assert _structured_logger_disabled()
        assert read_trace_records(tmp_path) == []

    def test_disabled_logger_can_be_reenabled(self, tmp_path, read_trace_records):
        init_structured_logger(
            source="training",
            output_dir=str(tmp_path),
            rank=0,
            enable=False,
        )

        init_structured_logger(
            source="training",
            output_dir=str(tmp_path),
            rank=0,
            enable=True,
        )
        log_trace_instant("training_start")

        assert not _structured_logger_disabled()
        assert [record["log_type_name"] for record in read_trace_records(tmp_path)] == [
            "training_start"
        ]

    def test_shutdown_flushes_and_allows_reinitialization(
        self, tmp_path, read_trace_records
    ):
        init_structured_logger(
            source="training",
            output_dir=str(tmp_path),
            rank=0,
        )
        log_trace_instant("first_process")
        shutdown_structured_logger()

        assert _structured_logger_disabled()
        assert not any(
            isinstance(handler, TraceJsonlHandler)
            for handler in _structured_logger.handlers
        )

        init_structured_logger(
            source="resumed_training",
            output_dir=str(tmp_path),
            rank=0,
            attempt=2,
        )
        log_trace_instant("second_process")
        shutdown_structured_logger()

        assert {record["log_type_name"] for record in read_trace_records(tmp_path)} == {
            "first_process",
            "second_process",
        }

    def test_custom_factory_replaces_default_jsonl_handler(self, tmp_path):
        called = []

        def fake_factory(*, structured_logger, rank, source, output_dir, **kwargs):
            called.append((structured_logger, rank, source, output_dir, kwargs))

        with mock.patch.dict(
            os.environ,
            {"PIMM_STRUCT_LOGGER_HANDLERS": ("tests.fake_structured_logger_factory")},
        ):
            with mock.patch("importlib.import_module") as import_module:
                module = mock.MagicMock()
                module.fake_structured_logger_factory = fake_factory
                import_module.return_value = module
                init_structured_logger(
                    source="training",
                    output_dir=str(tmp_path),
                    rank=4,
                    world_size=8,
                    attempt=2,
                    job_id="12345",
                    trace_id="run-a",
                )

        assert len(called) == 1
        assert called[0][1:4] == (4, "training", str(tmp_path))
        assert called[0][4]["world_size"] == 8
        assert called[0][4]["attempt"] == 2
        assert called[0][4]["job_id"] == "12345"
        assert called[0][4]["trace_id"] == "run-a"
        assert not any(
            isinstance(handler, TraceJsonlHandler)
            for handler in _structured_logger.handlers
        )


class TestTraceEmission:
    def test_scalar_emits_one_instant_per_numeric_value(
        self, tmp_path, read_trace_records
    ):
        init_structured_logger(source="training", output_dir=str(tmp_path), rank=0)
        set_step(5)

        log_trace_scalar({"train.loss": 2.5, "batch.points": 45})

        records = read_trace_records(tmp_path)
        assert [record["event_name"] for record in records] == [
            "train.loss",
            "batch.points",
        ]
        assert [record["value"] for record in records] == [2.5, 45.0]
        assert all(record["log_type"] == "instant" for record in records)
        assert all(record["step"] == 5 for record in records)

    def test_scalar_skips_non_numeric_and_boolean_values(
        self, tmp_path, read_trace_records, caplog
    ):
        init_structured_logger(source="training", output_dir=str(tmp_path), rank=0)

        with caplog.at_level(logging.WARNING):
            log_trace_scalar({"ok": 1, "bad": "one", "flag": True})

        assert [r["event_name"] for r in read_trace_records(tmp_path)] == ["ok"]
        assert "bad" in caplog.text
        assert "flag" in caplog.text

    def test_instant_emits_point_event(self, tmp_path, read_trace_records):
        init_structured_logger(source="training", output_dir=str(tmp_path), rank=0)

        log_trace_instant("training_start")

        record = read_trace_records(tmp_path)[0]
        assert record["log_type_name"] == "training_start"
        assert record["log_type"] == "instant"
        assert record["event_name"] is None

    def test_span_emits_start_end_and_duration(self, tmp_path, read_trace_records):
        init_structured_logger(source="training", output_dir=str(tmp_path), rank=0)
        set_step(5)

        with log_trace_span("forward"):
            pass

        records = read_trace_records(tmp_path)
        assert [record["log_type_name"] for record in records] == [
            "forward_start",
            "forward_end",
        ]
        assert records[1]["value"] >= 0
        assert all(record["step"] == 5 for record in records)

    def test_exception_emits_error_and_still_closes_span(
        self, tmp_path, read_trace_records
    ):
        init_structured_logger(source="training", output_dir=str(tmp_path), rank=0)

        with pytest.raises(ValueError, match="bad batch"):
            with log_trace_span("forward"):
                raise ValueError("bad batch")

        assert [record["log_type_name"] for record in read_trace_records(tmp_path)] == [
            "forward_start",
            "forward_error",
            "forward_end",
        ]

    def test_sync_decorator_brackets_function(self, tmp_path, read_trace_records):
        init_structured_logger(source="training", output_dir=str(tmp_path), rank=0)

        @log_trace_span("optimizer")
        def optimizer_step():
            log_trace_instant("inside")

        optimizer_step()

        assert [record["log_type_name"] for record in read_trace_records(tmp_path)] == [
            "optimizer_start",
            "inside",
            "optimizer_end",
        ]

    def test_async_decorator_brackets_coroutine_body(
        self, tmp_path, read_trace_records
    ):
        init_structured_logger(source="training", output_dir=str(tmp_path), rank=0)

        @log_trace_span("rollout")
        async def rollout():
            log_trace_instant("inside")

        asyncio.run(rollout())

        assert [record["log_type_name"] for record in read_trace_records(tmp_path)] == [
            "rollout_start",
            "inside",
            "rollout_end",
        ]


class TestJsonlHandler:
    def test_factory_registers_handler_with_expected_path(self, tmp_path):
        logger = logging.getLogger("pimm.test.structured")
        logger.handlers = []
        try:
            register_jsonl_handler(
                structured_logger=logger,
                rank=2,
                source="evaluation",
                output_dir=str(tmp_path),
            )

            assert len(logger.handlers) == 1
            handler = logger.handlers[0]
            assert isinstance(handler, TraceJsonlHandler)
            assert "structured_logs" in handler.baseFilename
            assert "evaluation.global_rank_2" in handler.baseFilename
            assert handler.baseFilename.endswith(".jsonl")
        finally:
            for handler in logger.handlers:
                handler.close()
            logger.handlers = []

    def test_rotation_bounds_files_per_rank(self, tmp_path):
        init_structured_logger(
            source="training",
            output_dir=str(tmp_path),
            rank=0,
            max_file_size_mb=0.001,
            backup_count=2,
        )

        for index in range(100):
            log_trace_instant(f"event_{index:03d}_{'x' * 100}")
        shutdown_structured_logger()

        files = list((tmp_path / "structured_logs").glob("*.jsonl*"))
        assert 2 <= len(files) <= 3
        assert any(path.name.endswith(".jsonl.1") for path in files)
