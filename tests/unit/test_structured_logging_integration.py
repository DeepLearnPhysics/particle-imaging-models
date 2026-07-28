from __future__ import annotations

from types import SimpleNamespace

import pytest

from pimm import train as train_entrypoint
from pimm.engines import train as train_module
from pimm.engines.train import Trainer


class _Config(dict):
    """Small dict/attribute hybrid matching the config surface used here."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _trainer_config(**structured_logging):
    config = _Config(
        epoch=1,
        hooks=[],
        pretty_text="test config",
        resume=False,
        save_path="/tmp/pimm-structured-logger-test",
    )
    if structured_logging:
        config["structured_logging"] = structured_logging
    return config


def _patch_trainer_build(monkeypatch):
    monkeypatch.setattr(
        train_module,
        "create_parallel_context",
        lambda cfg: SimpleNamespace(),
    )
    monkeypatch.setattr(
        train_module,
        "get_root_logger",
        lambda **kwargs: SimpleNamespace(info=lambda *args, **kw: None),
    )
    monkeypatch.setattr(Trainer, "build_model", lambda self: object())
    monkeypatch.setattr(Trainer, "build_train_loader", lambda self: [])
    monkeypatch.setattr(Trainer, "build_val_loader", lambda self: None)
    monkeypatch.setattr(Trainer, "build_test_loader", lambda self: None)
    monkeypatch.setattr(Trainer, "build_optimizer", lambda self: object())
    monkeypatch.setattr(Trainer, "build_scheduler", lambda self: object())
    monkeypatch.setattr(Trainer, "build_scaler", lambda self: None)
    monkeypatch.setattr(Trainer, "register_hooks", lambda self, hooks: None)
    monkeypatch.setattr(Trainer, "_call_hooks", lambda self, name, *args: None)
    monkeypatch.setattr(Trainer, "build_writer", lambda self: None)


def _patch_entrypoint_config(monkeypatch, cfg):
    parser = SimpleNamespace(
        parse_args=lambda: SimpleNamespace(
            config_file="test.py",
            options=None,
        )
    )
    monkeypatch.setattr(train_entrypoint, "default_argument_parser", lambda: parser)
    monkeypatch.setattr(
        train_entrypoint,
        "default_config_parser",
        lambda config_file, options: cfg,
    )


def test_disabled_config_avoids_per_batch_trace_work(monkeypatch):
    _patch_trainer_build(monkeypatch)
    monkeypatch.setattr(
        train_module,
        "_structured_logger_disabled",
        lambda: True,
    )

    trainer = Trainer(_trainer_config())

    assert trainer._trace_hooks is False
    assert trainer._trace_batch_stats_every == 0


def test_enabled_config_applies_trace_detail_controls(monkeypatch):
    _patch_trainer_build(monkeypatch)
    monkeypatch.setattr(
        train_module,
        "_structured_logger_disabled",
        lambda: False,
    )

    trainer = Trainer(_trainer_config(trace_hooks=True, batch_stats_every=7))

    assert trainer._trace_hooks is True
    assert trainer._trace_batch_stats_every == 7


def test_entrypoint_passes_structured_logging_config(monkeypatch, tmp_path):
    calls = []
    cfg = _Config(
        save_path=str(tmp_path),
        structured_logging=_Config(
            enabled=True,
            max_file_size_mb=64,
            backup_count=2,
        ),
    )
    _patch_entrypoint_config(monkeypatch, cfg)
    monkeypatch.setattr(
        train_entrypoint.sl,
        "init_structured_logger",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        train_entrypoint.sl,
        "log_trace_instant",
        lambda event_type: None,
    )
    monkeypatch.setattr(train_entrypoint.sl, "shutdown_structured_logger", lambda: None)
    monkeypatch.setattr(train_entrypoint.comm, "setup_distributed", lambda: None)
    monkeypatch.setattr(train_entrypoint.comm, "cleanup_distributed", lambda: None)
    monkeypatch.setattr(train_entrypoint, "main_worker", lambda cfg: None)

    train_entrypoint.main()

    assert calls == [
        {
            "source": "training",
            "output_dir": str(tmp_path),
            "enable": True,
            "max_file_size_mb": 64,
            "backup_count": 2,
        }
    ]


def test_entrypoint_trace_init_failure_propagates(monkeypatch, tmp_path):
    def fail_init(**kwargs):
        raise OSError("trace filesystem unavailable")

    cfg = _Config(
        save_path=str(tmp_path),
        structured_logging=_Config(
            enabled=True,
            max_file_size_mb=128,
            backup_count=3,
        ),
    )
    _patch_entrypoint_config(monkeypatch, cfg)
    monkeypatch.setattr(train_entrypoint.sl, "init_structured_logger", fail_init)
    setup_calls = []
    monkeypatch.setattr(
        train_entrypoint.comm,
        "setup_distributed",
        lambda: setup_calls.append(True),
    )

    with pytest.raises(OSError, match="trace filesystem unavailable"):
        train_entrypoint.main()

    assert setup_calls == []


def test_entrypoint_requires_complete_structured_logging_config(monkeypatch, tmp_path):
    cfg = _Config(
        save_path=str(tmp_path),
        structured_logging=_Config(enabled=True),
    )
    _patch_entrypoint_config(monkeypatch, cfg)

    with pytest.raises(AttributeError, match="max_file_size_mb"):
        train_entrypoint.main()


def test_entrypoint_shutdown_runs_when_distributed_cleanup_fails(monkeypatch):
    events = []
    cfg = _Config(
        save_path="/tmp/test",
        structured_logging=_Config(
            enabled=False,
            max_file_size_mb=128,
            backup_count=3,
        ),
    )
    _patch_entrypoint_config(monkeypatch, cfg)
    monkeypatch.setattr(
        train_entrypoint.sl,
        "init_structured_logger",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(train_entrypoint.comm, "setup_distributed", lambda: None)
    monkeypatch.setattr(
        train_entrypoint, "main_worker", lambda cfg: events.append("train")
    )

    def fail_cleanup():
        events.append("cleanup")
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(train_entrypoint.comm, "cleanup_distributed", fail_cleanup)
    monkeypatch.setattr(
        train_entrypoint.sl,
        "log_trace_instant",
        lambda name: events.append(name),
    )
    monkeypatch.setattr(
        train_entrypoint.sl,
        "shutdown_structured_logger",
        lambda: events.append("shutdown"),
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        train_entrypoint.main()

    assert events == [
        "process_start",
        "train",
        "cleanup",
        "process_end",
        "shutdown",
    ]


def test_resumed_loop_fetches_only_remaining_batches(monkeypatch):
    lifecycle = []
    step_contexts = []
    batches = [{"batch": "second"}, {"batch": "third"}]

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = SimpleNamespace(detect_anomaly=False)
    trainer.logger = SimpleNamespace(
        info=lambda message: lifecycle.append(("log", message))
    )
    trainer.model = SimpleNamespace(train=lambda: lifecycle.append(("model", "train")))
    trainer.train_loader = batches
    trainer.start_epoch = 1
    trainer.max_epoch = 2
    trainer.start_iter = 1
    trainer.global_step = 4
    trainer._train_is_iterable = False
    trainer.comm_info = {}
    trainer.writer = None

    trainer._iters_per_epoch = lambda: 3
    trainer._align_writer_step = lambda step: lifecycle.append(("align", step))
    trainer.before_train = lambda: lifecycle.append(("hook", "before_train"))
    trainer.before_epoch = lambda: lifecycle.append(("hook", "before_epoch"))
    trainer.before_step = lambda: lifecycle.append(
        ("before_step", trainer.comm_info["input_dict"])
    )
    trainer.run_step = lambda: lifecycle.append(
        ("run_step", trainer.comm_info["input_dict"])
    )
    trainer._record_step_state = lambda: lifecycle.append(
        ("record_step", trainer.comm_info["iter"])
    )
    trainer.after_step = lambda: lifecycle.append(
        ("after_step", trainer.comm_info["input_dict"])
    )
    trainer.after_epoch = lambda: lifecycle.append(("hook", "after_epoch"))
    trainer.after_train = lambda: lifecycle.append(("hook", "after_train"))
    trainer._log_trace_batch_stats = lambda step: lifecycle.append(
        ("batch_stats", step, trainer.comm_info["input_dict"])
    )

    monkeypatch.setattr(
        train_module,
        "set_dataloader_epoch",
        lambda loader, epoch, reset_position: lifecycle.append(
            ("set_epoch", epoch, reset_position)
        ),
    )
    monkeypatch.setattr(
        train_module.sl,
        "set_step",
        lambda step, **context: step_contexts.append((step, context)),
    )

    Trainer.train(trainer)

    assert step_contexts == [
        (
            5,
            {"relative_step": 1, "epoch": 1, "iteration": 1},
        ),
        (
            6,
            {"relative_step": 2, "epoch": 1, "iteration": 2},
        ),
    ]
    assert [event[1] for event in lifecycle if event[0] == "run_step"] == batches
    assert ("set_epoch", 1, False) in lifecycle
    assert trainer.start_iter == 0


def test_before_epoch_uses_upcoming_step_context(monkeypatch):
    current_context = {}
    before_epoch_contexts = []

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = SimpleNamespace(detect_anomaly=False)
    trainer.logger = SimpleNamespace(info=lambda message: None)
    trainer.model = SimpleNamespace(train=lambda: None)
    trainer.train_loader = [{"batch": "only"}]
    trainer.start_epoch = 0
    trainer.max_epoch = 2
    trainer.start_iter = 0
    trainer.global_step = 0
    trainer._train_is_iterable = False
    trainer.comm_info = {}
    trainer.writer = None

    trainer._iters_per_epoch = lambda: 1
    trainer._align_writer_step = lambda step: None
    trainer.before_train = lambda: None
    trainer.before_epoch = lambda: before_epoch_contexts.append(dict(current_context))
    trainer.before_step = lambda: None
    trainer.run_step = lambda: None
    trainer._record_step_state = lambda: None
    trainer.after_step = lambda: None
    trainer.after_epoch = lambda: None
    trainer.after_train = lambda: None
    trainer._log_trace_batch_stats = lambda step: None

    monkeypatch.setattr(
        train_module,
        "set_dataloader_epoch",
        lambda loader, epoch, reset_position: None,
    )

    def capture_step(step, **context):
        current_context.clear()
        current_context.update(step=step, **context)

    monkeypatch.setattr(train_module.sl, "set_step", capture_step)

    Trainer.train(trainer)

    assert before_epoch_contexts == [
        {
            "step": 1,
            "relative_step": 1,
            "epoch": 0,
            "iteration": 0,
        },
        {
            "step": 2,
            "relative_step": 2,
            "epoch": 1,
            "iteration": 0,
        },
    ]


def test_completed_resume_marker_uses_restored_step_context(monkeypatch):
    step_contexts = []

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = SimpleNamespace(detect_anomaly=False)
    trainer.logger = SimpleNamespace(info=lambda message: None)
    trainer.start_epoch = 2
    trainer.max_epoch = 2
    trainer.start_iter = 0
    trainer.global_step = 6
    trainer.writer = None

    trainer.before_train = lambda: None
    trainer._iters_per_epoch = lambda: 3
    trainer._align_writer_step = lambda step: None
    trainer._finish_completed_resume = lambda: None
    monkeypatch.setattr(
        train_module.sl,
        "set_step",
        lambda step, **context: step_contexts.append((step, context)),
    )

    Trainer.train(trainer)

    assert step_contexts == [
        (
            6,
            {
                "relative_step": 0,
                "epoch": 1,
                "iteration": 2,
            },
        )
    ]
