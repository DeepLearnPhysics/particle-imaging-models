"""The entrypoint shares seeds and writes execution metadata once per job."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from pimm import train
from pimm.engines import _execution


@pytest.fixture
def entrypoint(monkeypatch, tmp_path):
    events = []
    cfg = SimpleNamespace(
        seed=17,
        managed=True,
        resume=False,
        save_path=str(tmp_path / "original"),
        structured_logging=SimpleNamespace(
            enabled=False, max_file_size_mb=1, backup_count=1
        ),
    )
    args = SimpleNamespace(config_file="config.py", options={"batch_size": 4})
    monkeypatch.setattr(
        train, "default_argument_parser", lambda: SimpleNamespace(parse_args=lambda: args)
    )

    def parse_config(config_file, options, *, save_artifacts):
        assert (config_file, options) == (args.config_file, args.options)
        assert not save_artifacts
        return cfg

    def configure_training(config):
        config.save_path = str(tmp_path / "exex-output")
        return config.managed

    def gather(seed):
        events.append(("gather", seed))
        return [23, seed]

    monkeypatch.setattr(train, "default_config_parser", parse_config)
    monkeypatch.setattr(_execution, "configure_training", configure_training)
    monkeypatch.setattr(train.comm, "setup_distributed", lambda: events.append("setup"))
    monkeypatch.setattr(train.comm, "all_gather", gather)
    monkeypatch.setattr(train.comm, "is_main_process", lambda: True)
    monkeypatch.setattr(train.comm, "cleanup_distributed", lambda: events.append("cleanup"))
    monkeypatch.setattr(
        train,
        "_save_config_artifacts",
        lambda config, source, options: events.append(
            ("save", config.save_path, source, options)
        ),
    )
    monkeypatch.setattr(
        train, "main_worker", lambda config: events.append(("train", config.seed))
    )
    monkeypatch.setattr(train.sl, "log_trace_span", lambda name: nullcontext())
    monkeypatch.setattr(train.sl, "log_trace_instant", events.append)
    monkeypatch.setattr(
        train.sl,
        "init_structured_logger",
        lambda **kwargs: events.append(("logger", kwargs["output_dir"])),
    )
    monkeypatch.setattr(
        train.sl, "shutdown_structured_logger", lambda: events.append("shutdown")
    )
    return cfg, events


@pytest.mark.parametrize("main_process", [False, True])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("managed", [False, True])
def test_seed_is_shared_and_rank_zero_saves_each_managed_attempt(
    entrypoint, monkeypatch, main_process, resume, managed
):
    cfg, events = entrypoint
    cfg.resume = resume
    cfg.managed = managed
    monkeypatch.setattr(train.comm, "is_main_process", lambda: main_process)

    train.main()

    assert cfg.seed == 23
    assert events.index("setup") < events.index(("gather", 17))
    assert events.index(("gather", 17)) < events.index(("train", 23))
    assert ("logger", cfg.save_path) in events
    saves = [event for event in events if isinstance(event, tuple) and event[0] == "save"]
    assert saves == (
        [("save", cfg.save_path, "config.py", {"batch_size": 4})]
        if main_process and (managed or not resume)
        else []
    )
    assert events[-3:] == ["cleanup", "process_end", "shutdown"]


def test_training_failure_does_not_enter_synchronized_cleanup(entrypoint, monkeypatch):
    _, events = entrypoint

    def fail(config):
        raise RuntimeError("training failed")

    monkeypatch.setattr(train, "main_worker", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        train.main()

    assert "cleanup" not in events
    assert events[-2:] == ["process_end", "shutdown"]
