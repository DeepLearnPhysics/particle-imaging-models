"""pimm buffering/checkpoint adapter exercised through the real exex helper."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pimm.engines.defaults import default_config_parser, resolve_wandb_history
from pimm.engines.hooks import checkpoint as checkpoint_hooks
from pimm.utils import checkpoints, events
from pimm.utils.checkpoints import build_logger_state, configure_logger_from_checkpoint


class FakeRun:
    def __init__(self, backend, run_id, history, kwargs):
        self.backend, self.id, self.history = backend, run_id, history
        self.starting_step = max((row["_step"] for row in history), default=-1) + 1
        self.step = 0  # Real W&B 0.28.0 before the first resumed log.
        self.entity = kwargs.get("entity") or "team"
        self.project = kwargs.get("project") or "pimm"
        self.group = kwargs.get("group")
        self.disabled = kwargs.get("mode") == "disabled"
        self.url = None
        self.config = Mock()
        self.metric_definitions = []
        self.finished = False

    def define_metric(self, name, **kwargs):
        self.metric_definitions.append((name, kwargs))

    def log(self, data, **kwargs):
        self.backend.log_kwargs.append(kwargs)
        step = kwargs.get("step", max(self.step, self.starting_step))
        self.history.append({**data, "_step": step})
        self.step = step + 1

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.run = None
        self.histories, self.init_calls, self.log_kwargs = {}, [], []
        self.util = SimpleNamespace(generate_id=lambda: f"run-{len(self.init_calls)}")

    def setup(self):
        return SimpleNamespace(settings=SimpleNamespace(mode="online"))

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        run_id = kwargs["id"]
        if kwargs.get("fork_from"):
            source, query = kwargs["fork_from"].split("?_step=")
            self.histories[run_id] = [
                dict(row)
                for row in self.histories[source]
                if row["_step"] <= int(query)
            ]
        history = self.histories.setdefault(run_id, [])
        self.run = FakeRun(self, run_id, history, kwargs)
        return self.run


@pytest.fixture(autouse=True)
def clean_tracking_env(monkeypatch):
    for key in os.environ:
        if key.startswith("WANDB_") or key in {"PIMM_WANDB_HISTORY", "EXEX_LINK_DIR"}:
            monkeypatch.delenv(key)


@pytest.fixture
def backend(monkeypatch):
    sdk = FakeWandb()
    monkeypatch.setattr(events, "wandb", sdk)
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    return sdk


def trainer(writer, **cfg):
    return SimpleNamespace(cfg={"use_wandb": True, **cfg}, writer=writer)


@pytest.mark.parametrize("history", ["new", "append", "fork"])
def test_checkpoint_history_keeps_training_steps_separate(backend, history):
    parent = events.WandbSummaryWriter(project="pimm", id="parent")
    for step in (1, 2):
        parent.add_scalar("train/loss", step, step)
        parent.add_scalar("params/lr", 0.1, step)
        parent.flush_step()
    saved = parent.checkpoint_state()
    parent.add_scalar("train/loss", 3, 3)
    parent.close()
    original = list(backend.histories["parent"])

    child = events.WandbSummaryWriter(project="pimm", history=history)
    configure_logger_from_checkpoint(trainer(child), {"logger": {"wandb": saved}})
    immediate = child.checkpoint_state()
    expected = {"new": 0, "append": 4, "fork": 3}[history]
    assert immediate["next_step"] == expected
    assert (child.run.id == "parent") == (history == "append")
    child.add_scalar("train/loss", -3, 3)
    child.add_scalar("params/lr", 0.01, 3)
    child.flush_step()
    assert child.run.history[-1]["_step"] == max(expected, 3)
    assert child.run.history[-1]["train/global_step"] == 3
    assert child.run.history[-1]["params/lr"] == 0.01
    assert backend.histories["parent"][: len(original)] == original
    if history != "append":
        assert backend.histories["parent"] == original
    assert "resume_from" not in backend.init_calls[-1]
    if history == "fork":
        assert backend.init_calls[-1]["fork_from"] == "parent?_step=2"


def test_checkpoint_flushes_pending_row_and_records_actual_identity(backend):
    writer = events.WandbSummaryWriter(id="actual", entity="team", project="science")
    writer.add_scalar("train/loss", 1.0, 7)
    state = build_logger_state(
        trainer(writer, wandb_run_id="stale"),
        checkpoint_global_step=7,
        initialize_wandb=True,
    )["wandb"]
    assert state["run_id"] == "actual"
    assert state["entity"] == "team" and state["project"] == "science"
    assert state["next_step"] == 8 and state["checkpoint_global_step"] == 7
    assert "resume_step" not in state
    before = list(writer.run.history)
    writer.checkpoint_state()
    assert writer.run.history == before


@pytest.mark.parametrize("flat", [False, True])
def test_legacy_checkpoint_translation_requires_destination(backend, flat):
    saved = (
        {"wandb_run_id": "parent", "wandb_resume_step": 8}
        if flat
        else {"logger": {"wandb": {"run_id": "parent", "resume_step": 8}}}
    )
    writer = events.WandbSummaryWriter(history="fork")
    with pytest.raises(ValueError, match="Legacy W&B state needs"):
        configure_logger_from_checkpoint(trainer(writer), saved)
    configure_logger_from_checkpoint(
        trainer(writer, wandb_entity="team", wandb_project="science"), saved
    )
    assert writer._wandb_state["next_step"] == 8
    assert writer._wandb_state["entity"] == "team"


@pytest.mark.parametrize("history", ["append", "fork"])
def test_missing_state_does_not_fall_back(backend, history):
    writer = events.WandbSummaryWriter(history=history)
    configure_logger_from_checkpoint(trainer(writer), {})
    with pytest.raises(ValueError, match="require saved"):
        writer.initialize()
    assert not backend.init_calls


def test_fork_permission_failure_is_not_reinterpreted(backend, monkeypatch):
    init = Mock(side_effect=PermissionError("fork access denied"))
    monkeypatch.setattr(backend, "init", init)
    writer = events.WandbSummaryWriter(history="fork")
    writer.configure_from_checkpoint(
        {"run_id": "parent", "entity": "team", "project": "science", "next_step": 8}
    )
    with pytest.raises(PermissionError, match="fork access denied"):
        writer.initialize()
    assert init.call_count == 1
    assert init.call_args.kwargs["fork_from"] == "parent?_step=7"
    assert writer.run is None


@pytest.mark.parametrize("history", ["append", "fork"])
def test_continued_history_requires_online_mode(backend, history):
    writer = events.WandbSummaryWriter(history=history, mode="offline")
    writer.configure_from_checkpoint(
        {"run_id": "parent", "entity": "team", "project": "science", "next_step": 8}
    )
    with pytest.raises(ValueError, match="online mode"):
        writer.initialize()
    assert not backend.init_calls


def test_disabled_and_non_reporting_rank(backend):
    writer = events.WandbSummaryWriter(mode="disabled")
    assert writer.checkpoint_state() is None
    assert build_logger_state(trainer(writer), initialize_wandb=True)["wandb"] is None
    configure_logger_from_checkpoint(trainer(None), {})


def test_writer_owns_its_run_and_restoration_is_lazy(backend):
    first = events.WandbSummaryWriter().initialize()
    writer = events.WandbSummaryWriter()
    own = writer.initialize()
    assert own is writer.initialize() and own is not first
    with pytest.raises(RuntimeError, match="before initializing"):
        writer.configure_from_checkpoint({})
    writer.close()
    assert own.finished and not first.finished


@pytest.mark.parametrize(
    "cfg,env,expected",
    [
        ({}, None, "new"),
        ({}, "append", "append"),
        ({"wandb_history": "fork"}, "append", "fork"),
        ({"resume": True, "wandb_fork_run_on_resume": True}, None, "fork"),
        ({"resume": False, "wandb_fork_run_on_resume": True}, None, "new"),
        ({"resume": True, "wandb_fresh_run_on_resume": True}, None, "new"),
        ({"resume": True, "wandb_fork_run_on_resume": True}, "append", "append"),
        ({"wandb_history_resolved": "new"}, "append", "append"),
    ],
)
def test_history_precedence(monkeypatch, cfg, env, expected):
    if env is not None:
        monkeypatch.setenv("PIMM_WANDB_HISTORY", env)
    assert resolve_wandb_history(cfg) == expected


def test_trainer_passes_resolved_history_and_reports_only_on_main_rank(
    backend, monkeypatch
):
    from pimm.engines import train as train_engine
    from pimm.utils.config import Config

    monkeypatch.setenv("PIMM_WANDB_HISTORY", "append")
    owner = object.__new__(train_engine.Trainer)
    owner.cfg = Config({"use_wandb": True, "save_path": "/tmp/wandb-adapter-test"})
    owner.logger = Mock()
    monkeypatch.setattr(train_engine.comm, "is_main_process", lambda: True)
    assert owner.build_writer().history == "append"
    assert owner.cfg.wandb_history_resolved == "append"
    monkeypatch.setattr(train_engine.comm, "is_main_process", lambda: False)
    assert owner.build_writer() is None


@pytest.mark.parametrize(
    "cfg",
    [
        {"wandb_resume_from": "parent?_step=2"},
        {
            "resume": True,
            "wandb_fresh_run_on_resume": False,
            "wandb_fork_run_on_resume": False,
        },
        {"wandb_history": "rewind"},
        {"wandb_fresh_run_on_resume": True, "wandb_fork_run_on_resume": True},
    ],
)
def test_unsupported_or_ambiguous_policy_is_not_reinterpreted(cfg):
    with pytest.raises(ValueError):
        resolve_wandb_history(cfg)


def test_resolved_env_policy_is_saved_without_becoming_an_override(
    monkeypatch, tmp_path
):
    from pimm.utils.config import Config

    monkeypatch.setenv("PIMM_WANDB_HISTORY", "append")
    config = tmp_path / "input.py"
    config.write_text(
        f"seed=1\nresume=False\nuse_wandb=True\nsave_path={str(tmp_path / 'out')!r}\n"
    )
    cfg = default_config_parser(str(config), {})
    saved = Config.fromfile(str(tmp_path / "out/config.py"))
    assert cfg.wandb_history_resolved == saved.wandb_history_resolved == "append"
    assert "wandb_history" not in saved
    monkeypatch.setenv("PIMM_WANDB_HISTORY", "new")
    assert resolve_wandb_history(saved) == "new"


def test_last_step_checkpoint_waits_for_epoch_metrics(backend, monkeypatch):
    saves = []

    class FakeCheckpointManager:
        def __init__(self, trainer):
            self.trainer = trainer

        def save_iteration_checkpoint(self, **kwargs):
            saves.append(self.trainer.writer.checkpoint_state())

    monkeypatch.setattr(checkpoint_hooks, "CheckpointManager", FakeCheckpointManager)
    writer = events.WandbSummaryWriter(project="pimm")
    owner = SimpleNamespace(
        writer=writer,
        global_step=1,
        start_epoch=0,
        start_iter=1,
        train_loader=[None, None],
        comm_info={"iter": 1, "iter_per_epoch": 2},
        cfg=SimpleNamespace(evaluate=False),
        logger=SimpleNamespace(info=lambda *a: None),
        best_metric_value=float("-inf"),
    )
    hook = checkpoint_hooks.CheckpointSaverIteration(save_freq=2)
    hook.trainer = owner
    hook.before_train()
    writer.add_scalar("train/loss", 1.0, 2)
    hook.after_step()
    assert saves == []
    writer.add_scalar("train/epoch_loss", 0.8, 2)
    hook.after_epoch()
    assert writer.run.history[-1]["train/epoch_loss"] == 0.8
    assert saves[0]["next_step"] == 3


def test_all_resume_checkpoint_savers_capture_wandb_state(monkeypatch):
    calls = []

    def build_payload(trainer, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(checkpoints, "build_checkpoint_payload", build_payload)
    manager = checkpoints.CheckpointManager(SimpleNamespace(cfg={}))
    monkeypatch.setattr(manager, "_write_checkpoint", lambda *a, **kw: None)
    manager.save_epoch_checkpoint()
    manager.save_iteration_checkpoint()
    assert calls == [{"distributed_rng": True, "initialize_logger": True}] * 2


@pytest.mark.parametrize("legacy", [False, True])
def test_dcp_restores_tracking_fields_without_initializing_writer(
    backend, monkeypatch, tmp_path, legacy
):
    saved = {
        "run_id": "parent",
        "entity": "original-team",
        "project": "original-project",
        "next_step": 8,
    }
    if legacy:
        saved = {"run_id": "parent", "resume_step": 8}
    owner = trainer(
        events.WandbSummaryWriter(history="fork"),
        wandb_entity="team",
        wandb_project="pimm",
    )

    def build_payload(owner, **kwargs):
        return {"logger": build_logger_state(owner), "trainer": {}}

    monkeypatch.setattr(checkpoints, "build_checkpoint_payload", build_payload)
    path = str(tmp_path / "checkpoint")
    checkpoints.save_dcp_checkpoint(
        {"logger": {"backend": "wandb", "wandb": saved}, "trainer": {"global_step": 7}},
        path,
    )
    loaded = checkpoints.load_dcp_checkpoint(path, owner)
    configure_logger_from_checkpoint(owner, loaded)
    state = owner.writer._wandb_state
    assert state["next_step"] == 8
    assert state["entity"] == ("team" if legacy else "original-team")
    assert state["project"] == ("pimm" if legacy else "original-project")
    assert not backend.init_calls
