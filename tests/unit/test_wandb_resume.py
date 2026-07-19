from types import SimpleNamespace

from pimm.engines.hooks import checkpoint as checkpoint_hooks
from pimm.utils import checkpoints, events
from pimm.utils.checkpoints import (
    build_logger_state,
    configure_logger_from_checkpoint,
)


class FakeRun:
    def __init__(self, backend, run_id, history):
        self.backend = backend
        self.id = run_id
        self.history = history
        self.step = max((row["_step"] for row in history), default=-1) + 1
        self.metric_definitions = []

    def define_metric(self, name, **kwargs):
        self.metric_definitions.append((name, kwargs))

    def log(self, data, **kwargs):
        self.backend.log_kwargs.append(kwargs)
        step = kwargs.get("step", self.step)
        row = {**data, "_step": step}
        self.history.append(row)
        if kwargs.get("commit", kwargs.get("step") is None):
            self.step = step + 1


class FakeWandb:
    def __init__(self):
        self.run = None
        self.histories = {}
        self.archived = {}
        self.init_calls = []
        self.log_kwargs = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        resume_from = kwargs.get("resume_from")
        if resume_from:
            run_id, _, query = resume_from.partition("?")
            resume_step = int(query.removeprefix("_step="))
            history = self.histories[run_id]
            archived = [row for row in history if row["_step"] >= resume_step]
            self.archived.setdefault(run_id, []).append(archived)
            history[:] = [row for row in history if row["_step"] < resume_step]
        else:
            run_id = kwargs.get("id") or "logical-run"
            history = self.histories.setdefault(run_id, [])
        self.run = FakeRun(self, run_id, history)
        return self.run

    def finish(self):
        self.run = None


def test_checkpoint_rewind_replaces_preempted_history(monkeypatch):
    backend = FakeWandb()
    monkeypatch.setattr(events, "wandb", backend)

    writer = events.WandbSummaryWriter(project="pimm")
    for global_step in range(1, 51):
        writer.add_scalar("train/loss", float(global_step), global_step)
        writer.add_scalar("params/lr", 0.1, global_step)
        writer.flush_step()
    checkpoint_state = writer.checkpoint_state()
    for global_step in range(51, 76):
        writer.add_scalar("train/loss", float(global_step), global_step)
        writer.add_scalar("params/lr", 0.1, global_step)
        writer.flush_step()

    backend.run = None
    resumed = events.WandbSummaryWriter(
        project="pimm",
        id="manually-configured-id",
        resume="must",
        fork_from="other-run?_step=10",
    )
    resumed.resume_from_checkpoint(checkpoint_state)
    for global_step in range(51, 76):
        resumed.add_scalar("train/loss", -float(global_step), global_step)
        resumed.add_scalar("params/lr", 0.01, global_step)
        resumed.flush_step()

    assert checkpoint_state == {"run_id": "logical-run", "resume_step": 51}
    assert backend.init_calls[-1]["resume_from"] == "logical-run?_step=51"
    assert "id" not in backend.init_calls[-1]
    assert "resume" not in backend.init_calls[-1]
    assert "fork_from" not in backend.init_calls[-1]
    assert backend.log_kwargs[0] == {"step": 0, "commit": True}
    assert backend.log_kwargs[1:] == [{}] * 100

    history = backend.histories["logical-run"]
    metric_rows = [row for row in history if "train/loss" in row]
    assert [row["_step"] for row in metric_rows] == list(range(1, 76))
    assert [row["train/global_step"] for row in metric_rows] == list(range(1, 76))
    assert [row["train/loss"] for row in metric_rows[50:]] == [
        -float(step) for step in range(51, 76)
    ]
    assert len(backend.archived["logical-run"][0]) == 25
    assert resumed.run.metric_definitions == [
        ("train/global_step", {"hidden": True}),
        ("*", {"step_metric": "train/global_step"}),
    ]


def test_checkpoint_state_uses_wandb_run_not_config(monkeypatch):
    backend = FakeWandb()
    monkeypatch.setattr(events, "wandb", backend)

    writer = events.WandbSummaryWriter(id="actual-run")
    writer.add_scalar("train/loss", 1.0, 7)

    assert writer.checkpoint_state() == {
        "run_id": "actual-run",
        "resume_step": 8,
    }


def test_checkpoint_logger_state_drives_resume_from(monkeypatch):
    backend = FakeWandb()
    monkeypatch.setattr(events, "wandb", backend)

    writer = events.WandbSummaryWriter(id="actual-run")
    writer.add_scalar("train/loss", 1.0, 7)
    trainer = type(
        "Trainer",
        (),
        {
            "cfg": {
                "use_wandb": True,
                "wandb_run_id": "stale-config-id",
                "wandb_resume": "must",
            },
            "writer": writer,
        },
    )()

    logger_state = build_logger_state(
        trainer,
        checkpoint_global_step=7,
        initialize_wandb=True,
    )

    assert logger_state["wandb"]["run_id"] == "actual-run"
    assert logger_state["wandb"]["resume_step"] == 8
    assert logger_state["wandb"]["checkpoint_global_step"] == 7

    backend.run = None
    resumed_writer = events.WandbSummaryWriter(
        id="stale-config-id",
        resume="must",
    )
    resumed_trainer = type(
        "Trainer",
        (),
        {
            "cfg": {"use_wandb": True},
            "writer": resumed_writer,
        },
    )()
    configure_logger_from_checkpoint(
        resumed_trainer,
        {"logger": logger_state},
    )
    resumed_writer.add_scalar("train/loss", 2.0, 8)

    assert backend.init_calls[-1]["resume_from"] == "actual-run?_step=8"


def test_checkpoint_logger_can_start_fresh_continuation(monkeypatch):
    backend = FakeWandb()
    monkeypatch.setattr(events, "wandb", backend)

    writer = events.WandbSummaryWriter(project="pimm", name="continuation")
    messages = []
    trainer = type(
        "Trainer",
        (),
        {
            "cfg": {
                "use_wandb": True,
                "wandb_fresh_run_on_resume": True,
            },
            "writer": writer,
            "logger": SimpleNamespace(info=messages.append),
        },
    )()

    configure_logger_from_checkpoint(
        trainer,
        {
            "logger": {
                "wandb": {
                    "run_id": "original-run",
                    "resume_step": 15001,
                }
            }
        },
    )
    writer.add_scalar("train/loss", 1.0, 15001)

    assert backend.init_calls[-1]["name"] == "continuation"
    assert "resume_from" not in backend.init_calls[-1]
    assert messages == [
        "Starting a fresh W&B run for this checkpoint continuation; "
        "model and trainer state still resume exactly."
    ]


def test_new_runs_keep_legacy_step_axis_with_one_row_per_iteration(monkeypatch):
    backend = FakeWandb()
    monkeypatch.setattr(events, "wandb", backend)

    writer = events.WandbSummaryWriter(project="pimm")
    for global_step in (1, 2):
        writer.add_scalar("train/loss", float(global_step), global_step)
        writer.add_scalar("params/lr", 0.1, global_step)
        writer.add_scalar("train/accuracy", 0.9, global_step)
        writer.flush_step()

    metric_rows = [
        row for row in backend.histories["logical-run"] if "train/loss" in row
    ]
    assert len(metric_rows) == 2
    assert [row["_step"] for row in metric_rows] == [1, 2]
    assert [row["train/global_step"] for row in metric_rows] == [1, 2]
    assert all("params/lr" in row and "train/accuracy" in row for row in metric_rows)


def test_last_step_checkpoint_waits_for_epoch_metrics(monkeypatch):
    backend = FakeWandb()
    monkeypatch.setattr(events, "wandb", backend)
    saves = []

    class FakeCheckpointManager:
        def __init__(self, trainer):
            self.trainer = trainer

        def save_iteration_checkpoint(self, **kwargs):
            saves.append((kwargs, self.trainer.writer.checkpoint_state()))

    monkeypatch.setattr(
        checkpoint_hooks,
        "CheckpointManager",
        FakeCheckpointManager,
    )
    writer = events.WandbSummaryWriter(project="pimm")
    trainer = SimpleNamespace(
        writer=writer,
        global_step=1,
        start_epoch=0,
        start_iter=1,
        train_loader=[None, None],
        comm_info={"iter": 1, "iter_per_epoch": 2},
        cfg=SimpleNamespace(evaluate=False),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        best_metric_value=float("-inf"),
    )
    hook = checkpoint_hooks.CheckpointSaverIteration(save_freq=2)
    hook.trainer = trainer
    hook.before_train()

    writer.add_scalar("train/loss", 1.0, 2)
    hook.after_step()
    assert saves == []

    writer.add_scalar("train/epoch_loss", 0.8, 2)
    hook.after_epoch()

    metric_rows = [
        row for row in backend.histories["logical-run"] if "train/loss" in row
    ]
    assert len(metric_rows) == 1
    assert metric_rows[0]["_step"] == metric_rows[0]["train/global_step"] == 2
    assert metric_rows[0]["train/epoch_loss"] == 0.8
    assert saves[0][1]["resume_step"] == 3


def test_all_resume_checkpoint_savers_capture_wandb_state(monkeypatch):
    calls = []

    def build_payload(trainer, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(checkpoints, "build_checkpoint_payload", build_payload)
    trainer = type("Trainer", (), {"cfg": {}})()
    manager = checkpoints.CheckpointManager(trainer)
    monkeypatch.setattr(manager, "_write_checkpoint", lambda *args, **kwargs: None)

    manager.save_epoch_checkpoint()
    manager.save_iteration_checkpoint()

    assert calls == [
        {"distributed_rng": True, "initialize_logger": True},
        {"distributed_rng": True, "initialize_logger": True},
    ]
