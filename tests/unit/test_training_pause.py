"""Cooperative pauses preserve completed work without final-training callbacks."""

from types import SimpleNamespace

import pytest

from pimm.engines import train as train_module
from pimm.engines.hooks.default import HookBase
from pimm.engines.train import Trainer, TrainerBase


class _Lifecycle(HookBase):
    def __init__(self, events, pause_at):
        self.events = events
        self.pause_at = pause_at

    def before_train(self):
        self.events.append("before_train")

    def before_epoch(self):
        self.events.append("before_epoch")
        if type(self.trainer) is TrainerBase:
            self.trainer.data_iterator = enumerate(self.trainer.train_loader)

    def before_step(self):
        self.events.append("before_step")

    def after_step(self):
        self.events.append("after_step")
        if self.pause_at == "step":
            self.trainer.request_pause()

    def after_epoch(self):
        self.events.append("after_epoch")
        if self.pause_at == "epoch":
            self.trainer.request_pause()

    def after_train(self):
        self.events.append("after_train")


@pytest.fixture(params=[TrainerBase, Trainer], ids=["base", "default"])
def make_trainer(request, monkeypatch):
    def make(*, pause_at=None, completed=False):
        events = []
        trainer = request.param.__new__(request.param)
        TrainerBase.__init__(trainer)
        trainer.cfg = SimpleNamespace(
            detect_anomaly=False,
            matmul_precision=None,
            batch_size_per_gpu=1,
            empty_cache_per_epoch=False,
        )
        trainer.logger = SimpleNamespace(info=lambda message: None)
        trainer.model = SimpleNamespace(train=lambda: None)
        trainer.train_loader = [{"batch": index} for index in range(3)]
        trainer.max_epoch = 2
        trainer.start_epoch = 2 if completed else 0
        trainer.global_step = 6 if completed else 0
        trainer.writer = SimpleNamespace(close=lambda: events.append("close"))
        trainer._log_trace_batch_stats = lambda step: None
        trainer._align_writer_step = lambda step: None

        hook = _Lifecycle(events, pause_at)
        hook.trainer = trainer
        # A subsequent callback still runs when the preceding hook requests pause.
        tail = HookBase()
        tail.after_step = lambda: events.append("after_step_tail")
        tail.after_epoch = lambda: events.append("after_epoch_tail")
        trainer.hooks = [hook, tail]

        def run_step():
            events.append("run_step")
            if request.param is TrainerBase:
                trainer.global_step += 1

        trainer.run_step = run_step
        monkeypatch.setattr(
            train_module, "set_dataloader_epoch", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            train_module.comm, "synchronize", lambda: events.append("synchronize")
        )
        monkeypatch.setattr(train_module.comm, "is_main_process", lambda: True)
        monkeypatch.setattr(train_module.comm, "get_world_size", lambda: 1)
        return trainer, events

    return make


def test_step_pause_finishes_callbacks_but_not_epoch_or_training(make_trainer):
    trainer, events = make_trainer(pause_at="step")

    trainer.train()

    assert trainer.pause_requested
    assert trainer.global_step == 1
    assert trainer.comm_info["iter"] == 0
    assert events == [
        "before_train",
        "before_epoch",
        "before_step",
        "run_step",
        "after_step",
        "after_step_tail",
        "synchronize",
        "close",
    ]
    if isinstance(trainer, Trainer):
        assert trainer.samples_seen == 1
        assert trainer.train_state.epoch == 0
        assert trainer.train_state.iter_in_epoch == 1
        assert trainer.train_state.global_step == 1


def test_epoch_pause_finishes_epoch_callbacks_but_not_final_hooks(make_trainer):
    trainer, events = make_trainer(pause_at="epoch")

    trainer.train()

    assert trainer.global_step == 3
    assert events.count("run_step") == 3
    assert events.count("before_epoch") == 1
    assert events[-4:] == [
        "after_epoch",
        "after_epoch_tail",
        "synchronize",
        "close",
    ]
    assert "after_train" not in events
    if isinstance(trainer, Trainer):
        assert trainer.train_state.epoch == 1
        assert trainer.train_state.iter_in_epoch == 0


def test_unpaused_training_keeps_complete_lifecycle(make_trainer):
    trainer, events = make_trainer()

    trainer.train()

    assert not trainer.pause_requested
    assert trainer.global_step == 6
    assert events.count("after_epoch") == 2
    assert events[-3:] == ["synchronize", "after_train", "close"]


def test_already_complete_resume_still_skips_training_and_final_hooks(make_trainer):
    trainer, events = make_trainer(completed=True)

    trainer.train()

    assert trainer.global_step == 6
    assert not trainer.pause_requested
    assert events == ["before_train", "synchronize", "close"]


def test_later_hook_failure_does_not_turn_into_successful_pause(make_trainer):
    trainer, events = make_trainer(pause_at="step")

    def fail():
        raise RuntimeError("hook failed")

    trainer.hooks[-1].after_step = fail
    expected = SystemExit if isinstance(trainer, Trainer) else RuntimeError
    with pytest.raises(expected):
        trainer.train()

    assert trainer.pause_requested
    assert "synchronize" not in events
    assert "close" not in events
    assert "after_train" not in events


def test_pause_request_is_idempotent_and_instance_local(make_trainer):
    paused, _ = make_trainer()
    untouched, _ = make_trainer()

    paused.request_pause()
    paused.request_pause()

    assert paused.pause_requested
    assert not untouched.pause_requested
