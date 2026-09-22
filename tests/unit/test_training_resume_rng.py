"""Restoring a loader cursor must not advance the application CPU RNG."""

from types import SimpleNamespace

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from pimm.datasets.stateful import (
    StatefulRandomSampler,
    dataloader_state_dict,
    load_dataloader_state_dict,
)
from pimm.engines._train_utils import apply_train_state_to_trainer
from pimm.engines.hooks.default import HookBase
from pimm.engines.train import Trainer, TrainerBase


def _trainer():
    trainer = Trainer.__new__(Trainer)
    TrainerBase.__init__(trainer)
    trainer.cfg = SimpleNamespace(
        detect_anomaly=False,
        matmul_precision=None,
        batch_size_per_gpu=2,
        empty_cache_per_epoch=False,
    )
    trainer.logger = SimpleNamespace(info=lambda message: None)
    trainer.model = SimpleNamespace(train=lambda: None)
    dataset = torch.arange(12)
    trainer.train_loader = StatefulDataLoader(
        dataset,
        sampler=StatefulRandomSampler(dataset, seed=73),
        batch_size=2,
        num_workers=0,
    )
    trainer.max_epoch = 2
    trainer._log_trace_batch_stats = lambda step: None
    records = []

    def run_step():
        before = torch.get_rng_state()
        draw = torch.rand(3)
        records.append(
            {
                "batch": trainer.comm_info["input_dict"].clone(),
                "before": before,
                "draw": draw,
                "after": torch.get_rng_state(),
            }
        )

    trainer.run_step = run_step
    return trainer, records


class _Pause(HookBase):
    def __init__(self, trainer, step):
        self.trainer = trainer
        self.step = step

    def after_step(self):
        if self.trainer.comm_info["iter"] + 1 < len(self.trainer.train_loader):
            self._pause()

    def after_epoch(self):
        self._pause()

    def _pause(self):
        if self.trainer.global_step != self.step:
            return
        self.cursor = dataloader_state_dict(self.trainer.train_loader)
        self.progress = self.trainer.train_state
        self.rng = torch.get_rng_state()
        self.trainer.request_pause()


def _after_worker_seed(state):
    generator = torch.Generator().set_state(state)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    return generator.get_state()


@pytest.mark.parametrize("pause_step", [2, 6], ids=["mid_epoch", "epoch_boundary"])
def test_stateful_loader_resume_preserves_exact_batches_and_rng(pause_step):
    with torch.random.fork_rng(devices=[]):
        initial_rng = torch.Generator().manual_seed(73).get_state()
        torch.set_rng_state(initial_rng)
        control, expected = _trainer()
        control.train()

        # Fresh epochs retain their normal worker-seed draw. This is not a
        # blanket suppression of randomness whenever a loader is iterated.
        assert torch.equal(expected[0]["before"], _after_worker_seed(initial_rng))
        assert torch.equal(
            expected[6]["before"], _after_worker_seed(expected[5]["after"])
        )

        torch.set_rng_state(initial_rng)
        paused, first = _trainer()
        pause = _Pause(paused, pause_step)
        paused.hooks = [pause]
        paused.train()

        resumed, remaining = _trainer()
        apply_train_state_to_trainer(resumed, pause.progress)
        if resumed.start_iter:
            load_dataloader_state_dict(resumed.train_loader, pause.cursor)
        torch.set_rng_state(pause.rng)
        resumed.train()

        actual = first + remaining
        assert len(first) == pause_step
        assert len(actual) == len(expected) == 12
        for left, right in zip(actual, expected):
            for key in ("batch", "before", "draw", "after"):
                assert torch.equal(left[key], right[key]), key
        assert resumed.global_step == control.global_step == 12
        assert resumed.samples_seen == control.samples_seen == 24


def test_reconstructing_torchdata_iterator_consumes_exactly_one_extra_seed():
    with torch.random.fork_rng(devices=[]):
        original, _ = _trainer()
        iterator = iter(original.train_loader)
        next(iterator)
        cursor = dataloader_state_dict(original.train_loader)
        saved_rng = torch.get_rng_state()

        resumed, _ = _trainer()
        load_dataloader_state_dict(resumed.train_loader, cursor)
        torch.set_rng_state(saved_rng)
        iter(resumed.train_loader)

        assert not torch.equal(torch.get_rng_state(), saved_rng)
        assert torch.equal(torch.get_rng_state(), _after_worker_seed(saved_rng))
