"""Long rank-zero evaluation must not exhaust the training collective timeout."""

from datetime import timedelta
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pimm.engines.hooks.eval.probes.events import rank_zero_evaluation


class EvaluationFailure(ValueError):
    pass


class Evaluator:
    def __init__(self, *, training=True):
        self.trainer = SimpleNamespace(model=torch.nn.Linear(1, 1))
        self.trainer.model.train(training)
        self.calls = 0
        self.failure = EvaluationFailure("probe extraction failed")

    @rank_zero_evaluation
    def evaluate(self, delay=0, fail=False):
        self.calls += 1
        self.trainer.model.eval()
        time.sleep(delay)
        if fail:
            raise self.failure
        return "complete"


def _distributed_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"file://{rendezvous}",
        timeout=timedelta(seconds=2),
    )
    try:
        evaluator = Evaluator()
        started = time.monotonic()
        result = evaluator.evaluate(delay=4)
        assert time.monotonic() - started >= 4
        assert evaluator.calls == int(rank == 0)
        assert result == ("complete" if rank == 0 else None)
        assert evaluator.trainer.model.training
        total = torch.tensor(rank + 1)
        dist.all_reduce(total)
        assert total.item() == 3

        evaluator = Evaluator(training=False)
        try:
            evaluator.evaluate(fail=True)
        except EvaluationFailure as exc:
            assert rank == 0
            assert exc is evaluator.failure
        except RuntimeError as exc:
            assert rank == 1
            assert "Rank-zero evaluation" in str(exc)
            assert "EvaluationFailure: probe extraction failed" in str(exc)
        else:
            raise AssertionError("Evaluation failure was not propagated")
        assert not evaluator.trainer.model.training
        total = torch.tensor(rank + 1)
        dist.all_reduce(total)
        assert total.item() == 3
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires PyTorch Gloo support",
)
def test_long_evaluation_and_failure_keep_training_group_usable(tmp_path):
    context = mp.spawn(
        _distributed_worker,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Distributed evaluation regression timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)


@pytest.mark.parametrize("training", [True, False])
def test_single_rank_return_exception_and_original_mode(training):
    evaluator = Evaluator(training=training)
    assert evaluator.evaluate() == "complete"
    assert evaluator.trainer.model.training == training
    with pytest.raises(EvaluationFailure) as caught:
        evaluator.evaluate(fail=True)
    assert caught.value is evaluator.failure
    assert evaluator.trainer.model.training == training
