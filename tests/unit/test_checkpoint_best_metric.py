"""Rank-zero metrics must reach every rank before checkpoint state is captured."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous):
    from pimm.engines.hooks.checkpoint import CheckpointSaver
    from pimm.utils.config import Config

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"file://{rendezvous}",
        timeout=timedelta(seconds=30),
    )
    try:
        writes = []
        trainer = SimpleNamespace(
            cfg=Config(dict(evaluate=True)),
            best_metric_value=-float("inf"),
            comm_info=dict(iter=0, iter_per_epoch=2),
            logger=SimpleNamespace(info=lambda *args: None),
        )
        saver = CheckpointSaver(evaluator_every_n_steps=1)
        saver.trainer = trainer
        saver._pending_epoch_save = None
        saver.checkpoint_manager = SimpleNamespace(
            save_epoch_checkpoint=lambda **kwargs: writes.append(kwargs)
        )
        if rank == 0:
            trainer.comm_info.update(
                current_metric_value=0.75, current_metric_name="score"
            )
        saver.after_step()
        assert trainer.best_metric_value == 0.75
        assert writes[-1]["is_best"]

        trainer.comm_info["iter"] = 1
        if rank == 0:
            trainer.comm_info["current_metric_value"] = 0.5
        saver.after_step()
        saver.after_epoch()
        assert trainer.best_metric_value == 0.75
        assert not writes[-1]["is_best"]

        saver.evaluator_every_n_steps = 0
        if rank == 0:
            trainer.comm_info["current_metric_value"] = 0.9
        saver.after_epoch()
        assert trainer.best_metric_value == 0.9
        assert writes[-1]["is_best"]
        assert len(writes) == 3
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires PyTorch Gloo",
)
def test_step_and_epoch_checkpoints_broadcast_rank_zero_best(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
