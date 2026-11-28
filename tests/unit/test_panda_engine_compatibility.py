"""Panda engine options preserve public defaults and optimizer-step semantics."""

import copy
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from pimm.engines._train_utils import (
    GradientAccumulationLoader,
    capture_rng_state,
    restore_distributed_rng_state,
)
from pimm.engines.hooks import checkpoint
from pimm.engines.hooks.eval.probes.points import LinearProbingTrainer
from pimm.engines.hooks.eval.probes.events import EventLinearProbeEvaluator
from pimm.engines.metrics import eval_instances
from pimm.engines.train import Trainer, TrainerBase
from pimm.utils.config import Config


def test_public_metrics_import_without_training_dependencies():
    script = """
import importlib.abc
import sys

class BlockTrainingDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sklearn'}:
            raise ImportError(fullname)

sys.meta_path.insert(0, BlockTrainingDependencies())
from pimm.engines.metrics import eval_instances, compute_semseg_metrics
assert 'torch' not in sys.modules and 'sklearn' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)


class _Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, batch):
        return {"loss": (self.linear(batch["x"]) - batch["y"]).square().mean()}


def test_accumulation_matches_a_single_combined_batch_and_exposes_unclipped_gradients():
    torch.manual_seed(3)
    model = _Regression()
    reference = copy.deepcopy(model)
    batches = [{"x": torch.randn(3, 2), "y": torch.randn(3, 1)} for _ in range(2)]
    expected_loss = reference(
        {key: torch.cat([batch[key] for batch in batches]) for key in ("x", "y")}
    )["loss"]
    expected_loss.backward()
    expected_gradient = reference.linear.weight.grad.clone()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
    torch.optim.SGD(reference.parameters(), lr=0.2).step()

    trainer = Trainer.__new__(Trainer)
    TrainerBase.__init__(trainer)
    trainer.cfg = Config(
        dict(
            gradient_accumulation_steps=2,
            enable_amp=False,
            amp_dtype="bfloat16",
            clip_grad=0.1,
            empty_cache=False,
        )
    )
    trainer.model = model
    trainer.parallel_context = SimpleNamespace(device=torch.device("cpu"))
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=5)
    trainer.comm_info = {"input_dict": batches[0], "micro_batches": batches}
    observed = []
    trainer.after_backward = lambda: observed.append(model.linear.weight.grad.clone())
    trainer.run_step()

    assert len(observed) == 1
    torch.testing.assert_close(observed[0], expected_gradient)
    torch.testing.assert_close(model.linear.weight, reference.linear.weight)
    torch.testing.assert_close(
        trainer.comm_info["model_output_dict"]["loss"], expected_loss
    )
    assert trainer.scheduler.last_epoch == 1
    assert trainer.comm_info["accumulated_batches"] == 2
    assert list(GradientAccumulationLoader(range(5), 2)) == [[0, 1], [2, 3]]


def test_pid_representation_is_explicit_only_when_requested():
    instances = np.array([0, 0, 1, 1])
    labels = np.array([0, 0, 1, 1])
    args = (instances, instances, labels, labels)
    tables = eval_instances(*args, class_names=("a", "b"), pid_tables=True)
    points = eval_instances(*args, class_names=("a", "b"))  # per-point PIDs by default
    np.testing.assert_array_equal(tables["classification_on_matched"]["support"], [2, 0])
    np.testing.assert_array_equal(points["classification_on_matched"]["support"], [1, 1])
    with pytest.raises(ValueError, match="pid_tables=True"):
        eval_instances(
            instances, instances, np.array([0, 1]), np.array([0, 1]), pid_tables=False
        )


def test_probe_head_selection_keeps_the_public_split_by_default():
    features = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    labels = torch.arange(20) % 2
    kwargs = dict(
        X_train=features[:10],
        y_train=labels[:10],
        X_test=features[10:],
        y_test=labels[10:],
        num_classes=2,
        logger=None,
    )
    config = dict(
        learning_rates=[0.01],
        epochs=1,
        device=torch.device("cpu"),
        criteria=[dict(type="CrossEntropyLoss")],
    )
    public = LinearProbingTrainer(**kwargs, config=config)
    panda = LinearProbingTrainer(**kwargs, config=dict(config, held_out_selection=True))
    assert len(public.X_train) == 10
    torch.testing.assert_close(public.X_select, features[10:])
    assert len(panda.X_train) == 8 and len(panda.X_select) == 2
    assert set(panda.X_train[:, 0].tolist()).isdisjoint(panda.X_select[:, 0].tolist())
    assert set(panda.X_train[:, 0].tolist()) | set(
        panda.X_select[:, 0].tolist()
    ) == set(features[:10, 0].tolist())
    EventLinearProbeEvaluator(class_names=["a", "b", "c"])
    with pytest.raises(TypeError, match="binary"):
        EventLinearProbeEvaluator(class_names=["a", "b", "c"], out_of_fold=True)


def test_unrepresented_rank_keeps_its_rng_stream(monkeypatch):
    from pimm.engines import _train_utils

    saved = {"world_size": 1, "states": [capture_rng_state()]}
    torch.manual_seed(29)
    fresh = torch.get_rng_state().clone()
    monkeypatch.setattr(_train_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(_train_utils.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(_train_utils.dist, "get_rank", lambda: 1)
    restore_distributed_rng_state(saved, strict=False)
    assert torch.equal(torch.get_rng_state(), fresh)


def test_epoch_checkpoint_participates_without_a_local_metric(monkeypatch):
    writes = []
    trainer = SimpleNamespace(
        cfg=Config(dict(evaluate=True)), best_metric_value=-float("inf"), comm_info={}
    )
    saver = checkpoint.CheckpointSaver(evaluator_every_n_steps=0)
    saver.trainer = trainer
    saver.step_count = 4
    saver._pending_epoch_save = None
    saver.checkpoint_manager = SimpleNamespace(
        save_epoch_checkpoint=lambda **kwargs: writes.append(kwargs)
    )
    monkeypatch.setattr(checkpoint, "get_world_size", lambda: 2)

    def receive(values, src):
        values[:] = [0.75, True]

    monkeypatch.setattr(checkpoint.dist, "broadcast_object_list", receive)
    saver.after_epoch()
    assert trainer.best_metric_value == 0.75
    assert len(writes) == 1 and writes[0]["is_best"]
