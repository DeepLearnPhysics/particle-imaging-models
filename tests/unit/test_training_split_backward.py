import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pimm.engines.train import Trainer, TrainerBase
from pimm.utils.config import Config


class Branches(nn.Module):
    def __init__(self, form):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.form = form
        self.split_backward = form == "iterator"
        self.sequential_backwards = 0

    def parts(self, batch):
        yield (self.linear(batch["x"]) - batch["y"]).square().mean()
        assert self.linear.weight.grad is not None
        self.sequential_backwards += 1
        yield (self.linear(batch["x"] * 2) - batch["y"]).square().mean()

    def forward(self, batch):
        if self.form == "iterator":
            return {"loss": self.parts(batch)}
        parts = [
            (self.linear(batch["x"] * scale) - batch["y"]).square().mean()
            for scale in (1, 2)
        ]
        return {"loss": parts if self.form == "list" else sum(parts)}


@pytest.mark.parametrize("form", ["tensor", "list", "iterator"])
@pytest.mark.parametrize("micro_batches", [1, 2])
@pytest.mark.parametrize("amp", [False, True])
def test_loss_parts_preserve_updates_and_accumulation(form, micro_batches, amp):
    torch.manual_seed(17)
    model = Branches(form)
    reference = copy.deepcopy(model)
    reference.form = "tensor"
    batches = [
        {"x": torch.randn(3, 2), "y": torch.randn(3, 1)} for _ in range(micro_batches)
    ]
    expected_losses = []
    for batch in batches:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            loss = reference(batch)["loss"] / micro_batches
        loss.backward()
        expected_losses.append(loss.detach())
    expected_gradients = [p.grad.clone() for p in reference.parameters()]
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.2)
    torch.optim.SGD(reference.parameters(), lr=0.1).step()

    trainer = Trainer.__new__(Trainer)
    TrainerBase.__init__(trainer)
    trainer.cfg = Config(
        dict(enable_amp=amp, amp_dtype="bfloat16", clip_grad=0.2, empty_cache=False)
    )
    trainer.parallel_context = SimpleNamespace(device=torch.device("cpu"))
    trainer.model = model
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=5)
    trainer.scaler = torch.amp.GradScaler("cpu", enabled=amp)
    trainer.comm_info = dict(input_dict=batches[0], micro_batches=batches)
    observed = []
    trainer.after_backward = lambda: observed.extend(
        p.grad.clone() for p in model.parameters()
    )
    trainer.run_step()

    for actual, expected in zip(observed, expected_gradients):
        torch.testing.assert_close(
            actual, expected, rtol=1e-2 if amp else 1e-6, atol=1e-5
        )
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(
            actual, expected, rtol=1e-2 if amp else 1e-6, atol=1e-5
        )
    torch.testing.assert_close(
        trainer.comm_info["model_output_dict"]["loss"], sum(expected_losses)
    )
    assert trainer.scheduler.last_epoch == 1
    assert model.sequential_backwards == (micro_batches if form == "iterator" else 0)


@pytest.mark.parametrize("split", [False, True])
def test_ddp_static_graph_follows_the_loss_schedule(monkeypatch, split):
    from pimm.distributed import distributed

    captured = {}

    def wrap(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(distributed, "DistributedDataParallel", wrap)
    model = Branches("iterator" if split else "tensor")
    context = SimpleNamespace(strategy="ddp", world_size=2, device=torch.device("cpu"))
    assert distributed.prepare_model(model, {}, context) is model
    assert captured["static_graph"] is not split
