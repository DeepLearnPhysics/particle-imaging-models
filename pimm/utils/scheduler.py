"""
Scheduler

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from .registry import Registry

SCHEDULERS = Registry("schedulers")


def resolve_iters(value, total_steps):
    return int(value) if value > 1 else int(value * total_steps)


@SCHEDULERS.register_module()
class MultiStepLR(lr_scheduler.MultiStepLR):
    """Multi-step scheduler that accepts milestone ratios of total steps."""

    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        last_epoch=-1,
    ):
        """Convert fractional milestones to integer optimizer steps."""
        super().__init__(
            optimizer=optimizer,
            milestones=[int(rate * total_steps) for rate in milestones],
            gamma=gamma,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class MultiStepWithWarmupLR(lr_scheduler.LambdaLR):
    """Multi-step decay with linear warmup expressed as total-step ratios."""

    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        warmup_rate=0.05,
        warmup_scale=1e-6,
        last_epoch=-1,
    ):
        """Create a lambda schedule with warmup followed by milestone decay."""
        milestones = [rate * total_steps for rate in milestones]

        def multi_step_with_warmup(s):
            """Return the lr multiplier at scheduler step ``s``."""
            factor = 1.0
            for i in range(len(milestones)):
                if s < milestones[i]:
                    break
                factor *= gamma

            if s <= warmup_rate * total_steps:
                warmup_coefficient = 1 - (1 - s / warmup_rate / total_steps) * (
                    1 - warmup_scale
                )
            else:
                warmup_coefficient = 1.0
            return warmup_coefficient * factor

        super().__init__(
            optimizer=optimizer,
            lr_lambda=multi_step_with_warmup,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class PolyLR(lr_scheduler.LambdaLR):
    """Polynomial decay scheduler over a fixed number of steps."""

    def __init__(
        self,
        optimizer,
        total_steps,
        power=0.9,
        last_epoch=-1,
    ):
        """Create a polynomial lr multiplier."""
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: (1 - s / (total_steps + 1)) ** power,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class ExpLR(lr_scheduler.LambdaLR):
    """Exponential decay scheduler normalized by total training steps."""

    def __init__(
        self,
        optimizer,
        total_steps,
        gamma=0.9,
        last_epoch=-1,
    ):
        """Create an exponential lr multiplier."""
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: gamma ** (s / total_steps),
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class CosineAnnealingLR(lr_scheduler.CosineAnnealingLR):
    """Cosine annealing scheduler using total training steps as ``T_max``."""

    def __init__(
        self,
        optimizer,
        total_steps,
        eta_min=0,
        last_epoch=-1,
    ):
        """Initialize cosine annealing for the configured step budget."""
        super().__init__(
            optimizer=optimizer,
            T_max=total_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class OneCycleLR(lr_scheduler.OneCycleLR):
    r"""
    torch.optim.lr_scheduler.OneCycleLR, Block total_steps
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
    ):
        # Allow pct_start to be given as an absolute warmup-step count (> 1); the
        # parent OneCycleLR requires a fraction in (0, 1).
        if pct_start > 1:
            assert (
                total_steps is not None
            ), "pct_start given in steps (> 1) requires total_steps"
            pct_start = resolve_iters(pct_start, total_steps) / total_steps
        super().__init__(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
        )


class CosineScheduler(object):
    """Standalone cosine value schedule with warmup and optional freeze phase."""

    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        start_value=0,
        warmup_iters=0,
        freeze_value=None,
        freeze_iters=0,
    ):
        """Precompute a scalar schedule for direct indexing or stepping."""
        self.base_value = base_value
        self.final_value = final_value
        self.total_iters = total_iters

        warmup_schedule = np.linspace(start_value, base_value, warmup_iters)

        if freeze_value is None:
            freeze_value = final_value
        freeze_schedule = np.ones(freeze_iters) * freeze_value

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((warmup_schedule, schedule, freeze_schedule))
        self.iter = 0

    def get(self, it):
        """Return the scheduled value at iteration ``it``."""
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]

    def step(self):
        """Return the current value and advance the internal cursor."""
        value = self.get(self.iter)
        self.iter += 1
        return value

    def reset(self):
        """Reset the internal cursor to the first schedule element."""
        self.iter = 0

    def __getitem__(self, it):
        """Return the scheduled value at iteration ``it``."""
        return self.get(it)


def build_scheduler(cfg, optimizer):
    """Build a registered scheduler from config and attach the optimizer."""
    cfg.optimizer = optimizer
    return SCHEDULERS.build(cfg=cfg)
