"""Seeded sampling helpers.

Every draw takes an explicit ``torch.Generator`` (on any device) so an episode
is a pure function of its index: deterministic validation and exact
dataloader resume.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def episode_seed(*components: int) -> int:
    """Well-mixed 63-bit seed from integer components (seed, split, index)."""
    state = np.random.SeedSequence([int(c) for c in components]).generate_state(
        1, dtype=np.uint64
    )
    return int(state[0] & np.uint64(2**63 - 1))


def randn(rng: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(shape, generator=rng, device=rng.device)


def rand(rng: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.rand(shape, generator=rng, device=rng.device)


def uniform(rng: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(rand(rng))


def log_uniform(rng: torch.Generator, low: float, high: float) -> float:
    return math.exp(uniform(rng, math.log(low), math.log(high)))


def bernoulli(rng: torch.Generator, p: float) -> bool:
    return float(rand(rng)) < p


def unit_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def remove_mean_direction(x: torch.Tensor) -> torch.Tensor:
    """Project out the all-ones direction, which LayerNorm removes anyway."""
    return x - x.mean(dim=-1, keepdim=True)


def orthonormal_columns(matrix: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(matrix)
    return q
