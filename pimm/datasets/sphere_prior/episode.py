"""Sample exclusive isotropic clusters on the decoder's LayerNorm sphere."""

from __future__ import annotations

import math

import torch

from . import sampling

POINTS, CHANNELS, RANK = 2048, 256, 32  # points per scene, feature width, rank of the centres' subspace


def sample_episode(rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """One scene: ``k`` (2 to 40, log-uniform) equal-size clusters around centres in a random rank-32 subspace, with a
    shared angular spread (2 to 80 degrees), plus an optional isotropic background, shuffled. Returns the unit-norm
    ``[2048, 256]`` features and each point's cluster index (-1 for background)."""
    k = min(round(sampling.log_uniform(rng, 2, 40)), 40)
    draw_angle = sampling.log_uniform if sampling.bernoulli(rng, 0.5) else sampling.uniform
    angle = math.radians(draw_angle(rng, 2.0, 80.0))
    background = 0.0 if sampling.bernoulli(rng, 0.35) else sampling.uniform(rng, 0.0, 0.5)

    subspace = sampling.orthonormal_columns(sampling.remove_mean_direction(sampling.randn(rng, RANK + 1, CHANNELS)).T)
    centres = sampling.unit_rows(sampling.randn(rng, k, RANK) @ subspace[:, 1:].T)
    widest = min(math.radians(85.0), math.acos(math.sqrt(0.03)))
    concentration = math.cos(min(max(angle, math.radians(1.0)), widest))

    num_background = round(background * POINTS)
    num_foreground = POINTS - num_background
    sizes = [num_foreground // k + (i < num_foreground % k) for i in range(k)]  # equal sizes, remainder spread
    features, owner = [], []
    for index, size in enumerate(sizes):
        noise = sampling.remove_mean_direction(sampling.randn(rng, size, CHANNELS)) / math.sqrt(CHANNELS - 1)
        features.append(concentration * centres[index] + math.sqrt(1.0 - concentration**2) * noise)
        owner.append(torch.full((size,), index, dtype=torch.long))
    if num_background:
        noise = sampling.remove_mean_direction(sampling.randn(rng, num_background, CHANNELS))
        features.append(sampling.unit_rows(noise))
        owner.append(torch.full((num_background,), -1, dtype=torch.long))

    order = torch.randperm(POINTS, generator=rng)
    features = sampling.unit_rows(sampling.remove_mean_direction(torch.cat(features)))
    return features[order], torch.cat(owner)[order]
