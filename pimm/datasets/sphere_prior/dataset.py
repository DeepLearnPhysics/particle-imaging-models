"""Deterministic scenes from the Panda decoder training prior."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ..builder import DATASETS
from .episode import sample_episode
from .sampling import episode_seed

SPLITS = ("train", "val", "test")


@DATASETS.register_module()
class SpherePriorDataset(Dataset):
    """Fixed 2,048-point, 256-channel scenes with exclusive instance labels.

    Each split uses an independent seeded stream of the same prior. Targets
    use the overlap loss format: one-hot ``inst_pe``, padded to 48 slots,
    with ``ring_valid`` and class-zero ``inst_class`` labels.
    """

    def __init__(self, length: int, split: str = "train", seed: int = 0) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.length = int(length)
        self.split = split
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.length:
            raise IndexError(index)
        seed = episode_seed(self.seed, SPLITS.index(self.split), index)
        rng = torch.Generator().manual_seed(seed)
        features, owner = sample_episode(rng)
        num_points = len(owner)
        num_instances = int(owner.max()) + 1
        ownership = torch.zeros((num_points, 48))
        foreground = owner >= 0
        ownership[foreground, owner[foreground]] = 1.0
        ring_valid = torch.arange(48) < num_instances
        return {
            "feat": features,
            # Trainer utilities expect coordinates even for feature-only scenes.
            "coord": torch.zeros((num_points, 3)),
            "instance": owner,
            "inst_pe": ownership,
            "ring_valid": ring_valid,
            "inst_class": torch.zeros(48, dtype=torch.long),
            "offset": torch.tensor([num_points], dtype=torch.int32),
            "name": f"sphere-prior-{self.split}-{index:09d}",
        }
