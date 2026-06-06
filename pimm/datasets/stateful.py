"""Checkpointable dataloader/sampler state helpers.

The sampler mirrors PyTorch distributed sampler padding semantics while tracking
the current position. This lets long jobs resume mid-epoch without reshuffling
or replaying already-consumed samples.
"""

from __future__ import annotations

import math
import pickle
from typing import Iterator, Sized

import torch
from torch.utils.data import DataLoader, Sampler

DATALOADER_STATE_FORMAT = "pimm.torchdata_state.v1"
UNSUPPORTED_DATALOADER_STATE_FORMAT = "pimm.dataloader_state.unsupported.v1"


class StatefulRandomSampler(Sampler[int]):
    """Rank-aware map-style sampler with checkpointable epoch and position."""

    def __init__(
        self,
        data_source: Sized,
        *,
        shuffle: bool = True,
        seed: int = 0,
        epoch: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
    ) -> None:
        """Create a deterministic per-rank sample order for one epoch."""
        self.data_source = data_source
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.position = 0
        self.num_samples = self._compute_num_samples()
        self.total_size = self.num_samples * self.num_replicas
        self._order = self._build_order()

    def _compute_num_samples(self) -> int:
        """Compute local sample count after distributed padding or dropping."""
        length = len(self.data_source)
        if length == 0:
            return 0
        if self.drop_last and length % self.num_replicas != 0:
            return math.ceil((length - self.num_replicas) / self.num_replicas)
        return math.ceil(length / self.num_replicas)

    def _build_order(self) -> list[int]:
        """Build this rank's padded/dropped index order for the current epoch."""
        length = len(self.data_source)
        if length == 0:
            return []
        if not self.shuffle:
            indices = list(range(length))
        else:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(length, generator=generator).tolist()

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self.total_size]

        return indices[self.rank : self.total_size : self.num_replicas]

    def __iter__(self) -> Iterator[int]:
        """Yield remaining indices and advance the stored position."""
        while self.position < len(self._order):
            index = self._order[self.position]
            self.position += 1
            yield index

    def __len__(self) -> int:
        """Return this rank's sample count for the current epoch."""
        return self.num_samples

    @property
    def remaining(self) -> int:
        """Number of indices left before this rank exhausts the epoch."""
        return max(0, len(self._order) - self.position)

    def set_epoch(self, epoch: int, *, reset_position: bool = True) -> None:
        """Change epoch seed and optionally rewind to the start of that order."""
        self.epoch = int(epoch)
        self.num_samples = self._compute_num_samples()
        self.total_size = self.num_samples * self.num_replicas
        self._order = self._build_order()
        if reset_position:
            self.position = 0

    def set_position(self, position: int) -> None:
        """Clamp and set the next index position within the current order."""
        self.position = max(0, min(int(position), len(self._order)))

    def state_dict(self) -> dict[str, object]:
        """Serialize enough state to resume this sampler mid-epoch."""
        return {
            "type": self.__class__.__name__,
            "seed": self.seed,
            "epoch": self.epoch,
            "shuffle": self.shuffle,
            "position": self.position,
            "length": len(self.data_source),
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "drop_last": self.drop_last,
            "num_samples": self.num_samples,
            "total_size": self.total_size,
        }

    def load_state_dict(self, state_dict: dict[str, object], *, strict: bool = True) -> None:
        """Restore sampler state, validating dataset length and replica count."""
        length = int(state_dict["length"])
        if length != len(self.data_source):
            raise ValueError(
                f"Sampler state length {length} does not match dataset length {len(self.data_source)}"
            )
        saved_replicas = int(state_dict.get("num_replicas", self.num_replicas))
        if strict and saved_replicas != self.num_replicas:
            raise ValueError(
                f"Sampler state was saved with num_replicas={saved_replicas}, "
                f"but current num_replicas={self.num_replicas}."
            )
        self.seed = int(state_dict["seed"])
        self.epoch = int(state_dict["epoch"])
        self.shuffle = bool(state_dict["shuffle"])
        self.drop_last = bool(state_dict.get("drop_last", self.drop_last))
        self.num_samples = self._compute_num_samples()
        self.total_size = self.num_samples * self.num_replicas
        self._order = self._build_order()
        self.set_position(int(state_dict["position"]))


def dataloader_state_dict(loader: DataLoader) -> dict[str, object]:
    """Return a stable wrapper around torchdata's private loader state."""
    if hasattr(loader, "state_dict") and hasattr(loader, "load_state_dict"):
        return {
            "format": DATALOADER_STATE_FORMAT,
            "state": pickle.dumps(loader.state_dict()),
        }
    return {
        "format": UNSUPPORTED_DATALOADER_STATE_FORMAT,
        "loader_type": loader.__class__.__name__,
    }


def is_unsupported_dataloader_state(state_dict: dict[str, object] | None) -> bool:
    """Return whether a captured loader state cannot support exact resume."""
    return (
        isinstance(state_dict, dict)
        and state_dict.get("format") == UNSUPPORTED_DATALOADER_STATE_FORMAT
    )


def load_dataloader_state_dict(
    loader: DataLoader,
    state_dict: dict[str, object] | None,
    *,
    strict: bool = True,
) -> None:
    """Load the stable pimm wrapper and delegate raw state to torchdata."""
    if not state_dict:
        return
    state_format = state_dict.get("format")
    if state_format == UNSUPPORTED_DATALOADER_STATE_FORMAT:
        raise RuntimeError(
            "Cannot restore exact dataloader position because the checkpoint "
            f"was written for unsupported train loader type "
            f"{state_dict.get('loader_type', 'unknown')!r}."
        )
    if state_format != DATALOADER_STATE_FORMAT:
        raise ValueError(
            f"Unsupported dataloader state format {state_format!r}; expected "
            f"{DATALOADER_STATE_FORMAT!r}."
        )
    if not hasattr(loader, "load_state_dict"):
        raise RuntimeError(
            f"Train loader type {loader.__class__.__name__!r} cannot restore "
            "exact dataloader state because it has no load_state_dict()."
        )
    raw_state = pickle.loads(state_dict["state"])
    loader.load_state_dict(raw_state)
    setattr(loader, "_pimm_loaded_state", True)


def assert_exact_dataloader_state_available(
    state_dict: dict[str, object] | None,
    *,
    loader: DataLoader,
    iter_in_epoch: int,
) -> None:
    """Fail early when a mid-epoch checkpoint cannot restore loader position."""
    if int(iter_in_epoch) <= 0 or not is_unsupported_dataloader_state(state_dict):
        return
    raise RuntimeError(
        "Cannot create an exact mid-epoch checkpoint for train loader type "
        f"{loader.__class__.__name__!r} because it does not implement "
        "state_dict()/load_state_dict(). Use torchdata StatefulDataLoader or "
        "save only at epoch boundaries."
    )


def set_dataloader_epoch(loader: DataLoader, epoch: int, *, reset_position: bool = True) -> None:
    """Set epoch on a loader's sampler while respecting a just-loaded state."""
    if getattr(loader, "_pimm_loaded_state", False) and not reset_position:
        return
    if reset_position and getattr(loader, "_pimm_loaded_state", False):
        loader.load_state_dict({})
        setattr(loader, "_pimm_loaded_state", False)
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        try:
            sampler.set_epoch(epoch, reset_position=reset_position)
        except TypeError:
            sampler.set_epoch(epoch)
