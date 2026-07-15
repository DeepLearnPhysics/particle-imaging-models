"""PILArNet-M readers backed by parquet (map-style Arrow-mmap + streaming)."""

import glob
import os
from typing import Literal

import numpy as np
from torch.utils.data import Dataset, IterableDataset

from pimm.utils.logger import get_root_logger

from ..builder import DATASETS
from ..transform import Compose
from .decode import decode_event
from .overlay import PILArNetOverlayMixin

# Splits are named "train"/"validation"/"test" in the HF parquet export; accept
# the H5 reader's "val" as an alias so a config can swap readers without edits.
_HF_SPLIT_ALIASES = {"val": "validation", "valid": "validation"}


def resolve_parquet_data_files(
    split: str,
    repo_id: str | None = None,
    data_root: str | None = None,
    parquet_revision: str = "refs/convert/parquet",
    config_name: str = "default",
):
    """Resolve the parquet file(s) for a split into a ``load_dataset`` spec.

    Prefers a local ``data_root`` (``<config>/<split>/*.parquet`` and looser
    fallbacks); otherwise builds an ``hf://`` glob against ``repo_id`` on the
    auto-converted parquet ref. Returns a string glob (or sorted list of local
    paths) suitable for ``load_dataset("parquet", data_files=...)``.
    """
    hf_split = _HF_SPLIT_ALIASES.get(split, split)
    if data_root is not None:
        patterns = (
            f"{config_name}/{hf_split}/*.parquet",
            f"{hf_split}/*.parquet",
            f"*{hf_split}*/*.parquet",
            f"*{hf_split}*.parquet",
        )
        for pat in patterns:
            files = sorted(glob.glob(os.path.join(data_root, pat)))
            if files:
                return files
        raise FileNotFoundError(
            f"No parquet files for split '{split}' under {data_root} "
            f"(tried patterns: {patterns})"
        )
    if repo_id is None:
        raise ValueError("Provide either data_root or repo_id for parquet loading")
    return (
        f"hf://datasets/{repo_id}@{parquet_revision}/"
        f"{config_name}/{hf_split}/*.parquet"
    )


def _event_dict_from_row(row, revision, *, energy_threshold,
                         remove_low_energy_scatters, old_pid_mapping,
                         name, split):
    """Decode one parquet row into the standard PILArNet ``data_dict``."""
    data_dict = decode_event(
        point=np.asarray(row["point"]),
        cluster=np.asarray(row["cluster"]),
        cluster_extra=(
            np.asarray(row["cluster_extra"]) if revision != "v1" else None
        ),
        revision=revision,
        energy_threshold=energy_threshold,
        remove_low_energy_scatters=remove_low_energy_scatters,
        old_pid_mapping=old_pid_mapping,
    )
    data_dict["name"] = name
    data_dict["split"] = split
    data_dict["revision"] = revision
    return data_dict


@DATASETS.register_module()
class PILArNetParquetDataset(PILArNetOverlayMixin, Dataset):
    """Map-style PILArNet reader backed by parquet (Arrow-mmap).

    Reads the same flat ``point``/``cluster``/``cluster_extra`` layout as
    :class:`PILArNetH5Dataset` from parquet shards -- either the HF
    auto-converted ref of a dataset repo (``repo_id``) or a local directory of
    parquet files (``data_root``) -- and emits identical ``data_dict``
    dictionaries via the shared :func:`decode_event`. Intended for fine-tuning,
    evaluation, probes,
    and anywhere reproducible random access matters; for large-scale pretraining
    prefer :class:`PILArNetParquetIterableDataset` (sequential + shuffle buffer).

    The backing table is memory-mapped, so a local split is not resident in RAM.
    Event overlay is supported with the same ``overlay_*`` kwargs as the H5
    reader (both mix in :class:`PILArNetOverlayMixin`), so a config can swap the
    two readers freely. ``test_mode`` is not supported here (use the H5 reader
    for the voxelized/augmented test path); passing it raises.

    Args:
        repo_id (str | None): HF dataset repo whose auto-converted parquet ref is
            read when ``data_root`` is ``None``. Defaults to
            ``"DeepLearnPhysics/PILArNet-M-mini"``.
        data_root (str | None): Local directory of parquet shards; takes
            precedence over ``repo_id`` when set (and over the default via the
            ``PILARNET_PARQUET_ROOT_<REV>`` env var). Defaults to ``None``.
        split (str): Split name; ``"val"`` is accepted as an alias for
            ``"validation"``. Defaults to ``"train"``.
        transform (list[dict]): Transform configs (NOT a prebuilt ``Compose``).
        revision ({"v1","v2","v3"}): Cluster/extra column layout. Defaults to
            ``"v2"``.
        parquet_revision (str): Git ref carrying the parquet export. Defaults to
            ``"refs/convert/parquet"``.
        config_name (str): Parquet builder config subdir. Defaults to
            ``"default"``.
        min_points (int): Minimum points per event to keep. Defaults to ``1024``.

    See :class:`PILArNetH5Dataset` for the remaining shared arguments and the
    emitted dictionary schema.
    """

    def __init__(
        self,
        repo_id: str | None = "DeepLearnPhysics/PILArNet-M-mini",
        data_root: str | None = None,
        split="train",
        transform=None,
        revision: Literal["v1", "v2", "v3"] = "v2",
        parquet_revision: str = "refs/convert/parquet",
        config_name: str = "default",
        loop=1,
        ignore_index=-1,
        energy_threshold=0.0,
        min_points=1024,
        max_len=-1,
        remove_low_energy_scatters=False,
        old_pid_mapping=False,
        test_mode=False,
        test_cfg=None,
        # event overlay parameters (shared with PILArNetH5Dataset)
        overlay_n_events=1,
        overlay_prob=1.0,
        overlay_allow_repeats=True,
    ):
        super().__init__()
        if test_mode:
            raise NotImplementedError(
                "PILArNetParquetDataset does not support test_mode; use "
                "PILArNetH5Dataset for the voxelized/augmented test path."
            )
        from datasets import load_dataset

        self.repo_id = repo_id
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.revision = revision
        self.loop = loop
        self.ignore_index = ignore_index
        self.energy_threshold = energy_threshold
        self.min_points = min_points
        self.max_len = max_len
        self.remove_low_energy_scatters = remove_low_energy_scatters
        self.old_pid_mapping = old_pid_mapping
        # event overlay parameters
        self.overlay_n_events = overlay_n_events
        self.overlay_prob = overlay_prob
        self.overlay_allow_repeats = overlay_allow_repeats

        # data_root precedence mirrors PILArNetH5Dataset: explicit arg >
        # PILARNET_PARQUET_ROOT_<REV> env var > repo_id (hf auto-parquet). The
        # env var lets a staged local root (e.g. on /lscratch) win over the
        # remote default without editing configs.
        if data_root is None:
            data_root = os.environ.get(f"PILARNET_PARQUET_ROOT_{revision.upper()}")
            self.data_root = data_root

        data_files = resolve_parquet_data_files(
            split, repo_id=repo_id, data_root=data_root,
            parquet_revision=parquet_revision, config_name=config_name,
        )
        hf_split = _HF_SPLIT_ALIASES.get(split, split)
        self.table = load_dataset(
            "parquet", data_files={hf_split: data_files}, split=hf_split
        )

        self._build_index()

        logger = get_root_logger()
        logger.info(
            "Total number of samples in PILArNet(parquet) {} set: {} x {}.".format(
                self.cumulative_length, self.loop, split
            )
        )

    def _build_index(self):
        """Filter to events with at least ``min_points`` points."""
        npoints = self._point_counts()
        self.index = np.argwhere(npoints >= self.min_points).flatten()
        self.cumulative_length = int(self.index.shape[0])

    def _num_source_events(self):
        """Count of distinct events (pre-``loop``); overlay samples from this."""
        return self.cumulative_length

    def _point_counts(self):
        """Per-event point count. Prefers the scalar ``n_points`` column (our
        converter writes it); falls back to the ``point`` list offsets
        (``element_count // 8``) for the HF export that lacks it."""
        import pyarrow.compute as pc

        tbl = self.table.data
        if "n_points" in tbl.column_names:
            return np.asarray(tbl.column("n_points"), dtype=np.int64)
        col = tbl.column("point")
        return np.asarray(pc.list_value_length(col), dtype=np.int64) // 8

    def get_data(self, idx):
        row_idx = int(self.index[idx])
        row = self.table[row_idx]
        name = row.get("event_id") or f"{self.split}_{row_idx}"
        return _event_dict_from_row(
            row,
            self.revision,
            energy_threshold=self.energy_threshold,
            remove_low_energy_scatters=self.remove_low_energy_scatters,
            old_pid_mapping=self.old_pid_mapping,
            name=name,
            split=self.split,
        )

    def get_data_name(self, idx):
        row_idx = int(self.index[idx])
        tbl = self.table.data
        if "event_id" in tbl.column_names:
            return tbl.column("event_id")[row_idx].as_py()
        return f"{self.split}_{row_idx}"

    def __getitem__(self, idx):
        real_idx = idx % len(self)
        data_dict = self.get_data(real_idx)
        data_dict = self._maybe_overlay(data_dict)
        return self.transform(data_dict)

    def __len__(self):
        length = self.cumulative_length
        if self.max_len > 0:
            length = min(self.max_len, length)
        return length * self.loop


@DATASETS.register_module()
class PILArNetParquetIterableDataset(IterableDataset):
    """Streaming PILArNet reader for large-scale pretraining.

    Streams parquet shards sequentially (no random single-row seeks -- the
    access pattern parallel/network filesystems reward) and approximates a full
    shuffle with shard-order shuffling plus a reservoir ``buffer_size`` buffer.
    Shards are split across DDP ranks (``split_dataset_by_node``) and DataLoader
    workers automatically. Emits the same ``data_dict`` dictionaries as the
    map-style reader via the shared :func:`decode_event`.

    Note:
        This is a PyTorch ``IterableDataset`` -- wiring it into pimm's training
        engine (which currently assumes map-style datasets with samplers/length)
        is a separate integration step and has not yet been exercised end to end.

    Args:
        shuffle (bool): Enable buffer + shard-order shuffle. Defaults to ``True``.
        shuffle_buffer_size (int): Reservoir buffer size for streaming shuffle.
            Defaults to ``1000``.
        seed (int): Base shuffle seed. Defaults to ``0``.

    See :class:`PILArNetParquetDataset` for the shared loading/decoding args.
    """

    def __init__(
        self,
        repo_id: str | None = "DeepLearnPhysics/PILArNet-M-mini",
        data_root: str | None = None,
        split="train",
        transform=None,
        revision: Literal["v1", "v2", "v3"] = "v2",
        parquet_revision: str = "refs/convert/parquet",
        config_name: str = "default",
        energy_threshold=0.0,
        min_points=1024,
        max_len=-1,
        remove_low_energy_scatters=False,
        old_pid_mapping=False,
        shuffle=True,
        shuffle_buffer_size=1000,
        seed=0,
        # Accepted for config-swap parity with the map-style readers; see notes.
        loop=1,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
    ):
        super().__init__()
        if test_mode:
            raise NotImplementedError(
                "PILArNetParquetIterableDataset does not support test_mode; use "
                "PILArNetH5Dataset for the voxelized/augmented test path."
            )
        self.repo_id = repo_id
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.revision = revision
        self.energy_threshold = energy_threshold
        self.min_points = min_points
        self.max_len = max_len
        self.remove_low_energy_scatters = remove_low_energy_scatters
        self.old_pid_mapping = old_pid_mapping
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.ignore_index = ignore_index
        self.epoch = 0
        # loop repeats a map-style epoch; a stream has no fixed length, so the
        # engine controls epoch length via iters_per_epoch instead. Warn rather
        # than silently ignore a non-default.
        if loop != 1:
            get_root_logger().warning(
                "PILArNetParquetIterableDataset ignores loop=%s (streaming); set "
                "`iters_per_epoch` in the config to control epoch length.",
                loop,
            )
        self.loop = 1
        if data_root is None:
            data_root = os.environ.get(f"PILARNET_PARQUET_ROOT_{revision.upper()}")
            self.data_root = data_root
        self._data_files = resolve_parquet_data_files(
            split, repo_id=repo_id, data_root=data_root,
            parquet_revision=parquet_revision, config_name=config_name,
        )
        self._hf_split = _HF_SPLIT_ALIASES.get(split, split)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so the next ``__iter__`` reshuffles (engine contract)."""
        self.epoch = int(epoch)

    def num_samples(self) -> int:
        """Count events with ``>= min_points``, read cheaply from the parquet
        ``n_points`` column. The engine uses this to size ``iters_per_epoch``
        when the config does not set it. Requires a local parquet root and the
        ``n_points`` column (written by the offline h5 -> parquet converter)."""
        import glob as _glob

        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        files = self._data_files
        if isinstance(files, str):
            files = sorted(_glob.glob(files))
        if not files:
            raise RuntimeError(
                "num_samples() needs a local parquet root; set `iters_per_epoch` "
                "in the config for a remote/streaming source instead."
            )
        total = 0
        for f in files:
            col = pq.read_table(f, columns=["n_points"]).column("n_points")
            total += int(pc.sum(pc.greater_equal(col, self.min_points)).as_py() or 0)
        if self.max_len > 0:
            total = min(total, self.max_len)
        return total

    def _make_stream(self):
        from datasets import load_dataset

        stream = load_dataset(
            "parquet",
            data_files={self._hf_split: self._data_files},
            split=self._hf_split,
            streaming=True,
        )
        if self.shuffle:
            stream = stream.shuffle(
                seed=self.seed, buffer_size=self.shuffle_buffer_size
            )
        # Per-epoch reshuffle/reshard: the engine calls set_epoch each epoch
        # (via set_dataloader_epoch), and HF folds the epoch into the shuffle.
        stream.set_epoch(self.epoch)
        # Shard across DDP ranks by file so each rank streams disjoint shards.
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            from datasets.distributed import split_dataset_by_node

            stream = split_dataset_by_node(
                stream, rank=dist.get_rank(), world_size=dist.get_world_size()
            )
        # Then shard remaining work across this rank's DataLoader workers.
        import torch

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            stream = stream.shard(
                num_shards=worker_info.num_workers, index=worker_info.id
            )
        return stream

    def __iter__(self):
        for i, row in enumerate(self._make_stream()):
            n_points = row.get("n_points")
            if n_points is None:
                n_points = len(row["point"]) // 8
            if n_points < self.min_points:
                continue
            data_dict = _event_dict_from_row(
                row,
                self.revision,
                energy_threshold=self.energy_threshold,
                remove_low_energy_scatters=self.remove_low_energy_scatters,
                old_pid_mapping=self.old_pid_mapping,
                name=row.get("event_id") or f"{self.split}_stream_{i}",
                split=self.split,
            )
            yield self.transform(data_dict)
