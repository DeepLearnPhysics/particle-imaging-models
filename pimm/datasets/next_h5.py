"""NEXT high-pressure gas-TPC event dataset.

The NEXT AI-summer-school release contains 0νββ signal (``0nubb_part_*``) and
²¹⁴Bi background (``Bi_part_*``) tarballs. Each extracts to ~500–1000 small
PyTables/pandas-``HDFStore`` HDF5 files; each file bundles ~65–95 events stored
**contiguously** in one flat compound table at ``MC/hits/table`` with pandas'
blocked column layout:

  - ``values_block_0`` (int64, 1 col): ``event_id``
  - ``values_block_1`` (float32, 4 cols): ``x, y, z`` [mm], ``energy`` [MeV]
  - ``values_block_2`` (bytes, 1 col): ``label`` (``b"Signal"`` / ``b"Bkg"``)

Hits sit on an exact 3 mm lattice inside a ~3.6 m cylinder, energy is quantized
in 24.8 eV steps (one ionization electron), and per-event energy is already cut
to a narrow window at Qββ, so classes differ only in track topology.

The reader is parsed directly with ``h5py`` (no ``pytables`` dependency). It
mirrors :class:`~pimm.datasets.pilarnet.PILArNetH5Dataset` for the flat
per-point dict + ``Compose`` transform pipeline. Per-source seeded holdouts
and the ``datasets`` / ``data_list`` / ``split`` / ``config_id`` contract
support the event-level linear-probe evaluators.

Coordinates and energies are on a totally different scale than PILArNet-M, so
configs MUST re-derive ``NormalizeCoord`` / ``GridSample`` / ``LogTransform``
(see ``configs/panda/v2/next-aiss/pretrain-panda-v2m1-0-base.py``).
"""

from __future__ import annotations

import glob
import hashlib
import inspect
import os
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import h5py
import numpy as np
from torch.utils.data import Dataset

from pimm.utils.logger import get_root_logger

from .builder import DATASETS
from .transform import Compose

HITS_TABLE_KEY = "MC/hits/table"
LABEL_TO_INT = {"Bkg": 0, "Signal": 1}
LABEL_NAMES = ("Bkg", "Signal")


def _decode_label_column(raw):
    """Decode the fixed-width bytes label column to Python strings."""
    return np.char.decode(np.asarray(raw).astype("S"), "utf-8")


def decode_next_hits(rows):
    """Turn a slice of the ``MC/hits/table`` compound array into a flat dict.

    Args:
        rows (np.ndarray): Structured array slice with fields
            ``values_block_0`` (event id), ``values_block_1`` (x, y, z, energy)
            and ``values_block_2`` (label).

    Returns:
        dict: ``coord`` (N, 3) float32 in mm, ``energy`` (N, 1) float32 in MeV.

    Example:
        >>> import numpy as np
        >>> dt = np.dtype([("index", "<i8"), ("values_block_0", "<i8", (1,)),
        ...                ("values_block_1", "<f4", (4,)), ("values_block_2", "S6", (1,))])
        >>> rows = np.zeros(2, dtype=dt)
        >>> rows["values_block_1"] = [[1, 2, 3, 0.5], [4, 5, 6, 0.25]]
        >>> out = decode_next_hits(rows)
        >>> out["coord"].shape, out["energy"].shape, out["energy"].dtype
        ((2, 3), (2, 1), dtype('float32'))
    """
    values = np.asarray(rows["values_block_1"], dtype=np.float32)
    coord = np.ascontiguousarray(values[:, :3])
    energy = np.ascontiguousarray(values[:, 3:4])
    return {"coord": coord, "energy": energy}


def index_next_file(path):
    """Enumerate the contiguous per-event row ranges of one NEXT HDF5 file.

    Reads only the ``event_id`` and ``label`` columns. Events are stored as
    contiguous runs of rows (verified on the released data); a file that
    violates this raises so a silent mis-slice can never happen.

    Returns:
        dict of 1-D arrays: ``row_start`` (int64), ``n_hits`` (int32),
        ``event_id`` (int64), ``label`` (int8; see ``LABEL_TO_INT``).
    """
    with h5py.File(path, "r", libver="latest", swmr=True, locking=False) as f:
        table = f[HITS_TABLE_KEY]
        event_id = np.asarray(table.fields("values_block_0")[:]).reshape(-1)
        label_raw = np.asarray(table.fields("values_block_2")[:]).reshape(-1)
    event_id = event_id.astype(np.int64, copy=False)
    n_rows = event_id.shape[0]
    if n_rows == 0:
        empty = dict(
            row_start=np.zeros(0, np.int64),
            n_hits=np.zeros(0, np.int32),
            event_id=np.zeros(0, np.int64),
            label=np.zeros(0, np.int8),
        )
        return empty

    changes = np.flatnonzero(np.diff(event_id) != 0) + 1
    row_start = np.concatenate(([0], changes)).astype(np.int64)
    row_end = np.concatenate((changes, [n_rows])).astype(np.int64)
    run_ids = event_id[row_start]
    if np.unique(run_ids).shape[0] != run_ids.shape[0]:
        raise ValueError(
            f"{path}: events are not stored contiguously in {HITS_TABLE_KEY}"
        )

    labels_str = _decode_label_column(label_raw)
    label = np.empty(row_start.shape[0], dtype=np.int8)
    for i, (a, b) in enumerate(zip(row_start, row_end)):
        seen = np.unique(labels_str[a:b])
        if seen.shape[0] != 1:
            raise ValueError(
                f"{path}: event {int(run_ids[i])} has mixed labels {seen.tolist()}"
            )
        name = str(seen[0])
        if name not in LABEL_TO_INT:
            raise ValueError(
                f"{path}: unknown label {name!r}; expected one of {LABEL_NAMES}"
            )
        label[i] = LABEL_TO_INT[name]

    return dict(
        row_start=row_start,
        n_hits=(row_end - row_start).astype(np.int32),
        event_id=run_ids,
        label=label,
    )


def _filter_file(path, starts, counts, cfg):
    """Which of one file's events (row ``starts`` / ``counts``) pass the ``event_filter`` pipeline ``cfg``."""
    keep = Compose(cfg)
    lo, hi = int(starts.min()), int((starts + counts).max())
    with h5py.File(path, "r", libver="latest", swmr=True, locking=False) as handle:
        rows = handle[HITS_TABLE_KEY][lo:hi]
    return np.array(
        [
            keep(decode_next_hits(rows[s - lo : s - lo + c])) is not None
            for s, c in zip(starts, counts)
        ],
        dtype=np.int8,
    )


def _read_hits(path, starts, counts):
    """Hits (x, y, z, energy rows) of one file's events, concatenated in the given order."""
    lo, hi = int(starts.min()), int((starts + counts).max())
    with h5py.File(path, "r", libver="latest", swmr=True, locking=False) as handle:
        rows = np.asarray(
            handle[HITS_TABLE_KEY][lo:hi]["values_block_1"], dtype=np.float32
        )
    return np.concatenate([rows[s - lo : s - lo + c] for s, c in zip(starts, counts)])


class _LazyH5FileList:
    """Positionally indexed, bounded LRU cache of read-only HDF5 handles."""

    def __init__(self, files, max_open=8):
        self.files = list(files)
        self.max_open = int(max_open)
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, file_idx):
        file_idx = int(file_idx)
        if file_idx < 0:
            file_idx += len(self.files)
        if file_idx < 0 or file_idx >= len(self.files):
            raise IndexError(file_idx)
        handle = self._cache.pop(file_idx, None)
        if handle is None:
            handle = h5py.File(
                self.files[file_idx],
                "r",
                libver="latest",
                swmr=True,
                locking=False,
            )
        self._cache[file_idx] = handle
        while len(self._cache) > self.max_open:
            _, evicted = self._cache.popitem(last=False)
            evicted.close()
        return handle

    def close(self):
        for handle in self._cache.values():
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        self._cache.clear()


@DATASETS.register_module()
class NEXTH5Dataset(Dataset):
    """NEXT gas-TPC signal/background events read from the released HDF5 files.

    Each item is a flat ``dict`` with ``coord`` (N, 3 float32, mm), ``energy``
    (N, 1 float32, MeV), ``event_label`` (1,) int64 (the source's ``label``),
    ``config_id`` (1,) int64 (position of the source in ``configs``), plus
    ``name`` / ``split``. After collation a batch adds ``offset``. Registered as
    ``NEXTH5Dataset`` -- use as ``type`` under ``data.train`` / ``data.val``.

    Args:
        data_root (str): Directory containing the extracted ``<part>/`` folders
            (e.g. ``.../next/extracted`` holding ``0nubb_part_8/``, ``Bi_part_10/``).
        configs (Sequence): Source names or mappings. Mapping entries may set
            ``name``, ``label``, ``pattern`` (glob of part
            directories under ``data_root``; defaults to ``"<name>*"``),
            ``data_root`` (override), and ``max_events``. A bare string is a
            name with ``label = list position``.
        split (str): ``"train"`` uses all events except the holdout;
            ``"holdout"`` / ``"val"`` use the holdout events;
            ``"all"`` uses everything. Defaults to ``"train"``.
        holdout_events_per_config (int | None): Events reserved per source
            before the train/holdout split. Defaults to ``0``.
        holdout_seed (int): Seed for the random holdout permutation (offset by
            source index). Defaults to ``0``.
        max_events_per_config (int): Global cap on events kept per source
            (-1 = no cap); a per-source ``max_events`` overrides it.
            Defaults to ``-1``.
        min_points (int): Keep only events with at least this many hits (uses
            the cached per-event count; no I/O). Defaults to ``0``.
        subset (tuple | None): ``(seed, start, stop)``: keep the events at
            positions ``[start, stop)`` of ``np.random.default_rng(seed)``'s
            permutation of the selected events, in that order (a pretraining
            pool is ``(0, 0, K)``; the events no pool reached, ``(0, K, None)``).
            Defaults to ``None``.
        event_filter (list[dict] | None): Transform configs run on each raw
            event before ``transform``; events for which they return ``None``
            are dropped (e.g. ``NEXTTopologyCut``). The verdicts are cached next
            to the event index. Defaults to ``None``.
        transform (list[dict] | None): List of transform configs (NOT a prebuilt
            ``Compose``). ``None`` / ``[]`` means no preprocessing.
            Defaults to ``None``.
        max_per_source (int | tuple): After ``subset`` and ``event_filter``,
            keep the first this many events of each source, in order, or the
            events at positions ``[start, stop)`` of each source (a label draw);
            -1: all. Defaults to ``-1``.
        max_len (int): Then keep the first this many events overall (-1: all).
            Defaults to ``-1``.
        preload (bool): Read every selected event's hits into one array at
            construction; forked loader workers share it, so a read is a slice,
            not a file open (the release is ~15K small files). About 16 bytes a
            hit. Defaults to ``False``.
        loop (int): Epoch multiplier. Defaults to ``1``.
        index_cache_dir (str | None): Directory for the per-source event-index
            cache; falls back to ``$PIMM_NEXT_INDEX_CACHE`` then
            ``~/.cache/pimm/next_event_index``. Defaults to ``None``.
        index_workers (int): Threads used for the cold index build.
            Defaults to ``8``.
        max_open_files (int): Per-worker bound on simultaneously open HDF5
            handles. Defaults to ``8``.
        test_mode (bool): Unsupported; raises ``NotImplementedError`` when
            ``True``. Defaults to ``False``.

    Note:
        The random holdout is seeded by ``holdout_seed + source_idx``, so a
        source shared between the train and an eval/probe dataset must sit at
        the same list index in both, or holdout events leak into training (the
        probe evaluators' leakage guard will refuse to run). The per-source
        index cache is keyed on every file's size + mtime, so it
        self-invalidates when parts are re-extracted or added.

    Example:
        .. code-block:: python

            >>> from pimm.datasets.builder import build_dataset
            >>> ds = build_dataset(dict(                              # doctest: +SKIP
            ...     type="NEXTH5Dataset", data_root="/data/next/extracted",
            ...     configs=[dict(name="Bi", label=0, pattern="Bi_part_*"),
            ...              dict(name="0nubb", label=1, pattern="0nubb_part_*")],
            ...     split="train", holdout_events_per_config=5000, transform=[]))
            >>> sample = ds[0]                                        # doctest: +SKIP
            >>> # sample keys: coord (N, 3), energy (N, 1), event_label (1,),
            >>> #   config_id (1,), name, split
    """

    _HOLDOUT_SPLITS = {"holdout", "val"}

    def __init__(
        self,
        data_root,
        configs,
        split="train",
        holdout_events_per_config=0,
        holdout_seed=0,
        max_events_per_config=-1,
        min_points=0,
        subset=None,
        event_filter=None,
        transform=None,
        max_per_source=-1,
        max_len=-1,
        preload=False,
        loop=1,
        index_cache_dir=None,
        index_workers=8,
        max_open_files=8,
        test_mode=False,
    ):
        super().__init__()
        if test_mode:
            raise NotImplementedError(
                "NEXTH5Dataset is intended for train/val loaders, not "
                "fragmented test-time inference."
            )
        if not configs:
            raise ValueError("configs must contain at least one NEXT source")

        self.data_root = data_root
        self.config_specs = [
            self._normalize_config_spec(cfg, default_label=i)
            for i, cfg in enumerate(configs)
        ]
        self.split = split
        self.holdout_events_per_config = holdout_events_per_config
        self.holdout_seed = int(holdout_seed)
        self.max_events_per_config = int(max_events_per_config)
        self.min_points = int(min_points)
        self.transform = Compose(transform or [])
        self.loop = int(loop)
        self.index_cache_dir = (
            index_cache_dir
            or os.environ.get("PIMM_NEXT_INDEX_CACHE")
            or os.path.join(
                os.path.expanduser("~"), ".cache", "pimm", "next_event_index"
            )
        )
        self.index_workers = max(1, int(index_workers))
        self.max_open_files = max(1, int(max_open_files))
        self.test_mode = False

        self.datasets = self._build_sources()
        self.data_list = self._build_data_list()
        if subset is not None:
            seed, start, stop = subset
            order = np.random.default_rng(seed).permutation(len(self.data_list))
            self.data_list = [self.data_list[i] for i in order[start:stop]]
        self.event_filter = Compose(event_filter or [])
        if event_filter:
            self._apply_event_filter(event_filter)
        if (
            max_per_source != -1
        ):  # a label budget: a block of each source's events, in order
            lo, hi = (
                max_per_source
                if isinstance(max_per_source, (tuple, list))
                else (0, max_per_source)
            )
            seen = [0] * len(self.datasets)
            keep = []
            for s, e in self.data_list:
                if lo <= seen[s] < hi:
                    keep.append((s, e))
                seen[s] += 1
            self.data_list = keep
        if max_len > 0:
            self.data_list = self.data_list[:max_len]
        self._hits = self._preload() if preload else None

        # per-worker handle caches, created lazily (fork-safe)
        self._handles = None
        self._pid = None

        get_root_logger().info(
            "NEXTH5Dataset: %d events x %d loop, split=%s, sources=%s",
            len(self.data_list),
            self.loop,
            self.split,
            [spec["name"] for spec in self.config_specs],
        )

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _normalize_config_spec(config, default_label):
        if isinstance(config, str):
            return {
                "name": config,
                "label": int(default_label),
                "pattern": f"{config}*",
                "data_root": None,
                "max_events": None,
            }
        if not isinstance(config, Mapping):
            raise TypeError(
                "Each config must be a string or mapping, got "
                f"{type(config).__name__}"
            )
        name = config.get("name")
        if name is None:
            raise ValueError("Config mapping must provide 'name'")
        max_events = config.get("max_events", None)
        return {
            "name": str(name),
            "label": int(config.get("label", default_label)),
            "pattern": str(config.get("pattern", f"{name}*")),
            "data_root": config.get("data_root"),
            "max_events": (int(max_events) if max_events is not None else None),
        }

    def _discover_files(self, spec):
        root = spec["data_root"] or self.data_root
        pattern = os.path.join(root, spec["pattern"], "*.h5")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"NEXTH5Dataset: no HDF5 files match {pattern} for source "
                f"{spec['name']!r}"
            )
        return root, files

    def _build_sources(self):
        sources = []
        for spec in self.config_specs:
            root, files = self._discover_files(spec)
            stats = self._parallel_stat(files)
            index = self._load_or_build_index(spec["name"], files, stats)
            sources.append(
                {
                    "name": spec["name"],
                    "label": spec["label"],
                    # probe evaluators key sources on realpath(source_root)
                    "source_root": os.path.join(root, spec["name"]),
                    "pattern": spec["pattern"],
                    "files": files,
                    "index": index,
                    "index_path": self._index_cache_path(spec["name"], files, stats),
                    "max_events": spec["max_events"],
                }
            )
        return sources

    @staticmethod
    def _parallel_stat(files):
        if not files:
            return []
        with ThreadPoolExecutor(max_workers=min(8, len(files))) as pool:
            return list(pool.map(os.stat, files))

    def _index_cache_path(self, name, files, stats):
        parts = [
            f"{os.path.realpath(p)}:{st.st_size}:{int(st.st_mtime)}"
            for p, st in zip(files, stats)
        ]
        digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
        return os.path.join(
            self.index_cache_dir, f"{name}_{len(files)}files_{digest}.npz"
        )

    _INDEX_KEYS = ("file_idx", "row_start", "n_hits", "event_id", "label")

    def _load_or_build_index(self, name, files, stats):
        """Per-source flat event index, cached as an ``.npz`` sidecar."""
        logger = get_root_logger()
        path = self._index_cache_path(name, files, stats)
        if os.path.isfile(path):
            try:
                with np.load(path, allow_pickle=False) as sidecar:
                    if set(self._INDEX_KEYS) <= set(sidecar.files):
                        index = {k: sidecar[k] for k in self._INDEX_KEYS}
                        logger.info(
                            "NEXTH5Dataset: loaded event index for %s "
                            "(%d events, %d files) from %s",
                            name,
                            index["row_start"].shape[0],
                            len(files),
                            path,
                        )
                        return index
                    logger.warning(
                        "NEXTH5Dataset: index cache %s missing keys, rebuilding",
                        path,
                    )
            except Exception as exc:  # noqa: BLE001 - corrupt cache -> rebuild
                logger.warning(
                    "NEXTH5Dataset: unreadable index cache %s (%s), rebuilding",
                    path,
                    exc,
                )

        logger.info(
            "NEXTH5Dataset: building event index for %s over %d files -> %s",
            name,
            len(files),
            path,
        )
        per_file = self._index_files_parallel(files)

        file_idx = np.concatenate(
            [
                np.full(entry["row_start"].shape[0], i, dtype=np.int32)
                for i, entry in enumerate(per_file)
            ]
        )
        index = {"file_idx": file_idx}
        for key in ("row_start", "n_hits", "event_id", "label"):
            index[key] = np.concatenate([entry[key] for entry in per_file])

        try:
            os.makedirs(self.index_cache_dir, exist_ok=True)
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "wb") as fh:
                np.savez(fh, **index)
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            logger.warning(
                "NEXTH5Dataset: could not write index cache %s (%s)", path, exc
            )
        return index

    def _index_files_parallel(self, files):
        # h5py serializes HDF5 calls under one global lock, so threads barely
        # overlap the per-file decode; processes do. Fall back to threads when
        # a process pool is unavailable.
        workers = min(self.index_workers, len(files))
        if workers > 1:
            try:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    return list(pool.map(index_next_file, files, chunksize=8))
            except (OSError, RuntimeError) as exc:
                get_root_logger().warning(
                    "NEXTH5Dataset: process pool unavailable (%s); indexing "
                    "with threads",
                    exc,
                )
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(index_next_file, files))

    # ------------------------------------------------------------------ split
    def _split_indices(self, source_idx, n_events):
        indices = np.arange(n_events, dtype=np.int64)
        holdout = max(0, min(int(self.holdout_events_per_config or 0), n_events))
        if holdout == 0:
            holdout_indices = indices[:0]
            train_indices = indices
        else:
            rng = np.random.default_rng(self.holdout_seed + source_idx)
            perm = rng.permutation(indices)
            holdout_indices = np.sort(perm[:holdout])
            train_indices = np.sort(perm[holdout:])

        if self.split == "train":
            return train_indices
        if self.split in self._HOLDOUT_SPLITS:
            return holdout_indices
        if self.split == "all":
            return indices
        raise ValueError(
            "split must be one of {'train', 'holdout', 'val', 'test', 'all'}, "
            f"got {self.split!r}"
        )

    def _build_data_list(self):
        data_list = []
        logger = get_root_logger()
        for source_idx, source in enumerate(self.datasets):
            n_events = int(source["index"]["row_start"].shape[0])
            selected = self._split_indices(source_idx, n_events)
            before_filter = len(selected)
            cap = source.get("max_events")
            if cap is None:
                cap = self.max_events_per_config
            if self.min_points > 0:
                n_hits = source["index"]["n_hits"]
                selected = selected[n_hits[selected] >= self.min_points]
                logger.info(
                    "NEXTH5Dataset: kept %d/%d %s events with >=%d hits",
                    len(selected),
                    before_filter,
                    source["name"],
                    self.min_points,
                )
            if cap > 0:
                selected = selected[:cap]
            data_list.extend((source_idx, int(event_idx)) for event_idx in selected)
        return data_list

    def _apply_event_filter(self, cfg):
        """Drop the events ``cfg`` rejects; per-source verdicts (-1 unknown, 0 drop, 1 keep) are cached and only
        unknown events are computed, one process per file."""
        tag = hashlib.sha1(
            (
                repr(cfg)
                + "".join(
                    inspect.getsource(Compose([c]).transforms[0].__class__) for c in cfg
                )
            ).encode()
        ).hexdigest()[
            :12
        ]  # config + code
        verdicts = []
        for s, source in enumerate(self.datasets):
            index, path = source["index"], source["index_path"].replace(
                ".npz", f"_filter_{tag}.npy"
            )
            flags = (
                np.load(path)
                if os.path.isfile(path)
                else np.full(len(index["row_start"]), -1, np.int8)
            )
            events = np.array(
                [e for src, e in self.data_list if src == s], dtype=np.int64
            )
            todo = np.unique(events[flags[events] < 0])
            if len(todo):
                by_file = np.split(
                    todo, np.flatnonzero(np.diff(index["file_idx"][todo])) + 1
                )
                with ProcessPoolExecutor(max_workers=self.index_workers) as pool:
                    jobs = [
                        pool.submit(
                            _filter_file,
                            source["files"][int(index["file_idx"][ev[0]])],
                            index["row_start"][ev],
                            index["n_hits"][ev],
                            cfg,
                        )
                        for ev in by_file
                    ]
                    for ev, job in zip(by_file, jobs):
                        flags[ev] = job.result()
                if os.path.isfile(path):  # keep what another process cached meanwhile
                    flags = np.where(flags < 0, np.load(path), flags)
                tmp = f"{path[:-4]}.tmp{os.getpid()}.npy"
                np.save(tmp, flags)
                os.replace(tmp, path)
            kept = int((flags[events] == 1).sum())
            get_root_logger().info(
                "NEXTH5Dataset: event_filter kept %d/%d %s events",
                kept,
                len(events),
                source["name"],
            )
            verdicts.append(flags)
        self.data_list = [(s, e) for s, e in self.data_list if verdicts[s][e] == 1]

    def _preload(self):
        """(hits [total, 4], start per data_list entry), read one process per file."""
        groups = {}
        for i, (s, e) in enumerate(self.data_list):
            groups.setdefault(
                (s, int(self.datasets[s]["index"]["file_idx"][e])), []
            ).append(i)
        keys = list(groups)
        with ProcessPoolExecutor(max_workers=self.index_workers) as pool:
            jobs = []
            for s, f in keys:
                ev = np.array([self.data_list[i][1] for i in groups[(s, f)]])
                index = self.datasets[s]["index"]
                jobs.append(
                    pool.submit(
                        _read_hits,
                        self.datasets[s]["files"][f],
                        index["row_start"][ev],
                        index["n_hits"][ev],
                    )
                )
            parts = [job.result() for job in jobs]
        start, base = np.zeros(len(self.data_list), dtype=np.int64), 0
        for key, part in zip(keys, parts):
            for i in groups[key]:
                s, e = self.data_list[i]
                start[i] = base
                base += int(self.datasets[s]["index"]["n_hits"][e])
        get_root_logger().info(
            "NEXTH5Dataset: preloaded %d events, %d hits", len(start), base
        )
        return np.concatenate(parts), start

    # ------------------------------------------------------------------- read
    def _worker_handles(self):
        pid = os.getpid()
        if self._handles is None or self._pid != pid:
            self._handles = [
                _LazyH5FileList(source["files"], max_open=self.max_open_files)
                for source in self.datasets
            ]
            self._pid = pid
        return self._handles

    def get_data(self, idx):
        source_idx, event_idx = self.data_list[idx % len(self.data_list)]
        source = self.datasets[source_idx]
        index = source["index"]
        file_idx = int(index["file_idx"][event_idx])
        start = int(index["row_start"][event_idx])
        n_hits = int(index["n_hits"][event_idx])

        if self._hits is not None:
            hits, first = self._hits[0], int(self._hits[1][idx % len(self.data_list)])
            values = hits[first : first + n_hits]
            data = {
                "coord": np.ascontiguousarray(values[:, :3]),
                "energy": np.ascontiguousarray(values[:, 3:4]),
            }
        else:
            handle = self._worker_handles()[source_idx][file_idx]
            data = decode_next_hits(handle[HITS_TABLE_KEY][start : start + n_hits])
        data["event_label"] = np.array([source["label"]], dtype=np.int64)
        data["config_id"] = np.array([source_idx], dtype=np.int64)
        data["name"] = self.get_data_name(idx)
        data["split"] = self.split
        return data

    def get_data_name(self, idx):
        source_idx, event_idx = self.data_list[idx % len(self.data_list)]
        source = self.datasets[source_idx]
        index = source["index"]
        file_name = os.path.splitext(
            os.path.basename(source["files"][int(index["file_idx"][event_idx])])
        )[0]
        return f'{source["name"]}/{file_name}_{int(index["event_id"][event_idx])}'

    def prepare_train_data(self, idx):
        return self.transform(
            self.get_data(idx)
        )  # data_list holds only events event_filter kept

    def __getitem__(self, idx):
        return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

    def __del__(self):
        handles = getattr(self, "_handles", None)
        if handles:
            for cache in handles:
                cache.close()
