"""PILArNet-M dataset read directly from clustered HDF5 shards."""

import glob
import os
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from torch.utils.data import Dataset

from pimm.utils.logger import get_root_logger

from ..builder import DATASETS
from ..transform import Compose
from .decode import decode_event, resolve_revision
from .overlay import PILArNetOverlayMixin
from .default import default_transform


@DATASETS.register_module()
class PILArNetH5Dataset(PILArNetOverlayMixin, Dataset):
    """PILArNet-M LArTPC dataset read directly from clustered HDF5 shards.

    Loads events straight from ``point``/``cluster``/``cluster_extra`` HDF5
    arrays (no per-event preprocessing), expands per-cluster truth to per-point
    arrays, and emits a flat ``dict``. ``v3_extra`` also consumes
    ``cluster_extra_2`` and ``interaction_extra``. Standard keys per item:
    ``coord`` (N, 3),
    ``energy`` (N, 1, raw), ``segment_motif`` (N, 1, semantic class),
    ``segment_pid`` (N, 1, PID), ``momentum`` (N, 1),
    ``vertex`` (N, 3, interaction vertex), ``is_primary`` (N, 1),
    ``instance_particle`` and ``instance_interaction`` (N, 1, remapped
    contiguous ids), ``segment_interaction`` (N, 1, background flag), plus
    ``name``/``split``/``revision``. After collation a batch adds ``offset``.
    Registered as ``PILArNetH5Dataset`` -- use as ``type`` under
    ``data.train``/``data.val``/``data.test``.

    Semantic classes (``segment_motif``): 0 shower, 1 track, 2 Michel, 3 delta,
    4 low-energy deposit. PID classes (``segment_pid``): 0 photon, 1 electron,
    2 muon, 3 pion, 4 proton, 5 none/LED (6 when ``old_pid_mapping``).

    Args:
        data_root (str | None): Root directory of the revision's HDF5 shards.
            When ``None``, falls back to the revision-specific
            ``$PILARNET_DATA_ROOT_V2``/``_V3``/``_V3_EXTRA`` variable, then to
            ``~/.cache/pimm/pilarnet/<revision>``; raises if none exist.
            Defaults to ``None``.
        split (str | Sequence[str]): Split name(s) used to glob ``*<split>/*.h5``
            under ``data_root``. Defaults to ``"train"``.
        transform (list[dict] | None): List of transform configs (NOT a prebuilt
            ``Compose``). ``None`` (the default) uses
            :func:`~pimm.datasets.pilarnet.default_transform`; ``[]`` means no
            preprocessing. Defaults to ``None``.
        test_mode (bool): Unsupported; must be ``False``. Defaults to ``False``.
        loop (int): Train-time epoch multiplier. Defaults to ``1``.
        ignore_index (int): Ignored-label value. Defaults to ``-1``.
        energy_threshold (float): Drop points with energy at or below this value
            when positive. Defaults to ``0.13``.
        min_points (int): Minimum points per event; smaller events are excluded
            from the index. Defaults to ``1024``.
        max_points (int | None): Maximum points per event; larger events are
            excluded from the index.
            Defaults to ``None`` (no cap).
        max_len (int): Cap on event count before the loop multiplier (-1 = no
            cap). Defaults to ``-1``.
        subset_seed (int | None): With ``max_len``, keep the events at positions
            ``default_rng(subset_seed).permutation(N)[:max_len]`` of the index instead of the first
            ``max_len`` so pretraining and downstream pools can share the draw.
            Overlay donors come from the same events. Defaults to ``None`` (the first ``max_len``).
        remove_low_energy_scatters (bool): Drop the first (LED scatter) cluster
            and its points. Defaults to ``False``.
        old_pid_mapping (bool): Map LED PID to ``6`` instead of ``5``. Defaults
            to ``False``.
        revision ({"v2", "v3", "v3_extra"}): On-disk truth schema. ``v2``
            uses the historical width-5 cluster-extra table; ``v3_extra``
            additionally requires momentum/lineage and interaction-vertex
            tables. Defaults to ``"v3"``.
        overlay_n_events (int | tuple[int, int]): Number (or inclusive range) of
            events to overlay into one point cloud; ``> 1`` enables overlay.
            Defaults to ``1``.
        overlay_prob (float): Probability of applying overlay to a given sample.
            Defaults to ``1.0``.
        overlay_rotate_90 (bool): Rotate each overlaid event by random 90-degree
            increments about the LAr detector centre. Defaults to ``True``.
        overlay_dedup (bool): Deduplicate colliding voxels by semantic priority;
            when ``False``, coincident points are left for ``GridSample`` to merge.
            Defaults to ``True``.
        overlay_max_instances (dict[str, int] | None): Largest number of
            instances per key (e.g. ``instance_particle``) an overlaid event may
            hold; donors that would exceed it are redrawn. Defaults to ``None``.

    Note:
        Loader settings (``batch_size``, ``num_worker``) live at the top level of
        the config, not on the dataset constructor. Split membership differs
        between v1 and v3, so numbers from a v1-era run are not comparable to a
        v3 run on the same split name. Event overlay deduplicates
        colliding voxels by semantic priority (track > shower > Michel > delta >
        LED) and rotates overlaid events by random 90-degree increments, unless
        ``overlay_dedup`` / ``overlay_rotate_90`` turn that off. Donors come from the
        selected ``max_len`` events.

    Example:
        .. code-block:: python

            >>> from pimm.datasets.builder import build_dataset
            >>> ds = build_dataset(dict(type="PILArNetH5Dataset", revision="v3",
            ...                         split="train", transform=[], min_points=1024))
            >>> sample = ds[0]
            >>> sorted(sample)[:6]
            ['coord', 'energy', 'instance_interaction', 'instance_particle', 'momentum', 'name']
            >>> sample["coord"].shape          # (N, 3) float32
            (7366, 3)
            >>> sample["segment_motif"].shape  # (N, 1) semantic class
            (7366, 1)
            >>> # in a config:
            >>> # data = dict(train=dict(type="PILArNetH5Dataset", split="train",
            >>> #             revision="v3", min_points=1024, transform=transform))
    """

    def __init__(
        self,
        data_root: str | None = None,
        split="train",
        transform=None,
        test_mode=False,
        loop=1,
        ignore_index=-1,
        energy_threshold=0.13,
        min_points=1024,
        max_len=-1,
        remove_low_energy_scatters=False,
        old_pid_mapping=False,
        revision: Literal["v2", "v3", "v3_extra"] = "v3",
        # event overlay parameters
        overlay_n_events=1,
        overlay_prob=1.0,
        overlay_rotate_90=True,
        overlay_dedup=True,
        overlay_max_instances=None,
        max_points=None,
        subset_seed=None,
    ):
        super().__init__()
        revision = resolve_revision(revision)
        self.data_root = data_root
        if self.data_root is None:
            env_var = f"PILARNET_DATA_ROOT_{revision.upper()}"
            # Revision-specific env vars keep each data root independent.
            self.data_root = os.environ.get(env_var)
        if self.data_root is None:
            # Fall back to the default download location
            default_path = str(Path.home() / ".cache" / "pimm" / "pilarnet" / revision)
            if os.path.isdir(default_path):
                self.data_root = default_path
            else:
                raise RuntimeError(
                    f"\nPILArNet data root not found for revision '{revision}'.\n\n"
                    f"Option 1 - Download the dataset (saves to ~/.cache/pimm/pilarnet/{revision}):\n"
                    f"    python scripts/pilarnet/download.py --revision {revision} "
                    "--split test\n\n"
                    f"Option 2 - Set the environment variable:\n"
                    f'    export {env_var}="/path/to/pilarnet/{revision}/data"\n\n'
                    f"Option 3 - Pass data_root directly in your config:\n"
                    f'    --options data.train.data_root="/path/to/data"\n'
                )
        self.split = split
        self.transform = Compose(
            default_transform() if transform is None else transform
        )
        assert not test_mode, "PILArNetH5Dataset has no test mode"
        self.loop = loop
        self.ignore_index = ignore_index
        self.old_pid_mapping = old_pid_mapping

        self.revision = revision
        # event overlay parameters
        self.overlay_n_events = overlay_n_events
        self.overlay_prob = overlay_prob
        self.overlay_rotate_90 = overlay_rotate_90
        self.overlay_dedup = overlay_dedup
        self.overlay_max_instances = overlay_max_instances

        # PILArNet specific parameters
        self.energy_threshold = energy_threshold
        self.min_points = min_points
        self.max_points = max_points
        self.remove_low_energy_scatters = remove_low_energy_scatters
        self.max_len = max_len
        # Get list of h5 files
        self.h5_files = self.get_h5_files()
        assert len(self.h5_files) > 0, "No h5 files found"
        self.initted = False
        self.file_events = []

        # Build index for faster access
        self._build_index()
        self.subset = None
        if subset_seed is not None:
            assert max_len > 0, "subset_seed draws max_len events"
            total = int(self.cumulative_lengths[-1])
            self.subset = np.random.default_rng(subset_seed).permutation(total)[:max_len]

        logger = get_root_logger()
        logger.info(
            "Total number of samples in PILArNet {} set: {} x {}.".format(
                self.cumulative_lengths[-1], self.loop, split
            )
        )
        if self._overlay_enabled():
            logger.info(f"Event overlay enabled: n_events={self.overlay_n_events}, prob={self.overlay_prob}")

    def get_h5_files(self):
        """Get list of h5 files based on the split."""
        if isinstance(self.split, str):
            split_pattern = f"*{self.split}/*.h5"
        else:
            split_pattern = [f"*{s}/*.h5" for s in self.split]

        if isinstance(split_pattern, list):
            h5_files = []
            for pattern in split_pattern:
                h5_files.extend(sorted(glob.glob(os.path.join(self.data_root, pattern))))
        else:
            h5_files = sorted(glob.glob(os.path.join(self.data_root, split_pattern)))

        return sorted(h5_files)

    def _build_index(self):
        """Build an index of valid point clouds for faster access."""
        log = get_root_logger()
        log.info("Building index for PILArNetH5Dataset")

        self.cumulative_lengths = []
        self.indices = []

        for h5_file in self.h5_files:
            # Check if points count file exists
            points_file = h5_file.replace(".h5", "_points.npy")
            if os.path.exists(points_file):
                npoints = np.load(points_file)
                index = np.argwhere(self._kept(npoints)).flatten()
            else:
                # No points file, count on the fly
                log.info(
                    f"No points count file for {h5_file}, counting points on the fly"
                )
                with h5py.File(h5_file, "r", libver="latest", swmr=True) as f:
                    # Get all point counts
                    npoints = []
                    for i in range(f['point'].shape[0]):
                        npoint = f['point'][i].size // 8
                        npoints.append(npoint)
                    npoints = np.array(npoints)
                    index = np.argwhere(self._kept(npoints)).flatten()
                    self.file_events.append(npoints.shape[0])
            if os.path.exists(points_file):
                self.file_events.append(int(npoints.shape[0]))

            self.cumulative_lengths.append(index.shape[0])
            self.indices.append(index)

        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info(
            f"Found {self.cumulative_lengths[-1]} point clouds with at least {self.min_points} points"
        )

    def _kept(self, npoints):
        """Select events within the inclusive point-count limits."""
        maximum = np.inf if self.max_points is None else self.max_points
        return (npoints >= self.min_points) & (npoints <= maximum)

    def h5py_worker_init(self):
        """Initialize h5py files for each worker."""
        self.h5data = []
        for h5_file in self.h5_files:
            self.h5data.append(h5py.File(h5_file, mode="r", libver="latest", swmr=True))
        self.initted = True

    def get_data(self, idx):
        """Load a point cloud from h5 file.

        Output dictionary:
        - coord: (N, 3) array of coordinates
        - energy: (N, 1) array of energies
        - momentum: (N, 1) array of particle momentum (v2/v3 only)
        - vertex: (N, 3) array of vertices (v2/v3 only; interaction vertex for v3)
        - is_primary: (N, 1) array of primary-particle flags (v3 only)
        - segment_motif: (N, 1) array of motif labels
        - segment_pid: (N, 1) array of PID labels (v2/v3 only)
        - instance_particle: (N, 1) array of particle instance labels
        - instance_interaction: (N, 1) array of interaction instance labels
        - segment_interaction: (N, 1) array of interaction labels
        """
        if not self.initted:
            self.h5py_worker_init()
        if self.subset is not None:
            idx = int(self.subset[idx % len(self.subset)])

        # Find which h5 file and index the point cloud is in
        h5_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")
        if h5_idx > 0:
            idx_in_file = idx - self.cumulative_lengths[h5_idx - 1]
        else:
            idx_in_file = idx

        h5_file = self.h5data[h5_idx]
        file_idx = self.indices[h5_idx][idx_in_file]

        # load raw arrays for this event and decode into a flat data_dict
        data_dict = decode_event(
            point=h5_file["point"][file_idx],
            cluster=h5_file["cluster"][file_idx],
            cluster_extra=h5_file["cluster_extra"][file_idx],
            cluster_extra_2=(
                h5_file["cluster_extra_2"][file_idx]
                if "cluster_extra_2" in h5_file
                else None
            ),
            interaction_extra=(
                h5_file["interaction_extra"][file_idx]
                if "interaction_extra" in h5_file
                else None
            ),
            revision=self.revision,
            energy_threshold=self.energy_threshold,
            remove_low_energy_scatters=self.remove_low_energy_scatters,
            old_pid_mapping=self.old_pid_mapping,
        )

        # add metadata
        h5_name = os.path.basename(self.h5_files[h5_idx])
        data_dict["name"] = f"{h5_name}_{file_idx}"
        data_dict["split"] = self.split if isinstance(self.split, str) else "custom"
        data_dict["revision"] = self.revision

        return data_dict

    def get_data_name(self, idx):
        """Get name for the point cloud."""
        if not self.initted:
            self.h5py_worker_init()
        if self.subset is not None:
            idx = int(self.subset[idx % len(self.subset)])

        # Find which h5 file and index the point cloud is in
        h5_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")
        if h5_idx > 0:
            idx_in_file = idx - self.cumulative_lengths[h5_idx - 1]
        else:
            idx_in_file = idx

        h5_name = os.path.basename(self.h5_files[h5_idx])
        file_idx = self.indices[h5_idx][idx_in_file]

        return f"{h5_name}_{file_idx}"

    def _num_source_events(self):
        """Count of donor events: the selected ``max_len`` events."""
        total = int(self.cumulative_lengths[-1])
        return min(self.max_len, total) if self.max_len > 0 else total

    def prepare_train_data(self, idx):
        """Prepare training data with transforms."""
        data_dict = self.get_data(idx % len(self))
        data_dict = self._maybe_overlay(data_dict)
        return self.transform(data_dict)

    def __getitem__(self, idx):
        return self.prepare_train_data(idx % len(self))

    def __len__(self):
        if self.max_len > 0:
            return min(self.max_len, self.cumulative_lengths[-1]) * self.loop
        return self.cumulative_lengths[-1] * self.loop

    def __del__(self):
        """Clean up open h5 files."""
        if hasattr(self, "initted") and self.initted:
            for h5_file in self.h5data:
                h5_file.close()
