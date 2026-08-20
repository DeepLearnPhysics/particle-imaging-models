"""PILArNet-M dataset read directly from clustered HDF5 shards."""

import glob
import os
from copy import deepcopy
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from torch.utils.data import Dataset

from pimm.utils.logger import get_root_logger

from ..builder import DATASETS
from ..transform import TRANSFORMS, Compose
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
            ``$PILARNET_DATA_ROOT_V3``/``_V3_EXTRA`` variable, then to
            ``~/.cache/pimm/pilarnet/<revision>``; raises if none exist.
            Defaults to ``None``.
        split (str | Sequence[str]): Split name(s) used to glob ``*<split>/*.h5``
            under ``data_root``. Defaults to ``"train"``.
        transform (list[dict] | None): List of transform configs (NOT a prebuilt
            ``Compose``). ``None`` (the default) uses
            :func:`~pimm.datasets.pilarnet.default_transform`; ``[]`` means no
            preprocessing. Defaults to ``None``.
        test_mode (bool): Emit voxelized/augmented test fragments and force
            ``loop = 1``. Defaults to ``False``.
        test_cfg (object): Test config (``voxelize``, ``crop``, ``post_transform``,
            ``aug_transform``); required when ``test_mode``. Defaults to ``None``.
        loop (int): Train-time epoch multiplier. Defaults to ``1``.
        ignore_index (int): Ignored-label value. Defaults to ``-1``.
        energy_threshold (float): Drop points with energy at or below this value
            when positive. Defaults to ``0.0``.
        min_points (int): Minimum points per event; smaller events are excluded
            from the index. Defaults to ``1024``.
        max_len (int): Cap on event count before the loop multiplier (-1 = no
            cap). Defaults to ``-1``.
        remove_low_energy_scatters (bool): Drop the first (LED scatter) cluster
            and its points. Defaults to ``False``.
        old_pid_mapping (bool): Map LED PID to ``6`` instead of ``5``. Defaults
            to ``False``.
        revision ({"v2", "v3", "v3_extra"}): v3-family schema. ``"v2"`` is
            served as v3 with a warning. ``v3_extra`` additionally requires
            momentum/lineage and interaction-vertex tables. Defaults to ``"v3"``.
        overlay_n_events (int | tuple[int, int]): Number (or inclusive range) of
            events to overlay into one point cloud; ``> 1`` enables overlay.
            Defaults to ``1``.
        overlay_prob (float): Probability of applying overlay to a given sample.
            Defaults to ``1.0``.
        overlay_allow_repeats (bool): Allow the same event to be sampled more
            than once when overlaying. Defaults to ``True``.

    Note:
        Loader settings (``batch_size``, ``num_worker``) live at the top level of
        the config, not on the dataset constructor. Split membership differs
        between v1 and v3, so numbers from a v1-era run are not comparable to a
        v3 run on the same split name. Event overlay deduplicates
        colliding voxels by semantic priority (track > shower > Michel > delta >
        LED) and rotates overlaid events by random 90-degree increments.

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
        test_cfg=None,
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
        overlay_allow_repeats=True,
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
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.loop = loop if not test_mode else 1
        self.ignore_index = ignore_index
        self.old_pid_mapping = old_pid_mapping

        self.revision = revision
        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        # event overlay parameters
        self.overlay_n_events = overlay_n_events
        self.overlay_prob = overlay_prob
        self.overlay_allow_repeats = overlay_allow_repeats

        # PILArNet specific parameters
        self.energy_threshold = energy_threshold
        self.min_points = min_points
        self.remove_low_energy_scatters = remove_low_energy_scatters
        self.max_len = max_len
        # Get list of h5 files
        self.h5_files = self.get_h5_files()
        assert len(self.h5_files) > 0, "No h5 files found"
        self.initted = False
        self.file_events = []

        # Build index for faster access
        self._build_index()

        logger = get_root_logger()
        logger.info(
            "Total number of samples in PILArNet {} set: {} x {}.".format(
                self.cumulative_lengths[-1], self.loop, split
            )
        )
        if self.overlay_n_events > 1 or (isinstance(self.overlay_n_events, (tuple, list)) and self.overlay_n_events[1] > 1):
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
            try:
                # Check if points count file exists
                points_file = h5_file.replace(".h5", "_points.npy")
                if os.path.exists(points_file):
                    npoints = np.load(points_file)
                    index = np.argwhere(npoints >= self.min_points).flatten()
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
                        index = np.argwhere(npoints >= self.min_points).flatten()
                        self.file_events.append(npoints.shape[0])
                if os.path.exists(points_file):
                    self.file_events.append(int(npoints.shape[0]))
            except Exception as e:
                log.warning(f"Error processing {h5_file}: {e}")
                index = np.array([])
                self.file_events.append(0)

            self.cumulative_lengths.append(index.shape[0])
            self.indices.append(index)

        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info(
            f"Found {self.cumulative_lengths[-1]} point clouds with at least {self.min_points} points"
        )

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
        """Count of distinct events (pre-``loop``); overlay samples from this."""
        return int(self.cumulative_lengths[-1])

    def prepare_train_data(self, idx):
        """Prepare training data with transforms."""
        data_dict = self.get_data(idx % len(self))
        data_dict = self._maybe_overlay(data_dict)
        return self.transform(data_dict)

    def prepare_test_data(self, idx):
        """Prepare test data with test transforms."""
        # Load data
        data_dict = self.get_data(idx % len(self))
        data_dict = self._maybe_overlay(data_dict)

        # Apply transforms
        if self.transform is not None:
            data_dict = self.transform(data_dict)

        # Test mode specific handling
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
        return result_dict

    def __getitem__(self, idx):
        real_idx = idx % len(self)
        if self.test_mode:
            return self.prepare_test_data(real_idx)
        else:
            return self.prepare_train_data(real_idx)

    def __len__(self):
        if self.max_len > 0:
            return min(self.max_len, self.cumulative_lengths[-1]) * self.loop
        return self.cumulative_lengths[-1] * self.loop

    def __del__(self):
        """Clean up open h5 files."""
        if hasattr(self, "initted") and self.initted:
            for h5_file in self.h5data:
                h5_file.close()
