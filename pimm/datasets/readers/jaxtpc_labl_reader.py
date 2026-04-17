"""
JAXTPCLablReader — reads per-volume track_id → label lookup tables.

The labl file stores a mapping from track_id to labels (particle, cluster,
interaction) per volume. This is used for:
  - 3D tasks: deposit's track_id → look up label directly
  - 2D tasks: pixel → group_id → group_to_track → track_id → look up label

Output: a dict of numpy arrays per volume, keyed by label name.
    labl_v0_track_ids:   (T,) int32  — unique track IDs
    labl_v0_particle:    (T,) int32  — particle type per track
    labl_v0_cluster:     (T,) int32  — cluster ID per track
    labl_v0_interaction: (T,) int32  — interaction ID per track
"""

import os
import glob
import numpy as np
import h5py
from pimm.utils.logger import get_root_logger


class JAXTPCLablReader:
    """Reads per-volume track_id → label lookup tables.

    Parameters
    ----------
    data_root : str
        Directory containing labl shard files.
    split : str
        Split name.
    dataset_name : str
        File prefix (e.g., 'sim' matches 'sim_labl_0000.h5').
    label_keys : list of str or None
        Which label datasets to load (default: all available).
    """

    def __init__(self, data_root, split='train', dataset_name='sim',
                 label_keys=None):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.label_keys = label_keys  # None = load all

        self.h5_files = self._find_files()
        assert len(self.h5_files) > 0, (
            f"No labl files found for '{dataset_name}' in {data_root}/{split}")

        self._initted = False
        self._h5data = []

        self._build_index()

    def _find_files(self):
        pattern = os.path.join(
            self.data_root, self.split,
            f'{self.dataset_name}_labl_*.h5')
        files = sorted(glob.glob(pattern))
        if not files:
            pattern = os.path.join(
                self.data_root, f'{self.dataset_name}_labl_*.h5')
            files = sorted(glob.glob(pattern))
        return files

    def _build_index(self):
        log = get_root_logger()
        self.cumulative_lengths = []
        self.indices = []

        for h5_path in self.h5_files:
            try:
                with h5py.File(h5_path, 'r', libver='latest', swmr=True) as f:
                    n_events = int(f['config'].attrs['n_events'])
                    index = np.arange(n_events, dtype=np.int64)
            except Exception as e:
                log.warning(f"Error processing {h5_path}: {e}")
                index = np.array([], dtype=np.int64)

            self.cumulative_lengths.append(len(index))
            self.indices.append(index)

        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info(f"JAXTPCLablReader: {self.cumulative_lengths[-1]} events "
                 f"from {len(self.h5_files)} files")

    def h5py_worker_init(self):
        self._h5data = [
            h5py.File(p, 'r', libver='latest', swmr=True)
            for p in self.h5_files
        ]
        self._initted = True

    def _locate_event(self, idx):
        file_idx = int(np.searchsorted(self.cumulative_lengths, idx, side='right'))
        local_idx = idx - (int(self.cumulative_lengths[file_idx - 1]) if file_idx > 0 else 0)
        event_num = self.indices[file_idx][local_idx]
        event_key = f'event_{event_num:03d}'
        f = self._h5data[file_idx]
        return f, event_key

    def read_event(self, idx):
        """Read one event's per-volume label lookup tables.

        Returns dict with keys like:
            labl_v0_track_ids:   (T,) int32
            labl_v0_particle:    (T,) int32
            labl_v0_cluster:     (T,) int32
            labl_v0_interaction: (T,) int32
        """
        if not self._initted:
            self.h5py_worker_init()

        f, event_key = self._locate_event(idx)
        evt = f[event_key]

        data_dict = {}
        for vol_key in evt:
            vol = evt[vol_key]
            if not isinstance(vol, h5py.Group):
                continue
            if 'track_ids' not in vol:
                continue

            # Volume index from key (e.g., 'volume_0' → 0)
            vol_idx = vol_key.replace('volume_', '')
            prefix = f'labl_v{vol_idx}'

            data_dict[f'{prefix}_track_ids'] = vol['track_ids'][:].astype(np.int32)

            for lk in vol:
                if lk == 'track_ids':
                    continue
                if self.label_keys is not None and lk not in self.label_keys:
                    continue
                data_dict[f'{prefix}_{lk}'] = vol[lk][:].astype(np.int32)

        return data_dict

    def __len__(self):
        return int(self.cumulative_lengths[-1]) if len(self.cumulative_lengths) > 0 else 0

    def close(self):
        if self._initted:
            for f in self._h5data:
                try:
                    f.close()
                except Exception:
                    pass
            self._h5data = []
            self._initted = False
