"""
JAXTPCRespReader — reads sparse wire signals from JAXTPC resp files.

Decodes delta-encoded (wire, time, value) triples per plane.
Output keys are dot-namespaced: plane.{plane_label}.wire/time/value

Handles both old format (planes directly under event) and new format
(planes under volume_N/ subgroups).
"""

import os
import glob
import numpy as np
import h5py
from pimm.utils.logger import get_root_logger


class JAXTPCRespReader:
    """Reads sparse wire signals from JAXTPC resp HDF5 files.

    Parameters
    ----------
    data_root : str
        Directory containing resp shard files.
    split : str
        Split name — used as subdirectory or glob pattern.
    dataset_name : str
        File prefix (e.g., 'sim' matches 'sim_resp_0000.h5').
    planes : str or list
        Which planes to load: 'all' or list like ['east_U', 'east_V'].
    decode_digitization : bool
        If True, subtract pedestal from uint16 values.
    """

    def __init__(self, data_root, split='train', dataset_name='sim',
                 planes='all', decode_digitization=True):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.planes = planes
        self.decode_digitization = decode_digitization

        self.h5_files = self._find_files()
        assert len(self.h5_files) > 0, (
            f"No resp files found for '{dataset_name}' in {data_root}/{split}")

        self._initted = False
        self._h5data = []

        self._build_index()

    def _find_files(self):
        """Discover resp shard files."""
        pattern = os.path.join(
            self.data_root, self.split,
            f'{self.dataset_name}_resp_*.h5')
        files = sorted(glob.glob(pattern))
        if not files:
            pattern = os.path.join(
                self.data_root, f'{self.dataset_name}_resp_*.h5')
            files = sorted(glob.glob(pattern))
        return files

    def _build_index(self):
        """Scan files, count events, build cumulative index."""
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
        log.info(f"JAXTPCRespReader: {self.cumulative_lengths[-1]} events "
                 f"from {len(self.h5_files)} files")

    def h5py_worker_init(self):
        """Lazily open file handles (called after DataLoader fork)."""
        self._h5data = [
            h5py.File(p, 'r', libver='latest', swmr=True)
            for p in self.h5_files
        ]
        self._initted = True

    def _locate_event(self, idx):
        """Map global index -> (file_handle, event_key)."""
        file_idx = int(np.searchsorted(self.cumulative_lengths, idx, side='right'))
        local_idx = idx - (int(self.cumulative_lengths[file_idx - 1]) if file_idx > 0 else 0)
        event_num = self.indices[file_idx][local_idx]
        event_key = f'event_{event_num:03d}'
        f = self._h5data[file_idx]
        return f, event_key

    def _iter_planes(self, evt):
        """Yield (plane_label, h5py.Group) for each plane in an event.

        Handles both formats:
          - Old: planes directly under event (east_U, east_V, ...)
          - New: planes under volume_N/ subgroups
        """
        for key in evt:
            obj = evt[key]
            if not isinstance(obj, h5py.Group):
                continue
            if key.startswith('volume_'):
                # New format: volume_0/U, volume_0/V, ...
                vol_label = key  # e.g., 'volume_0'
                for plane_key in obj:
                    pg = obj[plane_key]
                    if isinstance(pg, h5py.Group) and 'delta_wire' in pg:
                        yield f'{vol_label}_{plane_key}', pg
            elif 'delta_wire' in obj:
                # Old format: east_U, east_V, ...
                yield key, obj

    def _decode_plane(self, g):
        """Decode one plane's delta-encoded sparse data."""
        wire_start = int(g.attrs['wire_start'])
        time_start = int(g.attrs['time_start'])

        wire = wire_start + np.cumsum(g['delta_wire'][:]).astype(np.int32)
        time = time_start + np.cumsum(g['delta_time'][:]).astype(np.int32)

        raw_values = g['values'][:]
        if self.decode_digitization and raw_values.dtype == np.uint16:
            ped = int(g.attrs.get('pedestal', 0))
            values = raw_values.astype(np.float32) - ped
        else:
            values = raw_values.astype(np.float32)

        return wire, time, values

    def read_event(self, idx):
        """Read one event, return dict with plane-namespaced sparse arrays.

        Returns keys like:
            plane.east_U.wire:  (M,) int32
            plane.east_U.time:  (M,) int32
            plane.east_U.value: (M,) float32
        """
        if not self._initted:
            self.h5py_worker_init()

        f, event_key = self._locate_event(idx)
        evt = f[event_key]

        data_dict = {}
        for plane_label, pg in self._iter_planes(evt):
            if self.planes != 'all' and plane_label not in self.planes:
                continue

            wire, time, values = self._decode_plane(pg)
            prefix = f'plane.{plane_label}'
            data_dict[f'{prefix}.wire'] = wire
            data_dict[f'{prefix}.time'] = time
            data_dict[f'{prefix}.value'] = values

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
