"""
JAXTPCCorrReader — reads 3D→2D correspondence from JAXTPC corr files.

Decodes CSR-encoded per-plane correspondence into flat arrays:
per pixel entry: (wire, time, group_id, charge).

Also loads per-volume group_to_track lookup tables.

All decoding is fully vectorized (no Python loops over groups).
"""

import os
import glob
import numpy as np
import h5py
from pimm.utils.logger import get_root_logger


class JAXTPCCorrReader:
    """Reads 3D→2D correspondence from JAXTPC corr HDF5 files.

    Parameters
    ----------
    data_root : str
        Directory containing corr shard files.
    split : str
        Split name.
    dataset_name : str
        File prefix (e.g., 'sim' matches 'sim_corr_0000.h5').
    planes : str or list
        Which planes to load: 'all' or list like ['volume_0_U'].
    """

    def __init__(self, data_root, split='train', dataset_name='sim',
                 planes='all', **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.planes = planes

        self.h5_files = self._find_files()
        assert len(self.h5_files) > 0, (
            f"No corr files found for '{dataset_name}' in {data_root}/{split}")

        self._initted = False
        self._h5data = []

        self._build_index()

    def _find_files(self):
        pattern = os.path.join(
            self.data_root, self.split,
            f'{self.dataset_name}_corr_*.h5')
        files = sorted(glob.glob(pattern))
        if not files:
            pattern = os.path.join(
                self.data_root, f'{self.dataset_name}_corr_*.h5')
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
        log.info(f"JAXTPCCorrReader: {self.cumulative_lengths[-1]} events "
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

    @staticmethod
    def _decode_plane_vectorized(g):
        """Decode one plane's CSR correspondence — fully vectorized.

        Returns (wire, time, group_id, charge) arrays, all shape (E,).
        """
        group_ids = g['group_ids'][:]
        group_sizes = g['group_sizes'][:].astype(np.int32)
        center_wires = g['center_wires'][:]
        center_times = g['center_times'][:]
        peak_charges = g['peak_charges'][:]
        delta_wires = g['delta_wires'][:]
        delta_times = g['delta_times'][:]
        charges_u16 = g['charges_u16'][:]

        G = len(group_ids)
        if G == 0:
            empty = np.array([], dtype=np.int32)
            return empty, empty, empty, np.array([], dtype=np.float32)

        # Broadcast group-level arrays to per-entry arrays
        wires = (np.repeat(center_wires, group_sizes).astype(np.int32)
                 + delta_wires.astype(np.int32))
        times = (np.repeat(center_times, group_sizes).astype(np.int32)
                 + delta_times.astype(np.int32))
        gids = np.repeat(group_ids, group_sizes)
        charges = (np.repeat(peak_charges, group_sizes)
                   * charges_u16.astype(np.float32) / 65535.0)

        return wires, times, gids, charges

    def read_event(self, idx):
        """Read one event's correspondence data.

        Returns dict with:
            corr.{vol_plane}.wire:     (E,) int32  — wire index per entry
            corr.{vol_plane}.time:     (E,) int32  — time index per entry
            corr.{vol_plane}.group_id: (E,) int32  — group ID per entry
            corr.{vol_plane}.charge:   (E,) float32 — charge per entry
            g2t_v{N}:                  (G,) int32  — group_to_track per volume
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
            if not vol_key.startswith('volume_'):
                continue

            vol_idx = vol_key.replace('volume_', '')

            # group_to_track lookup for this volume
            if 'group_to_track' in vol:
                data_dict[f'g2t_v{vol_idx}'] = vol['group_to_track'][:].astype(np.int32)

            # Per-plane correspondence
            for plane_key in vol:
                pg = vol[plane_key]
                if not isinstance(pg, h5py.Group) or 'group_ids' not in pg:
                    continue

                plane_label = f'volume_{vol_idx}_{plane_key}'
                if self.planes != 'all' and plane_label not in self.planes:
                    continue

                wires, times, gids, charges = self._decode_plane_vectorized(pg)

                prefix = f'corr.{plane_label}'
                data_dict[f'{prefix}.wire'] = wires
                data_dict[f'{prefix}.time'] = times
                data_dict[f'{prefix}.group_id'] = gids
                data_dict[f'{prefix}.charge'] = charges

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
