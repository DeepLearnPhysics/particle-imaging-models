"""
JAXTPCSegReader — reads 3D truth deposits from JAXTPC seg files.

Produces raw geometry, physics, and IDs. Labels come from the labl file
via JAXTPCLablReader + JAXTPCDataset._apply_labl_to_3d(), or from
PDGToSemantic transform as a fallback.

Output dict:
    coord (N,3), energy (N,1), volume_id (N,1),
    track_ids (N,), group_ids (N,), pdg (N,), interaction_ids (N,),
    ancestor_track_ids (N,),
    and optionally: dx, theta, phi, t0_us, charge, photons, qs_fractions
"""

import os
import glob
import numpy as np
import h5py
from pimm.utils.logger import get_root_logger


class JAXTPCSegReader:
    """Reads 3D truth deposits from JAXTPC seg HDF5 files.

    Concatenates volumes into a single point cloud with a volume_id feature.
    No label computation — just raw data.

    Parameters
    ----------
    data_root : str
        Directory containing seg shard files.
    split : str
        Split name — used as subdirectory or glob pattern.
    dataset_name : str
        File prefix (e.g., 'sim' matches 'sim_seg_0000.h5').
    min_deposits : int
        Minimum deposits per event to include in index.
    include_physics : bool
        Whether to load dx, theta, phi, charge, photons, etc.
    """

    def __init__(self, data_root, split='train', dataset_name='sim',
                 min_deposits=0, include_physics=True, volume=None):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.min_deposits = min_deposits
        self.include_physics = include_physics
        self.volume = volume  # None = all volumes, int = single volume

        self.h5_files = self._find_files()
        assert len(self.h5_files) > 0, (
            f"No seg files found for '{dataset_name}' in {data_root}/{split}")

        self._initted = False
        self._h5data = []

        self._build_index()

    def _find_files(self):
        """Discover seg shard files."""
        pattern = os.path.join(
            self.data_root, self.split,
            f'{self.dataset_name}_seg_*.h5')
        files = sorted(glob.glob(pattern))
        if not files:
            pattern = os.path.join(
                self.data_root, f'{self.dataset_name}_seg_*.h5')
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
                    n_volumes = int(f['config'].attrs.get('n_volumes', 1))

                    if self.min_deposits > 0:
                        valid = []
                        for i in range(n_events):
                            evt_key = f'event_{i:03d}'
                            if evt_key not in f:
                                continue
                            evt = f[evt_key]
                            total = sum(
                                int(evt[f'volume_{v}'].attrs.get('n_actual', 0))
                                for v in range(n_volumes)
                                if f'volume_{v}' in evt
                            ) if n_volumes > 1 else (
                                evt['positions'].shape[0] if 'positions' in evt else 0
                            )
                            if total >= self.min_deposits:
                                valid.append(i)
                        index = np.array(valid, dtype=np.int64)
                    else:
                        index = np.arange(n_events, dtype=np.int64)

            except Exception as e:
                log.warning(f"Error processing {h5_path}: {e}")
                index = np.array([], dtype=np.int64)

            self.cumulative_lengths.append(len(index))
            self.indices.append(index)

        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info(f"JAXTPCSegReader: {self.cumulative_lengths[-1]} events "
                 f"from {len(self.h5_files)} files "
                 f"(min_deposits={self.min_deposits})")

    def h5py_worker_init(self):
        """Lazily open file handles (called after DataLoader fork)."""
        self._h5data = [
            h5py.File(p, 'r', libver='latest', swmr=True)
            for p in self.h5_files
        ]
        self._initted = True

    def _locate_event(self, idx):
        """Map global index → (file_handle, event_key, n_volumes)."""
        file_idx = int(np.searchsorted(self.cumulative_lengths, idx, side='right'))
        local_idx = idx - (int(self.cumulative_lengths[file_idx - 1]) if file_idx > 0 else 0)
        event_num = self.indices[file_idx][local_idx]
        event_key = f'event_{event_num:03d}'
        f = self._h5data[file_idx]
        n_volumes = int(f['config'].attrs.get('n_volumes', 1))
        return f, event_key, n_volumes

    def read_event(self, idx):
        """Read one event, return flat dict of numpy arrays.

        No label computation — just raw geometry, physics, and IDs.
        """
        if not self._initted:
            self.h5py_worker_init()

        f, event_key, n_volumes = self._locate_event(idx)
        evt = f[event_key]

        vol_arrays = []

        if n_volumes > 1:
            for v in range(n_volumes):
                if self.volume is not None and v != self.volume:
                    continue
                vk = f'volume_{v}'
                if vk not in evt:
                    continue
                vg = evt[vk]
                n = int(vg.attrs.get('n_actual', 0))
                if n == 0:
                    continue
                vol_arrays.append(self._read_volume(vg, n, v))
        else:
            # Legacy flat format
            if 'positions' in evt:
                n = evt['positions'].shape[0]
                vol_arrays.append(self._read_volume_flat(evt, n, 0))

        if not vol_arrays:
            return self._empty_dict()

        return self._concat_volumes(vol_arrays)

    def _read_volume(self, vg, n, vol_idx):
        """Read arrays from a volume group."""
        step = float(vg.attrs['pos_step_mm'])
        origin = np.array([vg.attrs['pos_origin_x'],
                           vg.attrs['pos_origin_y'],
                           vg.attrs['pos_origin_z']], dtype=np.float32)

        d = {
            'coord': vg['positions'][:].astype(np.float32) * step + origin,
            'energy': vg['de'][:].astype(np.float32),
            'volume_id': np.full(n, vol_idx, dtype=np.int32),
            'track_ids': vg['track_ids'][:].astype(np.int32),
            'group_ids': vg['group_ids'][:].astype(np.int32),
        }

        # Optional ID fields
        for key, dtype in [('pdg', np.int32), ('interaction_ids', np.int32),
                           ('ancestor_track_ids', np.int32)]:
            if key in vg:
                d[key] = vg[key][:].astype(dtype)
            else:
                d[key] = np.full(n, -1, dtype=dtype)

        # Optional physics
        if self.include_physics:
            for key in ('dx', 'theta', 'phi', 't0_us'):
                if key in vg:
                    d[key] = vg[key][:].astype(np.float32)
            for key in ('charge', 'photons', 'qs_fractions'):
                if key in vg:
                    d[key] = vg[key][:].astype(np.float32)

        return d

    def _read_volume_flat(self, evt, n, vol_idx):
        """Read from legacy flat event format (no volume subgroups)."""
        step = float(evt.attrs['pos_step_mm'])
        origin = np.array([evt.attrs['pos_origin_x'],
                           evt.attrs['pos_origin_y'],
                           evt.attrs['pos_origin_z']], dtype=np.float32)

        d = {
            'coord': evt['positions'][:].astype(np.float32) * step + origin,
            'energy': evt['de'][:].astype(np.float32),
            'volume_id': np.full(n, vol_idx, dtype=np.int32),
            'track_ids': evt['track_ids'][:].astype(np.int32),
            'group_ids': evt['group_ids'][:].astype(np.int32),
        }

        for key, dtype in [('pdg', np.int32), ('interaction_ids', np.int32),
                           ('ancestor_track_ids', np.int32)]:
            if key in evt:
                d[key] = evt[key][:].astype(dtype)
            else:
                d[key] = np.full(n, -1, dtype=dtype)

        if self.include_physics:
            for key in ('dx', 'theta', 'phi', 't0_us'):
                if key in evt:
                    d[key] = evt[key][:].astype(np.float32)
            for key in ('charge', 'photons', 'qs_fractions'):
                if key in evt:
                    d[key] = evt[key][:].astype(np.float32)

        return d

    def _concat_volumes(self, vol_arrays):
        """Concatenate per-volume dicts into a single flat dict."""
        keys = vol_arrays[0].keys()
        data_dict = {}
        for k in keys:
            arrays = [v[k] for v in vol_arrays if k in v]
            combined = np.concatenate(arrays, axis=0)
            if k == 'coord':
                data_dict[k] = combined  # already (N,3) float32
            elif k in ('energy', 'dx', 'theta', 'phi', 't0_us',
                       'charge', 'photons', 'qs_fractions'):
                data_dict[k] = combined[:, None]  # (N,1)
            elif k == 'volume_id':
                data_dict[k] = combined[:, None]  # (N,1)
            else:
                data_dict[k] = combined  # (N,) for IDs

        return data_dict

    def _empty_dict(self):
        """Minimal valid dict for empty events."""
        d = {
            'coord': np.zeros((0, 3), dtype=np.float32),
            'energy': np.zeros((0, 1), dtype=np.float32),
            'volume_id': np.zeros((0, 1), dtype=np.int32),
            'track_ids': np.zeros((0,), dtype=np.int32),
            'group_ids': np.zeros((0,), dtype=np.int32),
            'pdg': np.zeros((0,), dtype=np.int32),
            'interaction_ids': np.zeros((0,), dtype=np.int32),
            'ancestor_track_ids': np.zeros((0,), dtype=np.int32),
        }
        return d

    def __len__(self):
        return int(self.cumulative_lengths[-1]) if len(self.cumulative_lengths) > 0 else 0

    def close(self):
        """Close open file handles."""
        if self._initted:
            for f in self._h5data:
                try:
                    f.close()
                except Exception:
                    pass
            self._h5data = []
            self._initted = False
