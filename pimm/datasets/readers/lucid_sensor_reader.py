"""
LUCiDSensorReader — reads PMT sensor data from Water Cherenkov sensor files.

Format: flat CSR arrays (no per-event groups).
  - event_hit_offsets (n_events+1,) — CSR into hit arrays
  - event_hit_sensor_idx, event_hit_PE, event_hit_T (total_hits,) — per-hit
  - particle_event_offset (n_events+1,) — CSR into particle arrays
  - particle_hit_offsets (n_particles+1,) — CSR into per-particle hits
  - particle_hit_sensor_idx, particle_hit_PE, particle_hit_T (total_pp_hits,)
  - particle_category (n_particles,) — semantic labels

PMT 3D positions are NOT stored in the file — must be provided separately
(via pmt_positions_file or pmt_positions array).

Output (sensor response):
    pmt_pe (N_sensors,), pmt_t (N_sensors,)
    If pmt_positions provided: pmt_coord (N_sensors, 3)

Output (per-particle sparse, when include_labels=True):
    pp_sensor_idx (E,), pp_particle_idx (E,), pp_pe (E,),
    pp_t (E,), pp_category (E,)
"""

import os
import glob
import numpy as np
import h5py
from pimm.utils.logger import get_root_logger


class LUCiDSensorReader:
    """Reads PMT sensor data from WC sensor HDF5 files.

    Parameters
    ----------
    data_root : str
        Directory containing sensor shard files.
    split : str
        Split name.
    dataset_name : str
        File prefix.
    include_labels : bool
        Whether to load per-particle hit decomposition.
    pe_threshold : float
        Minimum PE for per-particle hits (already sparse in file, this
        applies additional filtering).
    pmt_positions : ndarray or None
        (N_sensors, 3) PMT positions. If None, coord won't be produced
        (sensor_idx is still available).
    pmt_positions_file : str or None
        Path to .npy file with PMT positions. Alternative to pmt_positions.
    """

    def __init__(self, data_root, split='', dataset_name='wc',
                 include_labels=True, pe_threshold=0.0,
                 pmt_positions=None, pmt_positions_file=None, **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.include_labels = include_labels
        self.pe_threshold = pe_threshold

        # PMT positions (optional)
        if pmt_positions is not None:
            self._pmt_positions = np.asarray(pmt_positions, dtype=np.float32)
        elif pmt_positions_file is not None:
            self._pmt_positions = np.load(pmt_positions_file).astype(np.float32)
        else:
            self._pmt_positions = None

        self.h5_files = self._find_files()
        assert len(self.h5_files) > 0, (
            f"No WC sensor files found in {data_root}/{split}")

        self._initted = False
        self._h5data = []
        self._n_sensors = None
        self._build_index()

    def _find_files(self):
        for pattern in [
            os.path.join(self.data_root, self.split, f'{self.dataset_name}_sensor_*.h5'),
            os.path.join(self.data_root, f'{self.dataset_name}_sensor_*.h5'),
            os.path.join(self.data_root, self.split, 'sensor_events_*.h5'),
            os.path.join(self.data_root, 'sensor_events_*.h5'),
            os.path.join(self.data_root, self.split, '*sensor*.h5'),
            os.path.join(self.data_root, '*sensor*.h5'),
        ]:
            files = sorted(glob.glob(pattern))
            if files:
                return files
        return []

    def _build_index(self):
        log = get_root_logger()
        self.cumulative_lengths = []
        self.indices = []

        for h5_path in self.h5_files:
            try:
                with h5py.File(h5_path, 'r', libver='latest', swmr=True) as f:
                    n_events = int(f.attrs.get('n_events', 0))
                    if n_events == 0 and 'event_number' in f:
                        n_events = f['event_number'].shape[0]
                    self._n_sensors = int(f.attrs.get('n_sensors', 0))
                    index = np.arange(n_events, dtype=np.int64)
            except Exception as e:
                log.warning(f"Error processing {h5_path}: {e}")
                index = np.array([], dtype=np.int64)

            self.cumulative_lengths.append(len(index))
            self.indices.append(index)

        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info(f"LUCiDSensorReader: {self.cumulative_lengths[-1]} events, "
                 f"{self._n_sensors} sensors from {len(self.h5_files)} files")

    def h5py_worker_init(self):
        self._h5data = [
            h5py.File(p, 'r', libver='latest', swmr=True)
            for p in self.h5_files
        ]
        # Try to load PMT positions from file config if not provided
        if self._pmt_positions is None:
            f = self._h5data[0]
            if 'config' in f and 'pmt_positions' in f['config']:
                self._pmt_positions = f['config']['pmt_positions'][:].astype(np.float32)
        self._initted = True

    def _locate_event(self, idx):
        file_idx = int(np.searchsorted(self.cumulative_lengths, idx, side='right'))
        local_idx = idx - (int(self.cumulative_lengths[file_idx - 1]) if file_idx > 0 else 0)
        event_num = self.indices[file_idx][local_idx]
        return self._h5data[file_idx], event_num

    def read_event(self, idx):
        if not self._initted:
            self.h5py_worker_init()

        f, event_num = self._locate_event(idx)
        data_dict = {}

        # --- Event-level sensor hits ---
        hit_offsets = f['event_hit_offsets']
        h0 = int(hit_offsets[event_num])
        h1 = int(hit_offsets[event_num + 1])

        sensor_idx = f['event_hit_sensor_idx'][h0:h1].astype(np.int32)
        pe = f['event_hit_PE'][h0:h1].astype(np.float32)
        t = f['event_hit_T'][h0:h1].astype(np.float32)

        # Build dense per-sensor arrays (sum PE, min T for sensors with hits)
        n_sensors = self._n_sensors
        pmt_pe = np.zeros(n_sensors, dtype=np.float32)
        pmt_t = np.full(n_sensors, -1.0, dtype=np.float32)
        np.add.at(pmt_pe, sensor_idx, pe)
        # First-hit time: use scatter-min
        for i in range(len(sensor_idx)):
            s = sensor_idx[i]
            if pmt_t[s] < 0 or t[i] < pmt_t[s]:
                pmt_t[s] = t[i]

        data_dict['pmt_pe'] = pmt_pe
        data_dict['pmt_t'] = pmt_t

        if self._pmt_positions is not None:
            data_dict['pmt_coord'] = self._pmt_positions.copy()

        # --- Per-particle hits (sparse) ---
        if self.include_labels and 'particle_hit_offsets' in f:
            # Which particles belong to this event
            p_offsets = f['particle_event_offset']
            p0 = int(p_offsets[event_num])
            p1 = int(p_offsets[event_num + 1])
            n_particles = p1 - p0

            if n_particles > 0:
                categories = f['particle_category'][p0:p1].astype(np.int32)

                # Per-particle hit ranges
                pp_hit_offsets = f['particle_hit_offsets']
                pp_h0 = int(pp_hit_offsets[p0])
                pp_h1 = int(pp_hit_offsets[p1])

                pp_sensor = f['particle_hit_sensor_idx'][pp_h0:pp_h1].astype(np.int32)
                pp_pe = f['particle_hit_PE'][pp_h0:pp_h1].astype(np.float32)
                pp_t = f['particle_hit_T'][pp_h0:pp_h1].astype(np.float32)

                # Build particle_idx for each hit
                hits_per_particle = np.diff(pp_hit_offsets[p0:p1 + 1]).astype(np.int32)
                pp_particle_idx = np.repeat(np.arange(n_particles, dtype=np.int32),
                                            hits_per_particle)
                pp_category = np.repeat(categories, hits_per_particle)

                # Optional threshold filter
                if self.pe_threshold > 0:
                    mask = pp_pe > self.pe_threshold
                    pp_sensor = pp_sensor[mask]
                    pp_pe = pp_pe[mask]
                    pp_t = pp_t[mask]
                    pp_particle_idx = pp_particle_idx[mask]
                    pp_category = pp_category[mask]

                data_dict['pp_sensor_idx'] = pp_sensor
                data_dict['pp_particle_idx'] = pp_particle_idx
                data_dict['pp_pe'] = pp_pe
                data_dict['pp_t'] = pp_t
                data_dict['pp_category'] = pp_category

        return data_dict

    def __len__(self):
        return int(self.cumulative_lengths[-1]) if len(self.cumulative_lengths) > 0 else 0

    def close(self):
        if self._initted:
            for fh in self._h5data:
                try:
                    fh.close()
                except Exception:
                    pass
            self._h5data = []
            self._initted = False
