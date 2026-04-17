"""
LUCiDSegReader — reads 3D track segments from Water Cherenkov segment files.

Format: flat CSR arrays (no per-event groups).
  - track_event_offset (n_events+1,) — CSR into track arrays
  - segment_offset (n_tracks+1,) — CSR into segment arrays
  - start_x/y/z, end_x/y/z, edep, time (total_segments,) — per-segment
  - track_id, pdg, parent_id, initial_energy (n_tracks,) — per-track

Output dict:
    coord (N,3), energy (N,1), time (N,1),
    track_ids (N,), pdg (N,), parent_ids (N,)
"""

import os
import glob
import numpy as np
import h5py
from pimm.utils.logger import get_root_logger


class LUCiDSegReader:
    """Reads 3D track segments from WC segment HDF5 files.

    Parameters
    ----------
    data_root : str
        Directory containing segment shard files.
    split : str
        Split name.
    dataset_name : str
        File prefix — matches files like '{dataset_name}_*segment*_*.h5'
        or 'segment_events_*.h5'.
    min_segments : int
        Minimum segments per event to include.
    """

    def __init__(self, data_root, split='', dataset_name='wc',
                 min_segments=0, **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.min_segments = min_segments

        self.h5_files = self._find_files()
        assert len(self.h5_files) > 0, (
            f"No WC seg files found in {data_root}/{split}")

        self._initted = False
        self._h5data = []
        self._build_index()

    def _find_files(self):
        """Find segment HDF5 files. Tries multiple naming patterns."""
        for pattern in [
            os.path.join(self.data_root, self.split, f'{self.dataset_name}_seg_*.h5'),
            os.path.join(self.data_root, f'{self.dataset_name}_seg_*.h5'),
            os.path.join(self.data_root, self.split, 'segment_events_*.h5'),
            os.path.join(self.data_root, 'segment_events_*.h5'),
            os.path.join(self.data_root, self.split, '*segment*.h5'),
            os.path.join(self.data_root, '*segment*.h5'),
        ]:
            files = sorted(glob.glob(pattern))
            if files:
                return files
        return []

    def _build_index(self):
        log = get_root_logger()
        self.cumulative_lengths = []
        self.indices = []
        self._file_n_events = []

        for h5_path in self.h5_files:
            try:
                with h5py.File(h5_path, 'r', libver='latest', swmr=True) as f:
                    n_events = int(f.attrs.get('n_events', 0))
                    if n_events == 0 and 'event_number' in f:
                        n_events = f['event_number'].shape[0]

                    if self.min_segments > 0 and 'segment_offset' in f and 'track_event_offset' in f:
                        track_offsets = f['track_event_offset'][:]
                        seg_offsets = f['segment_offset'][:]
                        valid = []
                        for i in range(n_events):
                            t0 = track_offsets[i]
                            t1 = track_offsets[i + 1]
                            if t1 > t0:
                                n_seg = int(seg_offsets[t1] - seg_offsets[t0])
                            else:
                                n_seg = 0
                            if n_seg >= self.min_segments:
                                valid.append(i)
                        index = np.array(valid, dtype=np.int64)
                    else:
                        index = np.arange(n_events, dtype=np.int64)
                    self._file_n_events.append(n_events)
            except Exception as e:
                log.warning(f"Error processing {h5_path}: {e}")
                index = np.array([], dtype=np.int64)
                self._file_n_events.append(0)

            self.cumulative_lengths.append(len(index))
            self.indices.append(index)

        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info(f"LUCiDSegReader: {self.cumulative_lengths[-1]} events "
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
        return self._h5data[file_idx], event_num

    def read_event(self, idx):
        if not self._initted:
            self.h5py_worker_init()

        f, event_num = self._locate_event(idx)

        # Track range for this event
        track_offsets = f['track_event_offset']
        t0 = int(track_offsets[event_num])
        t1 = int(track_offsets[event_num + 1])
        n_tracks = t1 - t0

        if n_tracks == 0:
            return self._empty_dict()

        # Segment range for these tracks
        seg_offsets = f['segment_offset']
        s0 = int(seg_offsets[t0])
        s1 = int(seg_offsets[t1])
        n_seg = s1 - s0

        if n_seg == 0:
            return self._empty_dict()

        # Segment data
        sx = f['start_x'][s0:s1]
        sy = f['start_y'][s0:s1]
        sz = f['start_z'][s0:s1]
        ex = f['end_x'][s0:s1]
        ey = f['end_y'][s0:s1]
        ez = f['end_z'][s0:s1]

        mid_x = (sx + ex) / 2
        mid_y = (sy + ey) / 2
        mid_z = (sz + ez) / 2

        # Per-track data, expanded to per-segment
        track_ids = f['track_id'][t0:t1].astype(np.int32)
        pdg = f['pdg'][t0:t1].astype(np.int32)
        parent_ids = f['parent_id'][t0:t1].astype(np.int32)

        # Number of segments per track (from segment_offset)
        n_segs_per_track = np.diff(seg_offsets[t0:t1 + 1]).astype(np.int32)

        seg_track_ids = np.repeat(track_ids, n_segs_per_track)
        seg_pdg = np.repeat(pdg, n_segs_per_track)
        seg_parent_ids = np.repeat(parent_ids, n_segs_per_track)

        return {
            'coord': np.stack([mid_x, mid_y, mid_z], axis=1).astype(np.float32),
            'energy': f['edep'][s0:s1].astype(np.float32)[:, None],
            'time': f['time'][s0:s1].astype(np.float32)[:, None],
            'track_ids': seg_track_ids,
            'pdg': seg_pdg,
            'parent_ids': seg_parent_ids,
        }

    def _empty_dict(self):
        return {
            'coord': np.zeros((0, 3), dtype=np.float32),
            'energy': np.zeros((0, 1), dtype=np.float32),
            'time': np.zeros((0, 1), dtype=np.float32),
            'track_ids': np.zeros((0,), dtype=np.int32),
            'pdg': np.zeros((0,), dtype=np.int32),
            'parent_ids': np.zeros((0,), dtype=np.int32),
        }

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
