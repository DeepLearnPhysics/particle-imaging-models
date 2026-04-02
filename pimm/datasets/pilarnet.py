"""
PILArNet-M Dataset

This module handles the PILArNet-M dataset for particle physics point cloud segmentation.
"""

import os
import glob
import random
import numpy as np
import h5py
from copy import deepcopy
from torch.utils.data import Dataset
from typing import Literal
from pimm.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from .hepdataset import HEPDataset

# priority for voxel deduplication: track (1) > shower (0) > michel (2) > delta (3) > led (4)
DEFAULT_LABEL_PRIORITY = {1: 0, 0: 1, 2: 2, 3: 3, 4: 4}


@DATASETS.register_module()
class PILArNetH5Dataset(HEPDataset):
    """
    PILArNet-M Dataset that loads directly from h5 files, avoiding the need for preprocessing to individual files.

    The dataset contains the following semantic classes:
    - 0: Shower
    - 1: Track
    - 2: Michel
    - 3: Delta
    - 4: Low energy deposit

    and the following PID classes:
    - 0: Photon
    - 1: Electron
    - 2: Muon
    - 3: Pion
    - 4: Proton
    - 5: None (Low energy deposit)

    PID, momentum, and vertex information is only available in v2.
    v1 is the original PILArNet dataset in the PoLAr-MAE paper; v2 is the reprocessed PILArNet-M dataset
    which contains PID, momentum, and vertex information, and is used in the Panda paper. Note that
    the events in the splits are different between v1 and v2, so care needs to be taken when evaluating a model
    that was trained on v1 on v2.

    Event Overlay:
        Set overlay_n_events > 1 to overlay multiple events into a single point cloud.
        Overlapping voxels are deduplicated with priority: track > shower > michel > delta > led.
        Overlay events are randomly rotated by 90-degree increments.
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
        energy_threshold=0.0,
        min_points=1024,
        max_len=-1,
        remove_low_energy_scatters=False,
        old_pid_mapping=False,
        revision: Literal["v1", "v2"] = "v2",
        # event overlay parameters
        overlay_n_events=1,
        overlay_prob=1.0,
        overlay_allow_repeats=True,
    ):
        super().__init__()
        self.data_root = data_root
        if self.data_root is None:
            # set PILARNET_DATA_ROOT_V1/V2 in .env
            self.data_root = os.environ.get(f"PILARNET_DATA_ROOT_{revision.upper()}")
            assert self.data_root is not None, f"PILARNET_DATA_ROOT_V1/V2 is not set; checked {f'PILARNET_DATA_ROOT_{revision.upper()}'}"
        self.split = split
        self.transform = Compose(transform)
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
                            npoint = f['point'][i].numel() // 8
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
        - momentum: (N, 1) array of particle momentum (v2 only)
        - vertex: (N, 3) array of vertices (v2 only)
        - segment_motif: (N, 1) array of motif labels
        - segment_pid: (N, 1) array of PID labels (v2 only)
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

        # load point cloud data
        data = h5_file["point"][file_idx].reshape(-1, 8)[:, [0, 1, 2, 3]]  # (x,y,z,e)
        
        if self.revision == "v1":
            # v1: cluster dataset is (-1, 5) without PID, no cluster_extra dataset
            cluster_size, group_id, interaction_id, semantic_id = (
                h5_file["cluster"][file_idx].reshape(-1, 5)[:, [0, 2, -2, -1]].T
            )
            # v1 doesn't have interaction_id or pid, set defaults
            pid = np.full_like(semantic_id, -1)  # -1
            # v1 doesn't have cluster_extra, set defaults for momentum and vertex
            mom = np.zeros_like(semantic_id, dtype=np.float32)
            vtx_x = np.zeros_like(semantic_id, dtype=np.float32)
            vtx_y = np.zeros_like(semantic_id, dtype=np.float32)
            vtx_z = np.zeros_like(semantic_id, dtype=np.float32)
        else:  # v2
            cluster_size, group_id, interaction_id, semantic_id, pid = (
                h5_file["cluster"][file_idx].reshape(-1, 6)[:, [0, 2, -3, -2, -1]].T
            )
            mom, vtx_x, vtx_y, vtx_z = h5_file["cluster_extra"][file_idx].reshape(-1, 5)[:, [1, 2, 3, 4]].T
            pid[pid == -1] = (
                5 if not self.old_pid_mapping else 6
            )  # -1 (LED) --> 5 (where Kaon is) or 6 (new ID)

        # Remove low energy scatters if configured
        if self.remove_low_energy_scatters:
            data = data[cluster_size[0] :]
            semantic_id, group_id, interaction_id, pid, cluster_size = (
                semantic_id[1:],
                group_id[1:],
                interaction_id[1:],
                pid[1:],
                cluster_size[1:],
            )
            mom, vtx_x, vtx_y, vtx_z = mom[1:], vtx_x[1:], vtx_y[1:], vtx_z[1:]

        # Compute semantic ids for each point
        data_semantic_id = np.repeat(semantic_id, cluster_size)
        data_group_id = np.repeat(group_id, cluster_size)
        data_interaction_id = np.repeat(interaction_id, cluster_size)
        data_pid = np.repeat(pid, cluster_size)
        data_mom = np.repeat(mom, cluster_size)
        data_vtx_x = np.repeat(vtx_x, cluster_size)
        data_vtx_y = np.repeat(vtx_y, cluster_size)
        data_vtx_z = np.repeat(vtx_z, cluster_size)
        
        # Apply energy threshold if needed
        if self.energy_threshold > 0:
            threshold_mask = data[:, 3] > self.energy_threshold
            data = data[threshold_mask]
            data_semantic_id = data_semantic_id[threshold_mask]
            data_group_id = data_group_id[threshold_mask]
            data_interaction_id = data_interaction_id[threshold_mask]
            data_pid = data_pid[threshold_mask]
            data_mom = data_mom[threshold_mask]
            data_vtx_x = data_vtx_x[threshold_mask]
            data_vtx_y = data_vtx_y[threshold_mask]
            data_vtx_z = data_vtx_z[threshold_mask]

        # Prepare return dictionary
        data_dict = {}

        # Get coordinates
        data_dict["coord"] = data[:, :3].astype(np.float32)

        # Process energy (raw)
        energy = data[:, 3].astype(np.float32)
        data_dict["energy"] = energy[:, None]

        # Momentum (V2 only)
        data_dict["momentum"] = data_mom.astype(np.float32)[:, None]
        data_dict["vertex"] = np.stack([data_vtx_x, data_vtx_y, data_vtx_z], axis=1).astype(np.float32)

        # Get semantic labels
        data_dict["segment_motif"] = data_semantic_id.astype(np.int32)[:, None]
        data_dict["segment_pid"] = data_pid.astype(np.int32)[:, None]
        # compute both particle- and interaction-level instances
        particle_ids = data_group_id.astype(np.int32)
        interaction_ids = data_interaction_id.astype(np.int32)

        instance_particle = map_instance_ids(particle_ids)
        instance_interaction = map_instance_ids(interaction_ids)

        # always return both flavors
        data_dict["instance_particle"] = instance_particle
        data_dict["instance_interaction"] = instance_interaction
        data_dict["segment_interaction"] = (interaction_ids[:, None] != -1).astype(
                np.int32
            )  # 1 if not background, 0 if background

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

    def _sample_overlay_n_events(self):
        """Sample the number of events to overlay."""
        if isinstance(self.overlay_n_events, (tuple, list)):
            return random.randint(self.overlay_n_events[0], self.overlay_n_events[1])
        return self.overlay_n_events

    @staticmethod
    def _get_rotation_matrix_90(axis, n_rotations):
        """Get rotation matrix for n * 90 degree rotation around axis."""
        angle = n_rotations * np.pi / 2
        c, s = np.cos(angle), np.sin(angle)
        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        elif axis == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        else:  # z
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    def _apply_random_90_rotation(self, coord, center=None):
        """Apply random 90-degree rotations around x, y, z axes centered at given point."""
        if center is None:
            center = np.array([384.0, 384.0, 384.0], dtype=np.float32)
        coord = coord - center
        for axis in ["x", "y", "z"]:
            n_rot = random.randint(0, 3)
            if n_rot > 0:
                rot_mat = self._get_rotation_matrix_90(axis, n_rot)
                coord = coord @ rot_mat.T
        coord = coord + center
        return coord

    def _deduplicate_voxels(self, data_dict, concat_keys):
        """
        Deduplicate overlapping voxels based on segment_motif priority.
        Priority: track (1) > shower (0) > michel (2) > delta (3) > led (4)
        """
        coord = data_dict.get("coord")
        if coord is None:
            return data_dict

        coord_int = np.round(coord).astype(np.int64)
        segment = data_dict.get("segment_motif")
        
        if segment is None:
            _, unique_idx = np.unique(coord_int, axis=0, return_index=True)
            unique_idx = np.sort(unique_idx)
            for key in concat_keys:
                if key in data_dict and data_dict[key] is not None:
                    data_dict[key] = data_dict[key][unique_idx]
            return data_dict

        segment = segment.flatten()
        n_points = coord_int.shape[0]
        priorities = np.array([DEFAULT_LABEL_PRIORITY.get(int(s), 999) for s in segment], dtype=np.int32)

        coord_min = coord_int.min(axis=0)
        coord_shifted = coord_int - coord_min
        coord_max = coord_shifted.max(axis=0) + 1

        voxel_hash = (
            coord_shifted[:, 0].astype(np.int64) * (coord_max[1] * coord_max[2]) +
            coord_shifted[:, 1].astype(np.int64) * coord_max[2] +
            coord_shifted[:, 2].astype(np.int64)
        )

        unique_hashes, inverse_indices = np.unique(voxel_hash, return_inverse=True)
        n_unique = len(unique_hashes)

        best_idx = np.full(n_unique, -1, dtype=np.int64)
        best_priority = np.full(n_unique, 1000, dtype=np.int32)

        for i in range(n_points):
            voxel_idx = inverse_indices[i]
            if priorities[i] < best_priority[voxel_idx]:
                best_priority[voxel_idx] = priorities[i]
                best_idx[voxel_idx] = i

        keep_idx = best_idx[best_idx >= 0]
        keep_idx = np.sort(keep_idx)

        for key in concat_keys:
            if key in data_dict and data_dict[key] is not None:
                data_dict[key] = data_dict[key][keep_idx]

        return data_dict

    def _apply_overlay(self, data_dict):
        """Overlay multiple events into a single point cloud."""
        n_events = self._sample_overlay_n_events()
        if n_events <= 1:
            return data_dict

        concat_keys = [
            "coord", "energy", "segment_motif", "segment_pid",
            "instance_particle", "instance_interaction",
            "momentum", "vertex", "segment_interaction",
        ]
        instance_keys = ("instance_particle", "instance_interaction")

        dataset_len = len(self)
        if self.overlay_allow_repeats:
            indices = [random.randint(0, dataset_len - 1) for _ in range(n_events - 1)]
        else:
            indices = random.sample(range(dataset_len), min(n_events - 1, dataset_len))

        additional_dicts = []
        for idx in indices:
            try:
                extra = self.get_data(idx)
                additional_dicts.append(extra)
            except Exception:
                continue

        if not additional_dicts:
            return data_dict

        # track max instance ID for offsetting
        max_instance = {}
        for key in instance_keys:
            if key in data_dict and data_dict[key] is not None:
                vals = data_dict[key]
                max_instance[key] = int(vals[vals != -1].max()) + 1 if (vals != -1).any() else 0
            else:
                max_instance[key] = 0


        for extra in additional_dicts:
            # offset instance IDs
            for key in instance_keys:
                if key in extra and extra[key] is not None:
                    inst = extra[key]
                    mask = inst != -1
                    inst[mask] += max_instance[key]
                    if mask.any():
                        max_instance[key] = int(inst[mask].max()) + 1

            # apply random 90-degree rotation around detector center
            if "coord" in extra:
                # rotation center is the detector volume center
                detector_center = np.array([384.0, 384.0, 384.0], dtype=np.float32)
                extra["coord"] = self._apply_random_90_rotation(extra["coord"], center=detector_center)

            # concatenate arrays
            for key in concat_keys:
                if key in data_dict and key in extra:
                    if data_dict[key] is not None and extra[key] is not None:
                        data_dict[key] = np.concatenate([data_dict[key], extra[key]], axis=0)

        # deduplicate overlapping voxels
        data_dict = self._deduplicate_voxels(data_dict, concat_keys)

        if "name" in data_dict:
            data_dict["name"] = f"{data_dict['name']}_overlay{n_events}"

        return data_dict

    def prepare_train_data(self, idx):
        """Prepare training data with transforms."""
        data_dict = self.get_data(idx % len(self))
        # apply event overlay if enabled
        if self.overlay_n_events > 1 or (isinstance(self.overlay_n_events, (tuple, list)) and self.overlay_n_events[1] > 1):
            if random.random() < self.overlay_prob:
                data_dict = self._apply_overlay(data_dict)
        return self.transform(data_dict)

    def prepare_test_data(self, idx):
        """Prepare test data with test transforms."""
        # Load data
        data_dict = self.get_data(idx % len(self))

        # apply event overlay if enabled
        if self.overlay_n_events > 1 or (isinstance(self.overlay_n_events, (tuple, list)) and self.overlay_n_events[1] > 1):
            if random.random() < self.overlay_prob:
                data_dict = self._apply_overlay(data_dict)


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

def map_instance_ids(instance_ids_array):
    """Map instance ids to new ids.

    i.e. instead of having instance ids like [0, 1, 23, 47, 52, 53, 54, 55, 56, 57],
            we want to have instance ids like [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    unique_ids_local = np.unique(instance_ids_array)
    id_mapping_local = {
        old_id: new_id
        for new_id, old_id in enumerate(unique_ids_local[unique_ids_local >= 0])
    }
    return np.array(
        [id_mapping_local.get(id_val, -1) for id_val in instance_ids_array],
        dtype=np.int32,
    )[:, None]