"""Event-overlay augmentation shared by the map-style PILArNet readers.

Overlay composites several events into one point cloud (rotating each by random
90-degree increments about the detector centre and deduplicating colliding
voxels by semantic priority). It operates purely on decoded ``data_dict``s plus
random ``get_data(idx)`` access, so it is identical for the HDF5 and parquet
readers -- both mix in :class:`PILArNetOverlayMixin`.

A host class must provide:
    - ``overlay_n_events`` / ``overlay_prob`` / ``overlay_allow_repeats`` attrs,
    - ``revision`` attr,
    - ``get_data(idx)`` returning one event's ``data_dict``,
    - ``_num_source_events()`` -> count of distinct events (pre-``loop``).
"""

import random

import numpy as np

from .decode import DEFAULT_LABEL_PRIORITY, V3_EXTRA_KEYS


class PILArNetOverlayMixin:
    """Random multi-event overlay for map-style PILArNet datasets."""

    def _overlay_enabled(self):
        """True when the configured overlay count can ever exceed one event."""
        n = self.overlay_n_events
        if isinstance(n, (tuple, list)):
            return n[1] > 1
        return n > 1

    def _maybe_overlay(self, data_dict):
        """Apply overlay to ``data_dict`` with probability ``overlay_prob``."""
        if self._overlay_enabled() and random.random() < self.overlay_prob:
            return self._apply_overlay(data_dict)
        return data_dict

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

    def _apply_random_90_rotation(self, coord, center=None, rotations=None):
        """Apply random 90-degree rotations around x, y, z axes centered at given point."""
        if center is None:
            center = np.array([384.0, 384.0, 384.0], dtype=np.float32)
        if rotations is None:
            rotations = {axis: random.randint(0, 3) for axis in ("x", "y", "z")}
        coord = coord - center
        for axis in ["x", "y", "z"]:
            n_rot = rotations[axis]
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
        if self.revision in ("v3", "v3_extra"):
            concat_keys.extend(("is_primary", *V3_EXTRA_KEYS))
        # Keep order stable while removing duplicates.
        concat_keys = list(dict.fromkeys(concat_keys))
        instance_keys = ("instance_particle", "instance_interaction")

        # Sample from the distinct-event pool (pre-loop) so overlay indices stay
        # valid regardless of the loop multiplier.
        pool = self._num_source_events()
        if self.overlay_allow_repeats:
            indices = [random.randint(0, pool - 1) for _ in range(n_events - 1)]
        else:
            indices = random.sample(range(pool), min(n_events - 1, pool))

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
                rotations = {axis: random.randint(0, 3) for axis in ("x", "y", "z")}
                extra["coord"] = self._apply_random_90_rotation(
                    extra["coord"], center=detector_center, rotations=rotations
                )
                if self.revision in ("v3", "v3_extra"):
                    for position_key in ("vertex", "interaction_vertex"):
                        if position_key not in extra:
                            continue
                        valid = ~(extra[position_key] == -1).all(axis=1)
                        extra[position_key][valid] = self._apply_random_90_rotation(
                            extra[position_key][valid],
                            center=detector_center,
                            rotations=rotations,
                        )
                    for vector_key in ("momentum_vector", "momentum_vec"):
                        if vector_key not in extra:
                            continue
                        valid = ~(extra[vector_key] == -1).all(axis=1)
                        extra[vector_key][valid] = self._apply_random_90_rotation(
                            extra[vector_key][valid],
                            center=np.zeros(3, dtype=np.float32),
                            rotations=rotations,
                        )
                    if "momentum_vector" in extra:
                        extra["px"] = extra["momentum_vector"][:, 0:1]
                        extra["py"] = extra["momentum_vector"][:, 1:2]
                        extra["pz"] = extra["momentum_vector"][:, 2:3]

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
