"""Hierarchical masked autoencoder transforms and collation."""

from .common import *


@TRANSFORMS.register_module()
class HierarchicalMaskGenerator(object):
    """
    Generate hierarchical masks for MAE style pretraining.

    Points are grouped into patches at patch_size granularity, then a fraction
    are randomly masked. The visible points go to the encoder, while masked
    patch information is stored for the decoder to reconstruct.
    
    Uses same grid-based hashing as GridSample to ensure exact alignment with
    PTv3's coarsest features after hierarchical pooling.
    
    Important: Centroids are grid cell centers (not point means) to ensure
    proper 1:1 correspondence between patches and coarse encoder features.
    """

    def __init__(
        self,
        patch_size: float = 0.016,
        mask_ratio: float = 0.6,
        points_per_patch: int = 128,
        min_points_per_patch: int = 0,
        view_keys: tuple = ("coord", "origin_coord", "energy"),
    ):
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.points_per_patch = points_per_patch
        self.min_points_per_patch = min_points_per_patch
        self.view_keys = view_keys

    @staticmethod
    def fnv_hash_vec(arr):
        """FNV64-1A hash for grid coordinates"""
        assert arr.ndim == 2
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        n_points = coord.shape[0]

        if n_points == 0:
            data_dict["hmae_valid"] = False
            return data_dict

        # grid coordinates aligned to patch_size, matching PTv3's grid structure
        # matches: floor((coord - coord.min()) / patch_size) after 4x stride-2 poolings
        coord_min = coord.min(axis=0)
        grid_coord = np.floor((coord - coord_min) / self.patch_size).astype(np.int64)

        # spatial hash using FNV (same as GridSample for consistency)
        patch_ids = self.fnv_hash_vec(grid_coord)

        # unique patches and assignment of each point to a patch
        unique_patches, inverse_indices, patch_counts = np.unique(
            patch_ids, return_inverse=True, return_counts=True
        )
        n_patches = len(unique_patches)

        # keep only patches with enough points
        valid_patch_mask = patch_counts >= self.min_points_per_patch
        valid_patch_indices = np.where(valid_patch_mask)[0]
        n_valid_patches = len(valid_patch_indices)

        if n_valid_patches < 2:
            data_dict["hmae_valid"] = False
            return data_dict

        # choose which patches are masked vs visible
        n_mask = max(1, int(n_valid_patches * self.mask_ratio))
        n_visible = n_valid_patches - n_mask

        perm = np.random.permutation(n_valid_patches)
        masked_patch_local_idx = perm[:n_mask]
        visible_patch_local_idx = perm[n_mask:]

        masked_patch_idx = valid_patch_indices[masked_patch_local_idx]
        visible_patch_idx = valid_patch_indices[visible_patch_local_idx]

        # vectorized visible mask: map patch index -> visible flag
        is_visible_patch = np.zeros(n_patches, dtype=bool)
        is_visible_patch[visible_patch_idx] = True
        visible_mask = is_visible_patch[inverse_indices]

        # extract visible data for encoder
        visible_data = {}
        for key in self.view_keys:
            if key in data_dict:
                visible_data[key] = data_dict[key][visible_mask]

        # sort points once by patch index for efficient masked patch processing
        # this avoids repeatedly doing (inverse_indices == patch_idx) per patch
        order = np.argsort(inverse_indices)
        sorted_coord = coord[order]
        sorted_grid_coord = grid_coord[order]
        has_energy = "energy" in data_dict
        if has_energy:
            sorted_energy = data_dict["energy"][order]

        # build CSR style offsets from patch_counts
        # patch_offsets[j] .. patch_offsets[j+1] is the slice for patch j
        patch_offsets = np.concatenate(
            [np.array([0], dtype=np.int64), np.cumsum(patch_counts, dtype=np.int64)]
        )

        # masked patch targets
        masked_centroids = []
        masked_target_coords = []  # list of (Ni, 3)
        masked_target_energy = []  # list of (Ni, 1) if available
        masked_point_counts = []

        norm_factor = self.patch_size / 2.0

        for patch_idx in masked_patch_idx:
            start = patch_offsets[patch_idx]
            end = patch_offsets[patch_idx + 1]
            if end <= start:
                continue  # should not happen if min_points_per_patch checked

            patch_coord = sorted_coord[start:end]
            patch_grid_coord = sorted_grid_coord[start:end]
            
            # centroid = geometric center of grid cell (not point mean)
            # all points in this patch share the same grid coordinate
            grid_cell = patch_grid_coord[0]  # same for all points in patch
            centroid = grid_cell * self.patch_size + self.patch_size / 2.0 + coord_min
            masked_centroids.append(centroid)

            # relative coords in [-1, 1]
            rel_coord = (patch_coord - centroid) / norm_factor
            masked_target_coords.append(rel_coord)

            if has_energy:
                patch_energy = sorted_energy[start:end]
                masked_target_energy.append(patch_energy)

            masked_point_counts.append(patch_coord.shape[0])

        if len(masked_centroids) == 0:
            # very rare case: all masked patches dropped for some reason
            data_dict["hmae_valid"] = False
            return data_dict

        # pack results
        data_dict["hmae_valid"] = True

        data_dict["visible_coord"] = visible_data.get("coord", np.array([]))
        data_dict["visible_origin_coord"] = visible_data.get(
            "origin_coord", visible_data.get("coord", np.array([]))
        )

        if "energy" in visible_data:
            data_dict["visible_energy"] = visible_data["energy"]
        else:
            v_n = data_dict["visible_coord"].shape[0]
            data_dict["visible_energy"] = np.zeros((v_n, 1), dtype=np.float32)

        data_dict["masked_centroids"] = np.asarray(masked_centroids, dtype=np.float32)
        data_dict["masked_point_counts"] = np.asarray(masked_point_counts, dtype=np.int64)

        # pack masked patch targets into flattened arrays with offsets (no padding)
        # This replaces the need for HMAECollate
        target_coords_list = []
        target_energy_list = []
        
        for i, coords in enumerate(masked_target_coords):
            target_coords_list.append(coords)
            if i < len(masked_target_energy):
                energy = masked_target_energy[i]
                if energy.ndim == 1:
                    energy = energy[:, None]
                target_energy_list.append(energy)
            else:
                # Create zeros if energy not available for this patch
                target_energy_list.append(np.zeros((coords.shape[0], 1), dtype=np.float32))
        
        # concatenate into flattened arrays
        target_coords_flat = np.concatenate(target_coords_list, axis=0)  # (total_points, 3)
        target_energy_flat = np.concatenate(target_energy_list, axis=0)  # (total_points, 1)
        
        # compute offset per batch sample (not per patch)
        # output just the total point count for this sample
        # batching will convert this to cumulative offsets per batch sample
        total_points = target_coords_flat.shape[0]
        target_offset = np.array([total_points], dtype=np.int64)
        
        data_dict["target_coords"] = target_coords_flat
        data_dict["target_energy"] = target_energy_flat
        data_dict["target_offset"] = target_offset

        data_dict["n_visible_patches"] = n_visible
        data_dict["n_masked_patches"] = len(masked_centroids)

        return data_dict

@TRANSFORMS.register_module()
class HMAECollate(object):
    """Pack HMAE variable-length masked patch targets before batch collation.

    Packs target coordinates/energies into flattened arrays with offsets (no padding).
    Assumes 'energy' will always be present in data_dict.
    """

    def __init__(
        self,
        points_per_patch: int = 128,
    ):
        """Store the patch-size contract used by the HMAE target builder."""
        self.points_per_patch = points_per_patch

    def __call__(self, data_dict):
        """Flatten masked target lists into target arrays plus target_offset."""
        if not data_dict.get("hmae_valid", False):
            return data_dict

        masked_target_coords = data_dict["masked_target_coords"]
        masked_target_energy = data_dict["masked_target_energy"]

        # pack into flattened arrays with offsets
        target_coords_list = []
        target_energy_list = []
        target_point_counts = []

        for i, coords in enumerate(masked_target_coords):
            n_pts = coords.shape[0]
            target_coords_list.append(coords)
            target_point_counts.append(n_pts)

            energy = masked_target_energy[i]
            if energy.ndim == 1:
                energy = energy[:, None]
            target_energy_list.append(energy)

        # concatenate into flattened arrays
        target_coords_flat = np.concatenate(target_coords_list, axis=0)  # (total_points, 3)
        target_energy_flat = np.concatenate(target_energy_list, axis=0)  # (total_points, 1)

        # compute offset per batch sample (not per patch)
        # output just the total point count for this sample
        # batching will convert this to cumulative offsets per batch sample
        total_points = target_coords_flat.shape[0]
        target_offset = np.array([total_points], dtype=np.int64)

        data_dict["target_coords"] = target_coords_flat
        data_dict["target_energy"] = target_energy_flat
        data_dict["target_offset"] = target_offset

        # clean up lists
        del data_dict["masked_target_coords"]
        del data_dict["masked_target_energy"]

        return data_dict
