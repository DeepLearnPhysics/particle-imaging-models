"""Coordinate, sampling, voxelization, and crop transforms."""

from .common import *


@TRANSFORMS.register_module()
class NormalizeCoord(object):
    def __init__(self, center=None, scale=None):
        self.center = center
        self.scale = scale

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # modified from pointnet2
            if self.center is None:
                centroid = np.mean(data_dict["coord"], axis=0)
            else:
                centroid = np.array(self.center)
            data_dict["coord"] -= centroid

            if self.scale is None:
                m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
                scale = m
            else:
                scale = self.scale
            data_dict["coord"] = data_dict["coord"] / scale
            _apply_to_v3_vertex(data_dict, lambda vertex: (vertex - centroid) / scale)
        return data_dict

@TRANSFORMS.register_module()
class PositiveShift(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
            _apply_to_v3_vertex(data_dict, lambda vertex: vertex - coord_min)
        return data_dict

@TRANSFORMS.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True, axes=("x", "y", "z")):
        self.apply_z = apply_z
        if not isinstance(axes, tuple):
            axes = (axes,)
        self.axes = axes
        
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            for axis in self.axes:
                if axis == "x":
                    x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                    x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                    shift = (x_min + x_max) / 2
                    data_dict["coord"][:, 0] -= shift
                    _apply_to_v3_vertex(
                        data_dict, lambda vertex, shift=shift: _translate_axis(vertex, 0, -shift)
                    )
                elif axis == "y":
                    x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                    x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                    shift = (y_min + y_max) / 2
                    data_dict["coord"][:, 1] -= shift
                    _apply_to_v3_vertex(
                        data_dict, lambda vertex, shift=shift: _translate_axis(vertex, 1, -shift)
                    )
                elif axis == "z":
                    x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                    x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                    shift = (z_min + z_max) / 2
                    data_dict["coord"][:, 2] -= shift
                    _apply_to_v3_vertex(
                        data_dict, lambda vertex, shift=shift: _translate_axis(vertex, 2, -shift)
                    )
        return data_dict

@TRANSFORMS.register_module()
class ConditionalRandomTransform(object):
    _max_value_pilarnet = 2 * pow(3, 0.5) / 3 # (768) / (768 * 3 ** 0.5 / 2)
    def __init__(self, p=0.5, axes=("x", "y", "z"), buffer_size=0.05, bounds=((-1, 1), (-1, 1), (-1, 1))):
        self.p = p
        if not isinstance(axes, tuple):
            axes = (axes,)
        self.axes = axes
        self.buffer_size = buffer_size
        self.bounds = bounds

    def __call__(self, data_dict):
        if "coord" not in data_dict.keys():
            return data_dict
        coord = data_dict["coord"]

        for dim, axis in enumerate(("x", "y", "z")):
            if axis not in self.axes:
                continue

            bounds = self.bounds[dim]
            min_val = np.min(coord[:, dim])
            max_val = np.max(coord[:, dim])

            # skip if entirely interior (not near any wall)
            if (min_val >= bounds[0] + self.buffer_size) and (max_val <= bounds[1] - self.buffer_size):
                continue

            if random.random() <= self.p:
                lower, upper = bounds
                near_lower = min_val <= lower + self.buffer_size
                near_upper = max_val >= upper - self.buffer_size

                # base feasibility to keep all points within bounds
                t_low = lower - min_val
                t_high = upper - max_val

                if near_lower and not near_upper:
                    # keep near lower wall
                    t_high = min(t_high, (lower + self.buffer_size) - min_val)
                elif near_upper and not near_lower:
                    # keep near upper wall
                    t_low = max(t_low, (upper - self.buffer_size) - max_val)
                elif near_lower and near_upper:
                    # keep near both walls
                    t_low = max(t_low, (upper - self.buffer_size) - max_val)
                    t_high = min(t_high, (lower + self.buffer_size) - min_val)
                else:
                    # interior (should have been caught above)
                    continue

                if t_low <= t_high:
                    translation = np.random.uniform(t_low, t_high)
                    coord[:, dim] += translation
                    _apply_to_v3_vertex(
                        data_dict,
                        lambda vertex, dim=dim, translation=translation: _translate_axis(
                            vertex, dim, translation
                        ),
                    )

        data_dict["coord"] = coord
        return data_dict

@TRANSFORMS.register_module()
class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            shift = np.array([shift_x, shift_y, shift_z])
            data_dict["coord"] += shift
            _apply_to_v3_vertex(data_dict, lambda vertex: vertex + shift)
        return data_dict

@TRANSFORMS.register_module()
class PointClip(object):
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1)):
        self.point_cloud_range = point_cloud_range

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            data_dict["coord"] = np.clip(
                data_dict["coord"],
                a_min=self.point_cloud_range[:3],
                a_max=self.point_cloud_range[3:],
            )
        return data_dict

@TRANSFORMS.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = np.unique(np.append(idx, data_dict["sampled_index"]))
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx])[0]
            data_dict = index_operator(data_dict, idx)
        return data_dict

@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            center = np.asarray(center)
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
            _apply_to_v3_vertex(
                data_dict,
                lambda vertex: np.dot(vertex - center, np.transpose(rot_t)) + center,
            )
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict

@TRANSFORMS.register_module()
class RandomRotateTargetAngle(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            center = np.asarray(center)
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
            _apply_to_v3_vertex(
                data_dict,
                lambda vertex: np.dot(vertex - center, np.transpose(rot_t)) + center,
            )
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict

@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            data_dict["coord"] *= scale
            _apply_to_v3_vertex(data_dict, lambda vertex: vertex * scale)
        return data_dict

@TRANSFORMS.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5, axes=("x", "y",)):
        self.p = p
        if not isinstance(axes, tuple):
            axes = (axes,)
        self.axes = axes

    def __call__(self, data_dict):
        for axis in self.axes:
            if axis == "x" and random.random() < self.p:
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
                _apply_to_v3_vertex(
                    data_dict,
                    lambda vertex: np.column_stack(
                        (-vertex[:, 0], vertex[:, 1], vertex[:, 2])
                    ),
                )
                if "normal" in data_dict.keys():
                    data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
            elif axis == "y" and random.random() < self.p:
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
                _apply_to_v3_vertex(
                    data_dict,
                    lambda vertex: np.column_stack(
                        (vertex[:, 0], -vertex[:, 1], vertex[:, 2])
                    ),
                )
                if "normal" in data_dict.keys():
                    data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
            elif axis == "z" and random.random() < self.p:
                data_dict["coord"][:, 2] = -data_dict["coord"][:, 2]
                _apply_to_v3_vertex(
                    data_dict,
                    lambda vertex: np.column_stack(
                        (vertex[:, 0], vertex[:, 1], -vertex[:, 2])
                    ),
                )
                if "normal" in data_dict.keys():
                    data_dict["normal"][:, 2] = -data_dict["normal"][:, 2]
        return data_dict

@TRANSFORMS.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05, keys=("coord",), p=1.0):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip
        if not isinstance(keys, tuple):
            keys = (keys,)
        self.keys = keys
        self.p = p
    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        for k in self.keys:
            if k in data_dict.keys():
                jitter = np.clip(
                    self.sigma
                    * np.random.randn(data_dict[k].shape[0], data_dict[k].shape[1]),
                    -self.clip,
                    self.clip,
                )
                data_dict[k] += jitter
            else:
                raise ValueError(f"Key {k} not found in data_dict")
        return data_dict

@TRANSFORMS.register_module()
class MultiplicativeRandomJitter(object):
    def __init__(self, sigma=0.05, clip=0.05, keys=("energy",), p=0.5):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip
        if not isinstance(keys, tuple):
            keys = (keys,)
        self.keys = keys
        self.p = p

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        for k in self.keys:
            if k in data_dict.keys():
                noise = np.clip(
                    np.random.randn(*data_dict[k].shape) * self.sigma,
                    -self.clip,
                    self.clip,
                )
                data_dict[k] *= 1.0 + noise
            else:
                raise ValueError(f"Key {k} not found in data_dict")
        return data_dict

@TRANSFORMS.register_module()
class SetRandomValue(object):
    def __init__(self, sigma=0.05, clip=0.05, keys=("energy",)):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip
        self.keys = keys if isinstance(keys, tuple) else (keys,)

    def __call__(self, data_dict):
        for k in self.keys:
            if k in data_dict.keys():
                data_dict[k] = np.clip(
                    np.random.randn(*data_dict[k].shape) * self.sigma,
                    -self.clip,
                    self.clip,
                )
            else:
                raise ValueError(f"Key {k} not found in data_dict")
        return data_dict

@TRANSFORMS.register_module()
class ClipGaussianJitter(object):
    def __init__(self, scalar=0.02, store_jitter=False):
        self.scalar = scalar
        self.mean = np.mean(3)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.random.multivariate_normal(
                self.mean, self.cov, data_dict["coord"].shape[0]
            )
            jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
            data_dict["coord"] += jitter
            if self.store_jitter:
                data_dict["jitter"] = jitter
        return data_dict

@TRANSFORMS.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(
                        data_dict["coord"], granularity, magnitude
                    )
        return data_dict

@TRANSFORMS.register_module()
class GridSample(object):
    """Voxel-grid sample point clouds for train or test-time fragmentation."""

    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
        sum_keys=None,
        min_keys=None,
    ):
        """Configure voxel size, hash function, return keys, and reducers."""
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement
        self.sum_keys = sum_keys or []
        self.min_keys = min_keys or []

    def __call__(self, data_dict):
        """Apply grid sampling and return one dict or test fragments."""
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            reduced = {}
            if self.sum_keys or self.min_keys:
                voxel_of_point = np.empty(len(key), dtype=inverse.dtype)
                voxel_of_point[idx_sort] = inverse
                num_voxels = len(count)
                for sk in self.sum_keys:
                    if sk in data_dict:
                        vals = data_dict[sk]
                        agg = np.zeros((num_voxels,) + vals.shape[1:], dtype=vals.dtype)
                        np.add.at(agg, voxel_of_point, vals)
                        reduced[sk] = agg
                for mk in self.min_keys:
                    if mk in data_dict:
                        vals = data_dict[mk]
                        if np.issubdtype(vals.dtype, np.floating):
                            fill = np.inf
                        else:
                            fill = np.iinfo(vals.dtype).max
                        agg = np.full((num_voxels,) + vals.shape[1:], fill, dtype=vals.dtype)
                        np.minimum.at(agg, voxel_of_point, vals)
                        reduced[mk] = agg
            data_dict = index_operator(data_dict, idx_unique)
            for sk, agg in reduced.items():
                data_dict[sk] = agg
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
                data_dict["index_valid_keys"].append("grid_coord")
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
                data_dict["index_valid_keys"].append("displacement")
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = index_operator(data_dict, idx_part, duplicate=True)
                data_part["index"] = idx_part
                if self.return_inverse:
                    data_part["inverse"] = np.zeros_like(inverse)
                    data_part["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                    data_dict["index_valid_keys"].append("grid_coord")
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                    data_dict["index_valid_keys"].append("displacement")
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

@TRANSFORMS.register_module()
class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        if data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    np.random.randint(data_dict["coord"].shape[0])
                ]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                :point_max
            ]
            data_dict = index_operator(data_dict, idx_crop)
        return data_dict

@TRANSFORMS.register_module()
class HardExampleCrop(object):
    def __init__(
        self,
        point_max=80000,
        sample_rate=None,
        hard_labels=(2, 3),
        min_hard_points=1,
        attempts=5,
        fallback="none",  # random | center | none
        p=0.5,
    ):
        self.point_max = point_max
        self.sample_rate = sample_rate
        self.hard_labels = tuple(hard_labels)
        self.min_hard_points = int(min_hard_points)
        self.attempts = int(attempts)
        assert fallback in ["random", "center", "none"]
        self.fallback = fallback
        self.p = p
    def __call__(self, data_dict):
        assert "coord" in data_dict
        if np.random.rand() >= self.p:
            return data_dict
        n_points = data_dict["coord"].shape[0]
        point_max = (
            int(self.sample_rate * n_points)
            if self.sample_rate is not None
            else self.point_max
        )
        if n_points <= point_max:
            return data_dict

        coord = data_dict["coord"]
        segment = data_dict.get("segment", None)

        if segment is not None:
            seg = segment.reshape(-1)
            hard_mask = np.isin(seg, self.hard_labels)
        else:
            hard_mask = np.zeros(n_points, dtype=bool)

        if hard_mask.any():
            hard_indices = np.where(hard_mask)[0]
            last_idx_crop = None
            for _ in range(max(1, self.attempts)):
                cidx = np.random.choice(hard_indices)
                center = coord[cidx]
                idx_crop = np.argsort(np.sum(np.square(coord - center), axis=1))[:point_max]
                last_idx_crop = idx_crop
                if self.min_hard_points <= np.count_nonzero(hard_mask[idx_crop]):
                    return index_operator(data_dict, idx_crop)
            return index_operator(data_dict, last_idx_crop)

        # fallback when no hard points present
        if self.fallback == "none":
            return data_dict
        elif self.fallback == "center":
            center = coord[n_points // 2]
        else:  # random
            center = coord[np.random.randint(n_points)]
        idx_crop = np.argsort(np.sum(np.square(coord - center), axis=1))[:point_max]
        return index_operator(data_dict, idx_crop)

@TRANSFORMS.register_module()
class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        data_dict = index_operator(data_dict, shuffle_index)
        return data_dict

@TRANSFORMS.register_module()
class CropBoundary(object):
    def __call__(self, data_dict):
        assert "segment" in data_dict
        segment = data_dict["segment"].flatten()
        mask = (segment != 0) * (segment != 1)
        data_dict = index_operator(data_dict, mask)
        return data_dict

@TRANSFORMS.register_module()
class RandomDrop(object):
    """Randomly set a fraction of one array-like key to a configured value."""

    def __init__(self, key: str, p_apply: float = 0.5, p_drop: float = 0.2, value: float = 0.0):
        """Configure key, application probability, drop fraction, and value."""
        self.key = key
        self.p_apply = p_apply
        self.p_drop = p_drop
        self.value = value

    def __call__(self, data_dict):
        """Apply random value replacement in place when selected."""
        if self.key in data_dict.keys() and np.random.rand() < self.p_apply:
            n = data_dict[self.key].shape[0]
            idx = np.random.choice(n, int(n*self.p_drop), replace=False)
            data_dict[self.key][idx][:] = self.value
        return data_dict
