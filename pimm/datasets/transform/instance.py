"""Instance, anchor, and local-geometry transforms."""

from .common import *


@TRANSFORMS.register_module()
class ComputeAnchors(object):
    """Compute anchors once per event and attach to data_dict['anchors'].

    Args:
        cfg: dict overriding anchor defaults
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or dict()

    def __call__(self, data_dict):
        if compute_anchors is None:
            return data_dict
        if "coord" not in data_dict or "energy" not in data_dict:
            return data_dict
        xyz = data_dict["coord"].astype(np.float32)
        # energy may be (N,) or (N,1); use (N,)
        e = data_dict["energy"]
        if e.ndim > 1 and e.shape[-1] == 1:
            e = e.reshape(-1)
        # Merge defaults with overrides
        cfg = dict(ANCHOR_DEFAULT_CFG)
        cfg.update(self.cfg)
        anchors = compute_anchors(xyz=xyz, energy=e, is_shower_like=None, cfg=cfg)
        data_dict["anchors"] = anchors
        # Exclude LEDs from being used inadvertently elsewhere
        return data_dict

@TRANSFORMS.register_module()
class InstanceParser(object):
    def __init__(
        self,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        compute_axis_stats=False,
        axis_min_points=5,
        axis_eps=1e-6,
        axis_default=(1.0, 0.0, 0.0),
        axis_normalize_half_extent=True,
    ):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.compute_axis_stats = bool(compute_axis_stats)
        self.axis_min_points = max(int(axis_min_points), 1)
        self.axis_eps = float(axis_eps)
        axis_default = np.asarray(axis_default, dtype=np.float32)
        if axis_default.shape != (3,):
            raise ValueError("axis_default must have shape (3,)")
        axis_norm = np.linalg.norm(axis_default)
        if axis_norm <= 0:
            raise ValueError("axis_default must be non-zero")
        self.axis_default = axis_default / axis_norm
        self.axis_normalize_half_extent = bool(axis_normalize_half_extent)

    def __call__(self, data_dict):
        coord = np.asarray(data_dict["coord"])
        coord_dtype = coord.dtype
        # ensure 1D arrays for correct boolean indexing
        segment = data_dict["segment"]
        if isinstance(segment, np.ndarray):
            segment = segment.reshape(-1)
        else:
            segment = np.asarray(segment).reshape(-1)
        instance = data_dict["instance"]
        if isinstance(instance, np.ndarray):
            instance = instance.reshape(-1)
        else:
            instance = np.asarray(instance).reshape(-1)
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = np.ones((coord.shape[0], 3), dtype=coord_dtype) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8), dtype=coord_dtype) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index

        if self.compute_axis_stats:
            axis_default = self.axis_default.astype(coord_dtype, copy=False)
            axis = np.tile(axis_default, (coord.shape[0], 1))
            axis_coord = np.zeros(coord.shape[0], dtype=coord_dtype)
            axis_coord_normalized = np.zeros(coord.shape[0], dtype=coord_dtype)
            axis_length = np.zeros(coord.shape[0], dtype=coord_dtype)
            axis_weight = np.zeros(coord.shape[0], dtype=coord_dtype)
        else:
            axis = axis_coord = axis_coord_normalized = axis_length = axis_weight = None

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            bbox_centroid = coord_.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_centroid.astype(coord_dtype, copy=False)
            bbox_row = np.concatenate([bbox_center, bbox_size, bbox_theta, bbox_class])
            bbox[instance_id] = bbox_row.astype(coord_dtype, copy=False)

            if self.compute_axis_stats:
                point_count = coord_.shape[0]
                valid_axis = False
                axis_vec = axis_default
                axis_coord_local = np.zeros(point_count, dtype=coord_dtype)
                axis_coord_norm_local = np.zeros(point_count, dtype=coord_dtype)
                axis_length_value = 0.0
                if point_count >= self.axis_min_points:
                    centered = coord_.astype(np.float32, copy=False) - bbox_centroid.astype(np.float32, copy=False)
                    if np.linalg.norm(centered, axis=1).max() > self.axis_eps:
                        cov = centered.T @ centered
                        cov /= max(point_count, 1)
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        principal_index = int(np.argmax(eigvals))
                        principal_val = float(eigvals[principal_index])
                        principal_vec = eigvecs[:, principal_index].astype(np.float32, copy=False)
                        principal_norm = float(np.linalg.norm(principal_vec))
                        if principal_norm > self.axis_eps and principal_val > self.axis_eps:
                            axis_vec = principal_vec / principal_norm
                            projections = centered @ axis_vec
                            max_proj = float(projections.max())
                            min_proj = float(projections.min())
                            axis_length_value = max_proj - min_proj
                            if axis_length_value > self.axis_eps:
                                valid_axis = True
                                axis_coord_local = projections.astype(coord_dtype, copy=False)
                                denom = axis_length_value * 0.5 if self.axis_normalize_half_extent else axis_length_value
                                denom = float(denom) + self.axis_eps
                                axis_coord_norm_local = (axis_coord_local / denom).astype(coord_dtype, copy=False)
                if not valid_axis:
                    axis_vec = axis_default
                    axis_coord_local.fill(0.0)
                    axis_coord_norm_local.fill(0.0)
                    axis_length_value = 0.0
                axis[mask_] = axis_vec.astype(coord_dtype, copy=False)
                axis_coord[mask_] = axis_coord_local
                axis_coord_normalized[mask_] = axis_coord_norm_local
                axis_length[mask_] = axis_length_value
                axis_weight_value = 1.0 if valid_axis else 0.0
                axis_weight[mask_] = axis_weight_value

        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        if self.compute_axis_stats:
            data_dict["instance_axis"] = axis
            data_dict["instance_axis_coord"] = axis_coord
            data_dict["instance_axis_coord_normalized"] = axis_coord_normalized
            data_dict["instance_axis_length"] = axis_length
            data_dict["instance_axis_weight"] = axis_weight
            data_dict["instance_axis_coord_weight"] = axis_weight
        return data_dict

@TRANSFORMS.register_module()
class LocalCovarianceFeatures(object):
    def __init__(
        self,
        k=16,
        include_self=False,
        gaussian_weight=False,
        gaussian_sigma=None,
        out_keys=("local_eigvals", "local_shape"),
    ):
        self.k = int(k)
        self.include_self = bool(include_self)
        self.gaussian_weight = bool(gaussian_weight)
        self.gaussian_sigma = gaussian_sigma
        self.out_keys = out_keys

    def __call__(self, data_dict):
        if "coord" not in data_dict:
            return data_dict
        coord = np.asarray(data_dict["coord"]).astype(np.float32, copy=False)
        n_points = coord.shape[0]
        if n_points == 0:
            return data_dict

        # query kNN
        k_query = min(self.k + (0 if self.include_self else 1), max(1, n_points))
        kd = cKDTree(coord)
        dists, idxs = kd.query(coord, k=k_query)

        # ensure shape (N, K)
        if dists.ndim == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        # drop self if requested
        if not self.include_self and idxs.shape[1] > 0:
            dists = dists[:, 1:]
            idxs = idxs[:, 1:]

        neighbors = coord[idxs]  # (N, K, 3)
        center = coord[:, None, :]  # (N, 1, 3)
        offsets = neighbors - center

        if self.gaussian_weight:
            if self.gaussian_sigma is None:
                sigma = np.median(dists, axis=1, keepdims=True) + 1e-6
            else:
                sigma = float(self.gaussian_sigma)
            w = np.exp(-0.5 * (dists / (sigma + 1e-6)) ** 2).astype(np.float32)
            w_sum = np.sum(w, axis=1, keepdims=True) + 1e-6
            mean = np.sum(offsets * w[..., None], axis=1, keepdims=True) / w_sum[..., None]
            centered = offsets - mean
            cov = np.einsum("nki,nkj->nij", centered * w[..., None], centered) / w_sum[..., None]
        else:
            mean = np.mean(offsets, axis=1, keepdims=True)
            centered = offsets - mean
            denom = float(max(1, centered.shape[1] - 1))
            cov = np.einsum("nki,nkj->nij", centered, centered) / denom

        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 0.0, None)[:, ::-1]

        l1 = eigvals[:, 0] + 1e-12
        l2 = eigvals[:, 1] + 1e-12
        l3 = eigvals[:, 2] + 1e-12
        r21 = l2 / l1
        r32 = l3 / l2
        r31 = l3 / l1
        suml = (l1 + l2 + l3) + 1e-12
        curvature = l3 / suml
        local_shape = np.stack([r21, r32, r31, curvature], axis=1).astype(coord.dtype, copy=False)
        local_eigvals = eigvals.astype(coord.dtype, copy=False)

        data_dict[self.out_keys[0]] = local_eigvals
        data_dict[self.out_keys[1]] = local_shape

        data_dict["index_valid_keys"].append(self.out_keys[0])
        data_dict["index_valid_keys"].append(self.out_keys[1])
        return data_dict
