"""Contrastive and multiview SSL transform builders."""

from .common import *
from .base import Compose


@TRANSFORMS.register_module()
class ContrastiveViewsGenerator(object):
    def __init__(
        self,
        view_keys=("coord", "color", "normal", "origin_coord"),
        view_trans_cfg=None,
    ):
        self.view_keys = view_keys
        self.view_trans = Compose(view_trans_cfg)

    def __call__(self, data_dict):
        view1_dict = dict()
        view2_dict = dict()
        for key in self.view_keys:
            view1_dict[key] = data_dict[key].copy()
            view2_dict[key] = data_dict[key].copy()
        view1_dict = self.view_trans(view1_dict)
        view2_dict = self.view_trans(view2_dict)
        for key, value in view1_dict.items():
            data_dict["view1_" + key] = value
        for key, value in view2_dict.items():
            data_dict["view2_" + key] = value
        return data_dict

@TRANSFORMS.register_module()
class MultiViewGenerator(object):
    def __init__(
        self,
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=None,
        global_transform=None,
        local_transform=None,
        max_size=65536,
        center_height_scale=(0, 1),
        shared_global_view=False,
        center_sampling="random",  # or cnms
        center_sampling_kwargs=None,
        view_keys=("coord", "origin_coord", "color", "normal"),
        # Anchor-biased sampling
        anchor_bias_ratio=0.6,
        anchor_radius_scale=1.5,
        anchor_keys=("endpoints", "branches_track", "branches_shower", "bragg"),
    ):
        self.global_view_num = global_view_num
        self.global_view_scale = global_view_scale
        self.local_view_num = local_view_num
        self.local_view_scale = local_view_scale
        self.global_shared_transform = Compose(global_shared_transform)
        self.global_transform = Compose(global_transform)
        self.local_transform = Compose(local_transform)
        self.max_size = max_size
        self.center_height_scale = center_height_scale
        self.shared_global_view = shared_global_view
        self.view_keys = view_keys
        assert "coord" in view_keys
        self.center_sampling = center_sampling
        self.center_sampling_kwargs = center_sampling_kwargs
        # Anchors
        self.anchor_bias_ratio = anchor_bias_ratio
        self.anchor_radius_scale = anchor_radius_scale
        self.anchor_keys = anchor_keys

    def get_view(self, point, center, scale, size_override: Optional[int] = None):
        coord = point["coord"]
        max_size = min(self.max_size, coord.shape[0])
        if max_size <= 0:
            raise ValueError("Cannot generate a view from an empty point cloud")
        if size_override is None:
            size = int(np.random.uniform(*scale) * max_size)
        else:
            size = int(size_override)
        size = max(1, min(max_size, size))
        index = np.argsort(np.sum(np.square(coord - center), axis=-1))[:size]
        view = dict(index=index)
        for key in point.keys():
            if key in self.view_keys:
                view[key] = point[key][index]

        if "index_valid_keys" in point.keys():
            # inherit index_valid_keys from point
            view["index_valid_keys"] = point["index_valid_keys"]
        return view

    def get_center(self, coord, mask=None):
        if mask is None:
            possible_centers = coord
        else:
            possible_centers = coord[np.where(mask)[0]]
        if self.center_sampling == "cnms":
            from cnms import cnms
            possible_centers, _, _ = cnms(possible_centers, **self.center_sampling_kwargs)
        return possible_centers[np.random.choice(possible_centers.shape[0])]

    def _build_global_views(self, point, major_view):
        major_coord = major_view["coord"]
        if not self.shared_global_view:
            global_views = [
                self.get_view(
                    point=point,
                    center=major_coord[np.random.randint(major_coord.shape[0])],
                    scale=self.global_view_scale,
                )
                for _ in range(self.global_view_num - 1)
            ]
        else:
            global_views = [
                {key: value.copy() for key, value in major_view.items()}
                for _ in range(self.global_view_num - 1)
            ]
        return [major_view] + global_views

    def _pack_views(self, data_dict, global_views, local_views):
        view_dict = {}
        for global_view in global_views:
            global_view.pop("index")
            global_view = self.global_transform(global_view)
            for key in self.view_keys:
                if f"global_{key}" in view_dict.keys():
                    view_dict[f"global_{key}"].append(global_view[key])
                else:
                    view_dict[f"global_{key}"] = [global_view[key]]
        view_dict["global_offset"] = np.cumsum(
            [data.shape[0] for data in view_dict["global_coord"]]
        )
        for local_view in local_views:
            local_view.pop("index")
            local_view = self.local_transform(local_view)
            for key in self.view_keys:
                if f"local_{key}" in view_dict.keys():
                    view_dict[f"local_{key}"].append(local_view[key])
                else:
                    view_dict[f"local_{key}"] = [local_view[key]]
        view_dict["local_offset"] = np.cumsum(
            [data.shape[0] for data in view_dict["local_coord"]]
        )
        for key in view_dict.keys():
            if "offset" not in key:
                view_dict[key] = np.concatenate(view_dict[key], axis=0)
        data_dict.update(view_dict)
        return data_dict

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        point = self.global_shared_transform(copy.deepcopy(data_dict))
        z_min = coord[:, 2].min()
        z_max = coord[:, 2].max()
        z_min_ = z_min + (z_max - z_min) * self.center_height_scale[0]
        z_max_ = z_min + (z_max - z_min) * self.center_height_scale[1]
        center_mask = np.logical_and(coord[:, 2] >= z_min_, coord[:, 2] <= z_max_)
        # get major global view
        major_center = coord[np.random.choice(np.where(center_mask)[0])]
        major_view = self.get_view(point, major_center, self.global_view_scale)
        major_coord = major_view["coord"]
        # get global views: restrict the center of left global view within the major global view
        global_views = self._build_global_views(point, major_view)

        # get local views: restrict the center of local view within the major global view
        cover_mask = np.zeros_like(major_view["index"], dtype=bool)
        local_views = []
        # Prepare anchor pool if available (exclude LEDs)
        anchors_pool = []
        if isinstance(data_dict.get("anchors"), dict):
            for k in self.anchor_keys:
                if k == "led":
                    continue
                v = data_dict["anchors"].get(k)
                if v is not None and len(v) > 0:
                    anchors_pool.append(v)
        anchors_pool = np.concatenate(anchors_pool, axis=0) if len(anchors_pool) > 0 else np.zeros((0,3), dtype=np.float32)

        # Map anchors to nearest point inside major view to keep locality consistent
        kd_major = cKDTree(major_coord) if major_coord.shape[0] > 0 else None
        # Estimate size override for anchor crops: approximate radius scaling via cubic relation
        # size' ~= size * (radius_scale^3)
        size_base = int(np.mean([np.random.uniform(*self.local_view_scale) * min(self.max_size, coord.shape[0]) for _ in range(4)]))
        size_override = int(max(8, min(self.max_size, size_base * (self.anchor_radius_scale ** 3))))

        # Determine counts
        num_anchor_locals = int(np.ceil(self.local_view_num * float(self.anchor_bias_ratio))) if anchors_pool.shape[0] > 0 else 0
        num_random_locals = self.local_view_num - num_anchor_locals

        # Anchor-centered locals
        for i in range(num_anchor_locals):
            if sum(~cover_mask) == 0:
                cover_mask[:] = False
            if anchors_pool.shape[0] == 0:
                break
            aidx = np.random.randint(0, anchors_pool.shape[0])
            acoord = anchors_pool[aidx]
            # Project to nearest major point to keep within major global view
            if kd_major is not None and kd_major.n > 0:
                _, nn = kd_major.query(acoord, k=1)
                center = major_coord[nn]
            else:
                center = acoord
            local_view = self.get_view(
                point=data_dict,
                center=center,
                scale=self.local_view_scale,
                size_override=size_override,
            )
            local_views.append(local_view)
            cover_mask[np.isin(major_view["index"], local_view["index"])] = True

        # Uniform random locals
        for i in range(num_random_locals):
            if sum(~cover_mask) == 0:
                cover_mask[:] = False
            local_view = self.get_view(
                point=data_dict,
                center=major_coord[np.random.choice(np.where(~cover_mask)[0])],
                scale=self.local_view_scale,
            )
            local_views.append(local_view)
            cover_mask[np.isin(major_view["index"], local_view["index"])] = True

        return self._pack_views(data_dict, global_views, local_views)

@TRANSFORMS.register_module()
class MixedScaleGeometryMultiViewGenerator(MultiViewGenerator):
    """Multi-view generator with normal coarse locals plus fine local crops.

    Fine local crop centers can be sampled uniformly or from a simple local PCA
    directional-complexity score. This keeps the SSL objective unchanged while
    changing which local regions feed the local-global loss.
    """

    def __init__(
        self,
        fine_local_view_num=3,
        fine_local_view_scale=(0.01, 0.04),
        fine_center_mode="geometry",
        fine_center_top_frac=0.05,
        fine_center_k=24,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert 0 <= fine_local_view_num <= self.local_view_num
        assert fine_center_mode in ("geometry", "random")
        self.fine_local_view_num = int(fine_local_view_num)
        self.fine_local_view_scale = fine_local_view_scale
        self.fine_center_mode = fine_center_mode
        self.fine_center_top_frac = float(fine_center_top_frac)
        self.fine_center_k = int(fine_center_k)

    @staticmethod
    def _directional_complexity(coord, k):
        coord = np.asarray(coord, dtype=np.float32)
        n = coord.shape[0]
        if n < 4:
            return np.zeros(n, dtype=np.float32)
        k_eff = min(int(k) + 1, n)
        tree = cKDTree(coord)
        try:
            _, idx = tree.query(coord, k=k_eff, workers=-1)
        except TypeError:
            _, idx = tree.query(coord, k=k_eff)
        if idx.ndim == 1:
            idx = idx[:, None]
        idx = idx[:, 1:]
        if idx.shape[1] < 3:
            return np.zeros(n, dtype=np.float32)
        neigh = coord[idx]
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered) / centered.shape[1]
        eig = np.maximum(np.linalg.eigvalsh(cov), 0.0)
        return (eig[:, 1] / (eig[:, 2] + 1.0e-8)).astype(np.float32)

    def _geometry_pool(self, coord, major_index):
        if self.fine_center_mode == "random":
            return major_index
        score = self._directional_complexity(coord, self.fine_center_k)
        n_top = max(1, int(np.ceil(score.shape[0] * self.fine_center_top_frac)))
        top_index = np.argpartition(score, -n_top)[-n_top:]
        in_major = np.zeros(coord.shape[0], dtype=bool)
        in_major[major_index] = True
        pool = top_index[in_major[top_index]]
        return pool if pool.shape[0] > 0 else major_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        point = self.global_shared_transform(copy.deepcopy(data_dict))
        z_min = coord[:, 2].min()
        z_max = coord[:, 2].max()
        z_min_ = z_min + (z_max - z_min) * self.center_height_scale[0]
        z_max_ = z_min + (z_max - z_min) * self.center_height_scale[1]
        center_mask = np.logical_and(coord[:, 2] >= z_min_, coord[:, 2] <= z_max_)

        major_center = coord[np.random.choice(np.where(center_mask)[0])]
        major_view = self.get_view(point, major_center, self.global_view_scale)
        major_coord = major_view["coord"]

        global_views = self._build_global_views(point, major_view)

        cover_mask = np.zeros_like(major_view["index"], dtype=bool)
        local_views = []
        fine_pool = self._geometry_pool(coord, major_view["index"])

        for _ in range(self.fine_local_view_num):
            center = coord[fine_pool[np.random.randint(fine_pool.shape[0])]]
            local_views.append(
                self.get_view(
                    point=data_dict,
                    center=center,
                    scale=self.fine_local_view_scale,
                )
            )

        num_random_locals = self.local_view_num - self.fine_local_view_num
        for _ in range(num_random_locals):
            if sum(~cover_mask) == 0:
                cover_mask[:] = False
            local_view = self.get_view(
                point=data_dict,
                center=major_coord[np.random.choice(np.where(~cover_mask)[0])],
                scale=self.local_view_scale,
            )
            local_views.append(local_view)
            cover_mask[np.isin(major_view["index"], local_view["index"])] = True

        return self._pack_views(data_dict, global_views, local_views)
