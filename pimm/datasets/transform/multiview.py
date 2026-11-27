"""Contrastive and multiview SSL transform builders."""

from .common import *
from .base import Compose


@TRANSFORMS.register_module()
class ContrastiveViewsGenerator(object):
    """Generate two independently-augmented views for contrastive SSL.

    Copies the keys in ``view_keys`` into two sub-dicts, runs the same transform
    pipeline (built from ``view_trans_cfg``) independently on each, then writes
    the results back under ``view1_<key>`` and ``view2_<key>`` prefixes (for
    every key produced by the pipeline). Registered as
    ``ContrastiveViewsGenerator`` — use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        view_keys (tuple): Keys copied from the source sample into each view.
            Defaults to ``("coord", "color", "normal", "origin_coord")``.
        view_trans_cfg (list, optional): Transform config dicts composed into the
            per-view pipeline applied to both views. Defaults to ``None``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> np.random.seed(0)
            >>> data = {"coord": np.random.rand(10, 3).astype("f4"),
            ...         "color": np.random.rand(10, 3).astype("f4")}
            >>> out = ContrastiveViewsGenerator(
            ...     view_keys=("coord", "color"),
            ...     view_trans_cfg=[dict(type="RandomFlip", p=1.0)])(data)
            >>> sorted(k for k in out if k.startswith("view"))
            ['view1_color', 'view1_coord', 'view2_color', 'view2_coord']
            # two independently-augmented copies of each view_key
    """

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
    """Generate multiple global and local crops for DINO/iBOT-style SSL.

    Crops a major global view around a randomly chosen center (restricted in
    height by ``center_height_scale``), then additional global views and several
    local views, each a nearest-``k`` neighborhood whose size is a random
    fraction of the cloud (per ``global_view_scale`` / ``local_view_scale``).
    Local centers are pushed to cover the major view and, when
    ``data_dict["anchors"]`` is available, a fraction (``anchor_bias_ratio``) of
    local crops are centered on anchors. Reads ``data_dict["coord"]`` and the
    keys in ``view_keys`` (and optional ``anchors``); writes concatenated
    ``global_<key>`` / ``local_<key>`` arrays with ``global_offset`` /
    ``local_offset`` boundaries. Registered as ``MultiViewGenerator`` — use this
    string as the ``type`` in a ``transform=[...]`` config list.

    Args:
        global_view_num (int): Number of global views (including the major one).
            Defaults to ``2``.
        global_view_scale (tuple): ``(min, max)`` fraction of points per global
            crop. Defaults to ``(0.4, 1.0)``.
        local_view_num (int): Number of local views. Defaults to ``4``.
        local_view_scale (tuple): ``(min, max)`` fraction of points per local
            crop. Defaults to ``(0.1, 0.4)``.
        global_shared_transform (list, optional): Transform config applied once
            to the whole sample before any cropping. Defaults to ``None``.
        global_transform (list, optional): Transform config applied to each
            global view. Defaults to ``None``.
        local_transform (list, optional): Transform config applied to each local
            view. Defaults to ``None``.
        max_size (int): Upper bound on the points considered per view. Defaults
            to ``65536``.
        center_height_scale (tuple): ``(min, max)`` fractional z-band restricting
            where the major-view center is sampled. Defaults to ``(0, 1)``.
        shared_global_view (bool): If ``True``, the extra global views are copies
            of the major view rather than freshly cropped. Defaults to
            ``False``.
        center_sampling (str): Center-selection strategy, ``"random"`` or
            ``"cnms"`` (centrality-based non-maximum suppression). Defaults to ``"random"``.
        center_sampling_kwargs (dict, optional): ``radius``, ``overlap_factor``
            and ``K`` for ``pimm.models.polarmae.ops.cnms`` when
            ``center_sampling="cnms"``. Defaults to ``None``.
        view_keys (tuple): Keys cropped into each view; must include ``"coord"``.
            Defaults to ``("coord", "origin_coord", "color", "normal")``.
        anchor_bias_ratio (float): Fraction of local views centered on anchors
            (when anchors are present). Defaults to ``0.6``.
        anchor_radius_scale (float): Radius scale for anchor-centered local crops
            (size scales roughly cubically with it). Defaults to ``1.5``.
        anchor_keys (tuple): Anchor sub-keys pooled as candidate centers; ``led``
            is always excluded. Defaults to
            ``("endpoints", "branches_track", "branches_shower", "bragg")``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> np.random.seed(0)
            >>> data = {"coord": np.random.rand(200, 3).astype("f4"),
            ...         "color": np.random.rand(200, 3).astype("f4"),
            ...         "index_valid_keys": ["coord", "color"]}
            >>> out = MultiViewGenerator(global_view_num=2, local_view_num=4,
            ...                          view_keys=("coord", "color"))(data)
            >>> out["global_coord"].shape, out["global_offset"]   # 2 global crops concatenated
            ((356, 3), array([170, 356]))
            >>> out["local_coord"].shape, out["local_offset"]     # 4 local crops concatenated
            ((223, 3), array([ 66, 134, 155, 223]))
            # exact sizes vary with the RNG; offsets mark per-view boundaries
    """

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
            from pimm.models.polarmae.ops import cnms

            points = torch.from_numpy(possible_centers).float()[None]
            lengths = torch.tensor([points.shape[1]])
            centers, count = cnms(points, lengths=lengths, **self.center_sampling_kwargs)
            possible_centers = centers[0, : int(count)].numpy()
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
    """Multi-view generator with coarse locals plus extra fine local crops.

    Subclass of :class:`MultiViewGenerator` that replaces ``fine_local_view_num``
    of the local crops with much smaller (``fine_local_view_scale``) crops whose
    centers come either from uniform sampling (``fine_center_mode="random"``) or
    from a local-PCA directional-complexity score (``"geometry"``, keeping the
    top ``fine_center_top_frac`` of points by complexity). Keeps the SSL
    objective unchanged while steering which local regions feed the local-global
    loss; reads/writes the same keys as the base class. All base-class arguments
    are accepted via ``**kwargs``. Registered as
    ``MixedScaleGeometryMultiViewGenerator`` — use this string as the ``type`` in
    a ``transform=[...]`` config list.

    Args:
        fine_local_view_num (int): Number of fine local crops (must satisfy
            ``0 <= fine_local_view_num <= local_view_num``). Defaults to ``3``.
        fine_local_view_scale (tuple): ``(min, max)`` fraction of points per fine
            local crop. Defaults to ``(0.01, 0.04)``.
        fine_center_mode (str): ``"geometry"`` to bias fine centers toward
            high-complexity points, or ``"random"`` for uniform. Defaults to
            ``"geometry"``.
        fine_center_top_frac (float): Top fraction of points (by complexity)
            eligible as fine centers in ``"geometry"`` mode. Defaults to
            ``0.05``.
        fine_center_k (int): Neighbor count for the local-PCA complexity score.
            Defaults to ``24``.
        **kwargs: Forwarded to :class:`MultiViewGenerator`.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> np.random.seed(0)
            >>> data = {"coord": np.random.rand(300, 3).astype("f4"),
            ...         "energy": np.random.rand(300, 1).astype("f4"),
            ...         "index_valid_keys": ["coord", "energy"]}
            >>> out = MixedScaleGeometryMultiViewGenerator(
            ...     fine_local_view_num=2, local_view_num=4,
            ...     view_keys=("coord", "energy"))(data)
            >>> out["local_coord"].shape, out["local_offset"]
            ((193, 3), array([  8,  14, 105, 193]))
            # first 2 locals are tiny geometry-biased fine crops (8, 6 pts), then coarse locals
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
