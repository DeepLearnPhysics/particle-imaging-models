"""Coordinate, sampling, voxelization, and crop transforms."""

from .common import *


@TRANSFORMS.register_module()
class NormalizeCoord(object):
    """Recenter and rescale ``coord`` into a normalized frame.

    Computes ``coord = (coord - center) / scale`` in place. When ``center`` is
    ``None`` the per-sample centroid (mean of ``coord``) is used; when ``scale``
    is ``None`` the max point norm after centering is used, mapping the cloud
    into the unit ball. The same shift/scale is applied to v3 ``vertex``
    metadata and to opt-in ``aux_position_keys`` so position targets stay
    aligned; direction keys are left untouched (isotropic scale + translation
    preserve orientation). Requires ``coord``; a no-op if ``coord`` is absent.
    Registered as ``NormalizeCoord`` -- use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        center (Sequence[float], optional): fixed center to subtract. If
            ``None``, the per-sample centroid is used. Defaults to ``None``.
        scale (float, optional): fixed divisor applied after centering. If
            ``None``, the max post-centering point norm is used (unit ball).
            Defaults to ``None``.

    Note:
        ``NormalizeCoord(scale=X)`` computes ``(coord - center) / scale`` -- it
        divides by ``X``, it does not multiply. For PILArNet the standard fixed
        values are ``center=[384, 384, 384]`` and ``scale = 768 * sqrt(3) / 2
        approx 665.1076`` (the half-diagonal of the ``768^3`` volume), mapping
        the detector into roughly ``[-1, 1]^3``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import NormalizeCoord
            >>> data = {"coord": np.array([[0., 0., 0.], [1000., 0., 0.]], dtype="f4"),
            ...         "energy": np.array([[1.], [5.]], dtype="f4")}
            >>> out = NormalizeCoord(center=[500., 0., 0.], scale=500.)(data)
            >>> out["coord"]  # (coord - center) / scale: [0,1000] -> [-1,1]
            array([[-1.,  0.,  0.],
                   [ 1.,  0.,  0.]], dtype=float32)
    """

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
            _apply_to_aux_positions(data_dict, lambda p: (p - centroid) / scale)
            # directions: isotropic scale + translation preserve orientation -> untouched
        return data_dict

@TRANSFORMS.register_module()
class PositiveShift(object):
    """Translate ``coord`` so its minimum corner sits at the origin.

    Subtracts the per-axis minimum of ``coord`` in place, so all coordinates
    become non-negative (the bounding-box min moves to ``(0, 0, 0)``). The same
    shift is applied to v3 ``vertex`` metadata to keep it aligned. Requires
    ``coord``; a no-op if ``coord`` is absent. Registered as ``PositiveShift``
    -- use this string as the ``type`` in a ``transform=[...]`` config list.

    Note:
        Takes no constructor arguments.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import PositiveShift
            >>> data = {"coord": np.array([[2., -1., 5.], [4., 3., 5.]], dtype="f4")}
            >>> out = PositiveShift()(data)  # subtract per-axis min -> bbox min at origin
            >>> out["coord"]
            array([[0., 0., 0.],
                   [2., 4., 0.]], dtype=float32)
    """

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
            _apply_to_v3_vertex(data_dict, lambda vertex: vertex - coord_min)
        return data_dict

@TRANSFORMS.register_module()
class CenterShift(object):
    """Center ``coord`` on the midpoint of its bounding box per axis.

    For each requested axis, subtracts the bounding-box midpoint
    ``(min + max) / 2`` from that coordinate column in place, centering the
    cloud about the origin axis-by-axis. The same per-axis shift is applied to
    v3 ``vertex`` metadata and to opt-in ``aux_position_keys``. Requires
    ``coord``; a no-op if ``coord`` is absent. Registered as ``CenterShift`` --
    use this string as the ``type`` in a ``transform=[...]`` config list.

    Args:
        apply_z (bool): retained for backward compatibility; the axes actually
            centered are controlled by ``axes``. Defaults to ``True``.
        axes (str | Sequence[str]): which of ``"x"``, ``"y"``, ``"z"`` to
            center. A bare string is wrapped into a single-element tuple.
            Defaults to ``("x", "y", "z")``.

    Note:
        Each axis is centered independently using its own min/max, so the result
        is centered on the bounding-box center rather than the centroid.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import CenterShift
            >>> data = {"coord": np.array([[0., 0., 0.], [10., 4., 8.]], dtype="f4")}
            >>> out = CenterShift(axes=("x", "y"))(data)  # center x,y on bbox midpoint; z untouched
            >>> out["coord"]
            array([[-5., -2.,  0.],
                   [ 5.,  2.,  8.]], dtype=float32)
    """

    def __init__(self, apply_z=True, axes=("x", "y", "z")):
        self.apply_z = apply_z
        axes = tuple(axes) if isinstance(axes, (list, tuple)) else (axes,)
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
                    _apply_to_aux_positions(
                        data_dict, lambda p, shift=shift: _translate_axis(p, 0, -shift)
                    )
                elif axis == "y":
                    x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                    x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                    shift = (y_min + y_max) / 2
                    data_dict["coord"][:, 1] -= shift
                    _apply_to_v3_vertex(
                        data_dict, lambda vertex, shift=shift: _translate_axis(vertex, 1, -shift)
                    )
                    _apply_to_aux_positions(
                        data_dict, lambda p, shift=shift: _translate_axis(p, 1, -shift)
                    )
                elif axis == "z":
                    x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                    x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                    shift = (z_min + z_max) / 2
                    data_dict["coord"][:, 2] -= shift
                    _apply_to_v3_vertex(
                        data_dict, lambda vertex, shift=shift: _translate_axis(vertex, 2, -shift)
                    )
                    _apply_to_aux_positions(
                        data_dict, lambda p, shift=shift: _translate_axis(p, 2, -shift)
                    )
        return data_dict

@TRANSFORMS.register_module()
class ConditionalRandomTransform(object):
    """Wall-aware random translation that keeps points inside fixed bounds.

    For each requested axis, only acts when the cloud touches a wall (its min or
    max comes within ``buffer_size`` of the axis ``bounds``); fully interior
    axes are skipped. When triggered (with probability ``p`` per qualifying
    axis), it draws a uniform translation from the feasible range that keeps all
    points within ``bounds`` while preserving the cloud's contact with whichever
    wall(s) it was already near. Shifts ``coord`` in place and applies the same
    translation to v3 ``vertex`` metadata. Requires ``coord``; a no-op if
    ``coord`` is absent. Registered as ``ConditionalRandomTransform`` -- use
    this string as the ``type`` in a ``transform=[...]`` config list.

    Args:
        p (float): per-axis probability of applying a translation when the
            cloud is near a wall on that axis. Defaults to ``0.5``.
        axes (str | Sequence[str]): which of ``"x"``, ``"y"``, ``"z"`` to
            consider. A bare string is wrapped into a single-element tuple.
            Defaults to ``("x", "y", "z")``.
        buffer_size (float): margin (in coordinate units) defining how close to
            a bound counts as "near a wall" and the contact band to preserve.
            Defaults to ``0.05``.
        bounds (tuple[tuple[float, float], ...]): per-axis ``(low, high)``
            limits the translated points must stay within. Defaults to
            ``((-1, 1), (-1, 1), (-1, 1))``.

    Note:
        Assumes coordinates are already normalized into the ``bounds`` frame
        (e.g. after ``NormalizeCoord``). Intended for detector-volume data where
        a particle's contact with a TPC wall is physically meaningful and must
        survive augmentation.

    Example:
        .. code-block:: python

            >>> import numpy as np, random
            >>> from pimm.datasets.transform import ConditionalRandomTransform
            >>> np.random.seed(0); random.seed(0)
            >>> data = {"coord": np.array([[-0.99, 0., 0.], [-0.5, 0., 0.]], dtype="f4")}
            >>> # x hugs the -1 wall, so a random translation is drawn that keeps it
            >>> # in bounds and still touching that wall:
            >>> out = ConditionalRandomTransform(p=1.0, axes=("x",))(data)
            >>> bool(out["coord"][:, 0].min() >= -1.0)  # never pushed past the wall
            True
            >>> bool(out["coord"][:, 0].min() <= -0.95)  # still within the contact band
            True
    """

    _max_value_pilarnet = 2 * pow(3, 0.5) / 3 # (768) / (768 * 3 ** 0.5 / 2)
    def __init__(self, p=0.5, axes=("x", "y", "z"), buffer_size=0.05, bounds=((-1, 1), (-1, 1), (-1, 1))):
        self.p = p
        axes = tuple(axes) if isinstance(axes, (list, tuple)) else (axes,)
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
    """Translate ``coord`` by a uniform random per-axis offset.

    Draws an independent uniform offset for each axis from its configured range
    and adds the resulting vector to ``coord`` in place; the same offset is
    applied to v3 ``vertex`` metadata. Requires ``coord``; a no-op if ``coord``
    is absent. Registered as ``RandomShift`` -- use this string as the ``type``
    in a ``transform=[...]`` config list.

    Args:
        shift (tuple[tuple[float, float], ...]): per-axis ``(low, high)`` ranges
            for the uniform offset along x, y, z. Defaults to
            ``((-0.2, 0.2), (-0.2, 0.2), (0, 0))`` (no z shift).

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import RandomShift
            >>> np.random.seed(0)
            >>> data = {"coord": np.zeros((2, 3), dtype="f4")}
            >>> out = RandomShift(shift=((-0.1, 0.1), (-0.1, 0.1), (0, 0)))(data)
            >>> bool(np.abs(out["coord"][:, :2]).max() <= 0.1)  # x,y shifted within +/-0.1
            True
            >>> out["coord"][:, 2].tolist()  # z range is (0, 0): no shift
            [0.0, 0.0]
    """

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
    """Clamp ``coord`` to an axis-aligned point-cloud range.

    Clips each coordinate column to the ``[min, max]`` box given by
    ``point_cloud_range`` in place (out-of-range values are saturated to the
    nearest face, not removed; the point count is unchanged). Requires
    ``coord``; a no-op if ``coord`` is absent. Registered as ``PointClip`` --
    use this string as the ``type`` in a ``transform=[...]`` config list.

    Args:
        point_cloud_range (tuple[float, ...]): six values
            ``(x_min, y_min, z_min, x_max, y_max, z_max)`` giving the clamp box.
            Defaults to ``(-80, -80, -3, 80, 80, 1)``.

    Note:
        This clamps coordinates; it does not drop points. Per-point arrays stay
        the same length, so coordinates collapsed onto a face remain present.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import PointClip
            >>> data = {"coord": np.array([[-2., 0.5, 0.], [2., -3., 0.]], dtype="f4")}
            >>> out = PointClip(point_cloud_range=(-1, -1, -1, 1, 1, 1))(data)
            >>> out["coord"]  # out-of-range coords saturated to the box faces; count unchanged
            array([[-1. ,  0.5,  0. ],
                   [ 1. , -1. ,  0. ]])
    """

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
    """Randomly drop a fraction of points from the sample.

    With probability ``dropout_application_ratio`` for the whole sample, keeps a
    uniformly random ``1 - dropout_ratio`` fraction of points and removes the
    rest via ``index_operator`` (so every per-point key in ``index_valid_keys``
    is subsampled together). If ``sampled_index`` is present (data-efficient
    setups), the labeled points are forced into the kept set and
    ``sampled_index`` is remapped to the new indexing. Requires ``coord`` (and
    ``segment``/``sampled_index`` when those are present). Registered as
    ``RandomDropout`` -- use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        dropout_ratio (float): fraction of points to drop when the transform
            fires. Defaults to ``0.2``.
        dropout_application_ratio (float): probability of applying dropout to a
            given sample. Defaults to ``0.5``.

    Note:
        Point selection is random and unaffected by spatial structure; this
        reduces the point count, so it must run before length-sensitive steps.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import RandomDropout
            >>> np.random.seed(0)
            >>> data = {"coord": np.arange(30, dtype="f4").reshape(10, 3),
            ...         "energy": np.arange(10, dtype="f4").reshape(10, 1)}
            >>> # application_ratio=1.0 always fires; drops 40%, keeping coord and energy aligned
            >>> out = RandomDropout(dropout_ratio=0.4, dropout_application_ratio=1.0)(data)
            >>> len(out["coord"]), len(out["energy"])  # 10 -> int(10 * 0.6) = 6 points
            (6, 6)
    """

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
    """Rotate the cloud by a random angle about one axis.

    With probability ``p`` (forced to 1 when ``always_apply``), draws an angle
    uniformly from ``angle`` (in units of pi radians) and rotates ``coord``
    about the chosen ``axis`` around ``center`` in place. The same rotation is
    applied to v3 ``vertex`` metadata, to ``aux_position_keys``, to ``normal``,
    and -- as the linear part only, re-normalized -- to ``aux_direction_keys``.
    When ``center`` is ``None`` the bounding-box center is used. Requires
    ``coord``. Registered as ``RandomRotate`` -- use this string as the
    ``type`` in a ``transform=[...]`` config list.

    Args:
        angle (Sequence[float], optional): ``[low, high]`` range, in units of
            pi radians, to sample the rotation angle from. ``None`` means
            ``[-1, 1]`` (full +/- pi). Defaults to ``None``.
        center (Sequence[float], optional): rotation center. ``None`` uses the
            per-sample bounding-box center. Defaults to ``None``.
        axis (str): rotation axis, one of ``"x"``, ``"y"``, ``"z"``. Defaults
            to ``"z"``.
        always_apply (bool): if ``True``, always rotate (overrides ``p`` to 1).
            Defaults to ``False``.
        p (float): probability of applying the rotation when not
            ``always_apply``. Defaults to ``0.5``.

    Note:
        The angle is expressed in multiples of pi, so ``angle=[-1, 1]`` spans a
        full turn. Chain three instances (one per axis) for full 3D rotation.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import RandomRotate
            >>> data = {"coord": np.array([[1., 0., 0.], [0., 1., 0.]], dtype="f4")}
            >>> # angle=[0.5, 0.5] -> exactly 0.5*pi (90 deg) about z; always_apply forces it
            >>> out = RandomRotate(angle=[0.5, 0.5], axis="z", center=[0, 0, 0],
            ...                    always_apply=True)(data)
            >>> np.round(out["coord"], 3)  # +x -> +y, +y -> -x
            array([[ 0.,  1.,  0.],
                   [-1.,  0.,  0.]])
    """

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
            _apply_to_aux_positions(
                data_dict, lambda p: np.dot(p - center, np.transpose(rot_t)) + center
            )
            # directions rotate about the origin (linear part only, no centering)
            _apply_to_aux_directions(data_dict, lambda d: np.dot(d, np.transpose(rot_t)))
            _apply_to_aux_vectors(data_dict, lambda d: np.dot(d, np.transpose(rot_t)))
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict

@TRANSFORMS.register_module()
class RandomRotateTargetAngle(object):
    """Rotate the cloud by a random angle drawn from a discrete set.

    Like ``RandomRotate``, but the angle is chosen uniformly from the discrete
    ``angle`` set (in units of pi radians) rather than a continuous range --
    useful for snapping to canonical orientations (e.g. quarter/half turns).
    With probability ``p`` (forced to 1 when ``always_apply``), rotates
    ``coord`` about ``axis`` around ``center`` in place, and applies the same
    rotation to v3 ``vertex`` metadata, ``aux_position_keys``, ``normal``, and
    (linear part, re-normalized) ``aux_direction_keys``. When ``center`` is
    ``None`` the bounding-box center is used. Requires ``coord``. Registered as
    ``RandomRotateTargetAngle`` -- use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        angle (Sequence[float]): discrete set of angles, in units of pi
            radians, to choose from. Defaults to ``(1/2, 1, 3/2)`` (90, 180,
            270 degrees).
        center (Sequence[float], optional): rotation center. ``None`` uses the
            per-sample bounding-box center. Defaults to ``None``.
        axis (str): rotation axis, one of ``"x"``, ``"y"``, ``"z"``. Defaults
            to ``"z"``.
        always_apply (bool): if ``True``, always rotate (overrides ``p`` to 1).
            Defaults to ``False``.
        p (float): probability of applying the rotation when not
            ``always_apply``. Defaults to ``0.75``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import RandomRotateTargetAngle
            >>> data = {"coord": np.array([[1., 0., 0.], [0., 1., 0.]], dtype="f4")}
            >>> # single-element angle set -> deterministic 1*pi (180 deg) about z
            >>> out = RandomRotateTargetAngle(angle=(1.0,), axis="z", center=[0, 0, 0],
            ...                               always_apply=True)(data)
            >>> np.round(out["coord"], 3)  # 180 deg flips x and y signs
            array([[-1.,  0.,  0.],
                   [-0., -1.,  0.]])
    """

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
            _apply_to_aux_positions(
                data_dict, lambda p: np.dot(p - center, np.transpose(rot_t)) + center
            )
            _apply_to_aux_directions(data_dict, lambda d: np.dot(d, np.transpose(rot_t)))
            _apply_to_aux_vectors(data_dict, lambda d: np.dot(d, np.transpose(rot_t)))
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict

@TRANSFORMS.register_module()
class RandomScale(object):
    """Scale ``coord`` by a random factor.

    Draws a uniform scale from ``scale`` -- a single shared factor when
    ``anisotropic`` is ``False`` or three independent per-axis factors when
    ``True`` -- and multiplies ``coord`` in place. The same factor is applied to
    v3 ``vertex`` metadata and ``aux_position_keys``; direction keys are left
    unchanged. Requires ``coord``; a no-op if ``coord`` is absent. Registered as
    ``RandomScale`` -- use this string as the ``type`` in a ``transform=[...]``
    config list.

    Args:
        scale (Sequence[float], optional): ``[low, high]`` range for the uniform
            scale factor. ``None`` means ``[0.95, 1.05]``. Defaults to ``None``.
        anisotropic (bool): if ``True``, sample an independent factor per axis;
            otherwise use one shared factor. Defaults to ``False``.

    Note:
        Direction keys are intentionally not scaled: isotropic scaling preserves
        orientation, and anisotropic scaling of unit-direction targets is
        unsupported.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import RandomScale
            >>> data = {"coord": np.array([[1., 2., 3.], [4., 5., 6.]], dtype="f4")}
            >>> # degenerate range [2, 2] -> a fixed 2x scale, so the effect is visible
            >>> out = RandomScale(scale=[2.0, 2.0])(data)
            >>> out["coord"]
            array([[ 2.,  4.,  6.],
                   [ 8., 10., 12.]], dtype=float32)
    """

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
            _apply_to_aux_positions(data_dict, lambda p: p * scale)
            # NOTE: direction keys are intentionally NOT scaled. Isotropic scale
            # preserves orientation (no-op after renorm); anisotropic scale would
            # change it, but is unsupported for unit-direction targets.
        return data_dict

@TRANSFORMS.register_module()
class RandomFlip(object):
    """Randomly mirror the cloud across one or more axes.

    For each requested axis, with probability ``p`` negates that coordinate
    column of ``coord`` in place (a reflection through the origin plane). The
    same sign flip is applied to v3 ``vertex`` metadata, ``aux_position_keys``,
    ``aux_direction_keys``, and ``normal``. Requires ``coord``. Registered as
    ``RandomFlip`` -- use this string as the ``type`` in a ``transform=[...]``
    config list.

    Args:
        p (float): per-axis probability of flipping. Defaults to ``0.5``.
        axes (str | Sequence[str]): which of ``"x"``, ``"y"``, ``"z"`` may be
            flipped. A bare string is wrapped into a single-element tuple.
            Defaults to ``("x", "y")``.

    Note:
        Each listed axis is tested independently, so multiple axes can flip in a
        single call.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import RandomFlip
            >>> data = {"coord": np.array([[1., 2., 3.], [4., 5., 6.]], dtype="f4")}
            >>> out = RandomFlip(p=1.0, axes=("x",))(data)  # p=1 forces an x reflection
            >>> out["coord"]  # x column negated, y and z untouched
            array([[-1.,  2.,  3.],
                   [-4.,  5.,  6.]], dtype=float32)
    """

    def __init__(self, p=0.5, axes=("x", "y",)):
        self.p = p
        axes = tuple(axes) if isinstance(axes, (list, tuple)) else (axes,)
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
                _apply_to_aux_positions(
                    data_dict, lambda p: p * np.array([-1.0, 1.0, 1.0])
                )
                _apply_to_aux_directions(
                    data_dict, lambda d: d * np.array([-1.0, 1.0, 1.0])
                )
                _apply_to_aux_vectors(
                    data_dict, lambda d: d * np.array([-1.0, 1.0, 1.0])
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
                _apply_to_aux_positions(
                    data_dict, lambda p: p * np.array([1.0, -1.0, 1.0])
                )
                _apply_to_aux_directions(
                    data_dict, lambda d: d * np.array([1.0, -1.0, 1.0])
                )
                _apply_to_aux_vectors(
                    data_dict, lambda d: d * np.array([1.0, -1.0, 1.0])
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
                _apply_to_aux_positions(
                    data_dict, lambda p: p * np.array([1.0, 1.0, -1.0])
                )
                _apply_to_aux_directions(
                    data_dict, lambda d: d * np.array([1.0, 1.0, -1.0])
                )
                _apply_to_aux_vectors(
                    data_dict, lambda d: d * np.array([1.0, 1.0, -1.0])
                )
                if "normal" in data_dict.keys():
                    data_dict["normal"][:, 2] = -data_dict["normal"][:, 2]
        return data_dict

@TRANSFORMS.register_module()
class RandomJitter(object):
    """Add clipped Gaussian noise to one or more point-aligned keys.

    With probability ``p``, adds independent Gaussian noise
    (std ``sigma``, clamped to ``[-clip, clip]``) to each listed key in place.
    The noise has the same shape as the target array, so by default it perturbs
    ``coord`` per point and per dimension. Requires each listed key to be
    present (raises ``ValueError`` otherwise). Registered as ``RandomJitter`` --
    use this string as the ``type`` in a ``transform=[...]`` config list.

    Args:
        sigma (float): standard deviation of the additive Gaussian noise.
            Defaults to ``0.01``.
        clip (float): magnitude the noise is clamped to (must be ``> 0``).
            Defaults to ``0.05``.
        keys (str | Sequence[str]): keys to jitter. A bare string is wrapped
            into a single-element tuple. Defaults to ``("coord",)``.
        p (float): probability of applying the jitter. Defaults to ``1.0``.

    Note:
        The noise is additive; for multiplicative scalar noise (e.g. on energy)
        use ``MultiplicativeRandomJitter``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import RandomJitter
            >>> np.random.seed(0)
            >>> data = {"coord": np.zeros((3, 3), dtype="f4")}
            >>> out = RandomJitter(sigma=0.01, clip=0.05, keys=("coord",), p=1.0)(data)
            >>> out["coord"].shape  # one independent noise value per point per axis
            (3, 3)
            >>> bool(np.any(out["coord"] != 0)) and bool(np.abs(out["coord"]).max() <= 0.05)
            True
    """

    def __init__(self, sigma=0.01, clip=0.05, keys=("coord",), p=1.0):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip
        keys = tuple(keys) if isinstance(keys, (list, tuple)) else (keys,)
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
    """Multiply one or more keys by clipped Gaussian noise around 1.

    With probability ``p``, scales each listed key in place by
    ``1 + noise``, where ``noise`` is Gaussian (std ``sigma``) clamped to
    ``[-clip, clip]`` and has the same shape as the target. Intended for scalar
    per-point features such as ``energy``/charge where multiplicative
    fluctuation is the physical noise model. Requires each listed key to be
    present (raises ``ValueError`` otherwise). Registered as
    ``MultiplicativeRandomJitter`` -- use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        sigma (float): standard deviation of the multiplicative noise term.
            Defaults to ``0.05``.
        clip (float): magnitude the noise is clamped to (must be ``> 0``).
            Defaults to ``0.05``.
        keys (str | Sequence[str]): keys to scale. A bare string is wrapped into
            a single-element tuple. Defaults to ``("energy",)``.
        p (float): probability of applying the jitter. Defaults to ``0.5``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import MultiplicativeRandomJitter
            >>> np.random.seed(0)
            >>> data = {"energy": np.array([[100.], [100.], [100.]], dtype="f4")}
            >>> # scales each value by 1 + noise, noise in [-0.05, 0.05]
            >>> out = MultiplicativeRandomJitter(sigma=0.05, clip=0.05,
            ...                                  keys=("energy",), p=1.0)(data)
            >>> bool(np.all(out["energy"] >= 95.0)) and bool(np.all(out["energy"] <= 105.0))
            True
            >>> bool(np.any(out["energy"] != 100.0))  # values fluctuated around 100
            True
    """

    def __init__(self, sigma=0.05, clip=0.05, keys=("energy",), p=0.5):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip
        keys = tuple(keys) if isinstance(keys, (list, tuple)) else (keys,)
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
    """Overwrite one or more keys with clipped Gaussian noise.

    Replaces each listed key in place with freshly sampled Gaussian noise
    (std ``sigma``, clamped to ``[-clip, clip]``) of the same shape -- the
    original values are discarded entirely (unlike the additive/multiplicative
    jitter transforms). Useful as an ablation that destroys the information in a
    feature while preserving its shape. Always applies (no probability gate).
    Requires each listed key to be present (raises ``ValueError`` otherwise).
    Registered as ``SetRandomValue`` -- use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        sigma (float): standard deviation of the replacement noise. Defaults to
            ``0.05``.
        clip (float): magnitude the noise is clamped to (must be ``> 0``).
            Defaults to ``0.05``.
        keys (str | Sequence[str]): keys to overwrite. A bare string is wrapped
            into a single-element tuple. Defaults to ``("energy",)``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import SetRandomValue
            >>> np.random.seed(0)
            >>> data = {"energy": np.array([[100.], [100.], [100.]], dtype="f4")}
            >>> out = SetRandomValue(sigma=0.05, keys=("energy",))(data)
            >>> out["energy"].shape  # same shape, original values discarded
            (3, 1)
            >>> bool(np.abs(out["energy"]).max() <= 0.05)  # overwritten with clipped noise
            True
    """

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
    """Add isotropic clipped multivariate-Gaussian noise to ``coord``.

    Samples 3D noise from a standard multivariate normal, clips it to roughly
    +/- ``1.96`` sigma (the stored ``quantile``), scales by ``scalar``, and adds
    it to ``coord`` in place. Optionally records the applied noise under the
    ``jitter`` key. Requires ``coord``; a no-op if ``coord`` is absent.
    Registered as ``ClipGaussianJitter`` -- use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        scalar (float): multiplier applied to the clipped unit noise, setting
            the jitter magnitude. Defaults to ``0.02``.
        store_jitter (bool): if ``True``, save the applied per-point noise under
            ``data_dict["jitter"]``. Defaults to ``False``.

    Note:
        Assumes 3D coordinates (the noise covariance is the 3x3 identity).

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import ClipGaussianJitter
            >>> data = {"coord": np.zeros((2, 3), dtype="f4")}
            >>> # NOTE: as written the constructor sets mean = np.mean(3) (a scalar),
            >>> # so the multivariate-normal draw raises before any jitter is applied:
            >>> ClipGaussianJitter(scalar=0.02)(data)
            Traceback (most recent call last):
                ...
            ValueError: mean must be 1 dimensional
    """

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
    """Apply smooth random elastic warping to ``coord``.

    With 95% probability, displaces ``coord`` in place by a smoothed random
    noise field: for each ``(granularity, magnitude)`` pair, a coarse Gaussian
    noise grid of spacing ``granularity`` is blurred and trilinearly
    interpolated to each point, then added scaled by ``magnitude``. Stacking
    multiple pairs combines coarse and fine deformations. Requires ``coord``; a
    no-op if ``coord`` is absent or ``distortion_params`` is ``None``.
    Registered as ``ElasticDistortion`` -- use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        distortion_params (Sequence[Sequence[float]], optional): list of
            ``[granularity, magnitude]`` pairs applied in sequence; granularity
            is the noise-grid spacing in coordinate units and magnitude scales
            the displacement. ``None`` means ``[[0.2, 0.4], [0.8, 1.6]]``.
            Defaults to ``None``.

    Note:
        ``granularity`` and ``magnitude`` are in the same units as ``coord``, so
        place this relative to your normalization (before or after
        ``NormalizeCoord``) deliberately. Only the spatial coordinates are
        warped; other point-aligned arrays are unchanged.

    Example:
        .. code-block:: python

            >>> import numpy as np, random
            >>> from pimm.datasets.transform import ElasticDistortion
            >>> np.random.seed(0); random.seed(0)
            >>> coord = (np.random.rand(50, 3) * 2).astype("f4")  # spread over a 2-unit box
            >>> data = {"coord": coord.copy()}
            >>> out = ElasticDistortion(distortion_params=[[0.2, 0.4]])(data)
            >>> out["coord"].shape  # same points, smoothly warped (count unchanged)
            (50, 3)
            >>> bool(np.any(out["coord"] != coord))  # coordinates displaced by the noise field
            True
    """

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
    """Voxel-downsample a point cloud onto a hash grid.

    Floors ``coord / grid_size`` to integer voxel indices, hashes them, and
    deduplicates points per voxel. In ``mode="train"`` it keeps one random point
    per occupied voxel and subsamples every per-point key in ``index_valid_keys``
    via ``index_operator``, returning a single dict. In ``mode="test"`` it
    instead returns a list of fragment dicts that together cover all points
    (one point per voxel per fragment), each carrying an ``index`` key.
    Optionally aggregates selected keys per voxel (sum or min) and emits
    integer ``grid_coord`` for sparse convolutions. Requires ``coord``
    (asserted). Registered as ``GridSample`` -- use this string as the ``type``
    in a ``transform=[...]`` config list.

    Args:
        grid_size (float | Sequence[float]): voxel edge length in
            (normalized) coordinate units; smaller keeps more points. Defaults
            to ``0.05``.
        hash_type (str): voxel hash, ``"fnv"`` (FNV64-1A) or otherwise the
            ravel/raster hash. Defaults to ``"fnv"``.
        mode (str): ``"train"`` (one random point per voxel, single dict) or
            ``"test"`` (list of covering fragments). Defaults to ``"train"``.
        return_inverse (bool): if ``True``, store an ``inverse`` array mapping
            each original point to its voxel. Defaults to ``False``.
        return_grid_coord (bool): if ``True``, store integer ``grid_coord`` and
            append it to ``index_valid_keys``. Defaults to ``False``.
        return_min_coord (bool): if ``True``, store the per-sample
            ``min_coord`` (grid origin, shape ``[1, 3]``). Defaults to
            ``False``.
        return_displacement (bool): if ``True``, store each point's
            ``displacement`` from its voxel center in ``[-0.5, 0.5]``. Defaults
            to ``False``.
        project_displacement (bool): if ``True``, project the displacement onto
            the point ``normal`` to a scalar (requires ``normal``). Defaults to
            ``False``.
        sum_keys (Sequence[str], optional): keys summed over the points in each
            voxel (e.g. pooling energy). ``None`` means none. Defaults to
            ``None``.
        min_keys (Sequence[str], optional): keys reduced by per-voxel minimum.
            ``None`` means none. Defaults to ``None``.

    Note:
        Only keys in ``index_valid_keys`` are downsampled together; any
        point-aligned array missing from that list keeps its original length and
        silently desynchronizes from ``coord``. ``grid_size`` is in the same
        frame as ``coord`` -- after ``NormalizeCoord`` it is in normalized
        units. ``mode="test"`` returns a list of dicts, not a single dict, so it
        is normally the terminal geometric step of a test pipeline.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import GridSample
            >>> np.random.seed(0)
            >>> # 1000 points landing on a 5x5x5 integer lattice -> many coincide per voxel
            >>> coord = np.floor(np.random.rand(1000, 3) * 5).astype("f4")
            >>> data = {"coord": coord.copy(), "energy": np.random.rand(1000, 1).astype("f4")}
            >>> out = GridSample(grid_size=1.0, mode="train", return_grid_coord=True)(data)
            >>> len(out["coord"])  # one point kept per occupied voxel: 1000 -> 5**3 voxels
            125
            >>> out["grid_coord"].shape, out["grid_coord"].dtype  # integer voxel indices added
            ((125, 3), dtype('int64'))
    """

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
    """Crop to the nearest points around a center to cap the point count.

    When the cloud exceeds the point budget, picks a center and keeps the
    ``point_max`` (or ``sample_rate``-fraction) points closest to it -- a
    spherical crop -- subsampling every per-point key in ``index_valid_keys``
    via ``index_operator``. Smaller clouds are returned unchanged. The center is
    a random point (``mode="random"``) or the middle-indexed point
    (``mode="center"``). Requires ``coord`` (asserted). Registered as
    ``SphereCrop`` -- use this string as the ``type`` in a ``transform=[...]``
    config list.

    Args:
        point_max (int): maximum number of points to keep when ``sample_rate``
            is ``None``. Defaults to ``80000``.
        sample_rate (float, optional): if set, the budget is
            ``sample_rate * num_points`` instead of ``point_max``. Defaults to
            ``None``.
        mode (str): center selection, one of ``"random"``, ``"center"``, or
            ``"all"``. Defaults to ``"random"``.

    Note:
        ``mode="all"`` is accepted by the constructor but raises
        ``NotImplementedError`` if the crop branch is reached (i.e. when the
        cloud is over budget).

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import SphereCrop
            >>> np.random.seed(0)
            >>> data = {"coord": np.random.rand(100, 3).astype("f4"),
            ...         "energy": np.random.rand(100, 1).astype("f4")}
            >>> # over budget (100 > 10): keep the 10 points nearest the chosen center
            >>> out = SphereCrop(point_max=10, mode="center")(data)
            >>> len(out["coord"]), len(out["energy"])  # cropped together, aligned
            (10, 10)
    """

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
    """Spherical crop biased toward rare ("hard") segment labels.

    Like ``SphereCrop``, but when the cloud is over budget it centers the crop
    on a point whose ``segment`` is one of ``hard_labels``, retrying up to
    ``attempts`` times until the kept set contains at least ``min_hard_points``
    hard points (otherwise the last attempt is used). Only fires with
    probability ``p``; sub-budget clouds are returned unchanged. If no hard
    points are present, behaviour follows ``fallback``. Keeps the closest
    ``point_max`` (or ``sample_rate``-fraction) points via ``index_operator``.
    Requires ``coord`` (asserted) and reads ``segment`` when present. Registered
    as ``HardExampleCrop`` -- use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        point_max (int): maximum number of points to keep when ``sample_rate``
            is ``None``. Defaults to ``80000``.
        sample_rate (float, optional): if set, the budget is
            ``sample_rate * num_points`` instead of ``point_max``. Defaults to
            ``None``.
        hard_labels (Sequence[int]): ``segment`` label values treated as hard
            examples to center on. Defaults to ``(2, 3)``.
        min_hard_points (int): minimum hard points the crop must contain to
            accept an attempt early. Defaults to ``1``.
        attempts (int): number of random hard-centered crops to try. Defaults
            to ``5``.
        fallback (str): behaviour when no hard points exist, one of
            ``"random"``, ``"center"``, or ``"none"`` (return unchanged).
            Defaults to ``"none"``.
        p (float): probability of applying the crop at all. Defaults to ``0.5``.

    Note:
        With no ``segment`` key all points are treated as non-hard, so the
        ``fallback`` path is taken.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import HardExampleCrop
            >>> np.random.seed(0)
            >>> coord = np.random.rand(100, 3).astype("f4")
            >>> segment = np.zeros(100, dtype=int); segment[[5, 50]] = 3  # two rare points
            >>> data = {"coord": coord.copy(), "segment": segment.copy()}
            >>> # over budget: crop centers on a hard (label 3) point, keeping it in-frame
            >>> out = HardExampleCrop(point_max=10, hard_labels=(2, 3),
            ...                       min_hard_points=1, p=1.0)(data)
            >>> len(out["coord"])  # 100 -> 10 points (closest to a hard center)
            10
            >>> int(np.isin(out["segment"], (2, 3)).sum()) >= 1  # at least one hard point survives
            True
    """

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
    """Randomly permute the point ordering of the sample.

    Generates a random permutation of the point indices and reorders every
    per-point key in ``index_valid_keys`` together via ``index_operator``, so
    point-aligned arrays stay consistent. The point count is unchanged. Requires
    ``coord`` (asserted). Registered as ``ShufflePoint`` -- use this string as
    the ``type`` in a ``transform=[...]`` config list.

    Note:
        Takes no constructor arguments. Removes any incidental ordering bias
        before models or pooling that could be order-sensitive.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import ShufflePoint
            >>> np.random.seed(0)
            >>> data = {"coord": np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]], dtype="f4"),
            ...         "energy": np.array([[10.], [20.], [30.]], dtype="f4")}
            >>> out = ShufflePoint()(data)
            >>> len(out["coord"])  # same points, reordered (coord and energy together)
            3
            >>> sorted(out["energy"].ravel().tolist())  # the set of values is preserved
            [10.0, 20.0, 30.0]
    """

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        data_dict = index_operator(data_dict, shuffle_index)
        return data_dict

@TRANSFORMS.register_module()
class CropBoundary(object):
    """Drop points whose ``segment`` label is 0 or 1.

    Keeps only points with ``segment`` not in ``{0, 1}`` -- conventionally
    discarding unannotated/boundary classes -- and subsamples every per-point
    key in ``index_valid_keys`` together via ``index_operator``. Requires
    ``segment`` (asserted). Registered as ``CropBoundary`` -- use this string as
    the ``type`` in a ``transform=[...]`` config list.

    Note:
        Takes no constructor arguments. The removed labels (0 and 1) are
        hard-coded; ``segment`` is flattened before masking.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import CropBoundary
            >>> data = {"coord": np.array([[0., 0., 0.], [1., 1., 1.],
            ...                            [2., 2., 2.], [3., 3., 3.]], dtype="f4"),
            ...         "segment": np.array([0, 1, 2, 3])}
            >>> out = CropBoundary()(data)  # drop points whose segment is 0 or 1
            >>> out["segment"]
            array([2, 3])
            >>> out["coord"]  # coord cropped in lockstep with segment
            array([[2., 2., 2.],
                   [3., 3., 3.]], dtype=float32)
    """

    def __call__(self, data_dict):
        assert "segment" in data_dict
        segment = data_dict["segment"].flatten()
        mask = (segment != 0) * (segment != 1)
        data_dict = index_operator(data_dict, mask)
        return data_dict

@TRANSFORMS.register_module()
class RandomDrop(object):
    """Randomly overwrite a fraction of one key's rows with a constant.

    With probability ``p_apply`` (and only if ``key`` is present), selects a
    random ``p_drop`` fraction of the rows of ``data_dict[key]`` and sets them
    to ``value`` -- a feature-masking augmentation that keeps the point count
    unchanged. Registered as ``RandomDrop`` -- use this string as the ``type``
    in a ``transform=[...]`` config list.

    Args:
        key (str): the per-point key whose rows may be overwritten.
        p_apply (float): probability of applying the transform to a sample.
            Defaults to ``0.5``.
        p_drop (float): fraction of rows to overwrite when applied. Defaults to
            ``0.2``.
        value (float): constant written into the selected rows. Defaults to
            ``0.0``.

    Note:
        Operates on a single named key only; it does not touch ``coord`` or
        other point-aligned arrays, so the cloud geometry is preserved.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> from pimm.datasets.transform import RandomDrop
            >>> np.random.seed(0)
            >>> data = {"energy": np.array([[1.], [2.], [3.], [4.]], dtype="f4")}
            >>> out = RandomDrop(key="energy", p_apply=1.0, p_drop=0.5, value=0.0)(data)
            >>> # NOTE: the write uses data[key][idx][:] = value, which assigns into a
            >>> # fancy-index COPY -- so as written the rows are left unchanged:
            >>> out["energy"].ravel()
            array([1., 2., 3., 4.], dtype=float32)
    """

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
