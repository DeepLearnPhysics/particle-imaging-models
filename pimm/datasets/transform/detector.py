"""Detector-specific transforms for multimodal detector data."""

import numpy as np

from .common import TRANSFORMS


@TRANSFORMS.register_module()
class PDGToSemantic:
    """Fallback semantic labels derived from PDG codes.

    Use this only when no label file is available. For production training,
    prefer modalities that include labl labels loaded by the dataset reader.

    Schemes:
        motif_5cls: shower(0), track(1), michel(2), delta(3), led(4)
        pid_6cls: photon(0), electron(1), muon(2), pion(3), proton(4), other(5)
        custom: user-provided {pdg_code: class_index} dict
    """

    _MOTIF = {
        22: 0,
        11: 0,
        -11: 0,
        13: 1,
        -13: 1,
        211: 1,
        -211: 1,
        2212: 1,
        321: 1,
        -321: 1,
    }

    _PID = {
        22: 0,
        11: 1,
        -11: 1,
        13: 2,
        -13: 2,
        211: 3,
        -211: 3,
        2212: 4,
    }

    def __init__(self, scheme="motif_5cls", custom_map=None):
        """Select a PDG-to-label scheme or accept a custom mapping."""
        self.scheme = scheme
        if scheme == "motif_5cls":
            self.mapping = self._MOTIF
            self.default = 4
        elif scheme == "pid_6cls":
            self.mapping = self._PID
            self.default = 5
        elif scheme == "custom":
            assert custom_map is not None
            self.mapping = custom_map
            self.default = max(custom_map.values()) + 1
        elif scheme == "none":
            self.mapping = None
            self.default = -1
        else:
            raise ValueError(f"Unknown label scheme: {scheme}")

    def __call__(self, data_dict):
        """Add semantic, PID, and simple instance labels when absent."""
        if self.mapping is None or "pdg" not in data_dict:
            return data_dict

        if "segment" in data_dict or "segment_motif" in data_dict:
            return data_dict

        pdg = data_dict["pdg"]
        n = len(pdg)
        labels = np.full(n, self.default, dtype=np.int32)
        for code, cls in self.mapping.items():
            labels[pdg == code] = cls

        data_dict["segment_motif"] = labels[:, None]

        if self.scheme == "motif_5cls":
            pid = np.full(n, 5, dtype=np.int32)
            for code, cls in self._PID.items():
                pid[pdg == code] = cls
            data_dict["segment_pid"] = pid[:, None]
        elif self.scheme == "pid_6cls":
            data_dict["segment_pid"] = labels[:, None]

        if "instance_particle" not in data_dict and "track_ids" in data_dict:
            track_ids = data_dict["track_ids"]
            mask = track_ids >= 0
            if mask.any():
                _, inverse = np.unique(track_ids[mask], return_inverse=True)
                out = np.full(n, -1, dtype=np.int32)
                out[mask] = inverse
                data_dict["instance_particle"] = out[:, None]
            else:
                data_dict["instance_particle"] = np.full((n, 1), -1, dtype=np.int32)

        if "instance_interaction" not in data_dict and "interaction_ids" in data_dict:
            iids = data_dict["interaction_ids"]
            mask = iids >= 0
            if mask.any():
                _, inverse = np.unique(iids[mask], return_inverse=True)
                out = np.full(n, -1, dtype=np.int32)
                out[mask] = inverse
                data_dict["instance_interaction"] = out[:, None]
            else:
                data_dict["instance_interaction"] = np.full(
                    (n, 1), -1, dtype=np.int32
                )

            data_dict["segment_interaction"] = (iids[:, None] != -1).astype(np.int32)

        return data_dict


@TRANSFORMS.register_module()
class FourierTime:
    """Encode scalar nanosecond times with fixed sine/cosine clock phases.

    Output channels are ``[sin(P0), cos(P0), ..., sin(PK), cos(PK), coarse]``.
    The optional coarse channel is signed log time normalized to one at the
    longest period; it is not clipped. Fixed-amplitude phases retain sensitivity
    to a given time difference independently of the timestamp's magnitude.

    ``periods_ns`` must be a nonempty, strictly increasing sequence of finite,
    positive periods. Choose the shortest period for the required resolution
    and the longest beyond the supported event window. Commensurate periods
    repeat, so the coarse channel supplies additional ordering information.

    Reference subtraction and modulo reduction happen in float64, before the
    trigonometric evaluation and final float32 cast. Input timestamps must not
    already have lost precision (e.g. by casting raw ns to bfloat16). No time
    jitter, clipping, geometry, or time-of-flight correction is applied.

    For multiview training, subtract a shared event reference BEFORE cropping,
    and set ``subtract_min=False`` here to preserve that reference in all views.
    Each configured key must contain scalar times with shape (N,) or (N, 1).
    """

    def __init__(
        self, periods_ns, keys=("time",), subtract_min=True, include_coarse=True
    ):
        self.periods = np.array(periods_ns, dtype=np.float64, copy=True)
        if (
            self.periods.ndim != 1
            or self.periods.size == 0
            or not np.isfinite(self.periods).all()
            or np.any(self.periods <= 0)
            or np.any(np.diff(self.periods) <= 0)
        ):
            raise ValueError(
                "periods_ns must be finite, positive, and strictly increasing"
            )
        self.keys = (keys,) if isinstance(keys, str) else tuple(keys)
        self.subtract_min = bool(subtract_min)
        self.include_coarse = bool(include_coarse)
        self.coarse_denom = np.log1p(self.periods[-1] / self.periods[0])

    def __call__(self, data_dict):
        for key in self.keys:
            if key not in data_dict:
                raise ValueError(f"Key {key} not found in data_dict")
            t = np.asarray(data_dict[key], dtype=np.float64)
            if t.ndim not in (1, 2) or (t.ndim == 2 and t.shape[1] != 1):
                raise ValueError(
                    f"{key} must contain scalar times with shape (N,) or (N, 1)"
                )
            t = t.reshape(-1)
            if not np.isfinite(t).all():
                raise ValueError(f"{key} contains nonfinite timestamps")
            if self.subtract_min and t.size:
                t = t - t.min()
            phase = np.remainder(t[:, None], self.periods)
            phase *= 2 * np.pi / self.periods
            encoded = np.stack((np.sin(phase), np.cos(phase)), axis=-1)
            encoded = encoded.reshape(t.size, 2 * self.periods.size)
            if self.include_coarse:
                coarse = (
                    np.sign(t)
                    * np.log1p(np.abs(t) / self.periods[0])
                    / self.coarse_denom
                )
                encoded = np.concatenate((encoded, coarse[:, None]), axis=1)
            data_dict[key] = encoded.astype(np.float32)
        return data_dict


@TRANSFORMS.register_module()
class BinaryLabel(object):
    """Write a per-point 0/1 label: 1 where ``key`` takes one of ``values``, else 0.

    Leaves ``key`` itself untouched, so the source label and anything derived from it stay as they were.

    Args:
        key (str): Per-point label to read.
        values (sequence[int]): Values of ``key`` that map to 1.
        out_key (str): Key written with the 0/1 label.

    Example:
        Tag sPHENIX noise hits (motif 4) for a separate noise head:

        .. code-block:: python

            dict(type="BinaryLabel", key="segment_motif", values=(4,), out_key="segment_noise")

        With ``segment_motif = [0, 1, 4]`` this writes ``segment_noise = [0, 0, 1]``.
    """

    def __init__(self, key, values, out_key):
        self.key = key
        self.values = tuple(values)
        self.out_key = out_key

    def __call__(self, data_dict):
        if self.key not in data_dict:
            raise ValueError(f"BinaryLabel needs '{self.key}'")
        data_dict[self.out_key] = np.isin(np.asarray(data_dict[self.key]), self.values).astype(np.int64)
        return data_dict


# Topology selection from Simón et al. 2021, arXiv:2102.11931, Sec. 5.
@TRANSFORMS.register_module()
class NEXTTopologyCut(object):
    """NEXT's selection for the topological analysis (Simón et al. 2021, arXiv 2102.11931 Sec. 5): every hit at least
    ``margin`` inside the drift volume, and one track when binned into ``voxel`` cubes (voxels touching by a face,
    edge or corner are one track; their "gross single-track cut"). Returns ``None`` for a rejected event, so it
    belongs in ``NEXTH5Dataset(event_filter=[...])``, which drops those events, not in ``transform``.

    Args:
        radius (float): Drift-volume radius (mm). Defaults to the release's hit extent, 1820.
        z_range (tuple): Drift-volume extent along z (mm). Defaults to the release's (0, 3618).
        margin (float): Fiducial margin (mm). Defaults to ``20``.
        voxel (float): Cube size of the single-track test (mm). Defaults to ``15``.
    """

    OFFSETS = np.array(
        [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
            if (i, j, k) > (0, 0, 0)
        ]
    )

    def __init__(self, radius=1820.0, z_range=(0.0, 3618.0), margin=20.0, voxel=15.0):
        self.radius, self.z_range, self.margin, self.voxel = (
            radius,
            z_range,
            margin,
            voxel,
        )

    def tracks(self, coord):
        """Number of connected groups of occupied ``voxel`` cubes."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        cells = np.unique(np.floor(coord / self.voxel).astype(np.int64), axis=0)
        cells -= cells.min(0) - 1
        dims = cells.max(0) + 2
        key = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[
            :, 2
        ]  # ascending: np.unique sorts rows
        rows, cols = [], []
        for o in self.OFFSETS:
            nb = (
                ((cells[:, 0] + o[0]) * dims[1] + cells[:, 1] + o[1]) * dims[2]
                + cells[:, 2]
                + o[2]
            )
            j = np.searchsorted(key, nb).clip(max=len(key) - 1)
            hit = key[j] == nb
            rows.append(np.flatnonzero(hit))
            cols.append(j[hit])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        graph = coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(key), len(key))
        )
        return connected_components(graph, directed=False)[0]

    def __call__(self, data_dict):
        coord = data_dict["coord"].astype(np.float64)
        inside = (
            np.hypot(coord[:, 0], coord[:, 1]).max() <= self.radius - self.margin
            and coord[:, 2].min() >= self.z_range[0] + self.margin
            and coord[:, 2].max() <= self.z_range[1] - self.margin
        )
        return data_dict if inside and self.tracks(coord) == 1 else None


# Adapted from mmkekic/NEXT_torch @1c43507, src/utils/data_loaders.py (Kekic et al. 2021).
@TRANSFORMS.register_module()
class KekicVoxelize(object):
    """The sparse input of Kekic et al. 2021 (arXiv 2009.10783), as their loader builds it (github.com/mmkekic/NEXT_torch
    @1c43507, src/utils/data_loaders.py ``get_3d_input``): hits binned with ``np.digitize`` into the 10 x 10 x 5 mm
    voxels of a 40 x 40 x 55 cm box, energies divided by the event total, hits sharing a voxel summed (their SCN
    ``InputLayer`` default). ``augment`` adds their training augmentation, line for line: a shift of up to 5 cm in whole
    voxels, a flip of each axis and an x <-> y swap about the pre-shift extent, and a 0.8-1.5 zoom per axis. NEXT-White's
    box was its whole detector; here each event is first centred in the box. Outputs ``grid_coord`` (M, 3) and the
    per-voxel ``energy`` (M, 1).

    Args:
        augment (bool): Their training augmentation. Defaults to ``False``.
        voxel (tuple): Voxel size (mm). Defaults to theirs, ``(10, 10, 5)``.
        box (tuple): Box size (mm), x and y centred on 0, z from 0. Defaults to theirs, ``(400, 400, 550)``.
    """

    def __init__(
        self, augment=False, voxel=(10.0, 10.0, 5.0), box=(400.0, 400.0, 550.0)
    ):
        self.augment, self.voxel = augment, np.asarray(voxel, dtype=np.float64)
        lo = np.array([-box[0] / 2, -box[1] / 2, 0.0])
        self.centre = lo + np.asarray(box) / 2
        self.bins = [
            lo[k] + self.voxel[k] * np.arange(round(box[k] / self.voxel[k]) + 1)
            for k in range(3)
        ]

    def __call__(self, data_dict):
        c = data_dict["coord"].astype(np.float64)
        c += self.centre - (c.min(0) + c.max(0)) / 2
        if self.augment:
            lo, hi = c.min(0), c.max(0)
            centre = (hi + lo) / 2
            for k, step in enumerate(self.voxel):
                top = min(self.bins[k].max() - hi[k], 50) // step
                bottom = max(self.bins[k].min() - lo[k], -50) // step + 1
                if top > bottom:
                    c[:, k] += np.random.randint(int(bottom), int(top)) * step
            flip = np.random.choice([True, False], 3)
            c[:, flip] = 2 * centre[flip] - c[:, flip]
            if np.random.choice([True, False]):
                c[:, :2] = lo[:2] + c[:, [1, 0]] - lo[[1, 0]]
            c = centre + np.random.uniform(0.8, 1.5, 3) * (c - centre)
        idx = np.stack([np.digitize(c[:, k], b) for k, b in enumerate(self.bins)], 1)
        e = np.nan_to_num(data_dict["energy"][:, 0].astype(np.float64))
        mask = ((idx >= 0) & (idx < np.array([len(b) for b in self.bins]))).all(1) & (
            e > 0
        )
        e = e / e.sum()
        mask &= e > 1e-6
        grid, inverse = np.unique(idx[mask], axis=0, return_inverse=True)
        data_dict["grid_coord"] = grid
        data_dict["energy"] = np.bincount(inverse.ravel(), e[mask])[:, None].astype(
            np.float32
        )
        return data_dict


@TRANSFORMS.register_module()
class RescaleTotal(object):
    """Scale ``key`` so its event total is ``total``: the total (e.g. NEXT's event energy) is hidden from the model.

    Args:
        key (str): The per-point array. Defaults to ``"energy"``.
        total (float): The event total after scaling. Defaults to ``1.0``.
    """

    def __init__(self, key="energy", total=1.0):
        self.key, self.total = key, total

    def __call__(self, data_dict):
        v = data_dict[self.key]
        data_dict[self.key] = (v * (self.total / v.sum())).astype(v.dtype)
        return data_dict
