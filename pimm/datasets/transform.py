"""
3D point cloud augmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from __future__ import annotations

import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
from scipy.spatial import cKDTree
import numpy as np
import torch
import copy
from collections.abc import Sequence, Mapping

from pimm.utils.registry import Registry
from typing import Optional
# from cnms import cnms
# from pytorch3d import _C
# from pytorch3d.ops import ball_query, knn_points

TRANSFORMS = Registry("transforms")

# Anchor mining defaults (can be overridden from config)
try:
    from .anchors import compute_anchors, ANCHOR_DEFAULT_CFG
except Exception:
    compute_anchors = None
    ANCHOR_DEFAULT_CFG = dict()


def index_operator(data_dict, index, duplicate=False):
    # index selection operator for keys in "index_valid_keys"
    # custom these keys by "Update" transform in config
    if "index_valid_keys" not in data_dict:
        data_dict["index_valid_keys"] = [
            "coord",
            "color",
            "normal",
            "strength",
            "segment",
            "instance",
            "energy",
            "local_shape",
            "segment_motif",
            "segment_pid",
            "instance_particle",
            "instance_interaction",
            "momentum",
        ]
    if not duplicate:
        for key in data_dict["index_valid_keys"]:
            if key in data_dict:
                data_dict[key] = data_dict[key][index]
        return data_dict
    else:
        data_dict_ = dict()
        for key in data_dict.keys():
            if key in data_dict["index_valid_keys"]:
                data_dict_[key] = data_dict[key][index]
            else:
                data_dict_[key] = data_dict[key]
        return data_dict_


@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


@TRANSFORMS.register_module()
class Copy(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class Update(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class NormalizeColor(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 255
        return data_dict


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
                data_dict["coord"] -= centroid
            else:
                centroid = np.array(self.center)
                data_dict["coord"] -= centroid

            if self.scale is None:
                m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
                data_dict["coord"] = data_dict["coord"] / m
            else:
                data_dict["coord"] = data_dict["coord"] / self.scale
        return data_dict


@TRANSFORMS.register_module()
class LogTransform(object):
    def __init__(self, min_val=1.0e-2, max_val=20.0, log=True, keys=("energy",)):
        self.min_val = min_val
        self.max_val = max_val
        self.log = log
        if not isinstance(keys, tuple):
            keys = (keys,)
        self.keys = keys

    def log_transform(self, x):
        """Transform energy to logarithmic scale on [-1,1]"""
        # [emin, emax] -> [-1,1]
        y0 = np.log10(self.min_val)
        y1 = np.log10(self.max_val + self.min_val)
        return 2 * (np.log10(x + self.min_val) - y0) / (y1 - y0) - 1

    def linear_transform(self, x):
        """Transform energy to linear scale on [-1,1]"""
        return 2 * (x - self.min_val) / (self.max_val - self.min_val) - 1

    def __call__(self, data_dict):
        for k in self.keys:
            if k in data_dict.keys():
                data_dict[k] = (
                    self.log_transform(data_dict[k])
                    if self.log
                    else self.linear_transform(data_dict[k])
                )
            else:
                raise ValueError(f"Key {k} not found in data_dict")
        return data_dict

@TRANSFORMS.register_module()
class MomentumTransform(object):
    def __init__(self, keys=("momentum",)):
        self.keys = keys

    def __call__(self, data_dict):
        for k in self.keys:
            if k in data_dict.keys():
                mom = data_dict[k]
                mom = np.where(mom > 0, np.log10(np.clip(mom, 1e-6, None)), mom)
                data_dict[k] = mom
        return data_dict


@TRANSFORMS.register_module()
class PositiveShift(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
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
                    data_dict["coord"][:, 0] -= (x_min + x_max) / 2
                elif axis == "y":
                    x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                    x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                    data_dict["coord"][:, 1] -= (y_min + y_max) / 2
                elif axis == "z":
                    x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                    x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                    data_dict["coord"][:, 2] -= (z_min + z_max) / 2
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
            data_dict["coord"] += [shift_x, shift_y, shift_z]
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
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
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
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
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
                if "normal" in data_dict.keys():
                    data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
            elif axis == "y" and random.random() < self.p:
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
                if "normal" in data_dict.keys():
                    data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
            elif axis == "z" and random.random() < self.p:
                data_dict["coord"][:, 2] = -data_dict["coord"][:, 2]
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
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][
                :, :3
            ] + blend_factor * contrast_feat
        return data_dict


@TRANSFORMS.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict

@TRANSFORMS.register_module()
class EnergeticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "energy" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1) - 0.5) * 2 * self.ratio
            data_dict["energy"] = np.clip(tr + data_dict["energy"], -1, 1)
        return data_dict


@TRANSFORMS.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= self.std * 255
            data_dict["color"][:, :3] = np.clip(
                noise + data_dict["color"][:, :3], 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class EnergyJitter(object):
    def __init__(self, p=0.5, jitter_ratio=0.005, min_val=0.0):
        self.p = p
        self.jitter_ratio = jitter_ratio
        self.min_val = min_val

    def __call__(self, data_dict):
        if "energy" in data_dict.keys() and np.random.rand() < self.p:
            # jitter by +/- 0.5%
            jitter = (
                np.random.rand(*data_dict["energy"].shape) * 2 - 1
            ) * self.jitter_ratio
            data_dict["energy"] = np.clip(
                data_dict["energy"] * (1 + jitter), self.min_val, None
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomColorGrayScale(object):
    def __init__(self, p):
        self.p = p

    @staticmethod
    def rgb_to_grayscale(color, num_output_channels=1):
        if color.shape[-1] < 3:
            raise TypeError(
                "Input color should have at least 3 dimensions, but found {}".format(
                    color.shape[-1]
                )
            )

        if num_output_channels not in (1, 3):
            raise ValueError("num_output_channels should be either 1 or 3")

        r, g, b = color[..., 0], color[..., 1], color[..., 2]
        gray = (0.2989 * r + 0.587 * g + 0.114 * b).astype(color.dtype)
        gray = np.expand_dims(gray, axis=-1)

        if num_output_channels == 3:
            gray = np.broadcast_to(gray, color.shape)

        return gray

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            data_dict["color"] = self.rgb_to_grayscale(data_dict["color"], 3)
        return data_dict


@TRANSFORMS.register_module()
class RandomColorJitter(object):
    """
    Random Color Jitter for 3D point cloud (refer torchvision)
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.95):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.p = p

    @staticmethod
    def _check_input(
        value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def blend(color1, color2, ratio):
        ratio = float(ratio)
        bound = 255.0
        return (
            (ratio * color1 + (1.0 - ratio) * color2)
            .clip(0, bound)
            .astype(color1.dtype)
        )

    @staticmethod
    def rgb2hsv(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        eqc = maxc == minc
        cr = maxc - minc
        s = cr / (np.ones_like(maxc) * eqc + maxc * (1 - eqc))
        cr_divisor = np.ones_like(maxc) * eqc + cr * (1 - eqc)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = (h / 6.0 + 1.0) % 1.0
        return np.stack((h, s, maxc), axis=-1)

    @staticmethod
    def hsv2rgb(hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.astype(np.int32)

        p = np.clip((v * (1.0 - s)), 0.0, 1.0)
        q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
        t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6
        mask = np.expand_dims(i, axis=-1) == np.arange(6)

        a1 = np.stack((v, q, p, p, t, v), axis=-1)
        a2 = np.stack((t, v, v, q, p, p), axis=-1)
        a3 = np.stack((p, p, t, v, v, q), axis=-1)
        a4 = np.stack((a1, a2, a3), axis=-1)

        return np.einsum("...na, ...nab -> ...nb", mask.astype(hsv.dtype), a4)

    def adjust_brightness(self, color, brightness_factor):
        if brightness_factor < 0:
            raise ValueError(
                "brightness_factor ({}) is not non-negative.".format(brightness_factor)
            )

        return self.blend(color, np.zeros_like(color), brightness_factor)

    def adjust_contrast(self, color, contrast_factor):
        if contrast_factor < 0:
            raise ValueError(
                "contrast_factor ({}) is not non-negative.".format(contrast_factor)
            )
        mean = np.mean(RandomColorGrayScale.rgb_to_grayscale(color))
        return self.blend(color, mean, contrast_factor)

    def adjust_saturation(self, color, saturation_factor):
        if saturation_factor < 0:
            raise ValueError(
                "saturation_factor ({}) is not non-negative.".format(saturation_factor)
            )
        gray = RandomColorGrayScale.rgb_to_grayscale(color)
        return self.blend(color, gray, saturation_factor)

    def adjust_hue(self, color, hue_factor):
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError(
                "hue_factor ({}) is not in [-0.5, 0.5].".format(hue_factor)
            )
        orig_dtype = color.dtype
        hsv = self.rgb2hsv(color / 255.0)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = (h + hue_factor) % 1.0
        hsv = np.stack((h, s, v), axis=-1)
        color_hue_adj = (self.hsv2rgb(hsv) * 255.0).astype(orig_dtype)
        return color_hue_adj

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        b = (
            None
            if brightness is None
            else np.random.uniform(brightness[0], brightness[1])
        )
        c = None if contrast is None else np.random.uniform(contrast[0], contrast[1])
        s = (
            None
            if saturation is None
            else np.random.uniform(saturation[0], saturation[1])
        )
        h = None if hue is None else np.random.uniform(hue[0], hue[1])
        return fn_idx, b, c, s, h

    def __call__(self, data_dict):
        (
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if (
                fn_id == 0
                and brightness_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_brightness(
                    data_dict["color"], brightness_factor
                )
            elif (
                fn_id == 1 and contrast_factor is not None and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_contrast(
                    data_dict["color"], contrast_factor
                )
            elif (
                fn_id == 2
                and saturation_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_saturation(
                    data_dict["color"], saturation_factor
                )
            elif fn_id == 3 and hue_factor is not None and np.random.rand() < self.p:
                data_dict["color"] = self.adjust_hue(data_dict["color"], hue_factor)
        return data_dict


@TRANSFORMS.register_module()
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype("float")
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc
        )
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype("uint8")
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype("uint8")

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            # Assume color[:, :3] is rgb
            hsv = HueSaturationTranslation.rgb_to_hsv(data_dict["color"][:, :3])
            hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
            sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
            hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
            hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
            data_dict["color"][:, :3] = np.clip(
                HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomColorDrop(object):
    def __init__(self, p=0.2, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            data_dict["color"] *= self.color_augment
        return data_dict

    def __repr__(self):
        return "RandomColorDrop(color_augment: {}, p: {})".format(
            self.color_augment, self.p
        )


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
    ):
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

    def __call__(self, data_dict):
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
            summed = {}
            if self.sum_keys:
                voxel_of_point = np.empty(len(key), dtype=inverse.dtype)
                voxel_of_point[idx_sort] = inverse
                num_voxels = len(count)
                for sk in self.sum_keys:
                    if sk in data_dict:
                        vals = data_dict[sk]
                        agg = np.zeros((num_voxels,) + vals.shape[1:], dtype=vals.dtype)
                        np.add.at(agg, voxel_of_point, vals)
                        summed[sk] = agg
            data_dict = index_operator(data_dict, idx_unique)
            for sk, agg in summed.items():
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
        size = int(np.random.uniform(*scale) * max_size) if size_override is None else int(size_override)
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

        global_views = [major_view] + global_views

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

        # augmentation and concat
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
    """
    Custom collation for HMAE that handles variable-length masked patches.

    Packs target coordinates/energies into flattened arrays with offsets (no padding).
    Assumes 'energy' will always be present in data_dict.
    """

    def __init__(
        self,
        points_per_patch: int = 128,
    ):
        self.points_per_patch = points_per_patch

    def __call__(self, data_dict):
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


@TRANSFORMS.register_module()
class RandomDrop(object):
    def __init__(self, key: str, p_apply: float = 0.5, p_drop: float = 0.2, value: float = 0.0):
        self.key = key
        self.p_apply = p_apply
        self.p_drop = p_drop
        self.value = value

    def __call__(self, data_dict):
        if self.key in data_dict.keys() and np.random.rand() < self.p_apply:
            n = data_dict[self.key].shape[0]
            idx = np.random.choice(n, int(n*self.p_drop), replace=False)
            data_dict[self.key][idx][:] = self.value
        return data_dict

class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
