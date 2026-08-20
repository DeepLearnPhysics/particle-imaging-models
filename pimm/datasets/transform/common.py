"""Transform registry and point-cloud data-dict pipeline.

Transforms are config-built callables that receive and return a mutable
``data_dict``. Dataset wrappers establish modality-specific keys; transforms
then normalize, augment, voxelize, and collect those keys into the model-facing
batch contract consumed by the collation helpers.

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
"""Registry for config-built dataset transforms."""


def _valid_vertex_mask(data_dict):
    """Return a mask for valid 3D vertex rows, or None when not applicable."""
    vertex = data_dict.get("vertex")
    if vertex is None:
        return None
    if vertex.ndim != 2 or vertex.shape[1] != 3:
        return None
    # v3 uses (-1, -1, -1) as a missing-vertex sentinel. v2 vertices are
    # corrected labels and should transform with the coordinates.
    return ~(vertex == -1).all(axis=1)


def _apply_to_v3_vertex(data_dict, transform):
    """Apply a coordinate transform to valid vertex metadata in place."""
    mask = _valid_vertex_mask(data_dict)
    if mask is None or not mask.any():
        return
    data_dict["vertex"][mask] = transform(data_dict["vertex"][mask])


def _apply_to_aux_positions(data_dict, transform):
    """Apply a coordinate transform to opt-in auxiliary POSITION keys.

    Keys are listed in ``data_dict['aux_position_keys']`` (set via the Update
    transform). These ride the SAME translate/scale/rotate/flip as ``coord``
    so position targets (e.g. ``primary_vertex``) stay aligned under
    augmentation. No-op when the list is absent.
    """
    for key in data_dict.get("aux_position_keys", ()) or ():
        v = data_dict.get(key)
        if v is not None:
            data_dict[key] = transform(np.asarray(v))


def _apply_to_aux_directions(data_dict, transform):
    """Apply a LINEAR (norm-preserving) transform to opt-in DIRECTION keys.

    Keys listed in ``data_dict['aux_direction_keys']`` are unit vectors
    (e.g. ``primary_direction`` or PILArNet ``momentum_vec``): only
    rotation/flip are applied (NOT translation or scale), and the result is
    re-normalized to guard against float drift. All-``-1`` sentinel rows are
    preserved. Callers must pass the linear part only (no centering).
    """
    for key in data_dict.get("aux_direction_keys", ()) or ():
        v = data_dict.get(key)
        if v is None:
            continue
        v = np.asarray(v)
        if v.ndim == 2 and v.shape[1] == 3:
            valid = ~(v == -1).all(axis=1)
            out = v.copy()
            if valid.any():
                direction = transform(v[valid])
                norm = np.linalg.norm(direction, axis=-1, keepdims=True)
                out[valid] = direction / np.clip(norm, 1e-9, None)
            data_dict[key] = out.astype(np.float32, copy=False)
            continue
        v = transform(v)
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        data_dict[key] = (v / np.clip(norm, 1e-9, None)).astype(
            np.float32, copy=False
        )


def _apply_to_aux_vectors(data_dict, transform):
    """Apply a linear rotation/flip to magnitude-bearing vector keys.

    Keys listed in ``aux_vector_keys`` (for example PILArNet
    ``momentum_vector`` in GeV) transform like directions but are not
    re-normalized. All-``-1`` sentinel rows are preserved.
    """
    for key in data_dict.get("aux_vector_keys", ()) or ():
        value = data_dict.get(key)
        if value is None:
            continue
        value = np.asarray(value)
        if value.ndim == 2 and value.shape[1] == 3:
            valid = ~(value == -1).all(axis=1)
            out = value.copy()
            if valid.any():
                out[valid] = transform(value[valid])
            data_dict[key] = out.astype(np.float32, copy=False)
        else:
            data_dict[key] = transform(value)


def _translate_axis(points, dim, value):
    """Return a copy of points translated along one axis."""
    points = points.copy()
    points[:, dim] += value
    return points


# Anchor mining defaults (can be overridden from config)
try:
    from pimm.datasets.preprocessing.anchors import compute_anchors, ANCHOR_DEFAULT_CFG
except Exception:
    compute_anchors = None
    ANCHOR_DEFAULT_CFG = dict()


def index_operator(data_dict, index, duplicate=False):
    """Apply a point index to every per-point key listed in index_valid_keys."""
    # Configs can override index_valid_keys with the Update transform.
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
            "mass",
            "px",
            "py",
            "pz",
            "momentum_vector",
            "momentum_vec",
            "true_energy",
            "parent_pdg",
            "parent_track_id",
            "vertex",
            "interaction_vertex",
            "is_primary_4cls",
            # JAXTPCDataset keys (JAXTPC)
            "track_ids",
            "group_ids",
            "pdg",
            "volume_id",
            "interaction_ids",
            "ancestor_track_ids",
            "charge",
            "photons",
            "qs_fractions",
            "t0_us",
            "dx",
            "theta",
            "phi",
            "segment_interaction",
        ]
    if data_dict.get("revision") in ("v3", "v3_extra"):
        if not isinstance(data_dict["index_valid_keys"], list):
            data_dict["index_valid_keys"] = list(data_dict["index_valid_keys"])
        for key in ("is_primary", "is_primary_4cls"):
            if key not in data_dict["index_valid_keys"]:
                data_dict["index_valid_keys"].append(key)
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


__all__ = [
    "random",
    "numbers",
    "scipy",
    "np",
    "torch",
    "copy",
    "Sequence",
    "Mapping",
    "Optional",
    "cKDTree",
    "Registry",
    "TRANSFORMS",
    "_valid_vertex_mask",
    "_apply_to_v3_vertex",
    "_apply_to_aux_positions",
    "_apply_to_aux_directions",
    "_apply_to_aux_vectors",
    "_translate_axis",
    "compute_anchors",
    "ANCHOR_DEFAULT_CFG",
    "index_operator",
]
