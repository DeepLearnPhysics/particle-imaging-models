"""Shared test-time fragment assembly for point-cloud datasets."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import numpy as np


def build_test_fragments(
    data_dict: dict,
    *,
    aug_transform: Iterable,
    test_voxelize=None,
    test_crop=None,
    post_transform=None,
    add_index_without_voxelize: bool = False,
) -> list[dict]:
    """Apply test-time augmentation, voxelization, cropping, and post-transform."""
    data_dict_list = [aug(deepcopy(data_dict)) for aug in aug_transform]
    fragment_list = []
    for data in data_dict_list:
        if test_voxelize is not None:
            data_part_list = test_voxelize(data)
        else:
            if add_index_without_voxelize:
                data["index"] = np.arange(data["coord"].shape[0])
            data_part_list = [data]
        for data_part in data_part_list:
            if test_crop is not None:
                data_part = test_crop(data_part)
            else:
                data_part = [data_part]
            fragment_list += data_part

    if post_transform is not None:
        for i in range(len(fragment_list)):
            fragment_list[i] = post_transform(fragment_list[i])
    return fragment_list
