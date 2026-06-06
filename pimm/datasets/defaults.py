"""Default point-cloud datasets and dataset containers.

These generic adapters load preprocessed per-event directories of ``.npy``
assets, then hand flat dictionaries to the configured transform pipeline. More
detector-specific datasets in this package preserve the same output contract so
the model and collation code can stay shared.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import json
from re import split

import numpy as np
from torch.utils.data import Dataset
from collections.abc import Sequence

from pimm.utils.logger import get_root_logger
from pimm.utils.cache import shared_dict

from .builder import DATASETS, build_dataset
from .test_fragments import build_test_fragments
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class DefaultDataset(Dataset):
    """Load a directory or split-file collection of numpy assets.

    Each item is a flat dict keyed by asset name, with at minimum ``coord``,
    ``segment``, ``instance``, ``name``, and ``split``. Missing segmentation or
    instance labels are filled with ``ignore_index``-style ``-1`` arrays.
    """

    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
        "pose",
    ]

    def __init__(
        self,
        split="train",
        data_root="data/dataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        loop=1,
    ):
        """Initialize data discovery, transform pipeline, and test transforms."""
        super(DefaultDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.ignore_index = ignore_index
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} {} set.".format(
                len(self.data_list), self.loop, os.path.basename(self.data_root), split
            )
        )

    def get_data_list(self):
        """Resolve split names or split JSON files into event directories."""
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        for split in split_list:
            if os.path.isfile(os.path.join(self.data_root, split)):
                with open(os.path.join(self.data_root, split)) as f:
                    data_list += [
                        os.path.join(self.data_root, data) for data in json.load(f)
                    ]
            else:
                data_list += glob.glob(os.path.join(self.data_root, split, "*"))
        return data_list

    def get_data(self, idx):
        """Read one event directory into a flat numpy dictionary."""
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        if self.cache:
            cache_name = f"pimm-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["split"] = split

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        return data_dict

    def get_data_name(self, idx):
        """Return the basename used for logs, predictions, and cache keys."""
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def get_split_name(self, idx):
        """Return the split directory name for the indexed sample."""
        return os.path.basename(
            os.path.dirname(self.data_list[idx % len(self.data_list)])
        )

    def prepare_train_data(self, idx):
        """Load one sample and apply the train transform chain."""
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        """Build augmented and voxelized fragments for test-time inference."""
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        result_dict["fragment_list"] = build_test_fragments(
            data_dict,
            aug_transform=self.aug_transform,
            test_voxelize=self.test_voxelize,
            test_crop=self.test_crop,
            post_transform=self.post_transform,
            add_index_without_voxelize=True,
        )
        return result_dict

    def __getitem__(self, idx):
        """Return a transformed train item or a fragmented test item."""
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        """Return length after applying the train-time loop multiplier."""
        return len(self.data_list) * self.loop


@DATASETS.register_module()
class ConcatDataset(Dataset):
    """Concatenate configured datasets while preserving per-dataset ratios.

    ``loop`` is used by ``MultiDatasetDataloader`` as the epoch multiplier for
    the main dataset; child dataset ``loop`` values can also act as sampling
    ratios when mixed batches are built.
    """

    def __init__(self, datasets, loop=1):
        """Build child datasets from configs and flatten their index space."""
        super(ConcatDataset, self).__init__()
        self.datasets = [build_dataset(dataset) for dataset in datasets]
        self.loop = loop
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in the concat set.".format(
                len(self.data_list), self.loop
            )
        )

    def get_data_list(self):
        """Return ``(dataset_index, local_index)`` pairs for all children."""
        data_list = []
        for i in range(len(self.datasets)):
            data_list.extend(
                zip(
                    np.ones(len(self.datasets[i]), dtype=int) * i,
                    np.arange(len(self.datasets[i])),
                )
            )
        return data_list

    def get_data(self, idx):
        """Dispatch a global index to the owning child dataset."""
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx][data_idx]

    def get_data_name(self, idx):
        """Return the child dataset's sample name for a global index."""
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx].get_data_name(data_idx)

    def __getitem__(self, idx):
        """Return the child dataset item for a global index."""
        return self.get_data(idx)

    def __len__(self):
        """Return concatenated length after this container's loop multiplier."""
        return len(self.data_list) * self.loop
