"""
pimm-data LUCiD event datasets for SSL.

This adapter keeps the pimm-data reader boundary intact and returns flat
point-cloud dictionaries compatible with pimm's existing transform and
collation pipeline.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
from torch.utils.data import Dataset

from pimm.utils.logger import get_root_logger

from .builder import DATASETS
from .transform import Compose


def _load_pimm_data_lucid_dataset():
    try:
        from pimm_data import LUCiDDataset
    except ImportError:
        from pimm_data.lucid import LUCiDDataset
    return LUCiDDataset


@DATASETS.register_module()
class LUCiDEventSSLDataset(Dataset):
    """LUCiD SK-like raw sensor event dataset for Sonata-style SSL.

    Parameters
    ----------
    data_root:
        Directory containing per-config LUCiD outputs.
    configs:
        Sequence of config names or dictionaries. Dict entries can contain
        ``name``, ``label``, ``label_name``, ``data_root``, and ``pimm_split``.
    split:
        ``"train"`` uses all events except the deterministic holdout.
        ``"holdout"``, ``"val"``, and ``"test"`` use the holdout events.
        ``"all"`` uses all events.
    holdout_events_per_config:
        Number of events reserved from each config before train/holdout split.
    holdout_fraction:
        Optional fraction used when ``holdout_events_per_config`` is ``None``.
    """

    _HOLDOUT_SPLITS = {"holdout", "val", "test"}

    def __init__(
        self,
        data_root,
        configs,
        split="train",
        dataset_name="wc",
        holdout_events_per_config=0,
        holdout_fraction=None,
        holdout_seed=0,
        holdout_strategy="random",
        max_events_per_config=-1,
        min_points=0,
        aggregate_sensor_hits=True,
        time_aggregation="earliest",
        transform=None,
        loop=1,
        test_mode=False,
        test_cfg=None,
    ):
        super().__init__()
        if test_mode:
            raise NotImplementedError(
                "LUCiDEventSSLDataset is intended for train/val loaders, not "
                "fragmented test-time inference."
            )
        if not configs:
            raise ValueError("configs must contain at least one LUCiD config")
        if holdout_strategy not in {"random", "head", "tail"}:
            raise ValueError(
                "holdout_strategy must be one of {'random', 'head', 'tail'}"
            )
        if time_aggregation not in {"earliest", "pe_weighted", "mean", "first"}:
            raise ValueError(
                "time_aggregation must be one of "
                "{'earliest', 'pe_weighted', 'mean', 'first'}"
            )

        self.data_root = data_root
        self.config_specs = [
            self._normalize_config_spec(cfg, default_label=i)
            for i, cfg in enumerate(configs)
        ]
        self.split = split
        self.dataset_name = dataset_name
        self.holdout_events_per_config = holdout_events_per_config
        self.holdout_fraction = holdout_fraction
        self.holdout_seed = int(holdout_seed)
        self.holdout_strategy = holdout_strategy
        self.max_events_per_config = int(max_events_per_config)
        self.min_points = int(min_points)
        self.aggregate_sensor_hits = bool(aggregate_sensor_hits)
        self.time_aggregation = time_aggregation
        self.transform = Compose(transform)
        self.loop = int(loop)
        self.test_mode = False
        self.test_cfg = test_cfg

        self.datasets = self._build_sources()
        self.data_list = self._build_data_list()

        logger = get_root_logger()
        logger.info(
            "LUCiDEventSSLDataset: %d events x %d loop, split=%s, configs=%s",
            len(self.data_list),
            self.loop,
            self.split,
            [spec["name"] for spec in self.config_specs],
        )

    @staticmethod
    def _normalize_config_spec(config, default_label):
        if isinstance(config, str):
            return {
                "name": config,
                "label": int(default_label),
                "label_name": config,
                "data_root": None,
                "pimm_split": "",
            }
        if not isinstance(config, Mapping):
            raise TypeError(
                "Each config must be a string or mapping, got "
                f"{type(config).__name__}"
            )

        name = config.get("name", config.get("config", config.get("split")))
        root = config.get("data_root", config.get("root"))
        if name is None and root is None:
            raise ValueError("Config mapping must provide 'name' or 'data_root'")
        if name is None:
            name = os.path.basename(os.path.normpath(root))
        label = config.get("label", default_label)
        return {
            "name": str(name),
            "label": int(label),
            "label_name": str(config.get("label_name", name)),
            "data_root": root,
            "pimm_split": config.get("pimm_split", ""),
        }

    def _build_sources(self):
        pimm_data_lucid = _load_pimm_data_lucid_dataset()
        sources = []
        for spec in self.config_specs:
            source_root = spec["data_root"]
            if source_root is None:
                source_root = os.path.join(self.data_root, spec["name"])
            dataset = pimm_data_lucid(
                data_root=source_root,
                split=spec["pimm_split"],
                modalities=("sensor",),
                dataset_name=self.dataset_name,
                transform=None,
                loop=1,
            )
            sources.append(
                {
                    "dataset": dataset,
                    "name": spec["name"],
                    "label": spec["label"],
                    "label_name": spec["label_name"],
                    "source_root": source_root,
                }
            )
        return sources

    def _resolve_holdout_count(self, n_events):
        if self.holdout_events_per_config is not None:
            holdout = int(self.holdout_events_per_config)
        elif self.holdout_fraction is not None:
            holdout = int(round(float(self.holdout_fraction) * n_events))
        else:
            holdout = 0
        return max(0, min(holdout, n_events))

    def _split_indices(self, source_idx, n_events):
        indices = np.arange(n_events, dtype=np.int64)
        holdout = self._resolve_holdout_count(n_events)
        if holdout == 0:
            holdout_indices = indices[:0]
            train_indices = indices
        elif self.holdout_strategy == "head":
            holdout_indices = indices[:holdout]
            train_indices = indices[holdout:]
        elif self.holdout_strategy == "tail":
            holdout_indices = indices[-holdout:]
            train_indices = indices[:-holdout]
        else:
            rng = np.random.default_rng(self.holdout_seed + source_idx)
            perm = rng.permutation(indices)
            holdout_indices = np.sort(perm[:holdout])
            train_indices = np.sort(perm[holdout:])

        if self.split == "train":
            selected = train_indices
        elif self.split in self._HOLDOUT_SPLITS:
            selected = holdout_indices
        elif self.split == "all":
            selected = indices
        else:
            raise ValueError(
                "split must be one of {'train', 'holdout', 'val', 'test', 'all'}, "
                f"got {self.split!r}"
            )

        return selected

    def _build_data_list(self):
        data_list = []
        logger = get_root_logger()
        for source_idx, source in enumerate(self.datasets):
            selected = self._split_indices(source_idx, len(source["dataset"]))
            before_filter = len(selected)
            if self.min_points > 0:
                accepted = []
                for event_idx in selected:
                    if (
                        self._event_point_count(source["dataset"], int(event_idx))
                        > self.min_points
                    ):
                        accepted.append(int(event_idx))
                        if (
                            self.max_events_per_config > 0
                            and len(accepted) >= self.max_events_per_config
                        ):
                            break
                selected = accepted
                self._close_source_reader(source["dataset"])
                logger.info(
                    "LUCiDEventSSLDataset: kept %d/%d %s events with >%d points",
                    len(selected),
                    before_filter,
                    source["name"],
                    self.min_points,
                )
            elif self.max_events_per_config > 0:
                selected = selected[: self.max_events_per_config]
            data_list.extend((source_idx, int(event_idx)) for event_idx in selected)
        return data_list

    def _event_point_count(self, dataset, event_idx):
        raw = dataset.sensor_reader.read_event(event_idx)
        sensor_idx = raw["sensor_idx"]
        if self.aggregate_sensor_hits:
            return int(np.unique(sensor_idx).size)
        return int(sensor_idx.size)

    @staticmethod
    def _close_source_reader(dataset):
        reader = getattr(dataset, "sensor_reader", None)
        if reader is not None:
            reader.close()

    @staticmethod
    def _aggregate_hits(data, time_aggregation):
        sensor_idx = data["sensor_idx"]
        if sensor_idx.size == 0:
            return data
        order = np.argsort(sensor_idx, kind="stable")
        sorted_sensor_idx = sensor_idx[order]
        unique_sensor_idx, starts = np.unique(
            sorted_sensor_idx, return_index=True
        )
        coord = data["coord"][order][starts]
        energy_sorted = data["energy"][order]
        time_sorted = data["time"][order]
        energy = np.add.reduceat(energy_sorted, starts, axis=0)

        if time_aggregation == "earliest":
            time = np.minimum.reduceat(time_sorted, starts, axis=0)
        elif time_aggregation == "mean":
            counts = np.diff(np.r_[starts, sorted_sensor_idx.shape[0]])
            time = np.add.reduceat(time_sorted, starts, axis=0) / counts[:, None]
        elif time_aggregation == "pe_weighted":
            weighted = np.add.reduceat(time_sorted * energy_sorted, starts, axis=0)
            time = weighted / np.maximum(energy, 1.0e-6)
        else:
            time = time_sorted[starts]

        data["coord"] = coord.astype(np.float32, copy=False)
        data["energy"] = energy.astype(np.float32, copy=False)
        data["time"] = time.astype(np.float32, copy=False)
        data["sensor_idx"] = unique_sensor_idx.astype(np.int64, copy=False)
        return data

    def get_data(self, idx):
        source_idx, event_idx = self.data_list[idx % len(self.data_list)]
        source = self.datasets[source_idx]
        sample = source["dataset"][event_idx]
        sensor = sample["sensor"]

        data = {
            "coord": sensor["coord"].astype(np.float32, copy=False),
            "energy": sensor["energy"].astype(np.float32, copy=False),
            "time": sensor["time"].astype(np.float32, copy=False),
            "sensor_idx": sensor["sensor_idx"].astype(np.int64, copy=False),
            "event_label": np.array([source["label"]], dtype=np.int64),
            "config_id": np.array([source_idx], dtype=np.int64),
            "name": f'{source["name"]}/{sample["name"]}',
            "split": self.split,
        }

        if self.aggregate_sensor_hits:
            data = self._aggregate_hits(data, self.time_aggregation)
        return data

    def get_data_name(self, idx):
        source_idx, event_idx = self.data_list[idx % len(self.data_list)]
        source = self.datasets[source_idx]
        return f'{source["name"]}/{source["dataset"].get_data_name(event_idx)}'

    def prepare_train_data(self, idx):
        return self.transform(self.get_data(idx))

    def __getitem__(self, idx):
        return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
