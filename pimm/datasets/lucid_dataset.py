"""
LUCiDDataset — dataset for Water Cherenkov detector simulation output.

Loads PMT sensor data and/or 3D track segments from co-indexed HDF5 files.
Produces flat dicts compatible with pimm's transform/collation pipeline.

Example configs:

    # PMT event classification (sensor response as fixed-geometry point cloud)
    data = dict(train=dict(type="LUCiDDataset", data_root="dataset_wc",
        modalities=("sensor",), dataset_name="wc", ...))

    # Per-sensor instance separation (sparse per-particle entries)
    data = dict(train=dict(type="LUCiDDataset", data_root="dataset_wc",
        modalities=("sensor",), include_labels=True, ...))

    # 3D track reconstruction
    data = dict(train=dict(type="LUCiDDataset", data_root="dataset_wc",
        modalities=("seg",), ...))
"""

import os
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset

from pimm.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from .readers.lucid_seg_reader import LUCiDSegReader
from .readers.lucid_sensor_reader import LUCiDSensorReader


@DATASETS.register_module()
class LUCiDDataset(Dataset):
    """Water Cherenkov detector dataset.

    Parameters
    ----------
    data_root : str
        Root directory with seg/ and/or sensor/ subdirectories.
    split : str
        Split name for file discovery.
    transform : list[dict]
        Transform pipeline config.
    modalities : tuple[str]
        Which to load: 'seg', 'sensor'.
    dataset_name : str
        File prefix (e.g., 'wc' for 'wc_seg_0000.h5').
    output_mode : str
        How to format sensor data for the model:
        - 'response': PMT point cloud with total PE/T features
        - 'labels': sparse per-particle entries with instance/semantic labels
        - 'separate': keep raw reader keys (pmt_coord, pmt_pe, pp_* keys)
    include_labels : bool
        Whether sensor reader loads per-particle decomposition.
    pe_threshold : float
        Minimum PE for sparsifying PE_per_particle.
    min_segments : int
        Minimum segments per event (seg reader filter).
    max_len : int
        Cap on dataset length.
    loop : int
        Dataset repetition per epoch.
    """

    def __init__(
        self,
        data_root,
        split='',
        transform=None,
        modalities=('sensor',),
        dataset_name='wc',
        output_mode='response',
        include_labels=True,
        pe_threshold=0.0,
        min_segments=0,
        max_len=-1,
        loop=1,
        test_mode=False,
        test_cfg=None,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.modalities = tuple(modalities)
        self.dataset_name = dataset_name
        self.output_mode = output_mode
        self.max_len = max_len
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        self.transform = Compose(transform)
        if test_mode and test_cfg is not None:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop)
                if self.test_cfg.crop else None)
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [
                Compose(aug) for aug in self.test_cfg.aug_transform]

        # Build readers
        self.seg_reader = None
        self.sensor_reader = None

        if 'seg' in self.modalities:
            seg_root = self._modality_root('seg')
            self.seg_reader = LUCiDSegReader(
                data_root=seg_root, split=split,
                dataset_name=dataset_name, min_segments=min_segments)

        if 'sensor' in self.modalities:
            sensor_root = self._modality_root('sensor')
            self.sensor_reader = LUCiDSensorReader(
                data_root=sensor_root, split=split,
                dataset_name=dataset_name,
                include_labels=include_labels,
                pe_threshold=pe_threshold)

        # Canonical reader and length
        active_readers = [r for r in (self.seg_reader, self.sensor_reader)
                          if r is not None]
        if not active_readers:
            raise ValueError(f"Need 'seg' or 'sensor' in modalities, got {self.modalities}")
        self._canonical_reader = active_readers[0]
        self._n_events = min(len(r) for r in active_readers)

        logger = get_root_logger()
        logger.info(f"LUCiDDataset: {self._n_events} events, "
                    f"modalities={self.modalities}, output_mode={output_mode}")

    def _modality_root(self, modality):
        mod_dir = os.path.join(self.data_root, modality)
        if os.path.isdir(mod_dir):
            return mod_dir
        return self.data_root

    def get_data(self, idx):
        data_dict = {}

        # --- Seg (3D track segments) ---
        if self.seg_reader is not None:
            seg_data = self.seg_reader.read_event(idx)
            if self.sensor_reader is not None and self.output_mode == 'separate':
                # Prefix 3D keys to avoid collision with sensor coord
                for k, v in seg_data.items():
                    data_dict[f'seg3d.{k}'] = v
            else:
                data_dict.update(seg_data)

        # --- Sensor (PMT response + optional per-particle labels) ---
        if self.sensor_reader is not None:
            sensor_data = self.sensor_reader.read_event(idx)

            if self.output_mode == 'response':
                # PMT response — one entry per sensor
                n = len(sensor_data['pmt_pe'])
                if 'pmt_coord' in sensor_data:
                    data_dict['coord'] = sensor_data['pmt_coord']
                else:
                    # No 3D positions — use sensor index as 1D coord
                    data_dict['coord'] = np.arange(n, dtype=np.float32)[:, None]
                data_dict['energy'] = sensor_data['pmt_pe'][:, None]
                data_dict['time'] = sensor_data['pmt_t'][:, None]

            elif self.output_mode == 'labels':
                # Sparse per-particle entries
                if 'pp_sensor_idx' in sensor_data:
                    sidx = sensor_data['pp_sensor_idx']
                    if 'pmt_coord' in sensor_data:
                        data_dict['coord'] = sensor_data['pmt_coord'][sidx]
                    else:
                        data_dict['coord'] = sidx.astype(np.float32)[:, None]
                    data_dict['energy'] = sensor_data['pp_pe'][:, None]
                    data_dict['segment'] = sensor_data['pp_category']
                    data_dict['instance'] = sensor_data['pp_particle_idx']
                    if 'pp_t' in sensor_data:
                        data_dict['time'] = sensor_data['pp_t'][:, None]

            elif self.output_mode == 'separate':
                data_dict.update(sensor_data)

        # Metadata
        data_dict['name'] = self.get_data_name(idx)
        data_dict['split'] = self.split if isinstance(self.split, str) else 'custom'
        return data_dict

    def get_data_name(self, idx):
        reader = self._canonical_reader
        file_idx = int(np.searchsorted(reader.cumulative_lengths, idx, side='right'))
        local = idx - (int(reader.cumulative_lengths[file_idx - 1]) if file_idx > 0 else 0)
        event_num = reader.indices[file_idx][local]
        fname = os.path.basename(reader.h5_files[file_idx])
        return f"{fname}_evt{event_num:03d}"

    def prepare_train_data(self, idx):
        return self.transform(self.get_data(idx % len(self)))

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx % len(self))
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        result_dict = dict(name=data_dict.pop("name"))
        if "segment" in data_dict:
            result_dict["segment"] = data_dict.pop("segment")
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part
        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __getitem__(self, idx):
        real_idx = idx % len(self)
        if self.test_mode:
            return self.prepare_test_data(real_idx)
        return self.prepare_train_data(real_idx)

    def __len__(self):
        n = self._n_events
        if self.max_len > 0:
            n = min(n, self.max_len)
        return n * self.loop

    def __del__(self):
        for reader in (self.seg_reader, self.sensor_reader):
            if reader is not None:
                reader.close()
