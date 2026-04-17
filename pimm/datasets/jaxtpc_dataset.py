"""
JAXTPCDataset — multimodal dataset for LArTPC detector simulation output.

Loads from co-indexed HDF5 files produced by JAXTPC's production pipeline:
seg (3D deposits), resp (2D wire signals), corr (3D→2D correspondence),
labl (track_id→label lookup tables).

Who owns ``coord``/``energy`` is determined by which modalities are loaded:

- seg present → coord is 3D (N,3) from deposits. Resp/corr stay namespaced.
- seg absent, resp present → all planes merged into coord (M,2) with plane_id.
- seg absent, corr+labl present → corr entries become coord (E,2) with labels.

Example configs::

    # 3D segmentation
    data = dict(train=dict(type="JAXTPCDataset",
        modalities=("seg", "labl"), label_key="particle", ...))

    # 2D segmentation (all planes)
    data = dict(train=dict(type="JAXTPCDataset",
        modalities=("resp", "corr", "labl"), label_key="particle", ...))

    # Mixed 3D + 2D
    data = dict(train=dict(type="JAXTPCDataset",
        modalities=("seg", "resp", "corr", "labl"), ...))
"""

import os
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset

from pimm.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from .readers.jaxtpc_seg_reader import JAXTPCSegReader
from .readers.jaxtpc_resp_reader import JAXTPCRespReader
from .readers.jaxtpc_labl_reader import JAXTPCLablReader
from .readers.jaxtpc_corr_reader import JAXTPCCorrReader


@DATASETS.register_module()
class JAXTPCDataset(Dataset):
    """Multimodal dataset for LArTPC detector simulation output.

    Parameters
    ----------
    data_root : str
        Root directory with seg/, resp/, corr/, labl/ subdirectories.
    split : str
        Split name for file discovery.
    transform : list[dict]
        Transform pipeline config.
    modalities : tuple[str]
        Which to load: 'seg', 'resp', 'labl', 'corr'.
    dataset_name : str
        File prefix (e.g., 'sim' for 'sim_seg_0000.h5').
    volume : int or None
        Load only this volume index. None = all volumes.
    label_key : str
        Which label to use as 'segment': 'particle', 'cluster', 'interaction'.
    min_deposits : int
        Minimum 3D deposits per event (seg reader filter).
    max_len : int
        Cap on dataset length (-1 = unlimited).
    loop : int
        Dataset repetition per epoch.
    include_physics : bool
        Whether seg reader loads dx, theta, phi, charge, photons, etc.
    label_keys : list or None
        Which label datasets to load from labl files.
    test_mode : bool
        Enable test-time transforms.
    test_cfg : object
        Test config (voxelize, crop, post_transform, aug_transform).
    """

    def __init__(
        self,
        data_root,
        split='train',
        transform=None,
        modalities=('seg',),
        dataset_name='sim',
        volume=None,
        label_key='particle',
        min_deposits=0,
        max_len=-1,
        loop=1,
        include_physics=True,
        label_keys=None,
        test_mode=False,
        test_cfg=None,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.modalities = tuple(modalities)
        self.dataset_name = dataset_name
        self.volume = volume
        self.label_key = label_key
        self.min_deposits = min_deposits
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
        self.resp_reader = None
        self.labl_reader = None
        self.corr_reader = None

        # Plane filter: if volume is set, only load that volume's planes
        planes = 'all'
        if volume is not None:
            planes = [f'volume_{volume}_U', f'volume_{volume}_V',
                      f'volume_{volume}_Y']

        if 'seg' in self.modalities:
            self.seg_reader = JAXTPCSegReader(
                data_root=self._modality_root('seg'), split=split,
                dataset_name=dataset_name, min_deposits=min_deposits,
                include_physics=include_physics, volume=volume)

        if 'resp' in self.modalities:
            self.resp_reader = JAXTPCRespReader(
                data_root=self._modality_root('resp'), split=split,
                dataset_name=dataset_name, planes=planes)

        if 'labl' in self.modalities:
            self.labl_reader = JAXTPCLablReader(
                data_root=self._modality_root('labl'), split=split,
                dataset_name=dataset_name, label_keys=label_keys)

        if 'corr' in self.modalities:
            self.corr_reader = JAXTPCCorrReader(
                data_root=self._modality_root('corr'), split=split,
                dataset_name=dataset_name, planes=planes)

        # Canonical reader and length
        active_readers = [r for r in (self.seg_reader, self.resp_reader,
                                       self.labl_reader, self.corr_reader)
                          if r is not None]
        if not active_readers:
            raise ValueError(f"Need at least one modality, got {self.modalities}")
        self._canonical_reader = (self.seg_reader or self.resp_reader
                                  or self.corr_reader or self.labl_reader)
        self._n_events = min(len(r) for r in active_readers)

        logger = get_root_logger()

        # Warn about modality combinations that won't produce labels
        if (self.resp_reader and self.labl_reader
                and not self.corr_reader and not self.seg_reader):
            logger.warning(
                "modalities=('resp','labl') without 'corr': labl provides "
                "track_id→label tables but resp pixels can't be mapped to "
                "track_ids without corr. No 'segment' will be produced. "
                "Add 'corr' for 2D labels or 'seg' for 3D labels.")

        logger.info(
            f"JAXTPCDataset: {self._n_events} events, "
            f"modalities={self.modalities}, "
            f"volume={volume}, split={split}")

    def _modality_root(self, modality):
        """Resolve root directory for a modality."""
        mod_dir = os.path.join(self.data_root, modality)
        if os.path.isdir(mod_dir):
            return mod_dir
        split_dir = os.path.join(self.data_root, self.split)
        if os.path.isdir(split_dir):
            return self.data_root
        return self.data_root

    def get_data(self, idx):
        """Load one event. Who owns coord depends on modalities:

        - seg present: coord = 3D deposits. Resp/corr as namespaced keys.
        - seg absent, corr+labl present: coord = 2D corr entries with labels.
        - seg absent, resp present (no corr): coord = 2D resp merged.
        """
        data_dict = {}

        # --- Seg (3D point cloud) → owns coord if present ---
        if self.seg_reader is not None:
            data_dict.update(self.seg_reader.read_event(idx))

        # --- Labl (track_id → label lookup) ---
        labl_data = {}
        if self.labl_reader is not None:
            labl_data = self.labl_reader.read_event(idx)

        # Apply labels to 3D seg data
        if self.seg_reader is not None and labl_data:
            self._apply_labl_to_3d(data_dict, labl_data)

        # --- Resp (2D wire planes) ---
        resp_data = {}
        if self.resp_reader is not None:
            resp_data = self.resp_reader.read_event(idx)

        # --- Corr (correspondence) ---
        corr_data = {}
        if self.corr_reader is not None:
            corr_data = self.corr_reader.read_event(idx)

        # --- Build point clouds for each spatial modality ---
        # Each gets its own prefixed keys. When only one spatial source
        # exists, its keys are also copied to the standard coord/energy
        # so the default pipeline (GridSample, Collect, etc.) works.

        has_seg = self.seg_reader is not None
        has_resp = bool(resp_data)
        has_corr = bool(corr_data)

        # Resp → resp_coord/resp_energy/resp_plane_id
        if has_resp:
            self._merge_resp_planes(data_dict, resp_data, prefix='resp_')
            # Also keep raw namespaced keys for per-plane access
            data_dict.update(resp_data)

        # Corr+labl → corr_coord/corr_energy/corr_segment/corr_instance
        if has_corr and labl_data:
            self._build_corr_pointcloud(data_dict, corr_data, labl_data, prefix='corr_')
        elif has_corr:
            # corr without labl — keep as namespaced keys only
            data_dict.update(corr_data)

        # --- Set standard coord/energy from the primary spatial source ---
        if has_seg:
            # seg already set coord/energy
            pass
        elif has_corr and labl_data:
            # corr is primary (has labels)
            data_dict['coord'] = data_dict['corr_coord']
            data_dict['energy'] = data_dict['corr_energy']
            data_dict['segment'] = data_dict['corr_segment']
            data_dict['instance'] = data_dict['corr_instance']
            data_dict['plane_id'] = data_dict['corr_plane_id']
        elif has_resp:
            # resp is primary
            data_dict['coord'] = data_dict['resp_coord']
            data_dict['energy'] = data_dict['resp_energy']
            data_dict['plane_id'] = data_dict['resp_plane_id']

        # Pass through labl lookup tables (for downstream use)
        if labl_data:
            for k, v in labl_data.items():
                if k not in data_dict:
                    data_dict[k] = v

        # Metadata
        data_dict['name'] = self.get_data_name(idx)
        data_dict['split'] = self.split if isinstance(self.split, str) else 'custom'
        return data_dict

    def _apply_labl_to_3d(self, data_dict, labl_data):
        """Map 3D deposits' track_ids to labels via labl lookup. Vectorized."""
        track_ids = data_dict.get('track_ids')
        volume_ids = data_dict.get('volume_id')
        if track_ids is None:
            return

        n = len(track_ids)
        labels = np.full(n, -1, dtype=np.int32)

        vol_indices = sorted(set(
            k.split('_')[1] for k in labl_data
            if k.startswith('labl_v') and k.endswith('_track_ids')
        ))

        for vi in vol_indices:
            tids_key = f'labl_{vi}_track_ids'
            label_key = f'labl_{vi}_{self.label_key}'
            if tids_key not in labl_data or label_key not in labl_data:
                continue

            vol_tids = labl_data[tids_key]
            vol_labels = labl_data[label_key]
            vol_num = int(vi[1:])

            if volume_ids is not None:
                vol_mask = volume_ids.ravel() == vol_num
            else:
                vol_mask = np.ones(n, dtype=bool)

            sort_idx = np.argsort(vol_tids)
            sorted_tids = vol_tids[sort_idx]
            sorted_labels = vol_labels[sort_idx]

            deposit_tids = track_ids[vol_mask]
            insert_pos = np.searchsorted(sorted_tids, deposit_tids)
            insert_pos = np.clip(insert_pos, 0, len(sorted_tids) - 1)
            matched = sorted_tids[insert_pos] == deposit_tids
            labels[vol_mask] = np.where(matched, sorted_labels[insert_pos], -1)

        data_dict['segment'] = labels

    def _merge_resp_planes(self, data_dict, resp_data, prefix=''):
        """Merge all planes into {prefix}coord (M,2), {prefix}energy (M,1), {prefix}plane_id (M,1)."""
        planes = sorted(set(
            k.split('.')[1] for k in resp_data if k.endswith('.wire')
        ))

        all_coord, all_energy, all_plane_id = [], [], []
        for i, plane in enumerate(planes):
            wire = resp_data[f'plane.{plane}.wire']
            time = resp_data[f'plane.{plane}.time']
            value = resp_data[f'plane.{plane}.value']
            n = len(wire)
            all_coord.append(np.stack([wire, time], axis=1).astype(np.float32))
            all_energy.append(value[:, None].astype(np.float32))
            all_plane_id.append(np.full((n, 1), i, dtype=np.int32))

        data_dict[f'{prefix}coord'] = np.concatenate(all_coord, axis=0)
        data_dict[f'{prefix}energy'] = np.concatenate(all_energy, axis=0)
        data_dict[f'{prefix}plane_id'] = np.concatenate(all_plane_id, axis=0)

    def _build_corr_pointcloud(self, data_dict, corr_data, labl_data, prefix=''):
        """Build 2D labeled point cloud from corr + labl.

        Each corr entry is a point: coord=(wire,time), feature=charge,
        instance=group_id, segment from g2t+labl chain.
        Overlapping instances at the same pixel are separate points.
        """
        planes = sorted(set(
            k.split('.')[1] for k in corr_data if k.endswith('.wire')
        ))

        all_coord, all_charge, all_gid, all_segment, all_plane_id = [], [], [], [], []

        for pi, plane in enumerate(planes):
            wire_key = f'corr.{plane}.wire'
            if wire_key not in corr_data:
                continue

            wire = corr_data[f'corr.{plane}.wire']
            time = corr_data[f'corr.{plane}.time']
            gid = corr_data[f'corr.{plane}.group_id']
            charge = corr_data[f'corr.{plane}.charge']
            n = len(wire)

            all_coord.append(np.stack([wire, time], axis=1).astype(np.float32))
            all_charge.append(charge[:, None].astype(np.float32))
            all_gid.append(gid.astype(np.int32))
            all_plane_id.append(np.full((n, 1), pi, dtype=np.int32))

            # group_id → g2t → track_id → labl → label
            vol_idx = plane.split('_')[1]  # 'volume_0_U' → '0'
            g2t = corr_data.get(f'g2t_v{vol_idx}')

            labels = np.full(n, -1, dtype=np.int32)
            if g2t is not None:
                valid_gid = (gid >= 0) & (gid < len(g2t))
                track_ids = np.where(valid_gid, g2t[gid], -1)

                tids_key = f'labl_v{vol_idx}_track_ids'
                lbl_key = f'labl_v{vol_idx}_{self.label_key}'
                if tids_key in labl_data and lbl_key in labl_data:
                    labl_tids = labl_data[tids_key]
                    labl_vals = labl_data[lbl_key]
                    sort_idx = np.argsort(labl_tids)
                    sorted_tids = labl_tids[sort_idx]
                    sorted_vals = labl_vals[sort_idx]
                    insert_pos = np.searchsorted(sorted_tids, track_ids)
                    insert_pos = np.clip(insert_pos, 0, len(sorted_tids) - 1)
                    matched = sorted_tids[insert_pos] == track_ids
                    labels[matched] = sorted_vals[insert_pos[matched]]

            all_segment.append(labels)

        if not all_coord:
            return

        data_dict[f'{prefix}coord'] = np.concatenate(all_coord, axis=0)
        data_dict[f'{prefix}energy'] = np.concatenate(all_charge, axis=0)
        data_dict[f'{prefix}instance'] = np.concatenate(all_gid, axis=0)
        data_dict[f'{prefix}segment'] = np.concatenate(all_segment, axis=0)
        data_dict[f'{prefix}plane_id'] = np.concatenate(all_plane_id, axis=0)

    def get_data_name(self, idx):
        reader = self._canonical_reader
        file_idx = int(np.searchsorted(reader.cumulative_lengths, idx, side='right'))
        local = idx - (int(reader.cumulative_lengths[file_idx - 1])
                       if file_idx > 0 else 0)
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
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")
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
        for attr in ('seg_reader', 'resp_reader', 'labl_reader', 'corr_reader'):
            reader = getattr(self, attr, None)
            if reader is not None:
                reader.close()
