"""
Verification script for LUCiDDataset — all output modes.

Run: /usr/bin/python3 tests/test_lucid_dataset.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from pimm.datasets.lucid_dataset import LUCiDDataset
from pimm.datasets.utils import collate_fn
from pimm.datasets.transform import Compose

DATA_ROOT = '/home/oalterka/desktop_linux/JAXTPC/dataset_wc'
PASSED = 0
FAILED = 0


def check(condition, msg):
    global PASSED, FAILED
    if condition:
        print(f"  OK: {msg}")
        PASSED += 1
    else:
        print(f"  FAIL: {msg}")
        FAILED += 1


def make_ds(**kwargs):
    defaults = dict(data_root=DATA_ROOT, split='', dataset_name='wc', max_len=4)
    defaults.update(kwargs)
    return LUCiDDataset(**defaults)


def test_sensor_response():
    """Sensor response — one entry per sensor."""
    print("\n=== Sensor response ===")
    ds = make_ds(modalities=('sensor',), output_mode='response', include_labels=False)
    d = ds.get_data(0)
    check('coord' in d, "coord present")
    check('energy' in d, f"energy: {d['energy'].shape}")
    check('time' in d, "time present")
    check('segment' not in d, "no segment (SSL)")
    n = d['coord'].shape[0]
    check(n > 1000, f"many sensors: {n}")


def test_sensor_labels():
    """Sensor with labels — sparse per-particle entries."""
    print("\n=== Sensor labels ===")
    ds = make_ds(modalities=('sensor',), output_mode='labels', include_labels=True)
    d = ds.get_data(0)
    check('coord' in d, "coord present")
    check('segment' in d, "segment present")
    check('instance' in d, "instance present")
    n = d['coord'].shape[0]
    check(n > 0, f"sparse entries: {n}")
    n_inst = len(np.unique(d['instance']))
    check(n_inst > 1, f"multiple instances: {n_inst}")
    cats = np.unique(d['segment'])
    check(len(cats) >= 1, f"categories: {cats}")


def test_sensor_separate():
    """Sensor separate — raw reader keys."""
    print("\n=== Sensor separate ===")
    ds = make_ds(modalities=('sensor',), output_mode='separate')
    d = ds.get_data(0)
    check('pmt_pe' in d, "pmt_pe present")
    check('pmt_t' in d, "pmt_t present")
    check('pp_sensor_idx' in d, "pp_sensor_idx present")
    check('pp_category' in d, "pp_category present")
    check('coord' not in d, "no top-level coord")


def test_seg_only():
    """3D track segments."""
    print("\n=== Seg only ===")
    ds = make_ds(modalities=('seg',))
    d = ds.get_data(0)
    check(d['coord'].shape[1] == 3, f"coord 3D: {d['coord'].shape}")
    check(d['energy'].shape[1] == 1, f"energy: {d['energy'].shape}")
    check('track_ids' in d, "track_ids present")
    check('pdg' in d, "pdg present")


def test_mixed_separate():
    """Seg + sensor separate."""
    print("\n=== Mixed separate ===")
    ds = make_ds(modalities=('seg', 'sensor'), output_mode='separate')
    d = ds.get_data(0)
    seg_keys = [k for k in d if k.startswith('seg3d.')]
    check(len(seg_keys) > 0, f"seg3d keys: {len(seg_keys)}")
    check('pmt_pe' in d, "pmt_pe present")


def test_pipeline_response():
    """Pipeline: response → transforms → collate."""
    print("\n=== Pipeline response ===")
    transform = [
        dict(type='ToTensor'),
        dict(type='Collect', keys=('coord',), feat_keys=('coord', 'energy', 'time')),
    ]
    ds = make_ds(modalities=('sensor',), output_mode='response',
                 include_labels=False, transform=transform)
    s0 = ds[0]
    coord_dim = s0['coord'].shape[1]
    feat_dim = s0['feat'].shape[1]
    check(feat_dim == coord_dim + 2, f"feat={s0['feat'].shape} (coord_dim+2)")
    check('offset' in s0, "offset present")

    batch = collate_fn([ds[0], ds[1]])
    n0 = ds.get_data(0)['coord'].shape[0]
    check(batch['coord'].shape[0] > n0, f"batch: {batch['coord'].shape}")
    check(len(batch['offset']) == 2, f"offset: {batch['offset'].tolist()}")


def test_pipeline_labels():
    """Pipeline: labels → transforms → collate → toy model."""
    print("\n=== Pipeline labels ===")
    transform = [
        dict(type='ToTensor'),
        dict(type='Collect', keys=('coord', 'segment', 'instance'),
             feat_keys=('coord', 'energy')),
    ]
    ds = make_ds(modalities=('sensor',), output_mode='labels',
                 include_labels=True, transform=transform)

    batch = collate_fn([ds[0], ds[1]])
    check('segment' in batch, "segment in batch")
    check(len(batch['offset']) == 2, "offset correct")

    device = torch.device('cpu')
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    n_classes = len(torch.unique(batch['segment']))
    model = nn.Linear(batch['feat'].shape[1], max(4, n_classes)).to(device)
    logits = model(batch['feat'])
    loss = nn.CrossEntropyLoss(ignore_index=-1)(logits, batch['segment'].long())
    loss.backward()
    check(model.weight.grad is not None, "gradients computed")


def test_pipeline_seg():
    """Pipeline: 3D segments → transforms → collate."""
    print("\n=== Pipeline seg ===")
    transform = [
        dict(type='ToTensor'),
        dict(type='Collect', keys=('coord',), feat_keys=('coord', 'energy')),
    ]
    ds = make_ds(modalities=('seg',), transform=transform)
    batch = collate_fn([ds[0], ds[1]])
    check(batch['coord'].shape[1] == 3, f"3D: {batch['coord'].shape}")
    check(len(batch['offset']) == 2, "offset correct")


def test_dataloader():
    """DataLoader with workers."""
    print("\n=== DataLoader ===")
    transform = [
        dict(type='ToTensor'),
        dict(type='Collect', keys=('coord',), feat_keys=('coord', 'energy')),
    ]
    ds = make_ds(modalities=('sensor',), output_mode='response',
                 include_labels=False, transform=transform)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=False, num_workers=2,
        collate_fn=collate_fn, persistent_workers=False)
    for i, batch in enumerate(loader):
        if i >= 1:
            break
        check(batch['coord'].shape[0] > 0, f"DataLoader batch: {batch['coord'].shape}")


if __name__ == '__main__':
    print(f"Testing LUCiDDataset\nData root: {DATA_ROOT}")

    test_sensor_response()
    test_sensor_labels()
    test_sensor_separate()
    test_seg_only()
    test_mixed_separate()
    test_pipeline_response()
    test_pipeline_labels()
    test_pipeline_seg()
    test_dataloader()

    print(f"\n{'='*50}")
    print(f"PASSED: {PASSED}, FAILED: {FAILED}")
    if FAILED > 0:
        sys.exit(1)
    print("ALL TESTS PASSED")
