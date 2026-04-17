"""
Verification script for JAXTPCDataset — all modality combinations.

Run: /usr/bin/python3 tests/test_jaxtpc_dataset.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from pimm.datasets.jaxtpc_dataset import JAXTPCDataset
from pimm.datasets.utils import collate_fn
from pimm.datasets.transform import Compose

DATA_ROOT = os.environ.get(
    'JAXTPC_DATA_ROOT',
    '/home/oalterka/desktop_linux/JAXTPC/dataset_1')

MAX_LEN = 4
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
    defaults = dict(data_root=DATA_ROOT, split='', dataset_name='sim', max_len=MAX_LEN)
    defaults.update(kwargs)
    return JAXTPCDataset(**defaults)


def test_seg_only():
    """seg only — 3D point cloud, no labels."""
    print("\n=== seg only ===")
    ds = make_ds(modalities=('seg',))
    d = ds.get_data(0)
    check(d['coord'].shape[1] == 3, f"coord 3D: {d['coord'].shape}")
    check(d['energy'].shape[1] == 1, f"energy: {d['energy'].shape}")
    check('segment' not in d, "no segment without labl")


def test_seg_labl():
    """seg + labl — 3D with labels from lookup."""
    print("\n=== seg + labl ===")
    ds = make_ds(modalities=('seg', 'labl'), label_key='particle')
    d = ds.get_data(0)
    check(d['coord'].shape[1] == 3, f"coord 3D: {d['coord'].shape}")
    check('segment' in d, "segment present")
    check(d['segment'].shape[0] == d['coord'].shape[0], "segment matches coord")


def test_resp_only():
    """resp only — all planes merged into 2D point cloud, no labels."""
    print("\n=== resp only ===")
    ds = make_ds(modalities=('resp',))
    d = ds.get_data(0)
    check(d['coord'].shape[1] == 2, f"coord 2D: {d['coord'].shape}")
    check('plane_id' in d, "plane_id present")
    check('segment' not in d, "no segment")
    n_planes = len(np.unique(d['plane_id']))
    check(n_planes > 1, f"multiple planes: {n_planes}")


def test_resp_corr_labl():
    """resp + corr + labl — 2D labeled point cloud from corr chain."""
    print("\n=== resp + corr + labl ===")
    ds = make_ds(modalities=('resp', 'corr', 'labl'), label_key='particle')
    d = ds.get_data(0)
    check(d['coord'].shape[1] == 2, f"coord 2D: {d['coord'].shape}")
    check('segment' in d, "segment present")
    check('instance' in d, "instance present")
    check('plane_id' in d, "plane_id present")
    # Resp signal also available as namespaced keys
    resp_keys = [k for k in d if k.startswith('plane.')]
    check(len(resp_keys) > 0, f"resp namespaced keys: {len(resp_keys)}")
    # Overlapping instances
    _, counts = np.unique(d['coord'], axis=0, return_counts=True)
    check(np.sum(counts > 1) > 0, f"overlapping pixels: {np.sum(counts > 1)}")


def test_seg_resp_corr_labl():
    """All modalities — seg owns coord, resp/corr as separate point clouds."""
    print("\n=== seg + resp + corr + labl ===")
    ds = make_ds(modalities=('seg', 'resp', 'corr', 'labl'), label_key='particle')
    d = ds.get_data(0)
    check(d['coord'].shape[1] == 3, f"3D coord: {d['coord'].shape}")
    check('segment' in d, "3D segment from labl")
    # Resp as separate point cloud
    check('resp_coord' in d, f"resp_coord present: {d.get('resp_coord', 'MISSING')}")
    check(d['resp_coord'].shape[1] == 2, f"resp_coord 2D: {d['resp_coord'].shape}")
    # Corr as separate point cloud
    check('corr_coord' in d, "corr_coord present")
    check('corr_segment' in d, "corr_segment present")
    check('corr_instance' in d, "corr_instance present")
    # Raw plane keys also available
    plane_keys = [k for k in d if k.startswith('plane.')]
    check(len(plane_keys) > 0, f"raw plane keys: {len(plane_keys)}")


def test_resp_corr():
    """resp + corr (no labl) — resp merged, corr namespaced."""
    print("\n=== resp + corr (no labl) ===")
    ds = make_ds(modalities=('resp', 'corr'))
    d = ds.get_data(0)
    check(d['coord'].shape[1] == 2, f"coord 2D from resp: {d['coord'].shape}")
    check('segment' not in d, "no segment without labl")
    corr_keys = [k for k in d if k.startswith('corr.')]
    check(len(corr_keys) > 0, f"corr namespaced: {len(corr_keys)}")


def test_volume_filter():
    """volume=0 — only volume 0 data (fewer points than all volumes)."""
    print("\n=== volume filter ===")
    ds_all = make_ds(modalities=('resp',))
    ds_v0 = make_ds(modalities=('resp',), volume=0)
    d_all = ds_all.get_data(0)
    d_v0 = ds_v0.get_data(0)
    check(d_v0['coord'].shape[0] < d_all['coord'].shape[0],
          f"volume_0 ({d_v0['coord'].shape[0]}) < all ({d_all['coord'].shape[0]})")


def test_different_label_keys():
    """All label_key options."""
    print("\n=== different label_keys ===")
    for lk in ['particle', 'cluster', 'interaction']:
        ds = make_ds(modalities=('seg', 'labl'), label_key=lk)
        d = ds.get_data(0)
        n = len(np.unique(d['segment']))
        check(n > 1, f"label_key={lk}: {n} classes")


def test_pipeline_3d():
    """Full 3D pipeline: transforms → collate."""
    print("\n=== 3D pipeline ===")
    transform = [
        dict(type='NormalizeCoord', center=[0, 0, 0], scale=4000.0),
        dict(type='GridSample', grid_size=0.001, hash_type='fnv',
             mode='train', return_grid_coord=True),
        dict(type='ToTensor'),
        dict(type='Collect', keys=('coord', 'grid_coord', 'segment'),
             feat_keys=('coord', 'energy')),
    ]
    ds = make_ds(modalities=('seg', 'labl'), label_key='particle',
                 min_deposits=1024, transform=transform)
    batch = collate_fn([ds[0], ds[1]])
    check(batch['coord'].shape[1] == 3, f"3D: {batch['coord'].shape}")
    check(len(batch['offset']) == 2, "offset correct")


def test_pipeline_2d():
    """Full 2D pipeline: transforms → collate → DataLoader."""
    print("\n=== 2D pipeline ===")
    transform = [
        dict(type='GridSample', grid_size=1.0, hash_type='fnv',
             mode='train', return_grid_coord=True),
        dict(type='ToTensor'),
        dict(type='Collect', keys=('coord', 'grid_coord', 'segment', 'instance'),
             feat_keys=('coord', 'energy')),
    ]
    ds = make_ds(modalities=('resp', 'corr', 'labl'),
                 label_key='particle', transform=transform)
    batch = collate_fn([ds[0], ds[1]])
    check(batch['coord'].shape[1] == 2, f"2D: {batch['coord'].shape}")
    check(len(batch['offset']) == 2, "offset correct")

    # DataLoader
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=False, num_workers=2,
        collate_fn=collate_fn, persistent_workers=False)
    for i, b in enumerate(loader):
        if i >= 1:
            break
        check(b['coord'].shape[1] == 2, f"DataLoader: {b['coord'].shape}")


def test_toy_model():
    """Toy model forward+backward."""
    print("\n=== toy model ===")
    transform = [
        dict(type='GridSample', grid_size=1.0, hash_type='fnv',
             mode='train', return_grid_coord=True),
        dict(type='ToTensor'),
        dict(type='Collect', keys=('coord', 'grid_coord', 'segment'),
             feat_keys=('coord', 'energy')),
    ]
    ds = make_ds(modalities=('resp', 'corr', 'labl'),
                 label_key='particle', transform=transform)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = collate_fn([ds[0], ds[1]])
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    model = nn.Linear(batch['feat'].shape[1], 5).to(device)
    logits = model(batch['feat'])
    loss = nn.CrossEntropyLoss(ignore_index=-1)(logits, batch['segment'].long())
    loss.backward()
    check(logits.shape[1] == 5, f"logits: {logits.shape}")
    check(model.weight.grad is not None, "gradients computed")


if __name__ == '__main__':
    print(f"Testing JAXTPCDataset\nData root: {DATA_ROOT}")

    test_seg_only()
    test_seg_labl()
    test_resp_only()
    test_resp_corr_labl()
    test_seg_resp_corr_labl()
    test_resp_corr()
    test_volume_filter()
    test_different_label_keys()
    test_pipeline_3d()
    test_pipeline_2d()
    test_toy_model()

    print(f"\n{'='*50}")
    print(f"PASSED: {PASSED}, FAILED: {FAILED}")
    if FAILED > 0:
        sys.exit(1)
    print("ALL TESTS PASSED")
