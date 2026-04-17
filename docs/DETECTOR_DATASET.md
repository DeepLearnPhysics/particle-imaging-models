# Detector Datasets

pimm supports multiple detector types through dedicated dataset classes. Each loads from co-indexed HDF5 files and produces flat dicts that flow through pimm's standard pipeline (transforms → Collect → collate → Point → model).

## JAXTPCDataset

For Liquid Argon TPC detectors (JAXTPC production output).

### Data Layout
```
dataset_root/
├── seg/   sim_seg_0000.h5   — 3D truth deposits
├── resp/  sim_resp_0000.h5  — sparse wire signals per plane
├── corr/  sim_corr_0000.h5  — 3D→2D correspondence
└── labl/  sim_labl_0000.h5  — per-volume track_id→label lookup
```

### How coord is assigned

Who owns `coord`/`energy` depends on which modalities are loaded:

- **seg present** → coord is 3D (N,3) from deposits. Resp/corr available as `resp_coord`, `corr_coord`.
- **seg absent, corr+labl present** → coord is 2D (E,2) from corr entries with labels.
- **seg absent, resp present** → coord is 2D (M,2) from all planes merged.

When both resp and corr are loaded, each gets its own point cloud:
- `resp_coord`, `resp_energy`, `resp_plane_id` — from resp signal
- `corr_coord`, `corr_energy`, `corr_segment`, `corr_instance`, `corr_plane_id` — from correspondence

Raw per-plane keys (`plane.*`, `corr.*`) are always passed through for per-plane access.

### Task → Config

| Task | `modalities` | What you get |
|------|-------------|-------------|
| 3D segmentation | `('seg', 'labl')` | `coord (N,3)`, `segment (N,)` |
| 3D seg (PDG fallback) | `('seg',)` | `coord (N,3)`, `pdg (N,)` — use PDGToSemantic |
| 2D segmentation | `('resp', 'corr', 'labl')` | `coord (E,2)` from corr + labels. `resp_coord` also available. |
| 2D self-supervised | `('resp',)` | `coord (M,2)` merged planes |
| Resp→corr denoising | `('resp', 'corr')` | `coord (M,2)` from resp. `corr.*` namespaced keys. |
| Everything | `('seg', 'resp', 'corr', 'labl')` | 3D `coord` + `resp_coord` + `corr_coord` + raw keys |

**Note:** `modalities=('resp', 'labl')` without `'corr'` will NOT produce labels — labl provides track_id→label tables but resp pixels can't be mapped to track_ids without corr.

### Config Parameters
```python
data = dict(train=dict(
    type="JAXTPCDataset",
    data_root="/path/to/dataset",
    split="",
    dataset_name="sim",
    modalities=("seg", "labl"),
    volume=None,                # None=all, 0=volume_0 only
    label_key="particle",       # 'particle', 'cluster', 'interaction'
    min_deposits=1024,
    transform=[...],
))
```

### Label Chain
- **3D**: `deposit.track_id → labl[track_id] → label`
- **2D**: `corr.group_id → g2t → track_id → labl[track_id] → label`

### Transforms Safe for 2D
GridSample, ToTensor, Copy, Collect, RandomDropout, ShufflePoint, RandomJitter, RandomScale, RandomFlip, PositiveShift.

**3D-only** (crash on 2D coords): RandomRotate, NormalizeCoord, SphereCrop.

---

## LUCiDDataset

For Water Cherenkov detectors (PMT-based).

### Data Layout

Two HDF5 files per dataset; readers accept both naming conventions:

```
dataset_root/
├── seg/    {dataset_name}_seg_NNNN.h5   or  segment_events_NNNN.h5
└── sensor/ {dataset_name}_sensor_NNNN.h5 or  sensor_events_NNNN.h5
```

Format is flat CSR arrays (events indexed via `*_offsets` datasets),
matching the PhotonSim/LUCiD production output.

### Task → Config

`coord` shape depends on whether PMT 3D positions are provided
(via `pmt_positions` / `pmt_positions_file` on LUCiDSensorReader, or stored
in the file's `config/pmt_positions` dataset). Without positions,
`coord` falls back to `(N, 1)` with sensor indices.

| Task | `modalities` | `output_mode` | Output |
|------|-------------|--------------|--------|
| Event classification | `('sensor',)` | `'response'` | `coord (N_pmt, 3\|1)`, `energy (N_pmt,1)` [PE], `time (N_pmt,1)` [T] |
| Per-sensor instance separation | `('sensor',)` | `'labels'` | `coord (E, 3\|1)`, `energy (E,1)`, `segment (E,)`, `instance (E,)` |
| 3D track reconstruction | `('seg',)` | any | `coord (N_seg,3)`, `energy (N_seg,1)`, `time`, `track_ids`, `pdg`, `parent_ids` |
| Joint 3D + sensor | `('seg', 'sensor')` | `'separate'` | `seg3d.*` + `pmt_*` + `pp_*` keys |

### Config Parameters
```python
data = dict(train=dict(
    type="LUCiDDataset",
    data_root="/path/to/dataset_wc",
    dataset_name="wc",
    modalities=("sensor",),
    output_mode="response",     # 'response', 'labels', 'separate'
    include_labels=True,
    pe_threshold=0.0,           # optional: filter per-particle entries below this PE
    transform=[...],
))
```

---

## Adding a New Detector

Each dataset class is self-contained — no base class. Copy the closest existing
dataset as a template and modify.

1. **Write reader(s)** in `pimm/datasets/readers/`. Readers follow a lightweight
   convention (not a forced ABC): `__init__` discovers files and builds an event
   index; `h5py_worker_init` lazily opens HDF5 handles (DataLoader-fork safe);
   `read_event(idx)` returns a `dict[str, np.ndarray]`; `__len__` returns the
   event count; `close` releases handles. Copy an existing reader to start.
2. **Write a dataset class** in `pimm/datasets/`, inheriting
   `torch.utils.data.Dataset` directly, registered via `@DATASETS.register_module()`.
   Orchestrate readers in `get_data`; define `__init__`, `__len__`, `__getitem__`.
3. **Add imports** in `pimm/datasets/__init__.py` and
   `pimm/datasets/readers/__init__.py`.

No changes needed to transforms, collation, models, or training infrastructure.

## Running Tests
```bash
/usr/bin/python3 tests/test_jaxtpc_dataset.py     # JAXTPC / LArTPC (38 tests)
/usr/bin/python3 tests/test_lucid_dataset.py      # LUCiD / Water Cherenkov (32 tests)
```
