# Supported detectors

Each reader is a registered dataset class; a config selects one with `type` under `data.train`, `data.val` and `data.test`.

| Dataset `type` | Detector | Source |
|---|---|---|
| `PILArNetH5Dataset` | LArTPC, PILArNet-M simulation | HDF5 shards |
| `PILArNetParquetDataset`, `PILArNetParquetIterableDataset` | LArTPC, PILArNet-M | Parquet, local or streamed from Hugging Face |
| `JAXTPCDataset` | LArTPC, JAXTPC simulation | HDF5: 3D deposits, 2D wire signals, correspondence, labels |
| `LUCiDDataset` | water Cherenkov, LUCiD simulation | HDF5: PMT response and 3D track segments |
| `LUCiDEventSSLDataset` | water Cherenkov | LUCiD events through the `pimm-data` readers |
| `DefaultDataset` | any | one directory of `.npy` files per event |
| `ConcatDataset` | — | several datasets as one |

## PILArNet-M

Download it with the bundled script:

```bash
uv run python scripts/pilarnet/download.py --revision v3                # all splits, about 168 GB
uv run python scripts/pilarnet/download.py --revision v3 --split test   # test split, about 7 GB
```

Files go to `~/.cache/pimm/pilarnet/v3` unless you pass `--output-dir`, and the script offers to add `PILARNET_DATA_ROOT_V3` to your shell configuration. The reader looks for data in the config's `data_root`, then in `$PILARNET_DATA_ROOT_<REVISION>`, then in `~/.cache/pimm/pilarnet/<revision>`.

The reader accepts revisions `v3` (the default) and `v3_extra`, which adds momentum vectors, particle lineage and interaction-vertex tables. It reads `v2` as `v3`, with a warning, and rejects `v1`. `DeepLearnPhysics/PILArNet-M-mini` is a 120-event sample in the same layout.

| Option | Default | Effect |
|---|---|---|
| `split` | `"train"` | split directory, or a list of them |
| `min_points` | `1024` | skip events with fewer points |
| `max_len` | `-1` | cap the number of events |
| `energy_threshold` | `0.13` | drop points at or below this energy |
| `remove_low_energy_scatters` | `False` | drop the low-energy-deposit cluster |
| `overlay_n_events` | `1` | overlay several events into one point cloud |

## JAXTPC

`JAXTPCDataset` reads co-indexed HDF5 files from a JAXTPC production: `seg` (3D deposits), `resp` (2D wire signals), `corr` (3D-to-2D correspondence) and `labl` (track-to-label tables). The modalities you load decide what `coord` is:

| Task | `modalities` | Output |
|---|---|---|
| 3D segmentation | `("seg", "labl")` | `coord` (N, 3), `segment` (N,) |
| 2D segmentation | `("resp", "corr", "labl")` | `coord` (E, 2) from the correspondence, with labels; `resp_coord` also present |
| 2D self-supervised | `("resp",)` | `coord` (M, 2), all wire planes merged, with `plane_id` |
| Everything | `("seg", "resp", "corr", "labl")` | 3D `coord` plus `resp_coord` and `corr_coord` |

`resp` and `labl` without `corr` give no labels: wire signals map to tracks only through the correspondence. Other options are `label_key` (`"particle"`, `"cluster"` or `"interaction"`), `volume` and `min_deposits`. The committed recipe `detector/semseg/semseg-pt-v3m2-jaxtpc-5cls` reads its root from `JAXTPC_DATA_ROOT`.

With 2D coordinates, use `GridSample`, `ToTensor`, `Copy`, `Collect`, `RandomDropout`, `ShufflePoint`, `RandomJitter`, `RandomScale`, `RandomFlip` and `PositiveShift`. `RandomRotate`, `NormalizeCoord` and `SphereCrop` assume three dimensions.

## LUCiD

`LUCiDDataset` reads water Cherenkov simulation output from a root with `sensor/` and `seg/` subdirectories, named `{dataset_name}_sensor_NNNN.h5` and `{dataset_name}_seg_NNNN.h5` (or `sensor_events_NNNN.h5` and `segment_events_NNNN.h5`).

| Task | `modalities` | `output_mode` | Output |
|---|---|---|---|
| Event classification | `("sensor",)` | `"response"` | one row per PMT: `coord`, `energy` (photoelectrons), `time` |
| Per-sensor instance separation | `("sensor",)` | `"labels"` | per-particle entries: `coord`, `energy`, `segment`, `instance` |
| 3D track reconstruction | `("seg",)` | any | `coord` (N, 3), `energy`, `time`, `track_ids`, `pdg`, `parent_ids` |
| Joint 3D and sensor | `("seg", "sensor")` | `"separate"` | the readers' raw keys |

PMT `coord` is a 3D position when the sensor reader has PMT positions (`pmt_positions`, `pmt_positions_file`, or the file's `config/pmt_positions`); otherwise it is a sensor index of shape (N, 1).

```python
data = dict(train=dict(
    type="LUCiDDataset",
    data_root="/path/to/dataset_wc",
    dataset_name="wc",
    modalities=("sensor",),
    output_mode="response",
    transform=[...],
))
```

## Your own format

[Bring your own data](your-data.md) covers `DefaultDataset` and writing a reader.
