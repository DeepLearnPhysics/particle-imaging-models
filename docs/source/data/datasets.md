# Supported detectors

Each reader is a registered dataset class; you select one in a config with `type` under `data.train`, `data.val` and `data.test`.

| Dataset `type` | Detector | Source |
|---|---|---|
| `PILArNetH5Dataset` | LArTPC, PILArNet-M simulation | HDF5 shards |
| `TPCpp10MDataset` | sPHENIX TPC, TPCpp-10M simulation | the release's `.npz` files |
| `JAXTPCDataset` | LArTPC, JAXTPC simulation | HDF5: 3D deposits, 2D wire signals, correspondence, labels |
| `LUCiDDataset` | water Cherenkov, LUCiD simulation | HDF5: PMT response and 3D track segments |
| `DefaultDataset` | any | one directory of `.npy` files per event |
| `ConcatDataset` | — | several datasets as one |

## Download

`pimm datasets list` prints the datasets you can download, and `pimm datasets download <name>` fetches one from the Hugging Face Hub; `--split` restricts the download to some splits and `--revision` picks a revision. The command asks how many download workers to use. Files go to `--output-dir`, else to `$PIMM_DATA_ROOT/<name>`, else to `~/.cache/pimm/datasets/<name>`; when you choose a folder of your own, the command offers to add `PIMM_DATA_ROOT` to your shell configuration.

| Name | Source | License | Size | Download | After the download |
|---|---|---|---|---|---|
| `pilarnet-m` | [DeepLearnPhysics/PILArNet-M](https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M) | Apache 2.0 | 168 GB, test split 7 GB | `pimm datasets download pilarnet-m --revision v3_extra` | — |
| `tpcpp-10m` | [DeepLearnPhysics/tpcpp-10m](https://huggingface.co/datasets/DeepLearnPhysics/tpcpp-10m), mirror of [Zenodo 16970029](https://doi.org/10.5281/zenodo.16970029) | CC BY 4.0 | 119 GB, labeled splits 1 GB | `pimm datasets download tpcpp-10m` | for `unlabeled`, write the shards uncompressed (139 GB more); the command offers to run it |
| `wand-sk-like` | [DeepLearnPhysics/wand-sk-like](https://huggingface.co/datasets/DeepLearnPhysics/wand-sk-like) | CC BY 4.0 | 328 GB, train 306 GB | `pimm datasets download wand-sk-like` | — |
| `next-aiss` | [DeepLearnPhysics/next-aiss](https://huggingface.co/datasets/DeepLearnPhysics/next-aiss), mirror of [Zenodo 18927784](https://doi.org/10.5281/zenodo.18927784) | CC BY 4.0 | 16 GB in 20 tar parts | `pimm datasets download next-aiss` | extract the parts; the command offers to do it and keeps the `.tar` files |

A dataset class looks for its files in the config's `data_root` (the recipes use `data/<name>`), then in `$PIMM_DATA_ROOT/<name>`, then in `~/.cache/pimm/datasets/<name>`; PILArNet-M revisions live in `<name>/<revision>`. To add a dataset, you add an entry to `CATALOG` in `pimm/datasets/catalog.py` and call `resolve_data_root` once in its class.

## PILArNet-M

The supported revisions are `v3` (the default) and `v3_extra`; `v3_extra` also has momentum vectors, particle lineage and interaction-vertex tables. `v2` is read as `v3`, with a warning, and `v1` is rejected. `DeepLearnPhysics/PILArNet-M-mini` is a 120-event sample in the same layout.

| Option | Default | Effect |
|---|---|---|
| `split` | `"train"` | split directory, or a list of them |
| `min_points` | `1024` | events with fewer points are skipped |
| `max_len` | `-1` | cap on the number of events |
| `energy_threshold` | `0.13` | points at or below this energy are dropped |
| `remove_low_energy_scatters` | `False` | the low-energy-deposit cluster is dropped |
| `overlay_n_events` | `1` | number of events overlaid into one point cloud |

## TPCpp-10M

`TPCpp10MDataset` reads TPCpp-10M, simulated proton-proton collisions in an sPHENIX-like Time Projection Chamber ([Li et al., 2025](https://doi.org/10.5281/zenodo.16970029), CC BY 4.0), from the files of the original release. The labeled splits are read from their `.npz` files as they are. The unlabeled shards are compressed, so before pretraining you write their points uncompressed once, which lets the dataset memory-map them (`pimm datasets download` offers to run this):

```bash
uv run python -m pimm.datasets.preprocessing.tpcpp10m data/tpcpp-10m --workers 8   # about 139 GB more
```

This writes `unlabeled/spacepoints_NNN.data.npy` next to each `spacepoints_NNN.npz`; an interrupted run can be restarted. Reading the `unlabeled` split without these files raises an error naming the command.

The splits are the release's: `train`, `validation` and `test` hold 70,000, 13,000 and 7,000 labeled events, and `unlabeled` holds 10,000,000. Items have the keys of `PILArNetH5Dataset` with `revision="v2"`: `segment_pid` holds the particle class (0 other, 1 pion, 2 kaon, 3 proton, 4 electron; 5 for unlabeled events), `instance_particle` the track, and `segment_motif` is 4 for noise points and 1 otherwise. The options are `min_points`, `max_points`, `max_len`, `subset_seed`, `loop` and the event overlay, as in `PILArNetH5Dataset`.

## JAXTPC

`JAXTPCDataset` is a reader for co-indexed HDF5 files from a JAXTPC production: `seg` (3D deposits), `resp` (2D wire signals), `corr` (3D-to-2D correspondence) and `labl` (track-to-label tables). What `coord` is depends on the modalities you load:

| Task | `modalities` | Output |
|---|---|---|
| 3D segmentation | `("seg", "labl")` | `coord` (N, 3), `segment` (N,) |
| 2D segmentation | `("resp", "corr", "labl")` | `coord` (E, 2) from the correspondence, with labels; `resp_coord` also present |
| 2D self-supervised | `("resp",)` | `coord` (M, 2), all wire planes merged, with `plane_id` |
| Everything | `("seg", "resp", "corr", "labl")` | 3D `coord` plus `resp_coord` and `corr_coord` |

Without `corr`, no labels are produced from `resp` and `labl`: wire signals are mapped to tracks only through the correspondence. Other options are `label_key` (`"particle"`, `"cluster"` or `"interaction"`), `volume` and `min_deposits`. In the committed recipe `detector/semseg/semseg-pt-v3m2-jaxtpc-5cls`, set the data root with `JAXTPC_DATA_ROOT`.

With 2D coordinates, use `GridSample`, `ToTensor`, `Copy`, `Collect`, `RandomDropout`, `ShufflePoint`, `RandomJitter`, `RandomScale`, `RandomFlip` and `PositiveShift`. `RandomRotate`, `NormalizeCoord` and `SphereCrop` require three dimensions.

## LUCiD

`LUCiDDataset` is a reader for water Cherenkov simulation output: a root with `sensor/` and `seg/` subdirectories of files named `{dataset_name}_sensor_NNNN.h5` and `{dataset_name}_seg_NNNN.h5` (or `sensor_events_NNNN.h5` and `segment_events_NNNN.h5`).

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

For `DefaultDataset` and for writing a reader, see [Bring your own data](your-data.md).
