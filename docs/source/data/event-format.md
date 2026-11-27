# Event format

Each event from a dataset is a flat dict of NumPy arrays with one row per point, and the input of every model is a packed batch of such events.

## One event

Point-aligned fields have one row per point; scalar quantities have a trailing axis of length 1. A PILArNet-M event from `PILArNetH5Dataset` contains these fields:

| Key | Shape | Content |
|---|---|---|
| `coord` | (N, 3) float32 | point position |
| `energy` | (N, 1) float32 | deposited energy |
| `segment_motif` | (N, 1) | semantic class |
| `segment_pid` | (N, 1) | particle class |
| `instance_particle` | (N, 1) | particle id within the event |
| `instance_interaction` | (N, 1) | interaction id within the event |
| `segment_interaction` | (N, 1) | background flag |
| `momentum` | (N, 1) | momentum of the point's particle (GeV) |
| `vertex` | (N, 3) | interaction vertex of the point's particle |
| `is_primary` | (N, 1) | primary-particle flag |
| `name`, `split`, `revision` | strings | bookkeeping |

| Code | `segment_motif` | `segment_pid` |
|---:|---|---|
| 0 | shower | photon |
| 1 | track | electron |
| 2 | Michel | muon |
| 3 | delta | pion |
| 4 | low-energy deposit | proton |
| 5 | — | none (low-energy deposit) |

Points to ignore have the label `-1`. PILArNet-M coordinates are in a cube of side 768; they are rescaled in recipes, as described in [Transforms](transforms.md). Events from other readers have the same core keys (`coord`, `energy`, `segment`, `instance`) plus reader-specific keys; see [Supported detectors](datasets.md).

## See one

<div class="event-viewer" data-figure="panda-event"></div>

PILArNet-M-mini test event 0 after the Panda transforms: 1,175 points, colored by their true labels. Drag to rotate, scroll to zoom, and switch between semantic and particle classes.

## A batch

In `Collect`, you name the keys to keep and the `feat_keys` to join into `feat`. Pass the transformed events to `collate_fn` to concatenate them and build `offset`:

| Key | Shape | Content |
|---|---|---|
| `coord` | (N_total, 3) | transformed positions |
| `grid_coord` | (N_total, 3) int | grid cells, when grid sampling is in the pipeline |
| `feat` | (N_total, C) | features in `feat_keys` order |
| `segment` | (N_total,) | training target, when present |
| `offset` | (B,) | cumulative end index of each event; `offset[-1] == N_total` |

Inside a model, the dict is wrapped in `Point(batch)`, and the per-point event index is derived from `offset`.
