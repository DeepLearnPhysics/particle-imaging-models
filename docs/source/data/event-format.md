# Event format

Every dataset returns an event as a flat dict of NumPy arrays with one row per point, and every model reads a packed batch of such events.

## One event

Point-aligned fields have one row per point; scalar quantities keep a trailing axis of length 1. A PILArNet-M event from `PILArNetH5Dataset` contains:

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

`-1` marks points to ignore. PILArNet-M coordinates lie in a cube of side 768; recipes rescale them, as described in [Transforms](transforms.md). Other readers emit the same core keys (`coord`, `energy`, `segment`, `instance`) plus their own; see [Supported detectors](datasets.md).

## See one

<div class="event-viewer" data-figure="panda-event"></div>

PILArNet-M-mini test event 0 after the Panda transforms: 1,175 points, colored by their true labels. Drag to rotate, scroll to zoom, and switch between semantic and particle classes.

## A batch

After the transforms, `Collect` keeps the keys you name and joins `feat_keys` into `feat`. `collate_fn` concatenates the events and builds `offset`:

| Key | Shape | Content |
|---|---|---|
| `coord` | (N_total, 3) | transformed positions |
| `grid_coord` | (N_total, 3) int | grid cells, when the pipeline grid-samples |
| `feat` | (N_total, C) | features in `feat_keys` order |
| `segment` | (N_total,) | training target, when present |
| `offset` | (B,) | cumulative end index of each event; `offset[-1] == N_total` |

Inside a model, `Point(batch)` wraps the dict and derives the per-point event index from `offset`.
