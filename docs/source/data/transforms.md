# Transforms

A transform takes an event dict and returns it, changed. A config lists transforms in order, and `Compose` runs them on every event.

## The Panda pipeline, step by step

| Step | Event after the step |
|---|---|
| raw event | `coord` (N, 3) in detector units, `energy` (N, 1) |
| `NormalizeCoord(center=[384, 384, 384], scale=768·√3/2)` | `coord` inside the unit ball |
| `LogTransform(min_val=0.01, max_val=20.0)` | `energy` on [-1, 1], log-scaled |
| `GridSample(grid_size=0.001, return_grid_coord=True)` | one point per occupied cell, M ≤ N points; integer `grid_coord` added |
| `Copy(keys_dict={"segment_motif": "segment"})` | training target under the name models expect |
| `ToTensor()` | every array is a torch tensor |
| `Collect(keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "energy"))` | only the listed keys, plus `feat` (M, 4) |

`Copy` appears in training recipes only; inference needs no target.

## Common transforms

| Transform | What it does |
|---|---|
| `NormalizeCoord(center, scale)` | `coord = (coord - center) / scale`; uses the centroid and the largest radius when omitted |
| `LogTransform(min_val, max_val, keys=("energy",))` | log-scales the keys onto [-1, 1] |
| `GridSample(grid_size, mode, return_grid_coord)` | keeps one point per occupied cell and subsamples every point-aligned key |
| `Copy(keys_dict)` | copies keys under new names |
| `ToTensor()` | converts NumPy arrays to tensors |
| `Collect(keys, feat_keys)` | keeps `keys`, concatenates `feat_keys` into `feat`, records the point count for `offset` |
| `RandomRotate`, `RandomFlip`, `RandomScale`, `RandomJitter` | geometric augmentation |
| `RandomDropout`, `ShufflePoint`, `SphereCrop` | drop, reorder or crop points |

There are 46 registered transforms in all; [Python API](../reference/api.md#transforms) lists each one.

## Point-aligned keys

Transforms that drop or reorder points, such as `GridSample`, crops and dropout, subsample every key named in the event's `index_valid_keys` list. A new per-point key must be in that list before the first of those transforms, or it no longer lines up with `coord`.

## Write a transform

See [Add a component](../develop/add-component.md#a-transform).
