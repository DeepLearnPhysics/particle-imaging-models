# Transforms

The input of a transform is an event dict, and the output is the same dict, changed. You list transforms in order in a config; they are chained with `Compose` and applied to every event.

## The Panda pipeline, step by step

| Step | Event after the step |
|---|---|
| raw event | `coord` (N, 3) in detector units, `energy` (N, 1) |
| `NormalizeCoord(center=[384, 384, 384], scale=768·√3/2)` | `coord` inside the unit ball |
| `LogTransform(min_val=0.01, max_val=20.0)` | `energy` on [-1, 1], log-scaled |
| `GridSample(grid_size=0.001, return_grid_coord=True)` | one point per occupied cell, M ≤ N points; integer `grid_coord` added |
| `Copy(keys_dict={"segment_motif": "segment"})` | training target under the name used by models |
| `ToTensor()` | every array is a torch tensor |
| `Collect(keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "energy"))` | only the listed keys, plus `feat` (M, 4) |

`Copy` is used in training recipes only; no target is needed at inference.

## Common transforms

| Transform | Effect |
|---|---|
| `NormalizeCoord(center, scale)` | `coord = (coord - center) / scale`; the centroid and the largest radius are used when omitted |
| `LogTransform(min_val, max_val, keys=("energy",))` | the keys are log-scaled onto [-1, 1] |
| `GridSample(grid_size, mode, return_grid_coord, return_inverse)` | one point per occupied cell is kept, and every point-aligned key is subsampled; with `return_inverse=True`, `inverse` gives the cell of every original point, as needed to [evaluate](../training/evaluate.md#points-that-are-scored) `DefaultSegmentorV2` |
| `Copy(keys_dict)` | keys are copied under new names |
| `ToTensor()` | NumPy arrays are converted to tensors |
| `Collect(keys, feat_keys, offset_keys_dict)` | `keys` are kept, `feat_keys` are concatenated into `feat`, and the point count is recorded for `offset`; `offset_keys_dict=dict(offset="coord", origin_offset="inverse")` also records the original point count |
| `RandomRotate`, `RandomFlip`, `RandomScale`, `RandomJitter` | geometric augmentation |
| `RandomDropout`, `ShufflePoint`, `SphereCrop` | points are dropped, reordered or cropped |

There are 46 registered transforms in all; see [Python API](../reference/api.md#transforms) for the full list.

## Point-aligned keys

When points are dropped or reordered, as in `GridSample`, crops and dropout, every key named in the event's `index_valid_keys` list is subsampled to match. Add a new per-point key to that list before the first of those transforms, or the key will no longer be aligned with `coord`.

## Write a transform

See [Add a component](../develop/add-component.md#a-transform).
