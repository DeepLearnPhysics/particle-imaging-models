# Experiment anatomy

This page follows one training batch from command line to checkpoint. Use it as
a map when debugging or adding a component.

## Two configuration layers

| Layer | Source | Owns |
|---|---|---|
| Execution | `launch/defaults.yaml`, `launch/sites/*.yaml`, optional recipe YAML, launcher flags | node/GPU/CPU counts, Slurm, container, paths, environment, run name, resume |
| Experiment | `configs/*.py`, inherited bases, post-`--` overrides | data, transforms, model, loss, optimizer, schedule, hooks, epochs, global batch sizes |

`pimm launch` executes on the current node. `pimm submit` executes through
Slurm. Both call the same training path after resolving execution settings.

```bash
uv run pimm launch \
  --train.config tests/tiny_semseg \
  --resources.nproc-per-node 2 \
  -- batch_size=8 optimizer.lr=0.001
```

Here `resources.nproc_per_node=2` is execution state; `batch_size=8` and
`optimizer.lr` are experiment state.

## Python config resolution

{py:meth}`Config.fromfile <pimm.utils.config.Config.fromfile>` loads the requested file and its `_base_`
chain. Dictionaries merge recursively; replacing a list replaces the full
list. CLI overrides are applied last using dotted keys.

```python
# configs/my_project/semseg.py
_base_ = ["../panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py"]

batch_size = 16  # global across every rank
data = dict(train=dict(max_len=100_000))
optimizer = dict(lr=3e-5)
```

The resolved values are saved as `config.py` and `resolved_config.json`; these,
not the original base file, are the record of what started.

## Registries construct components

Configs describe objects with a `type` key:

```python
model = dict(
    type="DefaultSegmentorV2",
    # A real recipe supplies the complete PT-v3m2 architecture block.
    backbone=dict(type="PT-v3m2", in_channels=4),
    criteria=[dict(type="CrossEntropyLoss", ignore_index=-1)],
)
```

Builders resolve those names through registries. The relevant package must
import a decorated class before it can be found.

| Registry | Typical config fields |
|---|---|
| `MODELS` | `model`, nested `backbone`, task heads |
| `DATASETS` | `data.train`, `data.val`, `data.test` |
| `TRANSFORMS` | `transform` pipelines |
| `LOSSES` | `criteria` |
| `HOOKS` | `hooks` |
| `TRAINERS` | `train.type` |

The generated {doc}`registry API <../api/index>` is exhaustive. The
{doc}`extension map <../extend/architecture>` explains which interfaces are
stable enough to build against.

## Samples become a packed batch

A dataset returns one event as a dictionary. Common per-point fields are
two-dimensional even for scalar quantities:

```text
coord          float tensor  (num_points, 3)
energy         float tensor  (num_points, 1)
segment_motif  integer tensor (num_points, 1)
name           string
```

Transforms normalize, augment, voxelize, copy target fields, convert tensors,
and collect `feat`. Collation concatenates per-point arrays and adds cumulative
event boundaries:

```text
coord    float tensor   (total_points, 3)
feat     float tensor   (total_points, channels)
segment  integer tensor (total_points,)
offset   integer tensor (batch_size,)
```

For lengths $[120, 80, 300]$, `offset` is $[120, 200, 500]$. See
{doc}`Data conventions <../data/conventions>` for coordinate, unit, feature,
label, and transform contracts.

## The trainer owns the loop

{py:class}`DefaultTrainer <pimm.engines.train.Trainer>` creates distributed state, loaders, model, optimizer,
scheduler, scaler, and hooks. Its central rule is simple:

```python
output = model(batch)
loss = output["loss"]
loss.backward()
```

A trainable top-level model must return `loss`. Evaluators and testers consume
additional task-specific keys such as segmentation logits or detector masks.
Document those keys with a new task model; they are part of its public output
contract.

## Hooks observe lifecycle events

Hooks attach behavior without replacing the loop. Their order matters. Typical
default hooks load a checkpoint, time iterations, write information, compute
semantic-segmentation metrics, save checkpoints, and perform final evaluation.

Use a hook for behavior that responds to lifecycle events. Use a transform for
sample preprocessing, a model for differentiable computation, and a tester or
evaluator for task-specific predictions and metrics.

## Distributed settings are derived per rank

`batch_size`, `batch_size_val`, `batch_size_test`, and `num_worker` are global
totals. {py:func}`~pimm.engines.defaults.default_setup` divides them by world
size and asserts that explicit
batch sizes divide evenly.

| Config | World size | Per-rank value |
|---:|---:|---:|
| `batch_size=16` | 1 | 16 |
| `batch_size=16` | 4 | 4 |
| `num_worker=32` | 4 | 8 |
| `batch_size_val=None` | 4 | 1 (automatic) |

The event count does not bound memory: packed batches with unusually large
events can still exhaust VRAM.

## A run is a provenance bundle

By default, the launcher copies the code and starts training from the snapshot:

```text
exp/<group>/<run>/
├── code/
├── config.py
├── resolved_config.json
├── model_config.json
├── run_metadata.json
├── train.log
└── model/
```

`MODEL_DIR` may redirect `model/` to another filesystem through a symlink. Keep
the small provenance files with the run even when moving weights.

## Checkpoint and export are different products

- A **training checkpoint** includes weights plus trainer state required to
  continue optimization.
- A **portable export** includes consolidated model weights and, when the config
  can be found, a sanitized `config.json` used to reconstruct the model.

Same-topology structured resume can restore a mid-epoch dataloader cursor.
Changing world size or workers discards that cursor and restarts the saved epoch;
model and optimizer state can still reshard. Read {doc}`Checkpoint semantics
<../operations/checkpoints>` before claiming an interrupted run continued
bit-for-bit.

## Continue by intent

- Modify an experiment: {doc}`Configuration <../operations/configuration>`.
- Understand or add data: {doc}`Data conventions <../data/conventions>`.
- Run on more devices: {doc}`Distributed training <../workflows/distributed>`.
- Add a component: {doc}`Extend pimm <../extend/architecture>`.
- Load or export weights: {doc}`Models <../models/index>`.
