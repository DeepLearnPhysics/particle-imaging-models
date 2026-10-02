# How pimm works

A pimm run is a Python config executed by a launcher. This page follows one batch from the command line to a checkpoint.

## Two kinds of configuration

| | Experiment config | Launch config |
|---|---|---|
| Describes | what to train | where and how to run it |
| Lives in | `configs/*.py` | `launch/defaults.yaml`, `launch/sites/*.yaml`, launcher flags |
| Holds | data, transforms, model, loss, optimizer, schedule, hooks, epochs, batch sizes | site, nodes, GPUs, CPUs, Slurm, containers, paths, run name |
| Change it with | `key=value` after `--`, or a child config | `--group.field value` flags, or a site profile |

The same experiment config runs on one workstation GPU or across Slurm nodes; only the launch config changes. [Configs and overrides](../training/configs.md) has the details.

## Components are named by registry

Every component in a config is a dict with a `type`:

```python
model = dict(
    type="DefaultSegmentorV2",
    num_classes=5,
    backbone=dict(type="PT-v3m2", in_channels=4),
    criteria=[dict(type="CrossEntropyLoss", ignore_index=-1)],
)
```

pimm looks each `type` up in a registry and calls the class with the remaining keys. There are registries for models, datasets, transforms, losses, hooks, trainers, testers and schedulers; [Python API](../reference/api.md) lists every name.

## Events become a packed batch

A dataset returns one event as a dict of NumPy arrays, one row per point:

```text
coord          (N, 3)  float32
energy         (N, 1)  float32
segment_motif  (N, 1)  int
```

The transform list turns that into model input. It normalizes coordinates, scales energy, subsamples, converts to tensors, and finally `Collect` joins the feature keys into `feat`. Events have different numbers of points, so pimm concatenates them instead of padding and records where each event ends:

```text
coord   (N_total, 3)
feat    (N_total, C)
offset  (B,)          cumulative end index of each event
```

Three events with 120, 80 and 300 points give `offset = [120, 200, 500]`. [Event format](../data/event-format.md) lists every field.

## The trainer runs the loop; hooks do the rest

The trainer builds the loaders, model, optimizer and scheduler, then for each batch calls:

```python
output = model(batch)
output["loss"].backward()
```

Everything else is a hook: loading weights, timing, logging, evaluation and saving checkpoints. Hooks run in list order. The default list, from `configs/_base_/default_runtime.py`:

```python
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="FinalEvaluator", test_last=False),
]
```

## Batch sizes are global

`batch_size`, `batch_size_val`, `batch_size_test` and `num_worker` count across all GPUs. With `batch_size=16` on four GPUs, each GPU gets four events. Explicit batch sizes must divide evenly by the number of GPUs; `batch_size_val=None` means one event per GPU.

## A run is a directory

```text
exp/<config group>/<run name>/
├── code/                  copy of pimm/ and configs/ (skipped with --train.no-code-copy)
├── config.py              the config as launched
├── resolved_config.json   after inheritance and overrides
├── model_config.json
├── run_metadata.json
├── train.log
└── model/                 checkpoints
```

Checkpoints in `model/` continue training. An export made with `pimm export` loads in one call for inference or fine-tuning. See [Checkpoints and resume](../training/checkpoints.md) and [Export and publish](../models/export.md).
