# How pimm works

A pimm run is a Python config run through a launcher. Below, we follow one batch from the command line to a checkpoint.

## Two kinds of configuration

| | Experiment config | Launch config |
|---|---|---|
| Scope | what to train | where and how to run it |
| Defined in | `configs/*.py` | `launch/defaults.yaml`, `launch/sites/*.yaml`, launcher flags |
| Contains | data, transforms, model, loss, optimizer, schedule, hooks, epochs, batch sizes | site, nodes, GPUs, CPUs, Slurm, containers, paths, run name |
| Change it with | `key=value` after `--`, or a child config | `--group.field value` flags, or a site profile |

You can run the same experiment config on one workstation GPU or across Slurm nodes and change only the launch config. [Configs and overrides](../training/configs.md) has the details.

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

Each `type` is looked up in a registry, and the class is called with the remaining keys. There are registries for models, datasets, transforms, losses, hooks, trainers, testers and schedulers; you'll find every name in [Python API](../reference/api.md).

## Packed batches

Each dataset item is one event, as a dict of NumPy arrays with one row per point:

```text
coord          (N, 3)  float32
energy         (N, 1)  float32
segment_motif  (N, 1)  int
```

You turn that dict into model input with the transform list: normalize coordinates, scale energy, subsample points, convert arrays to tensors, and finally join the feature keys into `feat` with `Collect`. Events have different numbers of points, so the events are concatenated instead of padded, and the end of each event is recorded:

```text
coord   (N_total, 3)
feat    (N_total, C)
offset  (B,)          cumulative end index of each event
```

For three events with 120, 80 and 300 points, `offset = [120, 200, 500]`. See [Event format](../data/event-format.md) for every field.

## The training loop and hooks

Inside the trainer, the loaders, model, optimizer and scheduler are built first, and the step for each batch is:

```python
output = model(batch)
output["loss"].backward()
```

Everything else is a hook: loading weights, timing, logging, evaluation and saving checkpoints. Hooks are called in list order. The default list is in `configs/_base_/default_runtime.py`:

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

`batch_size`, `batch_size_val`, `batch_size_test` and `num_worker` are totals across all GPUs. `batch_size=16` on four GPUs means four events per GPU. Explicit batch sizes must be divisible by the number of GPUs; `batch_size_val=None` means one event per GPU.

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

You resume training from the checkpoints in `model/`. You can load an export made with `pimm export` in one call, for inference or fine-tuning. See [Checkpoints and resume](../training/checkpoints.md) and [Export and publish](../models/export.md).
