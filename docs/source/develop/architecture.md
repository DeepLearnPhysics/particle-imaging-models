# Architecture

Each part of pimm owns one job. Knowing which part owns what tells you where a change belongs.

| Component | Owns | Leaves to others |
|---|---|---|
| dataset | finding events and decoding one into arrays | features, training |
| transform | preprocessing, augmentation, label mapping, the final sample | file access, the training loop |
| collator and loader | packing samples, sampling, worker processes | what features mean |
| model | the forward pass, loss and task outputs | resources, metric aggregation |
| trainer | device and distributed setup, the optimization loop | task metrics |
| hook | logging, diagnostics, evaluation, checkpoints | replacing the loop |
| tester and evaluator | predictions, aggregation, metrics | parameter updates |
| launcher | environment, paths, processes, Slurm and containers | model and data choices |

## One batch through the code

```text
dataset.get_data(index)                  raw NumPy event
  → Compose(transforms)                  pimm/datasets/transform/
  → StatefulDataLoader + collate_fn      packed batch with offset
  → model(batch)                         {"loss": ..., task outputs}
  → trainer: backward, optimizer step    pimm/engines/train.py
  → hooks: log, evaluate, checkpoint     pimm/engines/hooks/
```

## Registries

Models, datasets, transforms, losses, hooks, trainers, testers and schedulers register under a name that configs use as `type`:

```python
from pimm.models.builder import MODELS


@MODELS.register_module("MyBackbone")
class MyBackbone(nn.Module):
    ...
```

The decorator runs when the module is imported, so a new module must be imported from its package's `__init__.py`. Configs saved by earlier runs refer to registered names, so a new architecture gets a new name rather than changing what an existing one builds.

## Choose the smallest change

| You need | Add |
|---|---|
| different hyperparameters or a different combination of parts | a child config |
| a weighted mix of existing losses | entries in the `criteria` list |
| new loss math | a registered loss |
| a new encoder that takes and returns a `Point` | a registered backbone |
| new task outputs or forward structure | a top-level model |
| a new file or truth format | a dataset |
| new per-event preprocessing or augmentation | a transform |
| periodic logging, evaluation or checkpoint behavior | a hook |
| a different optimization procedure | a trainer |
