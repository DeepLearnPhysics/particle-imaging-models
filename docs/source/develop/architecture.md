# Architecture

Each part of pimm has one job. If you know which part has which job, you know where a change belongs.

| Component | Responsibility | Left to others |
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

Models, datasets, transforms, losses, hooks, trainers, testers and schedulers are registered under a name, and you use that name as `type` in configs:

```python
from pimm.models.builder import MODELS


@MODELS.register_module("MyBackbone")
class MyBackbone(nn.Module):
    ...
```

Registration happens when the module is imported, so import a new module from its package's `__init__.py`. Configs saved by earlier runs contain registered names, so give a new architecture a new name rather than changing the architecture behind an existing name.

## Choose the smallest change

| You need | Add |
|---|---|
| different hyperparameters or a different combination of parts | a child config |
| a weighted mix of existing losses | entries in the `criteria` list |
| new loss math | a registered loss |
| a new encoder with a `Point` as input and output | a registered backbone |
| new task outputs or forward structure | a top-level model |
| a new file or truth format | a dataset |
| new per-event preprocessing or augmentation | a transform |
| periodic logging, evaluation or checkpoint behavior | a hook |
| a different optimization procedure | a trainer |
