# Architecture and extension seams

pimm is configuration-driven, but every config entry resolves to ordinary
Python. This map helps you change the right layer.

## Ownership map

| Component | Owns | Does not own |
|---|---|---|
| dataset | event discovery and raw decoding | model-ready feature concatenation, optimization |
| transform | preprocessing, augmentation, filtering, target mapping, final sample projection | file discovery, training loop |
| collator/loader | packing variable-length samples, sampling, worker execution | scientific feature meaning |
| model | differentiable forward, loss, task outputs | launch resources, metric aggregation |
| trainer | device/distributed setup and optimization loop | task-specific scientific metrics |
| hook | lifecycle side effects/diagnostics/evaluation/checkpointing | arbitrary replacement of the core loop |
| tester/evaluator | predictions, aggregation, metrics | parameter updates |
| launcher | environment, paths, processes, Slurm/container execution | architecture/data semantics |

## One batch through the system

```text
dataset.get_data(index)
  → raw NumPy event
  → Compose(transform list)
  → tensor sample + per-sample offset
  → StatefulDataLoader / collate_fn
  → packed batch
  → model(batch)
  → {loss, task outputs}
  → trainer backward/step
  → hooks log/evaluate/checkpoint
```

The {doc}`experiment anatomy <../getting_started/concepts>` explains this as a
user workflow; this page focuses on extension boundaries.

The concrete pipeline components are
{py:class}`~pimm.datasets.transform.base.Compose`,
{py:func}`~pimm.datasets.utils.collate_fn`, and, for the training loader,
{py:class}`~pimm.datasets.stateful.StatefulRandomSampler`.

## Registries

Models, datasets, transforms, losses, hooks, trainers, and testers register a
name used in config `type` fields. A decorator runs only when the module is
imported, so new modules must be re-exported from the relevant package
`__init__.py`.

```python
from pimm.models.builder import MODELS

@MODELS.register_module("MyBackbone")
class MyBackbone(...):
    ...
```

Prefer a distinctive, versioned name for materially different architecture
contracts. Avoid silently changing the meaning of an existing registered type
used by saved configs.

## Choose the smallest extension

| Need | Extension |
|---|---|
| change hyperparameters/composition | child config |
| combine existing losses | config criteria list |
| new differentiable loss | registered loss |
| new encoder with compatible {py:class}`Point <pimm.models.utils.structure.Point>` contract | registered backbone |
| new task output/forward structure | top-level model |
| new file/truth format | dataset |
| new per-event preprocessing or augmentation | transform |
| periodic logging/evaluation/checkpoint behavior | hook |
| fundamentally different optimization procedure | trainer, only after model/hook options are insufficient |

Most research additions should not create a trainer.

## Model contract

A top-level trainable model receives the packed dictionary and returns a
dictionary. During training, `loss` is required and must be a scalar tensor with
a gradient graph. Evaluation/test output keys are task contracts consumed by
their evaluator/tester.

Backbones commonly consume/return
{py:class}`Point <pimm.models.utils.structure.Point>`, which
holds features, packed offsets/batch indices, serialized order, and sparse
metadata.

## Hook lifecycle

```text
modify_config
before_train
  before_epoch
    before_step
    after_step
  after_epoch
after_train
```

Methods run in configured hook order on every rank. Rank-0 file/writer side
effects require a guard. Collectives require every rank. Arbitrary hook
attributes are not in the generic checkpoint payload.

## Compatibility and public API

The generated API is exhaustive, not a promise that every internal helper is
stable. Treat the following as supported extension concepts:

- registry/config construction;
- raw event dictionary → transform list → packed batch;
- `nn.Module` top-level model returning `loss` and documented outputs;
- {py:class}`HookBase <pimm.engines.hooks.default.HookBase>` lifecycle;
- saved resolved config and standard checkpoint/export artifacts.

If a contribution depends on an internal attribute, document and test it
explicitly. Prefer a small public helper over duplicating internal state logic.

## Test pyramid

1. pure CPU unit test for parsing/math/alignment where possible;
2. construction test through the registry/config;
3. one synthetic or pinned-fixture forward/backward test;
4. tiny end-to-end train/evaluate/checkpoint test;
5. distributed/GPU test only for behavior that actually depends on it;
6. docs example tied to the tested path.

## Next

- {doc}`Contributor setup <contributing>`.
- {doc}`Add a model <add_model>`.
- {doc}`Add a dataset <add_dataset>`.
- {doc}`Add a transform <add_transform>`.
- {doc}`Add a hook <add_hook>`.
