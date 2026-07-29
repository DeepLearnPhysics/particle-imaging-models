# Model contract

This record distinguishes the behavior present at the Milestone 00 baseline
from the contract accepted for the migration. Milestone 00 documents and tests
the boundary; it does not modify model construction or trainer behavior.

## Baseline behavior

At public commit `9491b0bf4b89bbee52a6383225a19f9c6a628a3c`:

- models are constructed from `pimm.models.builder.MODELS`;
- registry population depends on importing the module that runs a registration
  decorator;
- the default trainer passes one collated mapping to `model(input_dict)`;
- optimization reads `output["loss"]`;
- task evaluators consume additional task-specific output keys;
- `PointModel` and `PointModule` are conveniences for point-oriented models,
  not requirements imposed by PyTorch itself;
- `ModelHook` forwards lifecycle methods only to models that are instances of
  `HookBase`.

The baseline registry accepts a registered ordinary `torch.nn.Module`. The
synthetic fixture in `tests/fixtures/migration/plain_module_step.yaml` freezes
the minimum forward/loss behavior without importing a built-in model family.

## Accepted public contract

The migrated trainer must support:

```python
from collections.abc import Mapping

import torch


class MyModel(torch.nn.Module):
    def forward(self, batch: Mapping[str, object]) -> dict[str, object]:
        ...
        return {
            "loss": scalar_loss,
            # any additional task-defined entries
        }
```

The hard training requirements are:

- the model is a `torch.nn.Module`;
- `forward` accepts the configured collator's batch object;
- training output is a mapping with a scalar tensor at `loss`;
- the optimizer uses exactly `output["loss"]`.

No PIMM, point, task, or WarpConvNet base class is required. Backbones and
internal modules may use task-specific signatures and return types; the
contract applies at the trainer-facing model boundary.

`total_loss` is not a second optimization contract. Existing logging may
display it during the compatibility window, but it must not replace
`output["loss"]` as the optimizer input.

## Task-defined outputs

Evaluators, exporters, and inference helpers may require additional keys. Those
keys are task contracts rather than requirements for every model. Existing
examples include:

- `seg_logits` or `sem_logits` for semantic segmentation;
- `cls_logits` for classification;
- `point` for point-feature consumers;
- `pred_logits`, `pred_masks`, and regression outputs for Panda tasks.

Each evaluator must document and test the extra keys it consumes.

## Construction and lifecycle work

Milestone 01 will make the contract explicit in source and user documentation.
It will add colon import-path construction, lazy built-in imports, and
duck-typed model lifecycle forwarding. Until that milestone lands:

- project-local import-path strings are not yet supported by the generic
  registry;
- built-in registration still relies on package import side effects;
- models that want lifecycle callbacks must still satisfy the current
  `HookBase` check.

Checkpoint and resolved-config records must preserve the model `type` value
used for construction so a resumed run can rebuild the same class.
