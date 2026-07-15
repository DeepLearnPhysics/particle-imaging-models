# Add a model or loss

Most research changes do not require a new top-level model. Start with the
smallest extension that changes the behavior you need:

| Need | Add |
|---|---|
| reweight or combine existing objectives | a `criteria` config list |
| implement new differentiable objective math | a registered criterion |
| change the encoder while keeping the same packed-point contract | a registered backbone |
| change model inputs, task outputs, or the forward structure | a top-level model |

The architecture map explains the surrounding
{doc}`extension seams <architecture>`. This page covers the two code paths that
own differentiable computation: models and criteria.

## Model contract

A top-level pimm model is a {py:class}`torch.nn.Module` registered in
{py:data}`~pimm.models.builder.MODELS`. The
{py:class}`~pimm.engines.train.Trainer` moves the packed batch to the selected
device and calls the model as `model(input_dict)`. Do not call `forward`
directly; the {doc}`API reference <../api/index>` explains why.

### Input

`input_dict` follows the {doc}`packed-batch contract <../data/conventions>`.
The common fields are:

| Key | Shape | Meaning |
|---|---:|---|
| `coord` | `(N, 3)` | transformed point coordinates |
| `grid_coord` | `(N, 3)` | integer voxel coordinates, when required |
| `feat` | `(N, C)` | ordered model features |
| `offset` | `(B,)` | cumulative event boundaries; `offset[-1] == N` |
| `segment` | `(N,)` | semantic target, when supervised |

{py:class}`~pimm.datasets.transform.base.Collect` creates `feat`; its channel
count and order must match `backbone.in_channels`. Wrapping the dictionary in
{py:class}`~pimm.models.utils.structure.Point` derives packed-batch metadata
such as per-point batch indices when needed.

### Output

Return a dictionary. The exact keys are a contract between the model and its
trainer/evaluator:

| Situation | Required output |
|---|---|
| training with {py:class}`~pimm.engines.train.Trainer` | scalar, differentiable `loss` |
| semantic validation with {py:class}`~pimm.engines.hooks.eval.semantic_segmentation.SemSegEvaluator` | `loss` and `seg_logits` or `sem_logits` |
| classification following {py:class}`~pimm.models.default.DefaultClassifier` | `loss` in train/eval and `cls_logits` in eval/test |
| instance validation with {py:class}`~pimm.engines.hooks.eval.instance_segmentation.InstanceSegmentationEvaluator` | task-specific `point`, masks, class logits, and optional regression outputs |

Some feature and instance evaluators call
`model(input_dict, return_point=True)`. Accept that keyword when the model will
be used with one of those evaluators, and document what `point` contains.

## Minimal segmentation model

This example is intentionally small. It assumes the selected backbone returns
a {py:class}`~pimm.models.utils.structure.Point` at input-point resolution. If
the backbone pools points, follow the restoration logic in
{py:class}`~pimm.models.default.DefaultSegmentorV2` instead of assuming
`point.feat` already aligns with the input rows.

```python
# pimm/models/my_segmentor.py
import torch.nn as nn

from pimm.models.builder import MODELS, build_model
from pimm.models.losses import build_criteria
from pimm.models.utils.structure import Point


@MODELS.register_module("MySegmentor")
class MySegmentor(nn.Module):
    def __init__(self, backbone, backbone_out_channels, num_classes, criteria):
        super().__init__()
        self.backbone = build_model(backbone)
        self.head = nn.Linear(backbone_out_channels, num_classes)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict, return_point=False):
        point = self.backbone(Point(input_dict))
        logits = self.head(point.feat)

        output = {"point": point} if return_point else {}
        if self.training:
            output["loss"] = self.criteria(logits, input_dict["segment"])
        elif "segment" in input_dict:
            output["loss"] = self.criteria(logits, input_dict["segment"])
            output["seg_logits"] = logits
        else:
            output["seg_logits"] = logits
        return output
```

The code uses {py:func}`~pimm.models.builder.build_model` for the nested
backbone and {py:func}`~pimm.models.losses.builder.build_criteria` for the loss
list. Import the module from `pimm/models/__init__.py` so the model decorator
runs during normal pimm startup:

```python
# pimm/models/__init__.py
from .my_segmentor import MySegmentor  # noqa: F401
```

Select the registered name from a config:

```python
model = dict(
    type="MySegmentor",
    num_classes=5,
    backbone_out_channels=64,
    # Copy a complete backbone block from a tested recipe.
    backbone=dict(type="PT-v3m2", in_channels=4),
    criteria=[
        dict(type="CrossEntropyLoss", ignore_index=-1, loss_weight=1.0),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            ignore_index=-1,
            loss_weight=0.05,
        ),
    ],
)
```

`backbone_out_channels` must equal the feature width consumed by the head, not
merely a convenient hidden width. Begin with the full block from a known recipe
and change one contract at a time. See {doc}`Configuration
<../operations/configuration>` for inheritance and list-replacement behavior.

## Configure existing criteria

{py:func}`~pimm.models.losses.builder.build_criteria` builds every dictionary
through {py:data}`~pimm.models.losses.builder.LOSSES` and returns a
{py:class}`~pimm.models.losses.builder.Criteria` callable. For the ordinary
task path it calls each criterion with the same `(pred, target)` pair and sums
their scalar results. Browse the {doc}`loss registry
<../api/registry/losses>` for every available `type` and constructor.

Specialized detector criteria may instead return `(loss, info_dict)`; their
task models explicitly unpack that form. Do not return a tuple from an ordinary
segmentation/classification criterion unless the calling model handles it.

## Add a new criterion

Add Python code only when the objective itself is new. A different weight or
mixture belongs in the config above.

### 1. Implement and register the math

A criterion class is a {py:class}`torch.nn.Module` registered in
{py:data}`~pimm.models.losses.builder.LOSSES`. Its constructor receives every
config field except `type`; `forward` receives exactly what the task model
passes to `self.criteria(...)`.

This complete example implements a multiclass Tversky loss:

```python
# pimm/models/losses/tversky.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LOSSES


@LOSSES.register_module()
class TverskyLoss(nn.Module):
    """Multiclass Tversky loss for per-point class logits."""

    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        eps=1e-6,
        ignore_index=-1,
        loss_weight=1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        target = target.reshape(-1).long()
        valid = target != self.ignore_index

        # A batch with no labelled points still returns a differentiable zero.
        if not torch.any(valid):
            return pred.sum() * 0.0

        pred = pred[valid]
        target = target[valid]
        probability = pred.softmax(dim=-1)
        truth = F.one_hot(target, num_classes=pred.shape[-1]).to(probability.dtype)

        true_positive = (probability * truth).sum(dim=0)
        false_positive = (probability * (1.0 - truth)).sum(dim=0)
        false_negative = ((1.0 - probability) * truth).sum(dim=0)
        score = (true_positive + self.eps) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.eps
        )
        return self.loss_weight * (1.0 - score.mean())
```

Keep an ordinary criterion's contract narrow:

- return a scalar tensor without detaching it or converting it to a Python number;
- create new tensors from an input (`pred.new_tensor(...)`, `zeros_like`, and so on) so device and dtype follow the batch;
- expose weighting and ignore behavior as constructor fields so the resolved config records them;
- define behavior for empty/all-ignored targets and invalid labels;
- document input shapes, target convention, reduction, and return value in the class docstring.

:::{important}
{py:class}`~pimm.models.losses.builder.Criteria` is currently a callable
container, not a {py:class}`torch.nn.Module`, and its criterion list is not a
{py:class}`torch.nn.ModuleList`. Consequently, registered criteria are not
automatically moved with the task model, added to the optimizer, or included in
its state dict. Keep criteria built through this path stateless: do not give
them learnable parameters or persistent buffers. If an objective needs
learnable/stateful components, own that module directly on the task model (or
first change and test the criteria container contract).
:::

### 2. Import and configure it

Registration happens at import time. Re-export the class from the loss package:

```python
# pimm/models/losses/__init__.py
from .tversky import TverskyLoss  # noqa: F401
```

Then use its registry name:

```python
criteria = [
    dict(
        type="TverskyLoss",
        alpha=0.3,
        beta=0.7,
        ignore_index=-1,
        loss_weight=1.0,
    )
]
```

Add `TverskyLoss` to the appropriate matcher under `CATEGORIES["losses"]` in
`docs/gen_api.py`. An uncategorized class is still generated under **Other**,
and the generator prints its registered name so it cannot disappear silently.

### 3. Test registration, gradients, and edge cases

A device-agnostic criterion should have a CPU unit test. This catches a missing
package import, a non-scalar result, broken autograd, and the all-ignored case
without launching training:

```python
# tests/unit/models/losses/test_tversky.py
import torch

import pimm.models.losses  # runs registration imports
from pimm.models.losses.builder import LOSSES, build_criteria


def test_tversky_loss_builds_and_backpropagates():
    assert LOSSES.get("TverskyLoss") is not None
    criterion = build_criteria([
        dict(type="TverskyLoss", ignore_index=-1, loss_weight=0.5)
    ])

    pred = torch.randn(7, 4, requires_grad=True)
    target = torch.tensor([0, 1, 2, 3, 0, -1, 2])
    loss = criterion(pred, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_tversky_loss_all_ignored_stays_differentiable():
    criterion = build_criteria([
        dict(type="TverskyLoss", ignore_index=-1)
    ])
    pred = torch.randn(3, 4, requires_grad=True)
    loss = criterion(pred, torch.full((3,), -1))

    assert loss.item() == 0.0
    loss.backward()
    assert pred.grad is not None
```

## Model verification checklist

Before a long run:

1. build the model through {py:func}`~pimm.models.builder.build_model` from the smallest valid config;
2. run a synthetic or pinned packed batch through `model(batch)`, never `model.forward(batch)`;
3. assert that `loss` is finite, scalar, and has `requires_grad=True`;
4. call `loss.backward()` and check gradients on every intended trainable branch;
5. switch to evaluation and assert every evaluator-facing key, shape, and dtype;
6. save/reload a state dict and test {py:func}`~pimm.save_pretrained` if the model will be published;
7. run one tiny train/evaluate/checkpoint cycle;
8. run DDP/FSDP tests only for strategies the model claims to support.

Document the input fields and feature order, output keys, target convention,
losses, supported tasks, checkpoint mapping, parameter count, compute/memory
assumptions, limitations, provenance/citation, and one representative
prediction. Constructor and `forward` details belong in docstrings so the
generated {doc}`API reference <../api/index>` stays current.
