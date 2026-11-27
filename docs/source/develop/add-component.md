# Add a component

Every component is a registered class. Write it, import it from its package's `__init__.py`, and name it in a config.

## A model

The input of a top-level model is the packed batch dict, and its output is a dict. In training, the output must contain a scalar `loss`; in evaluation, it contains the outputs the model's evaluator needs, such as `seg_logits`, `segment` and `offset` for `SemSegEvaluator`, all at the same points.

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

    def forward(self, input_dict):
        point = self.backbone(Point(input_dict))
        logits = self.head(point.feat)
        output = {}
        if "segment" in input_dict:
            output["loss"] = self.criteria(logits, input_dict["segment"])
            if not self.training:
                output["segment"] = input_dict["segment"]
        if not self.training:
            output.update(seg_logits=logits, offset=input_dict["offset"])
        return output
```

`MySegmentor` is written for a backbone output with features for every input point, and only these points are scored. If you pool points in the backbone, map features back as in `DefaultSegmentorV2`; to score the points from before `GridSample`, map the logits through `inverse` as `DefaultSegmentorV2` does in evaluation. Register the module:

```python
# pimm/models/__init__.py
from .my_segmentor import MySegmentor
```

## A loss

Call `build_criteria` in the model to build each entry of a `criteria` list from the `LOSSES` registry; the entries' outputs are summed. The arguments of a new loss are the ones you pass to `self.criteria(...)` in the model:

```python
from pimm.models.losses.builder import LOSSES


@LOSSES.register_module()
class MyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, ignore_index=-1):
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        ...
        return self.loss_weight * loss
```

Use it as `criteria=[dict(type="MyLoss", loss_weight=0.5)]`, alongside existing losses if you like.

## A transform

```python
import numpy as np

from pimm.datasets.transform import TRANSFORMS


@TRANSFORMS.register_module()
class RandomEnergyDropout:
    def __init__(self, ratio=0.1, p=0.5):
        self.ratio = float(ratio)
        self.p = float(p)

    def __call__(self, data_dict):
        if "energy" in data_dict and np.random.rand() < self.p:
            mask = np.random.rand(len(data_dict["energy"])) < self.ratio
            data_dict["energy"][mask] = 0.0
        return data_dict
```

Import it from `pimm/datasets/transform/__init__.py` and use `dict(type="RandomEnergyDropout", ratio=0.1)`. Transforms are applied to NumPy arrays before `ToTensor`. If you change the number or order of points in a transform, apply the same change to every key in `index_valid_keys`.

## A hook

```python
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase
from pimm.utils.comm import is_main_process


@HOOKS.register_module()
class MyScalarHook(HookBase):
    def __init__(self, every=100):
        self.every = int(every)

    def after_step(self):
        step = int(self.trainer.global_step)
        if step % self.every == 0 and is_main_process() and self.trainer.writer is not None:
            self.trainer.writer.add_scalar("my/value", 1.0, step)
```

Import it from `pimm/engines/hooks/__init__.py` and add it to `hooks`. Hooks are run in list order. Guard file and logging side effects with `is_main_process()`, but let every process enter collective operations, or the other processes are left waiting forever. Checkpoints don't include hook attributes, so derive counters from `self.trainer.global_step`.

## A dataset

See [Bring your own data](../data/your-data.md#route-2-a-reader-class).
