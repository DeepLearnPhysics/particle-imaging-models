# Add a transform

A transform is a registered callable that reads and returns a sample
dictionary. Use one for preprocessing, augmentation, derived fields, or
method-specific views/masks.

## Skeleton

```python
import numpy as np

from pimm.datasets.transform import TRANSFORMS


@TRANSFORMS.register_module()
class RandomEnergyDropout:
    def __init__(self, ratio=0.1, p=0.5):
        self.ratio = float(ratio)
        self.p = float(p)

    def __call__(self, data_dict):
        if "energy" not in data_dict or np.random.rand() >= self.p:
            return data_dict
        mask = np.random.rand(len(data_dict["energy"])) < self.ratio
        data_dict["energy"][mask] = 0.0
        return data_dict
```

Import it from `pimm/datasets/transform/__init__.py`, then configure with
`dict(type="RandomEnergyDropout", ratio=0.1, p=0.5)`.

## Contract checklist

- required, optional, created, modified, and removed keys;
- input/output shapes and dtypes;
- whether it changes point count/order;
- how it updates every point-aligned field;
- how it transforms position and direction targets;
- randomness, seed source, probability, and train/eval use;
- behavior on missing keys, zero points, sentinels, and non-finite values;
- position relative to {py:class}`ToTensor
  <pimm.datasets.transform.base.ToTensor>` and {py:class}`Collect
  <pimm.datasets.transform.base.Collect>`.

## Tests

Use a tiny synthetic event with unique row IDs so misalignment is visible. Test
probability 0 and 1, a fixed seed, exact output for a simple geometry, all
auxiliary fields, empty/boundary cases, then composition with the preceding and
following standard transforms.

If a transform creates a new point field, update `index_valid_keys` before any
later selection. If it changes coordinates, update declared position/direction
truth in the same call.

Keep stochastic augmentation out of validation/test configs unless a documented
test-time aggregation protocol consumes it.
