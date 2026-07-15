# Add a dataset

A registered dataset indexes events, decodes one raw event into aligned NumPy
arrays, and applies a configured transform list.

The complete user-facing reader example is in {doc}`Custom data
<../data/custom>`. For a contribution, also satisfy these implementation and
maintenance requirements.

## Interface

```python
from pathlib import Path

import torch

from pimm.datasets.builder import DATASETS
from pimm.datasets.transform import Compose


@DATASETS.register_module()
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split="train", transform=None):
        self.transform = Compose(transform)
        self.data_list = sorted((Path(data_root) / split).glob("*.npz"))

    def get_data(self, index):
        raise NotImplementedError("decode one indexed event here")

    def __getitem__(self, index):
        return self.transform(self.get_data(index))

    def __len__(self):
        return len(self.data_list)
```

Return raw NumPy arrays before
{py:class}`~pimm.datasets.transform.base.ToTensor`. Open non-fork-safe HDF5/ROOT
handles
lazily inside each worker rather than in `__init__`. Keep indexed length and
`index → event` mapping stable so stateful sampling can resume.

## Required tests

- discovery/split counts on a deterministic tiny fixture;
- raw fields, dtypes, shapes, finite/sentinel behavior;
- corrupt/missing/empty-event errors;
- transformed sample and two-event packed batch invariants;
- label/instance distributions and selection cuts;
- multiprocessing/lazy handle behavior where applicable;
- deterministic validation transforms;
- stateful sampler/resume behavior for a map-style training dataset;
- download/conversion manifest checks.

## Registration

Import the new module from `pimm/datasets/__init__.py`. Add any custom
point-aligned key to transform alignment controls before subsampling. Add custom
position/direction truth to geometric auxiliary-key controls.

## Documentation

Create a dataset page with source, license, citation, revision, access, sizes,
directory tree, conversion/download, split construction, exact field table,
axes/units, transforms, validation command, rendered sample, known limitations,
and checksums. Do not advertise private/unpublished v3-like data as generally
downloadable.

## Review boundary

Dataset code should decode data, not encode a particular model's feature vector
or training augmentation. Keep those in transforms/config so the raw scientific
contract can be inspected and reused.
