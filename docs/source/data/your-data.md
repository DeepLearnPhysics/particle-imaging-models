# Bring your own data

Write your events as `.npy` files and use `DefaultDataset`, with no new code; or write a reader class that reads your format directly.

## Route 1: `.npy` files

`DefaultDataset` reads one directory per event:

```text
data/mydet/
├── train/
│   ├── event_0000/
│   │   ├── coord.npy      (N, 3) float32
│   │   ├── strength.npy   (N, 1) float32, the energy
│   │   └── segment.npy    (N,)   int, class of each point; -1 to ignore
│   └── event_0001/
└── val/
```

It loads files named `coord`, `color`, `normal`, `strength`, `segment`, `instance` and `pose`, and ignores everything else. There is no `energy` file, so store energy as `strength` and copy it in the transform list.

Make a few synthetic events to try the route:

```python
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
for split, count in [("train", 8), ("val", 2)]:
    for i in range(count):
        event = Path("data/mydet") / split / f"event_{i:04d}"
        event.mkdir(parents=True, exist_ok=True)
        n = int(rng.integers(500, 3000))
        np.save(event / "coord.npy", rng.uniform(0, 768, (n, 3)).astype(np.float32))
        np.save(event / "strength.npy", rng.exponential(1.0, (n, 1)).astype(np.float32))
        np.save(event / "segment.npy", rng.integers(0, 5, n))
```

Point a config at it:

```python
# configs/my_study/mydet_semseg.py
_base_ = ["../_base_/default_runtime.py"]

transform = [
    dict(type="Copy", keys_dict={"strength": "energy"}),
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="LogTransform", min_val=0.01, max_val=20.0),
    dict(type="GridSample", grid_size=0.001, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "energy")),
]

data = dict(
    num_classes=5,
    ignore_index=-1,
    names=["class0", "class1", "class2", "class3", "class4"],
    train=dict(type="DefaultDataset", data_root="data/mydet", split="train", transform=transform),
    val=dict(type="DefaultDataset", data_root="data/mydet", split="val", transform=transform),
)

# Then add model, optimizer, scheduler, hooks, train and test,
# for example copied from configs/tests/tiny_semseg.py.
```

Change `center` and `scale` so that `NormalizeCoord` maps your detector volume into the unit ball, and set `LogTransform`'s range to your energy scale.

Check one batch before you train:

```python
import pimm.datasets
from pimm.datasets.builder import build_dataset
from pimm.datasets.utils import collate_fn
from pimm.utils.config import Config

cfg = Config.fromfile("configs/my_study/mydet_semseg.py")
dataset = build_dataset(cfg.data.train)
batch = collate_fn([dataset[0], dataset[1]])
for key, value in batch.items():
    print(key, tuple(value.shape))
```

You should see `coord`, `grid_coord` and `segment` with one row per kept point, `feat` with four columns, and `offset` with two entries. Then train it like any recipe:

```bash
uv run pimm launch --train.config my_study/mydet_semseg --resources.nproc-per-node 1
```

## Route 2: a reader class

A reader is a registered `torch.utils.data.Dataset` whose `get_data` returns one event as a dict of NumPy arrays:

```python
# pimm/datasets/my_detector.py
import glob
import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class MyDetectorDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None):
        self.files = sorted(glob.glob(os.path.join(data_root, split, "*.h5")))
        self.index = []
        for path in self.files:
            with h5py.File(path, "r") as f:
                self.index += [(path, i) for i in range(len(f["coord_offsets"]) - 1)]
        self.transform = Compose(transform)
        self.handles = {}

    def get_data(self, idx):
        path, i = self.index[idx]
        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r")
        f = self.handles[path]
        start, stop = f["coord_offsets"][i], f["coord_offsets"][i + 1]
        return {
            "coord": f["coord"][start:stop].astype(np.float32),
            "energy": f["energy"][start:stop].astype(np.float32)[:, None],
            "segment": f["label"][start:stop].astype(np.int64),
            "name": f"{os.path.basename(path)}:{i}",
        }

    def __getitem__(self, idx):
        return self.transform(self.get_data(idx))

    def __len__(self):
        return len(self.index)
```

The `coord_offsets`, `coord`, `energy` and `label` datasets stand in for your file's layout. HDF5 handles open on first use, inside each data-loader worker, rather than in `__init__`. Keep the index order fixed so a resumed run sees the same events.

Register the class by importing it in `pimm/datasets/__init__.py`:

```python
from .my_detector import MyDetectorDataset
```

Then use `type="MyDetectorDataset"` in a config. The `JAXTPCDataset` and `LUCiDDataset` sources in `pimm/datasets/` are fuller examples: they split reading into reader classes in `pimm/datasets/readers/`, each with `read_event(idx)`, a lazy `h5py_worker_init`, `__len__` and `close`.
