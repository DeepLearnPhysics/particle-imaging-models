# Run a released model

`pimm.from_pretrained` downloads an export, rebuilds the model, loads its weights, moves it to a device and returns it in eval mode.

```python
import pimm

model = pimm.from_pretrained("DeepLearnPhysics/Panda-Semantic", device="cuda")
```

`device` defaults to `"cpu"`. Downloads go to the Hugging Face cache (`HF_HUB_CACHE`); pass `cache_dir=` to choose another directory and `revision=` to pin a branch, tag or commit.

## Label one event

This labels test event 0 of PILArNet-M-mini (see [First run](../get-started/first-run.md) for the download) with Panda Semantic. Panda needs CUDA.

```python
import numpy as np
import torch

import pimm
from pimm.datasets.pilarnet import PILArNetH5Dataset
from pimm.datasets.transform import Compose
from pimm.datasets.utils import collate_fn

model = pimm.from_pretrained("DeepLearnPhysics/Panda-Semantic", device="cuda")

transform = Compose([
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="LogTransform", min_val=0.01, max_val=20.0, keys=("energy",)),
    dict(type="GridSample", grid_size=0.001, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord"), feat_keys=("coord", "energy")),
])

events = PILArNetH5Dataset(data_root="data/PILArNet-M-mini", split="test",
                           revision="v3", transform=[], min_points=0)
event = events.get_data(0)

np.random.seed(0)
sample = transform({"coord": event["coord"].copy(), "energy": event["energy"].copy()})
batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in collate_fn([sample]).items()}

with torch.inference_mode():
    output = model(batch)

labels = output["seg_logits"].argmax(-1)
names = ["shower", "track", "Michel", "delta", "low-energy deposit"]
print({names[i]: int((labels == i).sum()) for i in range(5)})
```

To label several events at once, transform each one and pass the list to `collate_fn`; it concatenates them and builds `offset`. The labels refer to the points `GridSample` kept, in batch order.

## Outputs

Semantic models return `seg_logits`, one row per point of the packed batch.

Panda Particle and Panda Interaction return raw query predictions. `model.postprocess` turns them into per-point assignments:

```python
with torch.inference_mode():
    prediction = model.postprocess(model(batch))

prediction["instance_labels"]  # (N,) instance id of each point, -1 for none
prediction["class_labels"]     # (N,) class of each point
prediction["confidences"]      # (N,)
prediction["query_labels"]     # (N,) query that produced the point's instance, -1 for none
```

Instance ids are unique across the batch. `postprocess` accepts `mask_threshold`, `conf_threshold` and `min_points`.

Panda Base returns a `Point`; `point.feat` holds the features. PoLAr-MAE Pretrain returns its loss terms.

## What it loads

| Source | Example | Architecture comes from |
|---|---|---|
| Hugging Face repository | `DeepLearnPhysics/Panda-Semantic` or `hf://DeepLearnPhysics/Panda-Semantic` | the repository's `config.json` |
| Local export | `artifacts/my-model` | its `config.json` |
| Run checkpoint | `exp/<group>/<run>/model/model_best.pth` | the run's config, or `config_path=` |

A `trainer.dcp/` directory holds optimizer state, not a model; point at the checkpoint directory that contains `weights.pth`.

## Load part of a model

`model_type` picks a nested model out of the config, and `prefix` keeps only the matching weights:

```python
backbone = pimm.from_pretrained(
    "DeepLearnPhysics/Panda-Semantic",
    model_type="PT-v3m2",
    prefix="backbone.",
    device="cuda",
)
```

`model_config=` or `config_path=` supply an architecture, `key_mapping=` renames weights, and `filter_fn=` drops some. With `strict=False` and `return_metadata=True` you get the missing and unexpected keys:

```python
model, meta = pimm.from_pretrained("DeepLearnPhysics/Panda-Semantic", strict=False, return_metadata=True)
print(meta["incompatible_keys"])
```

## Loading errors

| Message | Meaning |
|---|---|
| `model_config, cfg, or config_path is required` | a weight file with no config beside it; pass `config_path=` |
| `No model weights found` | the directory has no `model.safetensors` or `model.bin` |
| `No keys found with prefix` | the prefix doesn't match any key in the checkpoint |
| `... is a raw DCP trainer-state directory` | you pointed at `trainer.dcp/`; use the directory above it |
