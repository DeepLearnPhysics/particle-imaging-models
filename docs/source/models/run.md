# Run a released model

Call `pimm.from_pretrained` to download an export, rebuild the model, load its weights, and get the model back on a device in eval mode.

```python
import pimm

model = pimm.from_pretrained("DeepLearnPhysics/Panda-Semantic", device="cuda")
```

The default `device` is `"cpu"`. Downloads are stored in the Hugging Face cache (`HF_HUB_CACHE`); pass `cache_dir=` to choose another directory and `revision=` to pin a branch, tag or commit.

## Label one event

Here you label test event 0 of PILArNet-M-mini (see [First run](../get-started/first-run.md) for the download) with Panda Semantic. Panda needs CUDA.

```python
import torch

import pimm
from pimm.datasets.pilarnet import PILArNetH5Dataset
from pimm.datasets.transform import Compose
from pimm.datasets.utils import collate_fn

model = pimm.from_pretrained("DeepLearnPhysics/Panda-Semantic", device="cuda")

transform = Compose([
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="LogTransform", min_val=0.01, max_val=20.0, keys=("energy",)),
    dict(type="GridSample", grid_size=0.001, hash_type="fnv", mode="deterministic", return_grid_coord=True,
         return_inverse=True),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "inverse"), feat_keys=("coord", "energy"),
         offset_keys_dict=dict(offset="coord", origin_offset="inverse")),
])

events = PILArNetH5Dataset(data_root="data/PILArNet-M-mini", split="test",
                           revision="v3", transform=[])
event = events.get_data(0)

sample = transform({"coord": event["coord"].copy(), "energy": event["energy"].copy()})
batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in collate_fn([sample]).items()}

with torch.inference_mode():
    output = model(batch)

labels = output["seg_logits"].argmax(-1)
names = ["shower", "track", "Michel", "delta", "low-energy deposit"]
print({names[i]: int((labels == i).sum()) for i in range(5)})
```

To label several events at once, transform each one and pass the list to `collate_fn` to concatenate the events and build `offset`. You get a label for every original point of each event, in input order; every point is given the prediction of its `GridSample` cell, through `inverse`.

## Outputs

The output of a semantic model is `seg_logits`, one row per original point, and `offset`, the cumulative point count at the end of each event. Panda Semantic needs `inverse` in the batch, as in the example above; without it, you get a `KeyError`.

The outputs of Panda Particle and Panda Interaction are raw query predictions. Call `model.postprocess` to turn these predictions into per-point assignments:

```python
with torch.inference_mode():
    prediction = model.postprocess(model(batch))

prediction["instance_labels"]  # (N,) instance id of each point, -1 for none
prediction["class_labels"]     # (N,) class of each point
prediction["confidences"]      # (N,)
prediction["query_labels"]     # (N,) query that produced the point's instance, -1 for none
```

Instance ids are unique across the batch. You can pass `mask_threshold`, `conf_threshold` and `min_points` to `postprocess`.

The output of Panda Base is a `Point`, and `point.feat` contains the features. The outputs of PoLAr-MAE Pretrain are its loss terms.

## Sources

| Source | Example | Architecture read from |
|---|---|---|
| Hugging Face repository | `DeepLearnPhysics/Panda-Semantic` or `hf://DeepLearnPhysics/Panda-Semantic` | the repository's `config.json` |
| Local export | `artifacts/my-model` | its `config.json` |
| Run checkpoint | `exp/<group>/<run>/model/model_best.pth` | the run's config, or `config_path=` |

A `trainer.dcp/` directory contains optimizer state, not a model; point at the checkpoint directory that contains `weights.pth`.

## Load part of a model

Use `model_type` to pick a nested model out of the config and `prefix` to keep only the matching weights:

```python
backbone = pimm.from_pretrained(
    "DeepLearnPhysics/Panda-Semantic",
    model_type="PT-v3m2",
    prefix="backbone.",
    device="cuda",
)
```

Pass `model_config=` or `config_path=` to supply an architecture, `key_mapping=` to rename weights, and `filter_fn=` to drop some of them. With `strict=False` and `return_metadata=True` you get the missing and unexpected keys:

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
