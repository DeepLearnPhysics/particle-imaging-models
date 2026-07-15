# Load, run, and fine-tune a model

Use {py:func}`~pimm.from_pretrained` when you want an instantiated model, not
just a state dict. The function resolves local or Hub weights, reconstructs the
architecture, loads the weights, moves the model to `device`, calls `eval()`,
and returns it.

```python
import torch
import pimm

device = "cuda"
model = pimm.from_pretrained(
    "DeepLearnPhysics/Panda-Semantic",
    device=device,
)  # Weights are cached under HF_HUB_CACHE when that variable is set.

def predict_event(coord, grid_coord, energy):
    """Run one preprocessed event containing N retained detector hits."""
    coord = coord.to(device)            # (N, 3), float32, normalized
    grid_coord = grid_coord.to(device)  # (N, 3), integer voxel coordinates
    energy = energy.to(device)          # (N, 1), float32, log-transformed

    input_dict = {
        "coord": coord,
        "grid_coord": grid_coord,
        "feat": torch.cat((coord, energy), dim=-1),  # (N, 4)
        "offset": torch.tensor(
            [coord.shape[0]],
            dtype=torch.int32,
            device=device,
        ),
    }

    with torch.inference_mode():
        output = model(input_dict)

    logits = output["seg_logits"]       # (N, 5)
    predicted_class = logits.argmax(-1)  # (N,)
    return predicted_class, logits

# coord, grid_coord, and energy come from the Panda inference transform below.
predicted_class, logits = predict_event(coord, grid_coord, energy)
```

`device` defaults to `"cpu"`. Pass `None` to skip the final `.to(device)` call;
the returned model is still in eval mode. `coord` and `energy` in this first
example are the transformed tensors, not the raw detector arrays; the complete
conversion is shown in [Panda input](#panda-input).

| I want to… | Go to |
|---|---|
| See the released checkpoints | [Model chooser](index.md#choose-a-model) |
| Run a semantic model | [Semantic outputs](#semantic-segmentation) |
| Run a Panda detector | [Panoptic outputs](#panda-panoptic-models) |
| Reconstruct a different architecture | [Control model construction](#control-model-construction) |
| Start a new training run from weights | [Fine-tune](#fine-tune) |

## What can be loaded?

| Source | Example | Where the architecture comes from |
|---|---|---|
| Hub export | `DeepLearnPhysics/Panda-Semantic` | The repository's `config.json` |
| Hub export with scheme | `hf://DeepLearnPhysics/Panda-Semantic` | The repository's `config.json` |
| Hub export at a revision | `hf://org/model` with `revision="v1"` | That revision's `config.json` |
| Local export directory | `artifacts/my-model` | A config in the directory |
| Local weight file | `exp/run/model/model_best.pth` | `model_config`, `config_path`, or an adjacent run config |

For a Hub repository, pimm downloads only the portable weights, recognized
config names, and model card. For a bare `hf://` repository that contains raw
checkpoints instead of a portable export, it chooses `model_best.pth`, then
`model_last.pth`, or the only top-level `.pth`. If a repository remains
ambiguous, download the intended checkpoint and pass its local path.

Repository snapshots use an explicit `cache_dir` when supplied. Otherwise the
loader respects `HF_HUB_CACHE`/`HF_HOME`, then uses pimm's shared model cache or
the Hugging Face default.

```python
model = pimm.from_pretrained(
    "DeepLearnPhysics/Panda-Semantic",
    cache_dir="/scratch/alice/huggingface",
    revision="main",                         # branch, tag, or commit
    device="cuda:0",
)
```

:::{warning}
A raw `trainer.dcp/` directory is trainer state, not a portable model. Point at
the surrounding split checkpoint (the directory containing `weights.pth`) or
[export it first](export.md).
:::

## Prepare an input batch

All released models consume pimm's packed point-cloud dictionary. At minimum,
the task models expect:

| Key | Shape | Meaning |
|---|---|---|
| `coord` | `(N, 3)` | Transformed point coordinates |
| `feat` | `(N, C)` | Concatenated model features; released models use coordinate + energy |
| `offset` | `(B,)` | Cumulative point count at the end of each event |
| `grid_coord` | `(N, 3)` | Integer grid coordinates required by Panda/PTv3 models |

Do not stack ragged events. Transform each event independently, then use
{py:func}`~pimm.datasets.utils.collate_fn` to concatenate them and construct the
cumulative `offset`.

### Panda input

The published Panda recipes use fixed PILArNet coordinates, log-energy with
`min_val=0.01`, and a `0.001` grid in normalized coordinates. This is the
minimal inference pipeline, with label-only transforms removed:

```python
import numpy as np
import torch

from pimm.datasets.transform import Compose
from pimm.datasets.utils import collate_fn

panda_pipeline = Compose([
    dict(
        type="NormalizeCoord",
        center=[384.0, 384.0, 384.0],
        scale=768.0 * 3**0.5 / 2,
    ),
    dict(type="LogTransform", min_val=0.01, max_val=20.0, keys=("energy",)),
    dict(
        type="GridSample",
        grid_size=0.001,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord"),
        feat_keys=("coord", "energy"),
    ),
])

def panda_batch(events, device="cuda"):
    """events: iterable of {'coord': (N, 3), 'energy': (N, 1)} arrays."""
    samples = [
        panda_pipeline({
            "coord": np.asarray(event["coord"], dtype=np.float32).copy(),
            "energy": np.asarray(event["energy"], dtype=np.float32).reshape(-1, 1).copy(),
        })
        for event in events
    ]
    batch = collate_fn(samples)
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
```

{py:class}`~pimm.datasets.transform.spatial.GridSample` with `mode="train"` is
the setting in the release evaluation configs.
It selects one representative when multiple points occupy a grid cell; seed
NumPy if that selection must be reproducible. Predictions are aligned to the
points retained by this pipeline, not automatically projected back to the raw
event.

### PoLAr-MAE input

The released PoLAr-MAE semantic model performs its fixed coordinate centering
and scaling inside the model. Its evaluation recipe log-transforms energy with
`min_val=0.13` and does **not** grid-sample:

```python
polar_semantic_pipeline = Compose([
    dict(type="LogTransform", min_val=0.13, max_val=20.0),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord",), feat_keys=("coord", "energy")),
])
```

The pretrained PoLAr-MAE release has a different contract: its recipe applies
`NormalizeCoord(center=[384, 384, 384], scale=665.1076)` and
`LogTransform(min_val=0.01, max_val=20.0)` before collation. Copy the recipe for
the specific model rather than inferring preprocessing from its architecture.

:::{important}
The released Hub repositories' `config.json` files currently carry model
construction only. A newly exported full run config may retain a `data`
section, but {py:func}`~pimm.from_pretrained` neither builds nor applies that
pipeline. Treat
transforms, label names, energy thresholds, and postprocessing as a separate,
versioned scientific contract in the model card and evaluation recipe.
:::

## Interpret the output

{py:func}`~pimm.from_pretrained` already puts the model in eval mode. Use inference mode as
well so PyTorch does not retain autograd state:

```python
batch = panda_batch([event])

with torch.inference_mode():
    output = model(batch)
```

### Semantic segmentation

Panda Semantic and PoLAr-MAE Semantic return a dictionary with packed,
per-point logits:

```python
logits = output["seg_logits"]  # (N_total, num_classes)
labels = logits.argmax(dim=-1)
```

Panda Semantic uses five classes in this order: shower, track, Michel, delta,
LED. PoLAr-MAE Semantic uses four: shower, track, Michel, delta. If a `segment`
target is present in the input while the model is in eval mode, the output also
contains `loss`.

### Panda panoptic models

Panda Particle and Panda Interaction return raw query predictions. Convert them
to point-level assignments with the loaded model's own postprocessor:

```python
with torch.inference_mode():
    raw = model(batch)
    prediction = model.postprocess(raw)

instance = prediction["instance_labels"]  # (N_total,), -1 for stuff/uncovered
category = prediction["class_labels"]     # (N_total,)
confidence = prediction["confidences"]    # (N_total,)
query = prediction["query_labels"]        # (N_total,), source query or -1
```

Instance IDs are offset to remain unique across the packed batch. Optional
regression heads appear under `instance_regression` and as `instance_<name>` /
`pred_<name>` aliases. Thresholds such as `mask_threshold`, `conf_threshold`,
and `min_points` can be passed to `model.postprocess(...)`; start with the
loaded model's effective defaults and state any override in an analysis record.

The raw dictionary exposes `pred_logits`, `pred_masks`, `seg_logits`,
`stuff_probs`, and `point_counts` for the primary task, plus label-specific
maps. These are useful for calibration or custom postprocessing; most analyses
should start from `postprocess()`.

### Representation and pretraining models

Panda Base is an encoder-only
{py:class}`PT-v3m2 <pimm.models.point_transformer_v3.point_transformer_v3m2_sonata.PointTransformerV3>`.
Its forward pass returns a
{py:class}`~pimm.models.utils.structure.Point`; `point.feat` contains the encoded
representation. It has no semantic or instance head.

PoLAr-MAE Pretrain executes its masked reconstruction objective and returns
`loss`, `chamfer_loss`, `energy_loss`, `mean_points`, and `mean_groups`. It is a
pretraining artifact, not a ready-made downstream predictor.

## Understand model construction

### Config selection

For a normal portable export, no config argument is needed. When you do supply
one, the effective precedence is:

1. `model_config={...}`
2. `config_path="..."` (when explicitly supplied)
3. a recognized config inside the export (`config.json`, then legacy names)
4. a recognized config or `config.py` inferred beside a local run/checkpoint

If a file contains a full training config, pimm extracts its top-level `model`
mapping. Keyword arguments not claimed by {py:func}`~pimm.from_pretrained` are
applied last as model-constructor overrides.

```python
model = pimm.from_pretrained(
    "weights/model_best.pth",
    config_path="configs/panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py",
    device="cuda",
)
```

The loader tolerates one narrow form of config drift: if construction fails
because a saved keyword is no longer accepted, it warns, drops unsupported
constructor keywords, and retries. Missing registry types, changed tensor
shapes, and other incompatibilities still fail visibly.

### Control model construction

Use `model_type` to select the first nested model config with that registry
type. This example extracts the PTv3 backbone from a full semantic export and
loads only keys below `backbone.`:

```python
backbone = pimm.from_pretrained(
    "DeepLearnPhysics/Panda-Semantic",
    model_type="PT-v3m2",
    prefix="backbone.",
    remove_prefix=True,
    device="cuda",
)
```

If no nested config matches, `model_type` replaces the root config's `type`.
`model_cls=MyModel` bypasses the registry and constructs that class from the
selected config (without its `type` field).

Weight transformations happen in this order:

1. `prefix` filters keys and optionally removes the prefix.
2. `key_mapping` applies exact or prefix rewrites. Unmapped keys are dropped by
   default; set `keep_unmapped_keys=True` to retain them.
3. `filter_fn(state_dict, model)` performs any project-specific filtering.
4. `model.load_state_dict(..., strict=strict)` loads the result.

### Inspect what was loaded

```python
model, metadata = pimm.from_pretrained(
    "DeepLearnPhysics/Panda-Semantic",
    strict=False,
    return_metadata=True,
)
```

| Metadata key | Value |
|---|---|
| `model_config` | Final construction config, after model selection and keyword overrides |
| `source_model_config` | Model config read before `model_type` selection and overrides |
| `path` | Resolved local export path or file |
| `weights` | Exact weight file loaded |
| `incompatible_keys` | Missing/unexpected keys; included only when `strict=False` |
| `config` | Reserved metadata field; currently `{}` |

`strict=True` is the default and is the right choice for ordinary inference.
Use `strict=False` only when missing and unused keys are expected and inspected.

## Fine-tune

There are two distinct workflows.

### Continue from a compatible task model

For domain adaptation with the same architecture and head, warm-start the
training config from a Hub export. Put this child config at
`configs/my_semantic_domain.py`:

```python
_base_ = ["./panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py"]

weight = "hf://DeepLearnPhysics/Panda-Semantic"
resume = False

# Replace with your dataset settings. Lists replace their base value;
# dictionaries merge recursively.
data = dict(
    train=dict(data_root="/path/to/train"),
    val=dict(data_root="/path/to/val"),
)
```

Then launch a new run:

```bash
uv run pimm launch --train.config my_semantic_domain --run.name my-semantic-domain
```

The inherited {py:class}`~pimm.engines.hooks.checkpoint.CheckpointLoader` loads
model weights only. It does not restore
optimizer, scheduler, step, RNG, or dataloader state. Check the reported loaded,
missing, and unexpected key counts before trusting the run.

:::{warning}
Do not combine a Hub weight with resume mode. Hub exports intentionally omit
trainer state; `resume=True` with a Hub weight is rejected.
:::

### Attach a new head to a backbone

For a different task, build the target model, load the backbone explicitly,
inspect incompatibilities, then train the target. Panda Base is encoder-only,
whereas the semantic recipe adds decoder stages, so missing decoder keys are
expected:

```python
import pimm
import pimm.models  # populate model registries

from pimm.models.builder import build_model
from pimm.utils.config import Config

cfg = Config.fromfile(
    "configs/panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py"
)
target = build_model(cfg.model)
source = pimm.from_pretrained("DeepLearnPhysics/Panda-Base")

missing, unexpected = target.backbone.load_state_dict(
    source.state_dict(),
    strict=False,
)
print("missing:", missing)
print("unexpected:", unexpected)

target.train()
```

This snippet only initializes a model; it does not create a pimm training run.
For launcher-based training, add a small hook that loads the source directly
into `model.backbone`, or save a mapped state dict whose keys carry the target's
`backbone.` prefix. Keep that mapping in the experiment config so it is
reviewable; the stock empty
{py:class}`~pimm.engines.hooks.checkpoint.CheckpointLoader` rule does not add a
prefix. See {py:func}`~pimm.models.builder.build_model` and
{py:class}`~pimm.utils.config.Config` for the two construction helpers used
above.

## Troubleshoot loading

`model_config, cfg, or config_path is required`
: The input is a raw weight file with no discoverable run config. Supply
  `config_path` or a model mapping.

`No model weights found`
: The directory or Hub snapshot has no `model.safetensors` or `model.bin`.
  Export it, use an unambiguous bare Hub repository, or pass the intended raw
  checkpoint as a local path.

`No keys found with prefix`
: The prefix describes a module layout that is not in this checkpoint. Inspect
  a few `state_dict` keys and correct the prefix instead of weakening strictness.

Size mismatch with `strict=False`
: PyTorch still rejects same-name tensors with different shapes. Drop only the
  known incompatible tensors in a `filter_fn`, then inspect
  `metadata["incompatible_keys"]`:

```python
def compatible_tensors(state_dict, model):
    target = model.state_dict()
    return {
        key: value
        for key, value in state_dict.items()
        if key in target and target[key].shape == value.shape
    }

model, metadata = pimm.from_pretrained(
    "path-or-repo",
    model_config=my_model_config,
    filter_fn=compatible_tensors,
    strict=False,
    return_metadata=True,
)
print(metadata["incompatible_keys"])
```

Predictions look plausible but wrong
: Verify the model repository, class order, energy transform, coordinate frame,
  grid sampling, and postprocessing settings. Architecture reconstruction alone
  does not reproduce the scientific input contract.
