# Fine-tune

Start a training run from released or saved weights. The launcher loads the model weights and starts the optimizer, schedule and step from zero.

## Warm-start a recipe

```bash
uv run pimm launch \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  --train.weight hf://DeepLearnPhysics/Panda-Base \
  --resources.nproc-per-node 1 \
  --run.name panda-semseg \
  -- data.train.revision=v3 data.val.revision=v3 data.test.revision=v3
```

The overrides point this recipe, which names PILArNet-M revision `v1`, at `v3`, the revision the reader accepts.

At startup the `CheckpointLoader` hook reports how many parameters loaded and lists missing and unexpected keys. Panda Base has no decoder or head, so those keys show up as missing. A mapping that loads zero parameters stops the run.

| Suffix | Trains |
|---|---|
| `-lin` | the head, with the encoder frozen |
| `-dec` | the decoder and head, with the encoder frozen |
| `-fft` | everything |
| `-scratch` | everything; run it without `--train.weight` |

## Fine-tune from a child config

```python
# configs/my_study/semseg.py
_base_ = ["../panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py"]

weight = "hf://DeepLearnPhysics/Panda-Semantic"
data = dict(
    train=dict(data_root="/data/pilarnet/v3", revision="v3"),
    val=dict(data_root="/data/pilarnet/v3", revision="v3"),
    test=dict(data_root="/data/pilarnet/v3", revision="v3"),
)
```

```bash
uv run pimm launch --train.config my_study/semseg --run.name my-semseg
```

Weights from `hf://` start new runs only: an export has no optimizer state, so `--train.resume` with an `hf://` weight is rejected.

## Attach a new head in Python

```python
import pimm
import pimm.models
from pimm.models.builder import build_model
from pimm.utils.config import Config

cfg = Config.fromfile("configs/panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py")
target = build_model(cfg.model)
source = pimm.from_pretrained("DeepLearnPhysics/Panda-Base")

missing, unexpected = target.backbone.load_state_dict(source.state_dict(), strict=False)
print("missing:", len(missing), "unexpected:", len(unexpected))
```

This builds and initializes a model; it doesn't start a pimm run. To train it with the launcher, put the weights in a recipe through `weight` and the loader's key mapping.

## Low-rank adapters

`LoRAAdapter` wraps a model, freezes it, and replaces matching `nn.Linear` layers with low-rank adapters:

```python
model = dict(
    type="LoRAAdapter",
    rank=8,
    alpha=16.0,
    dropout=0.0,
    target_modules=("attn.qkv", "attn.proj"),
    trainable_keywords=("seg_head",),
    model=dict(type="DefaultSegmentorV2", ...),
)
```

The adapters' `B` matrices start at zero, so the wrapped model initially computes the same output as the base model. `trainable_keywords` unfreezes base parameters whose names contain those strings. The wrapper raises an error if no layer matches `target_modules`. Wrapping adds a `model.` level to parameter names, so a checkpoint mapping that targeted `backbone` must target `model.backbone`.

`polarmae/semseg/semseg-polarmae-pilarnet-peft` takes another route: it sets `freeze_encoder=True` and trains only the segmentation head.

## Check what trains

```python
trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
print(sum(c for _, c in trainable), "trainable parameters")
print(*[n for n, _ in trainable[:20]], sep="\n")
```

The `ParameterCounter` hook logs the same totals at startup.
