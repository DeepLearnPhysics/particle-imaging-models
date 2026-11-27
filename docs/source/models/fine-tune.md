# Fine-tune

Start a training run from released or saved weights. The model weights are loaded, and the optimizer, schedule and step start from zero.

## Warm-start a recipe

```bash
uv run pimm launch \
  --train.config panda/v1/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  --train.weight hf://DeepLearnPhysics/Panda-Base \
  --resources.nproc-per-node 1 \
  --run.name panda-semseg \
  -- data.train.revision=v3 data.val.revision=v3 data.test.revision=v3
```

The recipe's PILArNet-M revision is `v1`, so you override it with `v3`, a supported revision.

At startup, the log has a `CheckpointLoader` report: the number of loaded parameters and the missing and unexpected keys. Panda Base has no decoder or head, so those keys are listed as missing. If zero parameters are loaded through a mapping, the run is stopped.

| Suffix | Trained |
|---|---|
| `-lin` | the head, with the encoder frozen |
| `-dec` | the decoder and head, with the encoder frozen |
| `-fft` | everything |
| `-scratch` | everything; run it without `--train.weight` |

## Fine-tune from a child config

```python
# configs/my_study/semseg.py
_base_ = ["../panda/v1/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py"]

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

You can use weights from `hf://` only to start new runs: an export has no optimizer state, so `--train.resume` with an `hf://` weight is rejected.

## Attach a new head in Python

```python
import pimm
import pimm.models
from pimm.models.builder import build_model
from pimm.utils.config import Config

cfg = Config.fromfile("configs/panda/v1/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py")
target = build_model(cfg.model)
source = pimm.from_pretrained("DeepLearnPhysics/Panda-Base")

missing, unexpected = target.backbone.load_state_dict(source.state_dict(), strict=False)
print("missing:", len(missing), "unexpected:", len(unexpected))
```

Here you build and initialize a model without starting a pimm run. To train the model with the launcher, put the weights in a recipe through `weight` and the loader's key mapping.

## Low-rank adapters

Wrap a model in `LoRAAdapter` to freeze it and replace matching `nn.Linear` layers with low-rank adapters:

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

The adapters' `B` matrices are initialized to zero, so at first the wrapped model's output is the same as the base model's. List strings in `trainable_keywords` to unfreeze base parameters whose names contain them. If no layer matches `target_modules`, you get an error. Wrapped parameter names have an extra `model.` level, so in a checkpoint mapping, replace `backbone` with `model.backbone`.

A different route is in `polarmae/semseg/semseg-polarmae-pilarnet-peft`: that recipe has `freeze_encoder=True`, so only the segmentation head is trained.

## Check which parameters are trained

```python
trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
print(sum(c for _, c in trainable), "trainable parameters")
print(*[n for n, _ in trainable[:20]], sep="\n")
```

You'll also find the same totals in the startup log, from the `ParameterCounter` hook.
