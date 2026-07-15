# Parameter-efficient fine-tuning

**Goal:** freeze a pretrained model, inject low-rank adapters into selected
linear projections, train only adapters plus an explicit task head, and verify
that the resulting parameter set matches the experiment claim.

pimm's {py:class}`~pimm.models.lora.LoRAAdapter` is a registered model wrapper.
{py:class}`PoLAr-MAE <pimm.models.polarmae.polarmae.PoLArMAE>` also has a
head-only frozen-encoder recipe; that is
parameter-efficient fine-tuning but does not use the LoRA wrapper.

## How the LoRA wrapper behaves

1. build the nested model;
2. freeze every base parameter;
3. replace matching `nn.Linear` modules with low-rank adapter layers;
4. enable the new `lora_A`/`lora_B` parameters;
5. optionally re-enable base parameters whose names match
   `trainable_keywords`.

Each adapter adds the scaled low-rank update
{math}`y = W_0x + b_0 + \frac{\alpha}{r}B(Ax)`, where {math}`r` is the adapter
rank and {math}`\alpha` controls its scale.

{math}`B` starts at zero, so injection is initially a no-op relative to the
loaded base model.

## Configure the wrapper

```python
model = dict(
    type="LoRAAdapter",
    rank=8,
    alpha=16.0,
    dropout=0.0,
    target_modules=("attn.qkv", "attn.proj"),
    trainable_keywords=("seg_head",),
    model=dict(
        type="DefaultSegmentorV2",
        num_classes=5,
        backbone_out_channels=64,
        backbone=dict(
            type="PT-v3m2",
            in_channels=4,
            # copy the complete compatible backbone config
        ),
        criteria=[dict(type="CrossEntropyLoss", ignore_index=-1)],
    ),
)
```

The inner task model above is
{py:class}`~pimm.models.default.DefaultSegmentorV2`; the `backbone_out_channels`
value must match the final decoder width in the complete PT-v3m2 backbone
configuration.

| Field | Meaning |
|---|---|
| `rank` | low-rank dimension; must be positive |
| `alpha` | adapter scale numerator |
| `dropout` | LoRA branch input dropout |
| `target_modules` | full-name or dotted-suffix matches for `nn.Linear` modules |
| `trainable_keywords` | additional nested parameter-name substrings to unfreeze |

The wrapper raises if no linear module matches. That prevents a plausible-looking
run with no adapters.

## Checkpoint mapping changes under the wrapper

Wrapping adds a `model.` level. A
{py:class}`~pimm.engines.hooks.checkpoint.CheckpointLoader` rule that previously
targeted `backbone` may now need `model.backbone`:

```python
dict(
    type="CheckpointLoader",
    keywords="module.student.backbone",
    replacement="module.model.backbone",
)
```

Use the mapping required by the exact source checkpoint and target config.
Inspect the grouped key report rather than assuming this generic example
matches every export.

## Assert the trainable set

Add {py:class}`~pimm.engines.hooks.diagnostics.ParameterCounter`, then
programmatically inspect names:

```python
names = [name for name, p in model.named_parameters() if p.requires_grad]
assert any("lora_A" in name for name in names)
assert any("lora_B" in name for name in names)
assert all(
    ("lora_A" in name or "lora_B" in name or "seg_head" in name)
    for name in names
)
print(*names, sep="\n")
```

Adapt this assertion to the chosen task head. Record the absolute and fractional
trainable parameter counts.

## Make a smoke run

```bash
uv run pimm launch \
  --train.config my_study/semseg_ptv3_lora \
  --train.weight 'hf://DeepLearnPhysics/Panda-Base' \
  --resources.nproc-per-node 1 \
  --run.name semseg-lora-smoke \
  -- \
  epoch=1 \
  data.train.max_len=32 \
  data.val.max_len=16 \
  batch_size=4 \
  num_worker=0 \
  use_wandb=False
```

Check base-weight loading, adapter count, trainable names, nonzero adapter
gradients, unchanged frozen weights after one optimizer step, validation, and
checkpoint/export round-trip.

Do not assume LoRA permits a larger batch or learning rate. Measure memory and
tune optimization as a new experiment.

## Head-only PoLAr-MAE alternative

`configs/polarmae/semseg/semseg-polarmae-pilarnet-peft.py` sets
`freeze_encoder=True` on
{py:class}`~pimm.models.polarmae.polarmae_semseg.PoLArMAESemSeg` and trains the
segmentation head. It uses PILArNet v1 and its own checkpoint-key mapping. Treat
its preprocessing and metric claim as specific to that recipe; do not combine
it with the Panda LoRA wrapper by analogy.

## Compare fairly

Compare PEFT, full fine-tuning, and a verified no-weight scratch run using identical data, evaluation,
seed set, and reporting. Include trainable/total parameters, peak memory,
training time, and final metric uncertainty. A lower trainable count is not by
itself evidence of better efficiency or transfer.

## Next

- {doc}`Fine-tuning workflow <../workflows/fine_tune>`.
- {doc}`Semantic segmentation <semantic_segmentation>`.
- {doc}`Export <../models/export>`.
