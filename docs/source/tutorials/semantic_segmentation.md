# Semantic segmentation

**Goal:** train a Point Transformer to assign one class to every detector hit,
starting with the tiny verified path and ending with a reviewable child config.

This tutorial uses PILArNet-M terminology. For another detector, first implement
and validate the reader in {doc}`Custom data <../data/custom>`.

## 1. Verify the batch contract

A semantic model needs:

```text
coord    float tensor  (total_points, 3)
feat     float tensor  (total_points, channels)
offset   integer tensor (batch_size,)
segment  integer tensor (total_points,)
```

Run {doc}`First experiment <../getting_started/quickstart>` through completion.
It proves that the dataset, transforms,
{py:class}`~pimm.models.default.DefaultSegmentorV2`, evaluator, and checkpoint
path work on a small 80/20-event split.

## 2. Read the tiny config

`configs/tests/tiny_semseg.py` is intentionally small but complete:

```python
_base_ = ["../_base_/default_runtime.py"]

batch_size = 4       # global across ranks
batch_size_val = 1
num_worker = 0
epoch = 1

model = dict(
    type="DefaultSegmentorV2",
    num_classes=5,
    backbone_out_channels=8,
    # The committed tiny config supplies the complete PT-v3m2 architecture.
    backbone=dict(type="PT-v3m2", in_channels=4),
    criteria=[dict(type="CrossEntropyLoss", ignore_index=-1)],
)
```

Its transform pipeline creates four input channels from normalized coordinates
and log-transformed energy, copies `segment_motif` to `segment`, and creates
`grid_coord` for PT-v3.

## 3. Choose an initialization

| Recipe | Intended initialization | Trainable scope |
|---|---|---|
| `semseg-pt-v3m2-pilarnet-ft-5cls-scratch` | no effective weight | task model from scratch |
| `...-lin` | supply a compatible pretrained backbone | restricted/linear-probe recipe |
| `...-dec` | supply a compatible pretrained backbone | decoder/head-focused recipe |
| `...-fft` | supply a compatible pretrained backbone | full fine-tuning |

The current `-scratch` child changes only W&B naming; it does not clear or
reject a launcher-supplied weight. For a real random-initialization baseline,
omit `--train.weight`, confirm the resolved `weight` is `None`, and check that
startup logs contain no checkpoint load. Add pretrained initialization only
after one batch and one optimizer step are correct.

## 4. Create a child config

```python
# configs/my_study/semseg_v1.py
_base_ = ["../panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-scratch.py"]

seed = 11
batch_size = 16
epoch = 50

data = dict(
    train=dict(max_len=100_000),
    val=dict(max_len=5_000),
    test=dict(max_len=5_000),
)
```

Launch this child without `--train.weight` for the intended scratch baseline.

If using custom data, replace the dataset type, root, class names/count,
transforms, and model class count together. A stale inherited PILArNet key can
survive a partial dictionary merge, so inspect `resolved_config.json` after a
dry run.

## 5. Smoke-test the real recipe

```bash
uv run pimm launch \
  --train.config my_study/semseg_v1 \
  --resources.nproc-per-node 1 \
  --run.name semseg-v1-smoke \
  -- \
  epoch=1 \
  data.train.max_len=32 \
  data.val.max_len=16 \
  batch_size=4 \
  num_worker=0 \
  use_wandb=False
```

Check:

- `feat.shape[1]` equals the backbone `in_channels`;
- target IDs are in `[0, num_classes)` or equal `ignore_index`;
- each loss component is finite;
- trainable parameter count matches the intended strategy;
- {py:class}`~pimm.engines.hooks.eval.semantic_segmentation.SemSegEvaluator`
  reports per-class statistics and mean intersection over union (mIoU);
- `model_best.pth` and a complete `model/last` exist.

## 6. Warm-start, if required

```bash
uv run pimm launch \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  --train.weight 'hf://DeepLearnPhysics/Panda-Base' \
  --resources.nproc-per-node 1 \
  --run.name semseg-fft-smoke
```

Use the exact URI and mapping from the {doc}`model card/chooser
<../models/index>`. Confirm that backbone keys loaded and that only expected
task-head keys are missing. Save the startup load report.

## 7. Scale without changing the science config

```bash
uv run pimm launch \
  --train.config my_study/semseg_v1 \
  --resources.nproc-per-node 4 \
  --run.name semseg-v1-seed11
```

The global batch must divide four. Use {doc}`Distributed training
<../workflows/distributed>` for rank/worker and resume semantics.

## 8. Evaluate and interpret

Semantic segmentation commonly reports per-class IoU and mean IoU. Also report
class support, ignored-point handling, point/event weighting, and dataset split.
A strong aggregate can hide a failed rare class; retain the full per-class
table and raw enough statistics to recompute it.

```bash
uv run sh scripts/test.sh -c my_study/semseg_v1 -n <run-name> -w model_best
```

This convenience command evaluates with the current checkout's config. See
{doc}`Evaluation <../workflows/evaluate>` for the saved-config/saved-code
command and provenance limitations of the current standalone tester.

## Next

- Inspect released semantic and instance outputs: {doc}`Explore Panda
  <explore_panda>`.
- Cheaper adaptation: {doc}`PEFT <peft>`.
- Publish the trained model: {doc}`Export <../models/export>`.
