# Train and pretrain

Pick a recipe, check it on a few events, then start the full run.

## Pick a recipe

`uv run pimm ls` prints every recipe; `uv run pimm ls panda semseg` lists one directory. The [recipe catalog](../reference/recipes.md) shows each recipe's model, data, warm start, epochs and batch size.

| Goal | Recipe |
|---|---|
| Pretrain Panda | `panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask` |
| Panda semantic segmentation | `panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft` |
| Panda particle detector | `panda/panseg/detector-v5-pt-v3m2-ft-pid-fft` |
| Panda interaction detector | `panda/panseg/detector-v5-pt-v3m2-ft-vtx-fft` |
| Pretrain PoLAr-MAE | `polarmae/pretrain-polarmae-pilarnet` |
| PoLAr-MAE semantic segmentation | `polarmae/semseg/semseg-polarmae-pilarnet-fft` |
| JAXTPC semantic segmentation | `detector/semseg/semseg-pt-v3m2-jaxtpc-5cls` |

Recipes that fine-tune take their starting weights from `--train.weight`; see [Fine-tune](../models/fine-tune.md).

Each recipe names a PILArNet-M revision. The reader accepts `v3` and `v3_extra`; recipes that name `v1` run on v3 data with one override per split, such as `data.train.revision=v3 data.val.revision=v3`.

## Check it on a few events

```bash
uv run pimm launch \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  --resources.nproc-per-node 1 \
  --run.name sonata-smoke \
  -- epoch=1 data.train.max_len=32 data.val.max_len=16 batch_size=4 num_worker=0 use_wandb=False \
     data.train.revision=v3 data.val.revision=v3
```

This needs PILArNet-M v3 on disk (see [Supported detectors](../data/datasets.md#pilarnet-m)). Run it once with `--dry-run` first. When it finishes, `model/last/.complete` exists in the run directory.

## Keep a variant as a child config

```python
# configs/my_study/sonata_v1.py
_base_ = ["../panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py"]

seed = 17
batch_size = 32
epoch = 100
optimizer = dict(lr=3e-5, weight_decay=0.2)
data = dict(train=dict(revision="v3"), val=dict(revision="v3"))
```

Dicts merge with the base; lists replace it. Redefining `hooks` or `transform` replaces the whole list.

## Start the run

```bash
uv run pimm launch --train.config my_study/sonata_v1 --resources.nproc-per-node 4 --run.name sonata-v1
```

The launcher appends a timestamp to the run name and copies `pimm/` and `configs/` into the run. Pass `--run.no-timestamp` for a fixed directory you can [resume](checkpoints.md#resume-a-run), and `--train.no-code-copy` to train from the checkout instead of the copy.

## From a notebook

Start training as a child process so it gets the launcher's setup:

```text
!uv run pimm launch --resources.nproc-per-node 1 --train.config tests/tiny_semseg --run.name notebook-smoke
```

A local launch stays attached to the notebook kernel; use `pimm submit` for a run that should outlive it. Loading data, transforms and models for inference works in ordinary notebook cells.
