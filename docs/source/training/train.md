# Train and pretrain

Pick a recipe, check it on a few events, then start the full run.

## Pick a recipe

Run `uv run pimm ls` to print every recipe, or `uv run pimm ls panda semseg` to list one directory. For each recipe's model, data, warm start, epochs and batch size, see the [recipe catalog](../reference/recipes.md).

| Goal | Recipe |
|---|---|
| Pretrain Panda | `panda/v1/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask` |
| Panda semantic segmentation | `panda/v1/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft` |
| Panda particle detector | `panda/v1/panseg/pt-v3m2-ft-pid-fft` |
| Panda interaction detector | `panda/v1/panseg/pt-v3m2-ft-vtx-fft` |
| Pretrain PoLAr-MAE | `polarmae/pretrain-polarmae-pilarnet` |
| PoLAr-MAE semantic segmentation | `polarmae/semseg/semseg-polarmae-pilarnet-fft` |
| JAXTPC semantic segmentation | `detector/semseg/semseg-pt-v3m2-jaxtpc-5cls` |

To fine-tune with a recipe, pass the starting weights with `--train.weight`; see [Fine-tune](../models/fine-tune.md).

Each recipe has a PILArNet-M revision. `v3` and `v3_extra` are the supported revisions; to run a recipe set to `v1`, point each split at v3 with an override, such as `data.train.revision=v3 data.val.revision=v3`.

## Check it on a few events

```bash
uv run pimm launch \
  --train.config panda/v1/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  --resources.nproc-per-node 1 \
  --run.name sonata-smoke \
  -- epoch=1 data.train.max_len=32 data.val.max_len=16 batch_size=4 num_worker=0 use_wandb=False \
     data.train.revision=v3 data.val.revision=v3
```

This command requires PILArNet-M v3 on disk (see [Supported detectors](../data/datasets.md#pilarnet-m)). Run the command once with `--dry-run` first. When the run finishes, you'll find `model/last/.complete` in the run directory.

## Keep a variant as a child config

```python
# configs/my_study/sonata_v1.py
_base_ = ["../panda/v1/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py"]

seed = 17
batch_size = 32
epoch = 100
optimizer = dict(lr=3e-5, weight_decay=0.2)
data = dict(train=dict(revision="v3"), val=dict(revision="v3"))
```

Dicts are merged with the base and lists are replaced, so if you redefine `hooks` or `transform`, you replace the whole list.

## Start the run

```bash
uv run pimm launch --train.config my_study/sonata_v1 --resources.nproc-per-node 4 --run.name sonata-v1
```

At launch, a timestamp is appended to the run name, and `pimm/` and `configs/` are copied into the run. Pass `--run.no-timestamp` for a fixed directory you can [resume](checkpoints.md#resume-a-run), and `--train.no-code-copy` to train from the checkout instead of the copy.

## From a notebook

Start training as a child process to get the launcher's setup:

```text
!uv run pimm launch --resources.nproc-per-node 1 --train.config tests/tiny_semseg --run.name notebook-smoke
```

A local launch stays attached to the notebook kernel; use `pimm submit` for a run that should outlive the kernel. You can load data, transforms and models for inference in ordinary notebook cells.
