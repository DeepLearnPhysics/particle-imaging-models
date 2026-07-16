# Train or pretrain

**Outcome:** choose an existing recipe, validate its data and resolved config on
a small subset, then start a named research run.

Start with {doc}`the first experiment <../getting_started/quickstart>` if you
have not completed it yet.

## 1. Choose by outcome

| Outcome | Good starting configuration | Notes |
|---|---|---|
| Panda/Sonata pretraining | `panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask` | PT-v3m2 backbone on PILArNet-M |
| Panda semantic segmentation | `panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft` | full fine-tune; requires compatible pretrained weight |
| Panda semantic segmentation from scratch | `panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-scratch` | intended no-weight baseline; verify effective `weight=None` |
| Panda Detector from Panda Base | `panda/panseg/detector-v5-pt-v3m2-ft-pid-fft` | full fine-tune after loading pretrained backbone weights |
| Further fine-tune Panda Particle | `panda/panseg/detector-v5-pt-v3m2-ft-pid-fft-detector` | strictly loads the published backbone and task decoder |
| PoLAr-MAE pretraining | `polarmae/pretrain-polarmae-pilarnet` | masked autoencoding |
| PoLAr-MAE semantic segmentation | `polarmae/semseg/semseg-polarmae-pilarnet-fft` | downstream semantic task |

The {doc}`model chooser <../models/index>` lists published artifacts. The
{doc}`config catalog <../reference/configuration>` lists every committed recipe
without implying that each is a supported benchmark.

## 2. Resolve the data contract

Read the config from bottom to top:

1. `data.train/val/test.type` chooses the dataset implementation.
2. `revision` and `split` choose the dataset version and partition.
3. the transform list defines coordinates, feature order, target copies,
   normalization, voxelization, and augmentation;
4. `model.in_channels`, target names, and loss settings must agree with the
   transformed batch.

Do not substitute a data revision merely because the file opens. Compare every
field against {doc}`Data conventions <../data/conventions>` and the checkpoint's
saved `config.json` or resolved run config.

## 3. Make a bounded smoke run

Use CLI overrides for temporary limits:

```bash
uv run pimm launch \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  --resources.nproc-per-node 1 \
  --run.name sonata-smoke \
  --run.no-timestamp \
  -- \
  epoch=1 \
  data.train.max_len=32 \
  data.val.max_len=16 \
  batch_size=4 \
  num_worker=0 \
  use_wandb=False
```

This assumes the config's required PILArNet-M revision is already available.
Run once with `--dry-run`, then remove it. A smoke run should prove that:

- one transformed sample and one packed batch have the documented shapes;
- loss is finite and gradients reach the intended parameters;
- validation runs;
- `model/last/.complete` and `model_best.pth` are written;
- the resolved config contains the intended data roots and checkpoint.

It does not establish convergence or physics performance.

## 4. Create a reviewable variant

For a real comparison, write a child config rather than preserving a long shell
history:

```python
# configs/my_study/sonata_v1.py
_base_ = ["../panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py"]

seed = 17
batch_size = 32
epoch = 100
data = dict(
    train=dict(max_len=500_000),
    val=dict(max_len=10_000),
)
optimizer = dict(lr=3e-5, weight_decay=0.2)
```

Lists are replaced, not merged. If you redefine `hooks`, `transform`,
`param_dicts`, or a class-name list, copy every entry you still need.

## 5. Name and start the run

```bash
uv run pimm launch \
  --train.config my_study/sonata_v1 \
  --resources.nproc-per-node 4 \
  --run.name sonata-v1-seed17
```

By default a timestamp is appended and the codebase is copied. For a resumable
fixed path, add `--run.no-timestamp`; never reuse that path for a different
resolved config.

## Required research record

Keep these together with reported results:

- pimm version and Git commit;
- `resolved_config.json`, `run_metadata.json`, and source snapshot;
- dataset name, revision, split, file manifest/checksums, and selection cuts;
- checkpoint URI/revision and load report;
- hardware, world size, global batch size, precision, seed, and determinism
  setting;
- metric definition, aggregation, class mapping, and evaluation split;
- relevant method and dataset citations.

## Next

- Starting from weights: {doc}`Fine-tuning <fine_tune>`.
- Scaling without changing the experiment: {doc}`Distributed training
  <distributed>` or {doc}`Slurm <slurm>`.
- Logging and diagnostics: {doc}`../operations/logging`.
- Resume guarantees: {doc}`../operations/checkpoints`.
