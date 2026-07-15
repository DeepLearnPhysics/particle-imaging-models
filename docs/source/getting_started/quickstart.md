# First experiment

**Outcome:** train and validate a tiny semantic-segmentation model on 100
PILArNet-M events, then inspect the exact artifacts pimm saved.

This is a pipeline check, not a scientifically meaningful benchmark.

## Prerequisites

- a completed {doc}`full installation <installation>`;
- one visible NVIDIA GPU;
- network access to download the small public dataset from Hugging Face;
- enough free space for the environment, mini dataset, and one tiny checkpoint.

:::{admonition} TODO
:class: pimm-todo
Add measured runtime and peak-memory ranges for the supported GPU families.
Until then, treat this as a small functional check rather than a timed
benchmark; runtime and memory still depend on the GPU, driver, filesystem, and
other processes.
:::

## 1. Download the mini dataset

From the repository root, run this in Python or a notebook:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="DeepLearnPhysics/PILArNet-M-mini",
    repo_type="dataset",
    local_dir="data/PILArNet-M-mini",
)
```

The download contains:

```text
data/PILArNet-M-mini/
├── train/generic_v2_80_v2.h5    80 events
├── val/generic_v2_20_v2.h5      20 events
└── test/generic_v2_20_v2.h5     20 events
```

## 2. Render the run

Dry-run exactly what will execute. The small Bash arrays keep each explanation
beside the option it describes while leaving the command copyable:

```bash
launcher_args=(
  --site local                         # use the local-machine profile
  --resources.nproc-per-node 1         # start one GPU process
  --resources.cpus-per-proc 2          # set two CPU threads per process
  --run.name tiny-semseg               # set the run-name prefix
  --train.config tests/tiny_semseg     # select the tiny training recipe
  --train.no-code-copy                 # run directly from this checkout
  --dry-run                            # render instead of launching
)

train_overrides=(
  "data.train.data_root=$PWD/data/PILArNet-M-mini"  # locate the training split
  "data.val.data_root=$PWD/data/PILArNet-M-mini"    # locate the validation split
)

uv run pimm launch "${launcher_args[@]}" -- "${train_overrides[@]}"
```

Everything before the bare `--` configures the launcher. Everything after it
overrides a training-config value. Inspect the rendered config path, experiment
path, process count, and data roots before removing `--dry-run`.

## 3. Train

Run the same arguments without `--dry-run`:

```bash
uv run pimm launch \
  --site local \
  --resources.nproc-per-node 1 \
  --resources.cpus-per-proc 2 \
  --run.name tiny-semseg \
  --train.config tests/tiny_semseg \
  --train.no-code-copy \
  -- \
  data.train.data_root="$PWD/data/PILArNet-M-mini" \
  data.val.data_root="$PWD/data/PILArNet-M-mini"
```

The launcher appends a timestamp and prints the exact experiment directory.
For example, a run started on July 14, 2026 might produce the following. The
run succeeds when `train.log` contains `Val result:` and these files exist:

```text
exp/tests/tiny-semseg-2026-07-14_14-30-00/
├── config.py
├── resolved_config.json
├── model_config.json
├── run_metadata.json
├── train.log
└── model/
    ├── last/
    │   ├── weights.pth
    │   ├── trainer.dcp/
    │   └── .complete
    └── model_best.pth
```

Because this command uses `--train.no-code-copy`, it runs directly from the
checkout and omits `code/`. Normal research runs copy the code by default.

## 4. Inspect what ran

Run this in Python or a notebook, using the directory that the launcher
printed:

```python
import json
from pathlib import Path

run = Path("exp/tests/tiny-semseg-2026-07-14_14-30-00")  # use your printed path
cfg = json.loads((run / "resolved_config.json").read_text())
meta = json.loads((run / "run_metadata.json").read_text())

print("model:", cfg["model"]["type"])
print("global batch:", cfg["batch_size"])
print("epochs:", cfg["epoch"])
print("config source:", meta["config_file"])
```

`batch_size=4` is the global event count across all ranks. With one rank it is
four events per step; with two ranks it must still be divisible by two and
becomes two events per rank.

## 5. Change one thing safely

Keep the tested config intact and override a value for a throwaway probe:

```bash
uv run pimm launch \
  --train.config tests/tiny_semseg \
  --resources.nproc-per-node 1 \
  --run.name tiny-semseg-two-epochs \
  --train.no-code-copy \
  -- \
  epoch=2 \
  data.train.data_root="$PWD/data/PILArNet-M-mini" \
  data.val.data_root="$PWD/data/PILArNet-M-mini"
```

For a change you intend to keep, create a child Python config instead of a long
command. See {doc}`Configuration <../operations/configuration>`.

## What happened

1. The launcher merged `launch/defaults.yaml`, `launch/sites/local.yaml`, and
   your flags.
2. The training config inherited `configs/_base_/default_runtime.py` and applied
   post-`--` overrides.
3. {py:class}`~pimm.datasets.pilarnet.h5.PILArNetH5Dataset` read individual
   events and the transform pipeline created coordinates, features, and
   semantic targets.
4. The collator packed four variable-length events and created `offset`.
5. {py:class}`~pimm.models.default.DefaultSegmentorV2` used a small `PT-v3m2`
   backbone and returned a loss and segmentation logits.
6. hooks timed the run, wrote metrics, evaluated the validation split, and saved
   the structured checkpoint.

Follow the complete object-level trace in {doc}`Experiment anatomy <concepts>`.

## Next

| If you want to… | Continue with… |
|---|---|
| choose a real recipe | {doc}`Train or pretrain <../workflows/train>` |
| fine-tune a checkpoint | {doc}`Fine-tuning <../workflows/fine_tune>` |
| use the full PILArNet-M dataset | {doc}`PILArNet-M <../data/pilarnet>` |
| add your own data | {doc}`Custom datasets <../data/custom>` |
| use multiple GPUs | {doc}`Distributed training <../workflows/distributed>` |
| understand saved state | {doc}`Checkpoints and resume <../operations/checkpoints>` |
