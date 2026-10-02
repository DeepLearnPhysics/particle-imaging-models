# First run

Train a tiny semantic-segmentation model on 100 events of PILArNet-M-mini, then look at what pimm saved. You need a finished [installation](install.md), one NVIDIA GPU, and network access to Hugging Face.

The recipe, `tests/tiny_semseg`, is a one-epoch smoke test with a deliberately small model.

## 1. Download the mini dataset

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('DeepLearnPhysics/PILArNet-M-mini', repo_type='dataset', local_dir='data/PILArNet-M-mini')"
```

```text
data/PILArNet-M-mini/
├── train/generic_v2_80_v2.h5    80 events
├── val/generic_v2_20_v2.h5      20 events
└── test/generic_v2_20_v2.h5     20 events
```

## 2. Dry-run the job

```bash
uv run pimm launch \
  --site local \
  --resources.nproc-per-node 1 \
  --resources.cpus-per-proc 2 \
  --run.name tiny-semseg \
  --train.config tests/tiny_semseg \
  --train.no-code-copy \
  --dry-run \
  -- \
  data.train.data_root="$PWD/data/PILArNet-M-mini" \
  data.val.data_root="$PWD/data/PILArNet-M-mini"
```

Flags before the bare `--` configure the launcher. The `key=value` pairs after it override values in the training config. The dry run prints the resolved `torchrun` command and the run directory, then exits.

## 3. Train

Run the same command without `--dry-run`. The launcher prints the run directory, for example `exp/tests/tiny-semseg-2026-07-14_14-30-00/`. Training has finished when `train.log` contains `Val result:` and the directory holds:

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

The log notes that revision `v2` is read as `v3`: the recipe names `v2`, and the reader serves it with the v3 layout that the mini files use.

With `--train.no-code-copy`, the run imports pimm straight from your checkout. Without it, pimm copies `pimm/` and `configs/` into the run's `code/` directory, and the run imports that copy.

## 4. Look at what ran

```bash
uv run python - <<'PY'
import json, pathlib
run = sorted(pathlib.Path("exp/tests").glob("tiny-semseg-*"))[-1]
cfg = json.loads((run / "resolved_config.json").read_text())
print(run)
print("model:", cfg["model"]["type"], "on", cfg["model"]["backbone"]["type"])
print("global batch:", cfg["batch_size"], "epochs:", cfg["epoch"])
PY
```

`resolved_config.json` is the config after inheritance and your overrides: what actually ran.

## 5. Change one thing

```bash
uv run pimm launch --train.config tests/tiny_semseg --resources.nproc-per-node 1 \
  --run.name tiny-semseg-2ep --train.no-code-copy \
  -- epoch=2 data.train.data_root="$PWD/data/PILArNet-M-mini" data.val.data_root="$PWD/data/PILArNet-M-mini"
```

For a change you want to keep, write a child config instead of a longer command; see [Configs and overrides](../training/configs.md).

## What happened

1. The launcher merged `launch/defaults.yaml`, `launch/sites/local.yaml` and your flags, then started `torchrun`.
2. `configs/tests/tiny_semseg.py` inherited `configs/_base_/default_runtime.py` and took your overrides.
3. `PILArNetH5Dataset` read each event. The transforms normalized coordinates, log-scaled energy, kept one point per grid cell and copied `segment_motif` to `segment`.
4. Collation packed four events into one batch and built `offset`.
5. `DefaultSegmentorV2` with a two-stage `PT-v3m2` backbone returned a loss in training and logits in validation.
6. Hooks timed the run, logged it, evaluated the validation split and saved checkpoints.

[How pimm works](how-pimm-works.md) covers each step.
