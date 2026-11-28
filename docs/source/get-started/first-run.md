# First run

Train a tiny semantic-segmentation model on 100 events of PILArNet-M-mini, then look at what was saved. You need a finished [installation](install.md), one NVIDIA GPU, and network access to Hugging Face.

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

Put launcher flags before the bare `--` and `key=value` overrides of the training config after it. Add `--dry-run` to print the resolved `torchrun` command and the run directory without running anything.

## 3. Train

Run the same command without `--dry-run`. The run directory is printed, for example `exp/tests/tiny-semseg-2026-07-14_14-30-00/`. Training has finished when `train.log` contains `Val result:` and the directory contains these files:

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

The recipe sets `revision="v3"` because the mini files have the v3 layout; the full PILArNet-M recipes read `v3_extra`.

Pass `--train.no-code-copy` to import pimm straight from your checkout. Without the flag, `pimm/` and `configs/` are copied into the run's `code/` directory, and that copy is imported.

## 4. Look at the saved run

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

`resolved_config.json` is the config that was actually run, after inheritance and your overrides.

## 5. Change one thing

```bash
uv run pimm launch --train.config tests/tiny_semseg --resources.nproc-per-node 1 \
  --run.name tiny-semseg-2ep --train.no-code-copy \
  -- epoch=2 data.train.data_root="$PWD/data/PILArNet-M-mini" data.val.data_root="$PWD/data/PILArNet-M-mini"
```

For a change you want to keep, write a child config instead of a longer command; see [Configs and overrides](../training/configs.md).

## What happened

1. `launch/defaults.yaml`, `launch/sites/local.yaml` and your flags were merged, then `torchrun` was started.
2. Your overrides were applied to `configs/tests/tiny_semseg.py`, which inherits from `configs/_base_/default_runtime.py`.
3. Each event was read with `PILArNetH5Dataset`. The transform steps were to normalize coordinates, log-scale energy, keep one point per grid cell and copy `segment_motif` to `segment`.
4. Four events were packed into one batch, and `offset` was built for it.
5. The model was `DefaultSegmentorV2` with a two-stage `PT-v3m2` backbone; its output was a loss in training and logits in validation.
6. Everything else was a hook: timing, logging, evaluating the validation split and saving checkpoints.

For more on each step, see [How pimm works](how-pimm-works.md).
