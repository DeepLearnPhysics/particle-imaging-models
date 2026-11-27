# Checkpoints and resume

Checkpoints saved during training are for resuming; for inference and fine-tuning, make a separate copy with `pimm export`.

## What's saved

```text
exp/<group>/<run>/model/
├── last/
│   ├── weights.pth       model weights
│   ├── trainer.dcp/      optimizer, scheduler, scaler, RNG, data-loader and metric state
│   └── .complete         written last
├── model_best.pth        weights, written when the selection metric improves
└── iter_<step>.pth       optional weight snapshots
```

Checkpoints are written to a temporary path and renamed into place, so a checkpoint without `.complete` is unfinished. Set `checkpoint_format="legacy"` to get a single `model_last.pth` instead of `last/`. If `MODEL_DIR` is set, `model/` is a symlink into that directory.

## Warm start and resume

| | Warm start | Resume |
|---|---|---|
| How | `--train.weight <path or hf://...>` | `--train.resume` on a fixed run directory |
| Restores | model weights | weights, optimizer, scheduler, step, RNG and data-loader position |
| Use it for | a new run from existing weights | continuing an interrupted run |

## Resume a run

Resume with the same config and run name, and without a timestamp:

```bash
uv run pimm launch --train.config <same-config> --run.name <same-name> --run.no-timestamp --train.resume
```

Training resumes from the newest complete checkpoint in `model/`. To see which checkpoint that is, run this command:

```bash
uv run python -m pimm.utils.path latest-checkpoint exp/<group>/<run>/model
```

You can't resume a run from `hf://` weights: exports contain no trainer state.

## Changing the number of GPUs

| On resume | Model and optimizer | Data-loader position |
|---|---|---|
| same GPUs and workers per GPU | restored | restored; training continues mid-epoch |
| different number of GPUs | resharded | dropped; the saved epoch restarts |
| different workers per GPU | restored | dropped; the saved epoch restarts |
| `resume_strict_state=False` | best effort | dropped |

When the position is dropped, the log contains a warning that batches from the restarted epoch may be seen twice.

## `model_best.pth`

To get `model_best.pth`, list an evaluator before `CheckpointSaver` in `hooks`. The file is written each time the evaluator's selection metric improves, such as mIoU for `SemSegEvaluator`. `model_best.pth` contains weights only, so use it for evaluation and export, not to resume a run.

## Errors

| Message | Meaning |
|---|---|
| `No weight found` | the weight path doesn't exist; training stops instead of starting from random weights |
| `Incomplete checkpoint directory` | the checkpoint lacks weights, trainer state or `.complete`; use an older complete one |
| resume from `hf://` rejected | exports have no trainer state; warm-start a new run instead |
