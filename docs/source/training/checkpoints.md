# Checkpoints and resume

During training, pimm saves checkpoints you can resume from; `pimm export` makes a separate copy for inference and fine-tuning.

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

Saves go to a temporary path and are renamed into place, so a checkpoint without `.complete` is unfinished. With `checkpoint_format="legacy"`, pimm writes a single `model_last.pth` instead of `last/`. If `MODEL_DIR` is set, `model/` is a symlink into that directory.

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

The launcher picks the newest complete checkpoint in `model/`. To see which one it will use:

```bash
uv run python -m pimm.utils.path latest-checkpoint exp/<group>/<run>/model
```

`hf://` weights can't be resumed: exports carry no trainer state.

## Changing the number of GPUs

| On resume | Model and optimizer | Data-loader position |
|---|---|---|
| same GPUs and workers per GPU | restored | restored; training continues mid-epoch |
| different number of GPUs | resharded | dropped; the saved epoch restarts |
| different workers per GPU | restored | dropped; the saved epoch restarts |
| `resume_strict_state=False` | best effort | dropped |

When the position is dropped, the log warns that batches from the restarted epoch may be seen twice.

## `model_best.pth`

`CheckpointSaver` writes `model_best.pth` when the evaluator listed before it reports a better selection metric, such as mIoU for `SemSegEvaluator`. It holds weights only, so it serves evaluation and export, not resume.

## Errors

| Message | Meaning |
|---|---|
| `No weight found` | the weight path doesn't exist; the run stops rather than start from random weights |
| `Incomplete checkpoint directory` | the checkpoint lacks weights, trainer state or `.complete`; use an older complete one |
| resume from `hf://` rejected | exports have no trainer state; warm-start a new run instead |
