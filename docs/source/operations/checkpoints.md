# Checkpoints and resume

A training checkpoint and a portable model export serve different purposes.
Use the checkpoint to continue optimization; use an export or weights file for
inference and fine-tuning.

## Standard format

```text
exp/<group>/<run>/model/
├── last/
│   ├── weights.pth       plain {"state_dict": ...} model weights
│   ├── trainer.dcp/      distributed trainer-state checkpoint
│   │   └── .complete
│   └── .complete         written last; marks the split checkpoint complete
├── model_best.pth        weights only, written when selection metric improves
└── iter_<step>.pth       optional weights-only snapshots
```

`trainer.dcp/` carries optimizer, scheduler, AMP scaler, structured trainer
state, distributed RNG state, dataloader state, and metric state. The separate
`weights.pth` stays easy to load without DCP.

Saves are published through temporary paths and renames. A `.prev` path exists
only transiently during rotation or after an interrupted rotation; a successful
save removes the old backup. Do not promise that `last.prev` is a persistent
history.

## Legacy format

```text
model/
├── model_last.pth
└── model_best.pth
```

The legacy file is monolithic and rank-0 written. It remains supported for old
runs, but the standard split format is the recommended default.

```bash
uv run pimm launch --train.config <config> -- checkpoint_format=standard
```

## Warm start

Warm start loads model weights into the model defined by a new config:

```bash
uv run pimm launch \
  --train.config <downstream-config> \
  --train.weight /path/to/weights.pth
```

It does not restore optimizer, schedule, step, RNG, or data cursor. Startup logs
show key rewrite rules and counts for loaded, missing, unexpected, and
shape-mismatched parameters. pimm raises if zero model parameters are selected.

Remote `hf://` weights are warm-start only; Hub uploads do not contain
`trainer.dcp/` and cannot resume a training run.

## Resume the same run

Use a fixed run path and the same experiment identity:

```bash
uv run pimm launch \
  --train.config <same-config> \
  --run.name <same-name> \
  --run.no-timestamp \
  --train.resume
```

The launcher selects the newest complete candidate under `model/`: normally
`last`, with recovery support for a complete interrupted-rotation `last.prev`
or legacy `model_last.pth`. Incomplete `.tmp` directories and missing
`.complete` markers are rejected.

Diagnose selection directly:

```bash
uv run python -m pimm.utils.path latest-checkpoint \
  exp/<group>/<run>/model
```

## What “exact resume” means

On the **same world size and worker topology**, a structured checkpoint with a
valid loader cursor can restore the saved epoch, iteration, global step,
sampler/loader position, optimizer, scheduler, scaler, and RNG state. Subject to
the usual deterministic-kernel and external-I/O limits, this is the path for a
mid-epoch continuation.

Topology changes deliberately narrow the guarantee:

| Resume condition | Behavior |
|---|---|
| same world size and per-rank workers | restore cursor when present; continue from saved iteration |
| world size changes | reshard compatible state, discard loader cursor, restart the saved epoch |
| per-rank worker count changes | discard loader cursor, restart the saved epoch |
| checkpoint has no loader state | restart the saved epoch |
| `resume_strict_state=False` | best-effort state restore; loader cursor skipped |
| legacy mid-epoch checkpoint | no structured cursor; replay from an epoch boundary |

When the cursor is skipped, pimm resets the in-epoch iteration/global-step view
to the saved epoch start and warns that already-completed batches may replay.
This is recovery and resharding—not bitwise-equivalent continuation.

## World-size resharding

PyTorch Distributed Checkpoint can redistribute compatible model and optimizer
state. That capability does not make rank-local RNG and sample order identical.
Record the topology change, warning, and replay boundary with the run.

If a changed topology is planned rather than emergency recovery, starting a new
run from model weights may be scientifically clearer than calling the result a
single uninterrupted training trajectory.

## Best checkpoint semantics

`model_best.pth` is model weights only in the standard format. It is written
when an evaluator runs before {py:class}`CheckpointSaver
<pimm.engines.hooks.checkpoint.CheckpointSaver>` and improves the configured
selection metric. It is suitable for evaluation/export, not full resume.

Check the metric name, validation split, and evaluator order before using
“best” in a report.

## Export

```bash
uv run pimm export \
  --run-dir exp/<group>/<run> \
  last \
  artifacts/<name>
```

An export writes `model.safetensors` (or `model.bin`) and writes `config.json`
when the training config is discoverable. Known load-trigger paths are cleared
and other absolute paths are redacted. See {doc}`Export a model
<../models/export>`.

## Recovery checklist

1. preserve the interrupted run directory before manual edits;
2. find the newest complete checkpoint with the path helper;
3. compare current and saved world size, workers, config, code, and data;
4. dry-run the resume command and confirm the same run path;
5. read startup logs for loader-cursor and key-load warnings;
6. record any replay/topology change in the provenance bundle;
7. compare the first resumed losses/schedule values with the pre-interruption
   log.

## Failure modes

### “No weight found”

The supplied path is non-empty and does not resolve; pimm fails rather than
silently training from scratch. Check the run path and completion marker.

### “Incomplete checkpoint directory”

The directory lacks required weights, DCP metadata, or `.complete`. Select a
complete backup/candidate; do not add the marker by hand.

### Resume from `hf://` is rejected

Expected: Hub model artifacts omit trainer state. Warm-start a new run or use a
local `model/last` directory from the original experiment.

### The saved epoch restarted

Read the warning. World size, per-rank workers, non-strict mode, or missing
loader state caused cursor restore to be skipped.
