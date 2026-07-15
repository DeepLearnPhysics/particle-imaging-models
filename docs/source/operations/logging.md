# Logging and diagnostics

pimm writes `train.log` and one experiment writer on rank 0. Set
`use_wandb=True` for Weights & Biases; otherwise the same writer calls go to a
TensorBoard event file in the experiment directory.

## Enable W&B

```bash
export WANDB_API_KEY=...
uv run pimm launch \
  --train.config <config> \
  --run.wandb-name <display-name> \
  --run.wandb-project <project> \
  -- use_wandb=True
```

`--run.wandb-name` controls the display name. If it is omitted, `pimm launch`
uses the generated experiment name. When running the lower-level training
script directly, {py:class}`~pimm.engines.hooks.logging.WandbNamer` can instead
derive a name from selected config fields. The project defaults to `pimm` when
neither the config nor the launcher sets one.

You can also authenticate with `wandb login` or put `WANDB_API_KEY` in the
local `.env`. Do not put a secret in the experiment config: the resolved
experiment config is sent to W&B.

## What pimm sends to W&B

The W&B run is an experiment index, not a backup of the experiment directory.

| Content | What is recorded |
|---|---|
| Run identity | project, display name, and optional `wandb_group` and `wandb_job_type` |
| Config | the complete serializable experiment config after inheritance, CLI overrides, per-rank runtime derivation, and hook config modifiers |
| Training | `params/lr`, scalar model outputs under `train_batch/*` each step, and their epoch means under `train/*` |
| Evaluation | the `val/*` metrics emitted by the configured evaluator, including optional per-class or probe metrics |
| Diagnostics | only measurements from diagnostic/resource hooks that are present in the config |
| Visuals | images, histograms, and other media only when a configured hook explicitly writes them; the MAE evaluator can emit W&B `Object3D` reconstructions |

The config includes model, data/transforms, optimizer, scheduler, hooks, seed,
batch sizes, worker counts, precision settings, checkpoint settings, and the
other experiment fields available to the trainer. It does **not** include the
whole launch YAML (site, container, Slurm allocation, and rendered submission
script) unless a value was explicitly forwarded as a training override.

pimm does not explicitly upload checkpoints, the code snapshot, `config.py`,
`resolved_config.json`, `run_metadata.json`, `launch.sbatch`, or Slurm logs as
W&B artifacts. Those stay under the experiment directory. Preserve that
directory separately; use the Hugging Face export workflow when a model should
be published.

The W&B client may add its own standard runtime and system telemetry according
to the user's W&B settings. That is separate from the fields pimm explicitly
logs above.

## Default information flow

{py:class}`~pimm.engines.hooks.logging.InformationWriter` filters the model
output down to Python numbers and scalar tensors. Large logits, masks,
embeddings, and non-scalar values are deliberately excluded.

```python
hooks = [
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_frequency=20),
]
```

`total_loss`, when present, is logged as `loss`; the redundant model-output
`loss` entry is omitted. The learning rate is always written as `params/lr`.
{py:class}`~pimm.engines.hooks.logging.IterationTimer` adds `data_time`,
`batch_time`, and the ETA to `train.log`, but does not create W&B/TensorBoard
curves for those timings by default.

`log_frequency` controls console lines, not metric sampling: writer scalars are
still collected every optimization step.

```text
model output                       experiment writer
loss_cls ────────────────────────▶ train_batch/loss_cls
total_loss ──────────────────────▶ train_batch/loss
optimizer LR ────────────────────▶ params/lr
epoch means ─────────────────────▶ train/*
evaluator metrics ───────────────▶ val/*
```

All curves use the hidden `train/global_step` field as their W&B step metric.
Writes from multiple hooks at the same optimizer step are combined into one
history row.

## Distributed runs

Only global rank 0 creates the W&B/TensorBoard writer, so a distributed job
creates one W&B run rather than one run per GPU. Evaluators that gather or
all-reduce their results log global validation metrics. Ordinary
`train_batch/*` values come from rank 0's model output unless the model reduced
that value itself; do not interpret them as a global mean without checking the
model.

## Resume and requeue

Current pimm checkpoints store the active W&B run ID and its next history row.
Before any metrics are written on resume,
{py:class}`~pimm.engines.hooks.checkpoint.CheckpointLoader` configures W&B's
`resume_from` cursor. W&B truncates the run history at that saved boundary and
then appends the replayed training steps, so a rollback from step 75 to a
checkpoint at step 50 does not leave two copies of steps 51--75.

Managed requeue attempts share the same experiment and W&B display name. Once
an attempt has produced a current checkpoint, later attempts recover the W&B
run identity from that checkpoint rather than creating `-job0001`,
`-job0002`, ... runs. Chain fields such as `wandb_group`, `wandb_job_type`,
`wandb_job_index`, and `chain_jobs` remain available in the resolved config.

If a legacy checkpoint has no W&B run ID/cursor, pimm warns and falls back to
the explicitly configured `wandb_run_id`/`wandb_resume` behavior. Without
those fields, W&B starts a new run (possibly with the same display name), and
rolled-back steps are not automatically reconciled. A weights-only warm start
also starts a new run; `log_step_offset` changes only its plotted x-axis and
does not make it a resume.

## TensorBoard fallback

```python
use_wandb = False
```

This writes the same scalar keys to TensorBoard event files under `save_path`:

```bash
uv run tensorboard --logdir exp
```

TensorBoard needs no W&B account or network connection. The resolved config
and provenance files remain beside the events rather than being embedded as a
W&B config. Resume continues with the trainer's restored global step, but
TensorBoard has no checkpointed history-rewind cursor; if a non-exact resume
replays steps, event files can contain overlapping step values.

## Useful read-only monitors

| Hook | Use |
|---|---|
| {py:class}`~pimm.engines.hooks.diagnostics.GradientNormLogger` | global and optional per-layer gradient norms |
| {py:class}`~pimm.engines.hooks.diagnostics.PrototypeUsageLogger` | prototype occupancy/entropy for prototype SSL |
| {py:class}`~pimm.engines.hooks.diagnostics.FeatureStdMonitor` | student/teacher feature spread and collapse diagnostics |
| {py:class}`~pimm.engines.hooks.resources.ResourceUtilizationLogger` | process CPU/RAM and GPU memory/utilization where available |
| {py:class}`~pimm.engines.hooks.diagnostics.ParameterCounter` | total/frozen/trainable parameters at startup |
| {py:class}`~pimm.engines.hooks.diagnostics.LogitEntropyLogger` | classification-head entropy |

Example for a short sizing run:

```python
hooks = [
    dict(type="ParameterCounter", show_details=True),
    dict(type="ResourceUtilizationLogger", log_frequency=10,
         log_per_gpu=True, per_process=True),
    dict(type="GradientNormLogger", log_frequency=10),
    # evaluator and checkpoint hooks follow
]
```

Interpret measurements in context. A gradient spike or low feature variance is
a symptom, not a diagnosis; compare data, objective schedule, precision,
optimizer, and model changes before prescribing a fix.

## Mutating runtime hooks

These change the experiment and belong in its provenance:

| Hook | Mutation |
|---|---|
| {py:class}`~pimm.engines.hooks.optimizer.WeightDecayExclusion` | rewrites optimizer parameter groups |
| {py:class}`~pimm.engines.hooks.optimizer.WeightDecayScheduler` | changes weight decay by global step |
| {py:class}`~pimm.engines.hooks.diagnostics.DtypeOverrider` | forces selected modules/methods/parameters to a dtype |
| {py:class}`~pimm.engines.hooks.diagnostics.AttentionMaskAnnealingHook` | advances a model's attention-mask schedule |
| {py:class}`~pimm.engines.hooks.resources.GarbageHandler` | controls Python GC and optional CUDA cache emptying |
| {py:class}`~pimm.engines.hooks.profiling.RuntimeProfiler` | instruments selected steps and may stop the run |

Hook order matters. For example, weight-decay exclusion must run before a
weight-decay scheduler uses the groups it created.

## Profile only a bounded run

{py:class}`~pimm.engines.hooks.profiling.RuntimeProfiler` adds overhead and can
write large traces under the run's
`logdir`. Enable it for a short named diagnostic, inspect the trace, then remove
it from the production config.

## What to watch first

| Symptom | Measurements |
|---|---|
| GPU idle or low utilization | `data_time` vs `batch_time`, CPU, loader workers, storage throughput |
| late OOM | total points per batch, allocated/reserved VRAM, event-length distribution |
| divergent loss | loss components, gradient norm, AMP scale/overflow, LR and schedule |
| frozen model not learning | trainable-parameter count, gradient presence, checkpoint load report |
| SSL collapse concern | feature std, prototype usage, probe metrics on held-out data |
| resumed run jumps | restored global step/schedule, cursor warning, topology, first resumed batches |

## Preserve diagnostic evidence

Keep `train.log`, writer events or immutable W&B run identity, resolved config,
and job logs together. For a failure report, include the first traceback and
the relevant resource/timing window rather than only the final scheduler exit.

See {doc}`Troubleshooting <troubleshooting>` for symptom-specific commands and
the {doc}`hook API <../api/index>` for current signatures.
