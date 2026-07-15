# Evaluate an experiment

**Outcome:** produce metrics from a named checkpoint and document exactly what
the values summarize.

## Evaluation lifecycle

```text
resolved config + checkpoint + data split
    → predictions
    → task evaluator
    → aggregated metrics and artifacts
    → selection metric (during training only)
```

An evaluator hook validates during training.
{py:class}`~pimm.engines.hooks.checkpoint.CheckpointSaver` reads the
evaluator's current selection metric and writes `model_best.pth` when it
improves. {py:class}`~pimm.engines.hooks.eval.final_tester.FinalEvaluator`
optionally runs a task tester after training.

:::{important}
Put the evaluator before
{py:class}`~pimm.engines.hooks.checkpoint.CheckpointSaver` in `hooks`. Reversing
them makes the
saver read a stale or absent metric.
:::

```python
hooks = [
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="FinalEvaluator", test_last=False),
]
test = dict(type="SemSegTester", verbose=True)
```

## Common evaluators

| Hook | Main outputs | Selection value |
|---|---|---|
| {py:class}`~pimm.engines.hooks.eval.semantic_segmentation.SemSegEvaluator` | loss, per-class counts, mIoU, mAcc, allAcc, macro precision/recall/F1 | mIoU |
| {py:class}`~pimm.engines.hooks.eval.instance_segmentation.InstanceSegmentationEvaluator` | instance/detection/class statistics and configured regression metrics | configured primary metric |
| {py:class}`~pimm.engines.hooks.eval.pretrain.semantic_segmentation_pretrain.PretrainEvaluator` | frozen-feature linear-probe metrics | mF1 |
| {py:class}`~pimm.engines.hooks.eval.pretrain.mae.MAEEvaluator` / {py:class}`~pimm.engines.hooks.eval.pretrain.hmae.HMAEEvaluator` | reconstruction losses and mask ratio | negative validation loss |

Instance evaluation commonly requires an event batch size of one; use the
exact evaluator/config contract rather than copying semantic-segmentation batch
settings.

## During training

`every_n_steps=0` evaluates after an epoch. A positive value evaluates at a
global-step interval. `evaluate=False` prevents validation/test loader creation
and makes evaluator hooks no-op.

The “best” model is best only with respect to the configured selection metric
on the configured validation split. It is not necessarily the best value for a
different class, threshold, or physics objective.

## Evaluate a finished run

For a convenience evaluation against a **current checkout config**, use the
wrapper through the locked environment:

```bash
uv run sh scripts/test.sh \
  -c panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  -n <experiment-name> \
  -w model_best
```

`scripts/test.sh` is single-process even though it accepts legacy `-g`/`-m`
options. It finds weights under the standard `exp/<config-group>/<name>`
layout and adds the run's `code/` snapshot to `PYTHONPATH`, but `-c` still
selects `configs/<value>.py` from the current checkout. That is useful for an
intentional reevaluation under new code/configuration; it is not exact
reproduction of the saved run config.

To use the run's saved config and code snapshot explicitly:

```bash
RUN=exp/panda/semseg/<experiment-name>
uv run python -u "$RUN/code/pimm/test.py" \
  --config-file "$RUN/config.py" \
  --options \
  save_path="$RUN" \
  weight="$RUN/model/model_best.pth"
```

If the run used `--train.no-code-copy`, replace
`"$RUN/code/pimm/test.py"` with `pimm/test.py` from the compatible pinned
checkout. In both cases, inspect `config.py`, the dataset revision/root, tester,
and weight path before reporting results.

## What to report

For every metric table or figure, record:

- checkpoint and immutable revision/checksum;
- pimm version and resolved config;
- dataset revision, split, event selection, and class mapping;
- preprocessing and any test-time transforms;
- metric definition, averaging, ignored labels, thresholds, and units;
- number of events and, where meaningful, uncertainty across events or seeds;
- hardware/precision only when they affect the procedure or performance claim.

Avoid writing only “accuracy” or “IoU.” State whether values are micro, macro,
per-class, event-weighted, or point-weighted.

## SSL probes

Linear probes estimate representation usefulness without fine-tuning the
backbone. They are diagnostics, not downstream-task results. Use held-out data,
keep probe train/test events disjoint from pretraining, and report the probe
data size, classifier, regularization search, and seed. Event-level probe hooks
perform leakage checks where dataset metadata permits; do not disable those
checks to obtain a number.

## Expected artifacts

A complete evaluation should leave the raw prediction/metric artifact required
to recompute reported tables, not only console output. Some current testers do
not yet standardize that artifact across tasks.

:::{admonition} TODO
:class: pimm-todo
Add one canonical saved-prediction schema per task and measured baseline tables
for the small public PILArNet-M-mini dataset. Until then, task pages should
describe their exact files rather than implying one universal format.
:::

## Next

- Metric-producing hook APIs: {doc}`../api/index`.
- Logging and diagnostic channels: {doc}`../operations/logging`.
- Reproduce a checkpoint: {doc}`../operations/checkpoints`.
