# Evaluate

Evaluator hooks measure a model on the validation split during training; testers run after training or on a saved checkpoint.

## During training

| Hook | Computes | Selects `model_best` by |
|---|---|---|
| `SemSegEvaluator` | loss, per-class counts, mIoU, mAcc, allAcc, precision, recall, F1 | mIoU |
| `InstanceSegmentationEvaluator` | instance, detection and class statistics | its configured primary metric |
| `PretrainEvaluator` | linear-probe metrics on frozen features | mF1 |
| `MAEEvaluator` | reconstruction losses and mask ratio | negative validation loss |

Put the evaluator before `CheckpointSaver` in `hooks`, so the saver reads the current metric:

```python
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="FinalEvaluator", test_last=False),
]
test = dict(type="SemSegTester", verbose=True)
```

An evaluator runs after each round of `eval_epoch`, or every N steps with `every_n_steps=N`. `evaluate=False` skips validation entirely. `FinalEvaluator` runs the `test` tester once training ends.

## On a finished run

With the current checkout's config:

```bash
uv run sh scripts/test.sh -c panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft -n <run-name> -w model_best
```

`scripts/test.sh` runs one process. It finds the weights under `exp/<config group>/<run-name>` and the run's `code/` copy, but `-c` reads the config from the current checkout.

With the config and code the run was launched with:

```bash
RUN=exp/panda/semseg/<run-name>
uv run python -u "$RUN/code/pimm/test.py" \
  --config-file "$RUN/config.py" \
  --options save_path="$RUN" weight="$RUN/model/model_best.pth"
```

For a run started with `--train.no-code-copy`, use `pimm/test.py` from the checkout instead.
