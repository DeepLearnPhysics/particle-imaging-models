# Evaluate

Use evaluator hooks to measure a model on the validation split during training, and a tester after training or on a saved checkpoint.

## During training

| Hook | Metrics | `model_best` selected by |
|---|---|---|
| `SemSegEvaluator` | loss, per-class counts, mIoU, mAcc, allAcc, precision, recall, F1 | mIoU |
| `PointClassificationEvaluator` | accuracy and macro precision, recall and F1 of each head | mean macro F1 over the heads |
| `InstanceSegmentationEvaluator` | instance, detection and class statistics | its configured primary metric |
| `PretrainEvaluator` | linear-probe metrics on frozen features | mF1 |
| `MAEEvaluator` | reconstruction losses and mask ratio | negative validation loss |

Put the evaluator before `CheckpointSaver` in `hooks`, so that the current metric is available to the saver:

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

Evaluators are run after each round of `eval_epoch`, or every N steps if you pass `every_n_steps=N`. Set `evaluate=False` to skip validation entirely. Keep `FinalEvaluator` in `hooks` to run the `test` tester once training ends.

## Points that are scored

With `SemSegEvaluator`, `PointClassificationEvaluator` and `SemSegTester`, only the model's output is scored: the logits, the labels and `offset`, all at the same points. In evaluation, the output of `DefaultSegmentorV2` is given at the original points of each event, before `GridSample`, and every point is given the logits of its cell. For this, keep a copy of the labels before `GridSample` and collect `inverse` in the validation and test transforms:

```python
eval_transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="LogTransform", min_val=0.01, max_val=20.0),
    dict(type="Copy", keys_dict={"segment_motif": "origin_segment"}),
    dict(type="GridSample", grid_size=0.001, hash_type="fnv", mode="deterministic", return_grid_coord=True,
         return_inverse=True),
    dict(type="Copy", keys_dict={"segment_motif": "segment"}),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
        feat_keys=("coord", "energy"),
        offset_keys_dict=dict(offset="coord", origin_offset="inverse"),
    ),
]
```

The loss is still computed on the cells, with `segment`. In the output, `segment` holds the batch's `origin_segment`; with several heads, each head's label holds `origin_<label>`. Without `inverse`, you get a `KeyError` in evaluation. With PoLAr-MAE, the input points are the original points, and no `GridSample` is needed.

## On a finished run

Evaluate with the current checkout's config:

```bash
uv run sh scripts/test.sh -c panda/v1/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft -n <run-name> -w model_best
```

`scripts/test.sh` is single-process. The weights come from `exp/<config group>/<run-name>` and the code from the run's `code/` copy, but the `-c` config comes from the current checkout.

Or evaluate with the config and code the run was launched with:

```bash
RUN=exp/panda/v1/semseg/<run-name>
uv run python -u "$RUN/code/pimm/test.py" \
  --config-file "$RUN/config.py" \
  --options save_path="$RUN" weight="$RUN/model/model_best.pth"
```

For a run started with `--train.no-code-copy`, use `pimm/test.py` from the checkout instead.
