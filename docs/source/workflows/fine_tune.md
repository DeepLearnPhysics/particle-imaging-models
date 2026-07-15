# Fine-tune a checkpoint

**Outcome:** initialize a downstream task from compatible pretrained weights,
verify the key mapping, and make the trainable parameter set explicit.

Fine-tuning is not the same as resuming:

| Operation | Restores | Use when |
|---|---|---|
| warm start / fine-tune | selected model weights | task, head, data, or optimization changes |
| resume | model, optimizer, scheduler, scaler, trainer cursor, RNG, and loader state where compatible | continuing the same run |

## 1. Choose a task recipe

Use an existing downstream config whose transforms, backbone, head, and targets
match your task. Common Panda semantic variants encode their strategy in the
filename:

| Suffix | Intent |
|---|---|
| `-lin` | linear or highly restricted probe |
| `-dec` | train a decoder/head while retaining more of the encoder |
| `-fft` | full fine-tuning |
| `-scratch` | label for the intended no-weight baseline; the child file does not enforce it |

These names describe the committed recipe, not a universal contract. Inspect
`param_dicts`, `requires_grad` handling, and the model's checkpoint mapping
before assuming which parameters train. The current `-scratch` children change
only W&B naming: omit `--train.weight`, confirm resolved `weight=None`, and
inspect startup logs to establish a random-initialization baseline.

## 2. Select compatible weights

Prefer a published Hub export or a local exported directory whose config is
available. The {doc}`model chooser <../models/index>` distinguishes consolidated
exports from raw training checkpoints. Record the resolved Hub revision in the
experiment metadata rather than embedding it in the user-facing URI.

Training can warm-start from a local file or a Hugging Face URI:

```bash
uv run pimm launch \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  --train.weight hf://DeepLearnPhysics/Panda-Base \
  --resources.nproc-per-node 1 \
  --run.name panda-semseg-smoke \
  --dry-run
```

Use the repository named by the model card and record the resolved revision in
your experiment provenance.

:::{important}
{py:func}`~pimm.from_pretrained` constructs an inference model. `--train.weight`
initializes the model already defined by the training config. They solve
different problems and may use different prefix/key mappings.
:::

## 3. Verify the preprocessing contract

Before launching, compare the checkpoint's saved config with the downstream
config:

- coordinate frame, units, and normalization;
- grid size and serialization order;
- feature names, order, and scaling;
- backbone `type`, channel widths, depth, and input channels;
- dataset revision and label mapping;
- masking or multi-view fields expected by a pretraining wrapper.

Architecture compatibility alone does not make features scientifically
compatible. See {doc}`Data conventions <../data/conventions>`.

## 4. Inspect the load report

On startup, {py:class}`~pimm.engines.hooks.checkpoint.CheckpointLoader` reports
how many model parameters loaded and
summarizes missing, unexpected, or shape-mismatched keys. pimm raises if a
mapping selects zero model parameters, but a partial load may still be
unintended.

Treat these as assertions for the experiment:

| Message | Interpretation |
|---|---|
| all intended backbone keys loaded | expected warm start |
| task-head keys missing | often expected for a new head |
| backbone blocks missing | usually a config or prefix mismatch |
| many unexpected teacher/student keys | mapping likely targets the wrong wrapper |
| zero selected keys | hard error; fix the URI, prefix, or mapping |

Save the complete startup log with the result.

## 5. Verify what can train

Before a long run, count parameters and inspect representative names:

```python
trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
print("trainable parameters:", sum(n for _, n in trainable))
print(*[name for name, _ in trainable[:30]], sep="\n")
```

For PEFT/LoRA, confirm adapters exist in the expected blocks and that the base
weights are frozen. For a linear probe, confirm only the classifier/head is
trainable. For full fine-tuning, confirm the backbone is not accidentally
frozen.

## 6. Run a small controlled comparison

Use the same data subset and seed for:

1. the pretrained initialization;
2. a verified no-weight `-scratch` baseline;
3. any freeze/PEFT alternative.

The first smoke run checks loss, gradients, validation, and checkpoint output.
It is not evidence that one strategy is better; use the full evaluation
protocol and report uncertainty across seeds where appropriate.

## Resume a fine-tuning run

Once optimization has started, use `--train.resume` with the same fixed run
path. Do not pass a new `--train.weight` to continue it.

```bash
uv run pimm launch \
  --train.config <same-config> \
  --run.name <same-run> \
  --run.no-timestamp \
  --train.resume
```

See {doc}`Checkpoint and resume semantics <../operations/checkpoints>` for
topology changes and mid-epoch behavior.

## Next

- {doc}`Pretrained loading and inference <../models/pretrained>`.
- {doc}`Semantic-segmentation tutorial <../tutorials/semantic_segmentation>`.
- {doc}`Panda Detector PEFT tutorial <../tutorials/peft>`.
- {doc}`Evaluation <evaluate>`.
