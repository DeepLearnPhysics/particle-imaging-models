# What is pimm?

pimm is a PyTorch research codebase for learning from **sparse,
variable-length 3D detector events**. It brings the pieces of an experiment
under one versioned workflow: data transforms, model construction, training,
evaluation, distributed launch, resumable state, and portable exports.

It is intended for two kinds of reader:

- researchers who need to reproduce, adapt, and scale an experiment without
  replacing their cluster or analysis workflow;
- students and new contributors who need one traceable path from a data file to
  a metric, checkpoint, and inference result.

## What you can do

| Workflow | pimm provides |
|---|---|
| Use a published model | registry-based model construction, checkpoint key mapping, local and Hugging Face loading |
| Fine-tune | full, linear-probe, frozen-encoder, and PEFT recipes where available |
| Pretrain | Panda/Sonata and PoLAr-MAE implementations and configs |
| Segment events | semantic segmentation, PointGroup, and Panda Detector task models |
| Scale a run | one typed launcher for local `torchrun` and Submitit-backed Slurm jobs |
| Reproduce a run | source snapshot, resolved config, metadata, logs, RNG/optimizer/scheduler/loader state |
| Publish a model | consolidated weights, sanitized config, model card, and optional Hub upload |

## What pimm is not

- It is not a general image library. The currently supported common input is a
  3D sparse point set; 2D wire-plane and optical waveform support is future
  work.
- It is not a detector-independent promise. A shared packed representation does
  not make coordinate frames, units, features, targets, or calibrations
  interchangeable.
- It is not a benchmark leaderboard. The repository contains research recipes,
  but metrics are only comparable when the dataset revision, split,
  preprocessing, checkpoint, and evaluation protocol match.
- It is not a stable production API. Research components evolve; pin a release
  and record the resolved config for published work.

## Why packed events?

Detector events contain different numbers of hits. Padding a batch to its
largest event wastes memory, so pimm concatenates points and records cumulative
event boundaries in `offset`:

```text
coord   [total_points, 3]
feat    [total_points, channels]
offset  [batch_size]              cumulative end indices
```

This supports point-transformer and sparse-convolution models without imposing
a common event length. It also means memory use depends on the total points in
each batch, not only the configured number of events. See {doc}`Data conventions
<../data/conventions>` before creating a dataset or interpreting a prediction.

## How the pieces fit

1. A Python config selects registered datasets, transforms, model, losses,
   optimizer, scheduler, hooks, trainer, and evaluator.
2. `pimm launch` or `pimm submit` resolves the execution environment and turns
   that config into a `torchrun` job.
3. The dataset emits one event; transforms create the exact coordinate and
   feature convention expected by the model.
4. Collation packs several events. The model returns a dictionary containing at
   least `loss` during training and task-specific predictions during evaluation.
5. Hooks log, evaluate, and checkpoint the run.
6. `pimm export` turns model weights plus their sanitized construction config
   into a portable directory.

The {doc}`experiment anatomy <concepts>` page names the actual modules and
artifacts for every step.

## Next

- {doc}`Install pimm <installation>`.
- {doc}`Complete the CI-sized first experiment <quickstart>`.
- If you already have a checkpoint, go to {doc}`Pretrained models
  <../models/pretrained>`.
