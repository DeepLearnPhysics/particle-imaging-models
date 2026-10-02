# Troubleshooting

Find the symptom, run the check, apply the fix.

## Installation

`error: Failed to spawn: pimm`
: `uv run` was called outside the checkout. Run from the repository root or use `uv run --project /path/to/particle-imaging-models`.

`ModuleNotFoundError` for `spconv`, `pointops` or another native package
: Run `uname -srm` (training needs Linux x86-64), `uv lock --check`, then `uv sync --locked`.

`torch.cuda.is_available()` is `False`
: Check `nvidia-smi` and that the driver supports CUDA 12.6. Containers need `--nv` (Apptainer) or `--gpus all` (Docker).

## Data

`PILArNet data root not found`
: Set `data_root` in the config, export `PILARNET_DATA_ROOT_V3`, or download into `~/.cache/pimm/pilarnet/v3`. On a cluster, put the variable in the site profile's `env:` so compute nodes see it.

`PILArNet revision='v1' is not supported`
: The recipe names revision `v1`. Override each split to `v3`, for example `data.train.revision=v3 data.val.revision=v3`.

`Key ... not found` in a transform
: The reader didn't emit the key, or an earlier transform removed or renamed it. Apply the transforms one at a time and print the keys after each.

Point fields have different lengths
: A per-point key is missing from `index_valid_keys`, so subsampling skipped it. See [Transforms](../data/transforms.md#point-aligned-keys).

## Launcher and Slurm

A launcher flag is rejected
: Launcher flags are dotted, such as `--resources.nproc-per-node`. Training values go after `--` as `key=value`. `uv run pimm launch --help` lists the flags.

The job asked for the wrong account, partition or GPUs
: Run the same command with `--dry-run` and read the script. Check the site profile; without `--site`, `pimm submit` uses `s3df`.

A multi-node job hangs at NCCL initialization
: Run one GPU on one node with the same image and data first, then check the site's network-interface settings and the rendezvous values in the dry run.

## Training

`batch_size` assertion fails
: Batch sizes are global and must divide by the number of GPUs.

CUDA runs out of memory after many steps
: Events differ in size, so one large event can overflow a batch. Log the total points per batch and the GPU memory to find it.

Loss is NaN
: On a few events, log the loss components and gradient norm (`GradientNormLogger`), then rerun with `enable_amp=False`.

No model parameters loaded
: The checkpoint's keys don't match the model. Check the weight path, prefix and key mapping in the startup report.

The wrong parameters train
: Add `ParameterCounter` and print the names with `requires_grad=True` before the first step.

## Checkpoints and evaluation

`Incomplete checkpoint directory`
: The checkpoint has no `.complete`. Use `uv run python -m pimm.utils.path latest-checkpoint <run>/model` to find the newest complete one.

The saved epoch restarted on resume
: The number of GPUs or workers changed; see [Checkpoints and resume](checkpoints.md#changing-the-number-of-gpus).

`model_best.pth` is never written
: `evaluate` is `False`, there is no validation split, or `CheckpointSaver` comes before the evaluator in `hooks`.

## Reporting a problem

Open an [issue](https://github.com/DeepLearnPhysics/particle-imaging-models/issues) with the pimm commit, the exact command, the resolved config, the first traceback, and your OS, GPU, driver and CUDA versions.
