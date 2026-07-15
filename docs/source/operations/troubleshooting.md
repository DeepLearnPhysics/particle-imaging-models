# Troubleshooting

Start with the exact symptom. Preserve the first traceback and run the narrow
diagnostic before changing dependencies or configs.

## Installation and command path

### `error: Failed to spawn: pimm`

**Cause:** `uv run` is outside the project that declares the command.

```bash
pwd
test -f pyproject.toml && rg '^name = "pimm"' pyproject.toml
uv run --project /path/to/particle-imaging-models pimm --help
```

### `ModuleNotFoundError` for `spconv`, `pointops`, or another native package

**Cause:** launcher-only environment, unsupported platform, or environment
drift.

```bash
uname -srm
uv lock --check
uv sync --locked
uv run python -c "import torch, spconv, pointops; print(torch.__version__, torch.version.cuda)"
```

Full training is supported on Linux x86-64/Python 3.10. Re-sync the complete
lock or use the release container; do not install an arbitrary wheel version on
top.

### `torch.cuda.is_available()` is false

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)"
```

The host driver must support the locked CUDA 12.6 runtime. Container execution
also needs `--nv` (Apptainer) or `--gpus` (Docker).

## Data

### `PILArNet data root not found`

```bash
printf '%s\n' "$PILARNET_DATA_ROOT_V2"
find /path/to/pilarnet/v2 -maxdepth 2 -name '*.h5' | head
```

Set an explicit `data_root`, export the revision-matched variable before the
job starts, or use the documented cache path. Confirm the site profile passes
the variable into compute nodes/containers.

### `Key ... not found` in a transform

Apply the reader and transform steps separately. The chosen revision/reader may
not emit the key, a previous transform may have removed it, or the order is
wrong. Print keys and shapes after each step as shown in {doc}`Transforms
<../data/transforms>`.

### Point fields have different lengths

A subsampling transform did not index a new point-aligned key. Add it to
`index_valid_keys` before the first filter/crop/grid operation and add an
alignment assertion to the transform test.

## Launcher and Slurm

### A launcher flag is rejected

```bash
uv run pimm launch --help
```

Launcher fields use dotted Tyro flags such as `--resources.nproc-per-node`.
Training overrides are bare `key=value` tokens after `--`.

### Slurm requested the wrong GPUs/account/partition

Run the identical command with `--dry-run` and inspect the rendered manifest.
Check the site profile's `gpu_directive`, account, partition, QOS, constraint,
and resource overrides. Clusters differ between `--gres=gpu:N` and
`--gpus-per-node=N`.

### Multi-node job hangs during NCCL initialization

Preserve the NCCL error and inspect rank/node environment, reachability, GPU
allocation, network-interface exclusions, and rendezvous settings. First prove
one GPU and one node with the same image and data. Do not copy NCCL variables
from another cluster without site validation.

## Training

### Can I start training from a notebook?

Yes, but launch the supported CLI as a child process. Do not use
`Trainer(cfg).train()` as the notebook entrypoint.

On a GPU workstation or inside an interactive allocation:

```text
# Jupyter/IPython cell
%cd /path/to/particle-imaging-models
!uv run pimm launch --site local --resources.nproc-per-node 1 --train.config tests/tiny_semseg --run.name notebook-smoke -- epoch=1 data.train.max_len=32 data.val.max_len=16 batch_size=4 num_worker=0 use_wandb=False
```

From a cluster login node, submit the job rather than training on the login
node:

```text
# Inspect first; then repeat without --dry-run.
!uv run pimm submit --site <cluster> --train.config <config> --run.name <name> --dry-run
!uv run pimm submit --site <cluster> --train.config <config> --run.name <name>
```

The `!` is IPython's shell-command syntax. It keeps the training runtime in a
separate process while output streams into the cell. A local `pimm launch`
cell remains attached to that notebook session; use `pimm submit` for a long
job that must survive a closed browser or restarted kernel.

{py:class}`~pimm.engines.train.Trainer` is a registry implementation, not a
standalone public launcher. Calling it directly skips parts of the supported
entrypoint around it: experiment-config parsing and artifact creation,
per-rank derivation through {py:func}`~pimm.engines.defaults.default_setup`,
`torchrun` process creation, distributed setup/cleanup, code snapshotting,
resume-checkpoint selection, and container/site setup. It also lets a notebook's
existing CUDA or W&B state leak into the run, and an interrupted cell can leave
resources alive in the kernel. Even for one GPU, use `pimm launch`; use normal
Python cells for loading data, transforms, models, and inference.

### `batch_size` assertion fails

Configured batch sizes are global and explicit values must divide world size.
For eight ranks, choose `batch_size` divisible by eight. `num_worker` is also a
global total and is integer-divided per rank.

### CUDA out of memory after many steps

Packed events vary in point count. Log total points and GPU allocated/reserved
memory, inspect the tail of the event-size distribution, and reproduce the
specific batch. Lowering only event batch count may not bound a pathological
event; use scientifically justified selection/crop limits or point-aware
batching when available.

### Loss is NaN or diverges

On a bounded subset, log loss components, gradient norm, AMP state, input
ranges/non-finite values, and schedule. Re-run without AMP as a diagnostic, not
as proof that precision was the root cause. Verify preprocessing and checkpoint
load before tuning optimization.

### No model parameters loaded

pimm raises rather than continuing from random initialization. Check the
checkpoint type, filename, prefix/key mapping, target backbone type, and config
revision. Inspect the grouped missing/unexpected-key report.

### Fine-tuning trains the wrong parameters

Add {py:class}`ParameterCounter <pimm.engines.hooks.diagnostics.ParameterCounter>`,
print all `requires_grad=True` names, and verify adapter
placement or freeze rules before the optimizer step.

## Checkpoints and resume

### `Incomplete checkpoint directory`

```bash
find exp/<group>/<run>/model -maxdepth 3 -name '.complete' -o -name 'weights.pth'
uv run python -m pimm.utils.path latest-checkpoint exp/<group>/<run>/model
```

Do not create `.complete` manually. Use the newest checkpoint that contains all
required artifacts.

### Resume restarts the saved epoch

This is expected when world size or per-rank workers changed, the loader cursor
is missing, or `resume_strict_state=False`. Read the warning and record possible
batch replay. See {doc}`Checkpoints <checkpoints>`.

### Resume from a Hub URI fails

Expected. Hub artifacts carry model weights/config, not optimizer and loader
state. Warm-start a new run or resume from the local original `model/last`.

## Evaluation

### `model_best.pth` is never written

Confirm `evaluate=True`, a validation loader exists, an evaluator writes the
selection metric, and {py:class}`CheckpointSaver
<pimm.engines.hooks.checkpoint.CheckpointSaver>` appears **after** that evaluator in the
hook list.

### Metrics differ after re-evaluation

Compare the source snapshot, resolved config, checkpoint hash, data revision and
split, transform/test-time augmentation, ignore labels, class map, and metric
implementation. `scripts/test.sh` expects the run's `code/` snapshot; a run made
with `--train.no-code-copy` needs the compatible checkout.

## Report a useful issue

Include:

- pimm version/commit and whether the worktree is modified;
- operating system, Python, PyTorch/CUDA, driver, GPU model/count;
- exact command and resolved config (redact secrets/absolute private paths);
- smallest reproducible data fixture or schema/shape summary;
- first traceback plus relevant preceding logs;
- for distributed failures, rank/node/job IDs and scheduler output;
- what you expected and what happened.

Search existing [issues](https://github.com/DeepLearnPhysics/particle-imaging-models/issues)
before opening a new one.
