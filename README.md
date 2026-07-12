<div align="center">

<h1><img src="assets/logo.svg" alt="pimm logo" height="48" valign="middle">&nbsp; Particle Imaging Models (pimm)</h1>

### Foundation model research for particle imaging detectors

</div>

A codebase for perception research for neutrino physics, built on the [Pointcept](https://github.com/Pointcept/Pointcept) training and inference framework.

This repository currently deals with 3D point clouds only, with plans to incorporate 2D
images (e.g., wireplane waveforms) and other modalities in the near future.

## Overview

**pimm** adapts methods in deep learning and computer vision for event reconstruction neutrino detectors. This repository provides:

- **Self-supervised pre-training**: discriminative pre-training ([Sonata](https://arxiv.org/abs/2503.16429)) for learning good representations
of LArTPC images.
- **Panoptic segmentation** (PointGroup, Panda Detector) models for particle and interaction instance/semantic segmentation
- **Semantic segmentation** models for per-pixel segmentation.

In sum, **pimm** integrates the following works:  
**Backbone**: 
[MinkUNet](https://github.com/NVIDIA/MinkowskiEngine), [SpUNet](https://github.com/traveller59/spconv) (see [SparseUNet](#sparseunet)),
[PTv1](https://arxiv.org/abs/2012.09164), [PTv2](https://arxiv.org/abs/2210.05666), [PTv3](https://arxiv.org/abs/2312.10035) (see [Point Transformers](#point-transformers)),
**Instance Segmentation**: 
[PointGroup](https://github.com/dvlab-research/PointGroup) (see [PointGroup](#pointgroup)),  
[Panda Detector](https://arxiv.org/abs/2512.01324) (see [Panda Detector](#detector));  
**Pre-training**: 
[Sonata](https://arxiv.org/abs/2503.16429) (see [Sonata](#sonata)),
[PoLAr-MAE](https://arxiv.org/abs/2502.02558) (see [PoLAr-MAE](#polar-mae));  
**Datasets**:
[PILArNet-M](https://arxiv.org/abs/2502.02558) (see [PILArNet-M](#pilarnet-m)) 

### TODO

We are looking at including the following models/modalities in the future:
- [ ] [SPINE](https://arxiv.org/abs/2102.01033), up until postprocessing module
- [x] [PoLAr-MAE](https://arxiv.org/abs/2502.02558) pre-training and fine-tuning
- [ ] 2D TPC waveforms/networks, e.g., [NuGraph](https://arxiv.org/abs/2403.11872)
- [ ] Optical waveforms

## Setup

### Prerequisites

Linux x86_64 and an NVIDIA GPU with a recent driver. The training stack
(PyTorch and the native extensions) installs as prebuilt CUDA 12.6 wheels,
so no CUDA toolkit or host compiler is required. Flash attention kernels
ship inside PyTorch, so no separate flash-attn package is needed.

### Quick setup

Set up pimm with one command:

```bash
curl -sSL https://raw.githubusercontent.com/DeepLearnPhysics/particle-imaging-models/main/install.sh | bash
```

It installs uv, clones the repo, and syncs the locked environment.

Run everything through `uv run` from inside the checkout (no need to activate
the environment; from elsewhere, `uv run` cannot find the project and fails
with ``error: Failed to spawn: `pimm` ``):

```bash
uv run pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  -- epoch=1 data.train.max_len=32 data.val.max_len=16 batch_size=4 num_worker=0 use_wandb=False
```

`pimm launch`, `pimm submit`, and `pimm export` are the three commands; each is
also reachable as `uv run python -m pimm.cli <cmd>`.

<details>
<summary>Manual install</summary>
<br>

1. Clone the repository

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models
```

2. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

3. Install the locked environment (Linux x86_64; prebuilt CUDA wheels, nothing compiles)

```bash
uv sync --locked
```

On a login or submit host that only runs `pimm launch` / `pimm submit`, install
the small launcher environment instead:

```bash
uv sync --locked --no-default-groups
```

4. Verify the CUDA stack imports

```bash
uv run python -c "import cnms, pointgroup_ops, pointops, pointrope, pytorch3d_ops, serialize_cuda, spconv, torch, torch_cluster, torch_scatter, torch_sparse"
```

`./install.sh` runs exactly these steps (plus the platform check); set
`SKIP_CLONE=1` to install into the current directory, or `PIMM_REPO` /
`PIMM_BRANCH` to target a fork or branch.
</details>

<details>
<summary>Container (Docker / Apptainer)</summary>

Prebuilt images resolve from the same `uv.lock` as the local install.

```bash
apptainer pull /path/to/pimm.sif docker://ghcr.io/deeplearnphysics/pimm:v0.3.1
```

The image ships only the locked environment (at `/opt/pimm/.venv`) - no pimm
source is baked in. Run commands from inside your clone (bound into the
container) and they import your code. Build it yourself with:

```bash
docker build -f .github/docker/Dockerfile -t pimm:local .   # or Dockerfile.nersc on Perlmutter
```

| Image | Description |
|-------|-------------|
| `ghcr.io/deeplearnphysics/pimm` | Standard image |
| `ghcr.io/deeplearnphysics/pimm-nersc` | NERSC variant with extra dependencies |
</details>

## Training & Testing

The primary entry point on local GPU(s) is `pimm launch`:

```bash
uv run pimm launch --train.config <config-path> [-- TRAIN_OVERRIDES...]
```

The launcher invokes `scripts/train.sh`, which prepares experiment paths, code
snapshots, resume checkpoints, and then calls `torchrun`.

Useful flags:

| Flag | Description |
|------|-------------|
| `--train.config` | Config path under `configs/`, with or without `.py` |
| `--run.name` | Experiment name (default: auto-generated) |
| `--resources.nproc-per-node` | Torchrun processes per node |
| `--resources.nnodes` | Number of nodes |
| `--train.weight` | Path to checkpoint |
| `--train.resume` | Resume training from last checkpoint |
| `--train.no-code-copy` | Skip code snapshot, run from repo source |
| `--dry-run` | Print rendered command/script |

```bash
# Override config values
uv run pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py \
  -- epoch=10 data.train.max_len=1000

# Fine-tune from pre-trained checkpoint
uv run pimm launch --resources.nproc-per-node 4 \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-lin \
  --train.weight /path/to/checkpoint.pth

# Resume
uv run pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py \
  --run.name my_experiment --train.resume
```

See [Config Structure](#config-structure) for more on how configs work.

By default, the launcher snapshots the codebase into
`exp/<dataset>/<name>/code/` and runs the code from this snapshot for
reproducibility. Use `--train.no-code-copy` to run directly from the repo source.

Model checkpoints, which can be quite large, are saved to `exp/<dataset>/<name>/model/`. To redirect to a separate disk, set `MODEL_DIR` in your `.env` file or environment; this will save the checkpoint to `MODEL_DIR` and symlink it to `exp/<dataset>/<name>/model`.


## Multi-Node Training

You can either run `pimm launch` inside your own Slurm script or use the
managed submitit path:

```bash
uv run pimm submit --site s3df \
  --resources.nnodes 2 \
  --resources.nproc-per-node 4 \
  --resources.time 02:00:00 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py
```

The key rule is one Slurm task per node. `torchrun` launches one process per
GPU on each node.

## HPC Launching

For routine Slurm runs, prefer `pimm submit` over copying one-off Slurm
scripts. It composes common defaults, a site profile, and optional run settings
into a submitit job.

If you just made a normal Python config and want to submit it with site
defaults:

```bash
uv run pimm submit --site s3df \
  --resources.nnodes 1 \          # 1 node
  --resources.nproc-per-node 4 \  # 4 gpus/node
  --resources.time 00:30:00 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py
```

For a saved run recipe with launch-time state such as checkpoint weights,
resource overrides, or W&B naming:

```bash
uv run pimm submit --site s3df --recipe launch/runs/e050_tail.yaml --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py
uv run pimm submit --site nersc --recipe launch/runs/e050_tail.yaml --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py
```

Always dry-run first when changing sites or resources:

```bash
uv run pimm submit --dry-run --site s3df \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py
```

To run directly on the current node without Slurm:

```bash
uv run pimm launch \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py \
  --resources.nproc-per-node 4
```

Slurm submission uses [submitit](https://github.com/facebookincubator/submitit). See `launch/README.md` for the YAML layer details and
override syntax.

Every login or submit host where you invoke `pimm launch` or `pimm submit` needs
the launcher dependencies.
Run `./install.sh --launcher-only`, then use `uv run pimm submit ...`.
Containerized jobs bind `paths.repo_root` at `container.repo_mount` (default
`/opt/pimm/src`); the image itself carries only the environment at
`/opt/pimm/.venv`.

## Exporting models

Export a training checkpoint to a portable pretrained directory for fine-tuning
or Hugging Face upload:

```bash
uv run pimm export --run-dir exp/panda/pretrain/my_run last ./artifacts/my-model
```

This supports split checkpoint directories such as `model/last` and legacy
`.pth` checkpoints. Use `pimm export --help` for direct checkpoint paths, safe
serialization, and Hub upload options.

## Configuration System

Configurations are Python dictionary-based files located in the `configs/` directory. Each config file defines the model architecture, dataset settings, training hyperparameters, and different hooks to run during training (checkpoint saving, logging, evaluation).

### Config Structure

Configs use a hierarchical structure with `_base_` inheritance:

```python
_base_ = ["../../_base_/default_runtime.py"]

# Override or add settings
model = dict(type="PT-v3m2", ...)
data = dict(train=dict(...), val=dict(...))
```

### Modifying Configs

You can modify configs in two ways:

1. Edit the config file directly
2. Override via command line after `--`:
   ```bash
   uv run pimm launch --train.config panda/pretrain/x -- epoch=50 data.train.max_len=500000
   ```

Example configs can be found in:
- `configs/panda/pretrain/` - Pre-training configurations
- `configs/panda/semseg/` - Semantic segmentation configurations  
- `configs/panda/panseg/` - Panoptic segmentation configurations

## Dataset Preparation

### PILArNet-M

Download the 168GB dataset from Hugging Face:

```bash
uv run python scripts/pilarnet/download.py --version v2 --output-dir /path/to/dir
```

Data saves to `~/.cache/pimm/pilarnet/v2` if `--output-dir` is not provided. After downloading the dataset, run `cp example.env .env` and set `PILARNET_DATA_ROOT_V2`. This will allow the dataloader to automatically find the data.

PILArNet has two revisions. **v2** is recommended for new models (adds PID, momentum, and vertex information). **v1** is the original dataset from the PoLAr-MAE paper. Events differ between splits, so models trained on v1 should be evaluated on v1.

## Data Format

Point cloud data should be organized with the following structure:

```python
{
    'coord': (N, 3),           # 3D hit positions [x, y, z]
    'feat': (N, C),            # Hit features (charge, time, etc.)
    'segment': (N,1),          # Semantic labels (optional, for training)
    'instance': (N,1),         # Instance IDs (optional, for training)
    ...                        # Extra attributes
}
```

The data often needs to be re-scaled to new domains that lead to more efficient training
(e.g., centering/scaling of coordinates to [-1,1]$^3$). This can be done within the Dataset class, or from a Transform. See the transform sections of configuration files for more details.

### Packed Data Format

This library works with packed data, where all batched quantities are in two dimensions instead of three, i.e. `(N, 3)` instead of `(B, N, 3)`. This is because point clouds are variable length, and getting to a 3 dimensional tensor would require padding. Instead of padding, there is an `offset` tensor, which is of length `B` and gives the indices in the packed tensors at which a point cloud ends and a new one starts.

`Offset` is conceptually similar to the concept of `Batch` in PyG, and can be seen as the cumulative sum of a `lengths` tensor. A visual illustration of batch and offset is as follows:

<p align="center">
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset.png">
    <img alt="pointcept" src="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset.png" width="480">
    </picture><br>
</p>

## Model Zoo

### Model Versioning

Models use `vXmY` naming (version X, mode Y). Different modes indicate small architecture variants, while versions indicate large architectural changes.

### Backbones

- **[PTv3](https://arxiv.org/abs/2312.10035)** (Point Transformer V3) — efficient backbone with FlashAttention. Requires `spconv` and CUDA 11.6+ for FlashAttention (which is optional)
- **SparseUNet** — SpConv-based UNet.
- **[PTv2](https://arxiv.org/abs/2210.05666)**, **[PTv1](https://arxiv.org/abs/2012.09164)** — earlier Point Transformer versions.

### Pre-training

- **[Panda](https://arxiv.org/abs/2512.01324)/[Sonata](https://arxiv.org/abs/2503.16429)** — DINO-style self-supervised learning with teacher-student framework and online prototype clustering.
- **[PoLAr-MAE](https://arxiv.org/abs/2502.02558)** — masked autoencoder with chamfer + energy reconstruction losses.

### Instance / Panoptic Segmentation

- **[PointGroup](https://github.com/dvlab-research/PointGroup)** — clustering-based instance segmentation.
- **[Panda Detector](https://arxiv.org/abs/2512.01324)** — Mask2Former-style detection modified to take low energy deposits into account.

## Logging

pimm writes either Weights & Biases or TensorBoard logs from rank 0. Configs that
set `use_wandb=True` use W&B; set `use_wandb=False` to write TensorBoard events
under the experiment directory instead. W&B run names and projects can be
supplied from the launcher:

```bash
export WANDB_API_KEY=...
uv run pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py \
  --run.name test \
  --run.wandb-name test-display \
  --run.wandb-project Pretraining-Sonata-PILArNet-M
```

You can also authenticate with `wandb login` or by adding
`WANDB_API_KEY=your_key` to `.env` (see `example.env`).

```python
hooks = [
    dict(type="WandbNamer", keys=("model.type", "data.train.max_len", "amp_dtype", "seed")),
    ...
]
```


## Checkpoint formats

Training uses **one checkpoint format for every parallelism** — single-GPU,
multi-GPU, and multi-node all write the same thing, so resume is predictable
regardless of how many devices you used.


```
exp/<dataset>/<name>/model/
  last/                 # resume from here
    weights.pth         # portable model weights — plain `torch.load(...)["state_dict"]`
    trainer.dcp/        # optimizer / scheduler / RNG / dataloader as a DCP checkpoint
    .complete           # written last; marks the checkpoint atomically complete
  model_best.pth        # best-metric model weights only (for eval / export)
```

- **Portable weights, always.** `last/weights.pth` and `model_best.pth` are
  ordinary single-file state dicts — load them anywhere without DCP.
- **Reshards automatically.** The DCP `trainer.dcp/` lets you resume on a
  different number of GPUs/nodes with no extra flags.
- **Atomic.** Each save publishes via a temp dir + rename and a `.complete`
  marker, so an interrupted save never corrupts the previous checkpoint.

### `legacy`

A single monolithic `model_last.pth` (model + trainer state in one file),
plus a `model_best.pth` copy. Simple and dependency-free, but it does not
reshard across world sizes (resume on a different GPU count needs
`resume_strict_state=False`). You can set this with:

```bash
uv run pimm launch --train.config <config> --run.name <name> -- checkpoint_format=legacy
```

## Acknowledgements

Built on [Pointcept](https://github.com/Pointcept/Pointcept), [torchtitan](https://github.com/pytorch/torchtitan), and [torchrl](https://github.com/pytorch/rl). Thanks to them!

## License

MIT (inherited from Pointcept).
