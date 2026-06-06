<div align="center">

# Particle Imaging Models (pimm)
### Foundation model research for particle imaging detectors

</div>

A codebase for perception research for time projection chambers (TPCs), with a focus on liquid argon TPCs, built on the [Pointcept](https://github.com/Pointcept/Pointcept) training and inference framework.

This repository currently deals with 3D charge clouds only, with plans to incorporate 2D
images (e.g., wireplane waveforms) and other modalities in the near future.

## Overview

**pimm** adapts methods in deep learning and computer vision for event reconstruction in LArTPC detectors. This repository provides:

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

## Quick Start

### Using the container (recommended)

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models
apptainer pull /path/to/pimm.sif docker://youngsm/pimm:pytorch2.5.0-cuda12.4
```

The image installs `pimm` as an editable package at `/opt/pimm/src`. Bind your
own local clone over that path so the `pimm` console script imports your local clone,
not the global install baked into the image.

#### Train (single GPU):

```bash
apptainer exec --nv \
  --bind "$PWD:/opt/pimm/src" \
  --pwd /opt/pimm/src \
  /path/to/pimm.sif \
  pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8 \
    -- epoch=1 data.train.max_len=32 data.val.max_len=16 batch_size=4 num_worker=0 use_wandb=False
```

Add extra binds for data/checkpoint filesystems when needed. For example, at
SLAC National Laboratory's S3DF cluster, include `--bind /sdf,/lscratch`.

#### Multi-GPU:

Use `--resources.nproc-per-node 4` for four local GPUs:

```bash
pimm launch --resources.nproc-per-node 4 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8
```

#### Multi-Node:

Inside a user-authored Slurm script, allocate one task per node and run:

```
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
```

```bash
srun pimm launch \
  --resources.nnodes "$SLURM_NNODES" \
  --resources.nproc-per-node "$SLURM_GPUS_ON_NODE" \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8
```

> See [Dataset Preparation](#dataset-preparation) to download PILArNet-M.

### From source

```bash
git clone https://github.com/youngsm/particle-imaging-models.git
cd particle-imaging-models
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 8.6 8.9 9.0}"
conda env create -f environment.yml
conda activate pimm-torch2.5.0-cu12.4
pip install -e .
pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8 \
  -- epoch=1 data.train.max_len=32 data.val.max_len=16 batch_size=4 num_worker=0 use_wandb=False
```

Requires CUDA 11.6+ for FlashAttention (set `enable_flash=False` in configs if unavailable).
The `TORCH_CUDA_ARCH_LIST` export lets the LitePT/PointROPE CUDA extension build
on login or container build hosts without a visible GPU.

## Multi-Node Training

You can either run `pimm launch` inside your own Slurm script or use the
managed submitit path:

```bash
pimm submit --site s3df \
  --resources.nnodes 2 \
  --resources.nproc-per-node 4 \
  --resources.time 02:00:00 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8
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
pimm submit --site s3df \
  --resources.nnodes 1 \
  --resources.nproc-per-node 4 \
  --resources.time 00:30:00 \
  --train.config panda/pretrain_geometry_combos/pretrain-sonata-v1m1-pilarnet-e050-head512-tail-wd20
```

For a saved run recipe with launch-time state such as checkpoint weights,
resource overrides, or W&B naming:

```bash
pimm submit --site s3df --recipe launch/runs/e050_tail.yaml --train.config panda/pretrain_geometry_combos/pretrain-sonata-v1m1-pilarnet-e050-head512-tail-wd20
pimm submit --site nersc --recipe launch/runs/e050_tail.yaml --train.config panda/pretrain_geometry_combos/pretrain-sonata-v1m1-pilarnet-e050-head512-tail-wd20
```

Always dry-run first when changing sites or resources:

```bash
pimm submit --dry-run --site s3df \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8
```

To run directly on the current node without Slurm:

```bash
pimm launch \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8 \
  --resources.nproc-per-node 4
```

`pimm launch` is local or in-allocation execution. `pimm submit` is managed
Slurm submission.

Slurm submission uses submitit. `--site s3df --submit.host iana` SSHes to `iana`,
activates the `pointcept-torch2.5.0-cu12.4` mamba environment, and submits from
the configured `paths.repo_root`. `--site nersc` assumes the launcher is run on a
NERSC login node and uses the Shifter/Perlmutter settings from
`launch/sites/nersc.yaml`. See `launch/README.md` for the YAML layer details and
override syntax.

Because `pimm` is a console-script entry point, every login/submit host where
you type `pimm launch` or `pimm submit` needs the checkout installed in the
active environment with `pip install -e .`. Containerized jobs additionally bind
`paths.repo_root` over `/opt/pimm/src` by default so the in-image editable install
also resolves to the same checkout.

## Training & Testing

The primary entry point is `pimm launch`:

```bash
pimm launch --train.config <config-path> [-- TRAIN_OVERRIDES...]
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
pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8 \
  -- epoch=10 data.train.max_len=1000

# Fine-tune from pre-trained checkpoint
pimm launch --resources.nproc-per-node 4 \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-lin \
  --train.weight /path/to/checkpoint.pth

# Resume
pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8 \
  --run.name my_experiment --train.resume
```

See [Config Structure](#config-structure) for more on how configs work.

By default, the launcher snapshots the codebase into
`exp/<dataset>/<name>/code/` and runs the code from this snapshot for
reproducibility. Use `--train.no-code-copy` to run directly from the repo source.

Model checkpoints, which can be quite large, are saved to `exp/<dataset>/<name>/model/`. To redirect to a separate disk, set `MODEL_DIR` in your `.env` file or environment; this will save the checkpoint to `MODEL_DIR` and symlink it to `exp/<dataset>/<name>/model`.

Export a training checkpoint to a portable pretrained directory for fine-tuning
or Hugging Face upload:

```bash
pimm export --run-dir exp/panda/pretrain/my_run last ./artifacts/my-model
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
   pimm launch --train.config panda/pretrain/x -- epoch=50 data.train.max_len=500000
   ```

Example configs can be found in:
- `configs/panda/pretrain/` - Pre-training configurations
- `configs/panda/semseg/` - Semantic segmentation configurations  
- `configs/panda/panseg/` - Panoptic segmentation configurations

## Dataset Preparation

### PILArNet-M

Download the 168GB dataset from Hugging Face:

```bash
python tools/download_pilarnet.py --version v2 --output-dir /path/to/dir
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

### Docker / Apptainer

Pre-built images are available on Docker Hub:

| Image | Description |
|-------|-------------|
| `youngsm/pimm:pytorch2.5.0-cuda12.4` | Standard image |
| `youngsm/pimm-nersc:pytorch2.5.0-cuda12.4` | NERSC variant with extra dependencies |

```bash
apptainer pull /path/to/pimm.sif docker://youngsm/pimm:pytorch2.5.0-cuda12.4
```

## Model Zoo

### Model Versioning

Models use `vXmY` naming (version X, mode Y). Different modes indicate small architecture variants, while versions indicate large architectural changes.

### Backbones

- **[PTv3](https://arxiv.org/abs/2312.10035)** (Point Transformer V3) — efficient backbone with FlashAttention. Requires `spconv` and CUDA 11.6+ for FlashAttention (which is optional)
- **SparseUNet** — SpConv-based UNet.
- **[PTv2](https://arxiv.org/abs/2210.05666)**, **[PTv1](https://arxiv.org/abs/2012.09164)** — earlier Point Transformer versions.

### Pre-training

- **[Panda/Sonata](https://arxiv.org/abs/2503.16429)** — DINO-style self-supervised learning with teacher-student framework and online prototype clustering.
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
pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8 \
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


## Acknowledgements

Built on [Pointcept](https://github.com/Pointcept/Pointcept). Thanks to them!

## License

MIT (inherited from Pointcept).
