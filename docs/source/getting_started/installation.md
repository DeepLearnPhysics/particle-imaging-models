# Installation

pimm uses [uv](https://docs.astral.sh/uv/) for Python, dependency resolution, and the project environment.
The exact dependency set is stored in `uv.lock` and is shared by local installs, container images, and documentation builds.

Clone the repository before choosing an installation path:

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models
```

## Prerequisites

A full local training installation requires:

- Linux on x86_64.
- An NVIDIA GPU and a recent driver.

That is all.
The locked training stack uses PyTorch 2.10.0, torchvision 0.25.0, Python 3.10, and CUDA 12.6.
All native extensions, including `torch-scatter`, `torch-sparse`, and `torch-cluster`, install as prebuilt RHEL 8-compatible wheels from pimm's compatibility release.
No CUDA toolkit, host compiler, or system libraries are needed for installation.
The installer does not invoke `sudo` or modify system packages.

Rebuilding a native extension from source (an occasional maintainer task) does need the CUDA 12.6 toolkit and a compatible GCC/G++ host compiler.

## Local uv environment

Bootstrap everything (uv, the clone, and the environment) with one command:

```bash
curl -sSL https://raw.githubusercontent.com/DeepLearnPhysics/particle-imaging-models/main/install.sh | bash
```

Or run the installer from an existing checkout:

```bash
./install.sh
```

### Manual installation

The installer only chains the following commands; run them yourself for full control.

1. Clone the repository:

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models
```

2. Install [uv](https://docs.astral.sh/uv/) with the official Astral installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

3. Create `.venv` and install the prebuilt wheels recorded in `uv.lock` (uv also provides Python 3.10):

```bash
uv sync --locked
```

4. Verify the CUDA packages and extensions import:

```bash
uv run python -c "import cnms, pointgroup_ops, pointops, pointrope, pytorch3d_ops, serialize_cuda, spconv, torch, torch_cluster, torch_scatter, torch_sparse"
```

Run project commands through uv without activating the environment:

```bash
uv run pimm launch --help
uv run python scripts/pilarnet/download.py --help
```

You can also activate `.venv` and use `pimm` or `python` directly:

```bash
source .venv/bin/activate
pimm launch --help
```

### Flash attention

Flash attention kernels ship inside PyTorch (`torch.nn.attention.varlen.varlen_attn`), so no separate flash-attn package is installed or compiled.
Models select the fast path with `enable_flash=True` in their configs.
On Volta and Turing GPUs (compute capabilities 7.0 and 7.5), configure models with `enable_flash=False` and disable BF16; pimm leaves those choices explicit rather than changing model or precision settings at runtime.

### Repeated installs

Installs pull prebuilt wheels, so running the installer again is fast and compiles nothing.
To reinstall a single package (for example after clearing a cache), run:

```bash
uv sync --locked --reinstall-package pointops
```

## Launcher-only environment

A login or submit host that only runs `pimm submit` does not need PyTorch or the CUDA extensions.
Install the small launcher environment with:

```bash
./install.sh --launcher-only
uv run pimm submit --help
```

Or directly, from a checkout with uv installed:

```bash
uv sync --locked --no-default-groups
```

The full training environment remains inside the selected container on compute nodes.

## Apptainer or Singularity

The prebuilt image is the shortest path on an HPC system:

```bash
apptainer pull pimm.sif docker://youngsm/pimm:pytorch2.10.0-cuda12.6
```

Run from the repository root so the image imports the current checkout:

```bash
apptainer run --nv pimm.sif \
  pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

For managed batch jobs, `pimm submit` wraps the training command in the configured image.
See {doc}`../hpc/sites`.

## Docker

Build the standard image from the same lockfile:

```bash
docker build -f .github/docker/Dockerfile -t pimm:local .
```

Run a command from the current checkout:

```bash
docker run --rm --gpus all -v "$PWD:$PWD" -w "$PWD" pimm:local \
  pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

The image ships only the locked environment at `/opt/pimm/.venv` - no pimm source is baked in.
Commands must run from inside a bound checkout, which the entrypoint places on `PYTHONPATH`.

The NERSC image adds MPICH, parallel HDF5, `mpi4py`, and an MPI-enabled build of `h5py`:

```bash
docker build -f .github/docker/Dockerfile.nersc -t pimm-nersc:local .
```

## Verify the environment

Check the locked Python and CUDA stack:

```bash
uv run python - <<'PY'
import pointops
import spconv
import torch
import torch_scatter

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name())
PY
```

Then run a short training job.
This command still requires PILArNet-M v1 on disk:

```bash
uv run pimm launch \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  -- epoch=1 data.train.max_len=32 data.val.max_len=16 \
     batch_size=4 num_worker=0 use_wandb=False
```

## Environment variables

Scripts load `.env` from the repository root when it exists, while existing shell variables take precedence:

```bash
cp example.env .env
```

- `PILARNET_DATA_ROOT_V1` and `PILARNET_DATA_ROOT_V2` select local PILArNet-M revisions.
- `MODEL_DIR` redirects checkpoints to another filesystem.
- `WANDB_API_KEY` authenticates Weights & Biases.
- `PIMM_DISABLE_SERIALIZE_CUDA=1` disables the optional CUDA serialization backend.

## Troubleshooting

:::{dropdown} ``error: Failed to spawn: `pimm` ``
`uv run` resolves commands from the project in the current directory, so outside the checkout it finds no `pimm`.
Run pimm commands from inside the `particle-imaging-models` checkout, or point uv at it explicitly:

```bash
uv run --project /path/to/particle-imaging-models pimm --help
```
:::

:::{dropdown} the training packages are missing
The training stack (`train`) is a default dependency group, so a plain `uv sync --locked` installs it.
If a host only needs the launcher, `uv sync --locked --no-default-groups` installs the minimal environment instead.
:::

## Next steps

- {doc}`quickstart` covers a complete training and fine-tuning run.
- {doc}`concepts` explains packed tensors, registries, and Python configs.
