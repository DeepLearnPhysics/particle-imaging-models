# Installation

**Outcome:** a locked pimm environment and a launcher dry run that succeeds
without data or a GPU job.

## Recommended: one-line installation

```bash
curl -sSL https://raw.githubusercontent.com/DeepLearnPhysics/particle-imaging-models/main/install.sh | bash
```

This installs `uv` when necessary, clones pimm into
`particle-imaging-models/`, runs `uv sync --locked`, then imports PyTorch and
the native operators. Enter the checkout and use `uv run`; no shell activation
is required:

```bash
cd particle-imaging-models
```

Verify the command path and config resolution:

```bash
uv run pimm --help
uv run pimm launch \
  --train.config tests/tiny_semseg \
  --resources.nproc-per-node 1 \
  --dry-run
```

Success is a rendered `torchrun` command or launch script and exit status 0.
No dataset is opened and no training process starts.

## Compatibility

| Component | Supported path |
|---|---|
| Operating system | Linux x86-64 for the full training environment |
| Python | 3.10 |
| PyTorch | 2.10.0 from the lockfile |
| CUDA runtime | 12.6, installed with PyTorch and the native wheels |
| Host requirement | NVIDIA driver compatible with the CUDA 12.6 runtime |
| Package manager | `uv` 0.11.1 or newer |

macOS can install the small launcher environment, but the CUDA training group
and native sparse operators are Linux-only. A host CUDA toolkit and compiler are
not required for the recommended installation because the extensions are
prebuilt.

### Supported GPUs

pimm's prebuilt CUDA extensions cover NVIDIA compute capabilities 7.0 through
9.0. The attention and automatic mixed precision (AMP) settings depend on the
GPU:

| GPU family | Compute capability | Flash Attention | Recommended AMP dtype |
|---|---:|---|---|
| V100 | 7.0 | Off | FP16 |
| RTX 20xx | 7.5 | Off | FP16 |
| A100 | 8.0 | On | BF16 |
| RTX 30xx | 8.6 | On | BF16 |
| RTX 40xx | 8.9 | On | BF16 |
| L40S | 8.9 | Off | BF16 |
| H100 / H200 | 9.0 | On | BF16 |

For a V100 or RTX 20xx recipe that normally uses BF16, select FP16 instead:

```bash
-- enable_amp=True amp_dtype=float16
```

Full-precision training is also available with `enable_amp=False`. On a V100,
RTX 20xx, or L40S, disable every `enable_flash` field in the selected recipe.
For the Panda semantic-segmentation recipes that is:

```bash
-- model.backbone.enable_flash=False
```

Panda detector recipes have a second attention stack, so disable both fields:

```bash
-- model.backbone.enable_flash=False model.enable_flash=False
```

These are training-config overrides and therefore go after the bare `--` in a
`pimm launch` or `pimm submit` command. Other model families may place the
field elsewhere; search the selected config for `enable_flash` and turn off
each occurrence.

## Manual installation

Use these steps when you want to manage the checkout yourself:

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models
curl -LsSf https://astral.sh/uv/install.sh | sh  # omit if uv is already installed
source "$HOME/.local/bin/env"                    # omit if uv was already installed
uv sync --locked
```

Do not omit `--locked`: the lockfile, native wheels, PyTorch, and CUDA version
form one tested environment. Use `uv lock --check` to diagnose a modified or
out-of-date lockfile without changing it.

## Container

The published image contains the locked environment at `/opt/pimm/.venv`, but
**not the source tree**. Run it from a pimm checkout so imports resolve to the
code you have checked out.

::::{tab-set}

:::{tab-item} Apptainer
```bash
apptainer pull pimm.sif \
  docker://ghcr.io/deeplearnphysics/pimm:main

apptainer exec --nv pimm.sif \
  pimm launch --train.config tests/tiny_semseg --dry-run
```

Run this from the repository root. Apptainer normally binds and preserves the
current working directory, so a separate `$PWD` to `/opt/pimm/src` bind is not
needed. If your site disables the default current-directory bind, bind the
checkout at the **same path** and set `--pwd` to that path.
:::

:::{tab-item} Docker
```bash
docker run --rm --gpus all \
  -v "$PWD:$PWD" \
  -w "$PWD" \
  ghcr.io/deeplearnphysics/pimm:main \
  pimm launch --train.config tests/tiny_semseg --dry-run
```

Docker does not expose the host checkout by default, so it still needs a
volume mount. The mount can remain at the checkout's host path; it does not
need to be `/opt/pimm/src`.
:::

::::

Use `ghcr.io/deeplearnphysics/pimm-nersc` only for the NERSC/Perlmutter
environment. For a published experiment, record the immutable image digest in
the experiment metadata.

When pimm itself launches a configured Singularity, Shifter, or Docker image,
it mounts `paths.repo_root` at `container.repo_mount` automatically. The
default in-container path is `/opt/pimm/src`; you do not add that bind to the
launch command yourself.

## Launcher-only hosts

Login nodes and remote submission hosts may need only YAML parsing, Tyro, and
Submitit:

```bash
./install.sh --launcher-only
# equivalent:
uv sync --locked --no-default-groups
```

This environment can render and submit jobs. It cannot import the full model
stack or train.

## Environment variables

Copy the template when you want repository-local settings:

```bash
cp example.env .env
```

`scripts/train.sh` and `scripts/test.sh` source `.env` with ordinary shell
semantics, so an assignment in the file replaces a same-named exported value.
Direct Python calls and `pimm export` do not source it.

| Variable | Purpose |
|---|---|
| `PILARNET_DATA_ROOT_V1` | PILArNet-M v1 directory |
| `PILARNET_DATA_ROOT_V2` | PILArNet-M v2 directory |
| `MODEL_DIR` | alternate filesystem for checkpoint directories |
| `WANDB_API_KEY` | non-interactive Weights & Biases login |
| `HF_TOKEN` | Hugging Face access/upload token |
| `HF_HUB_CACHE` | Hugging Face dataset and model cache location |

Site profiles may set additional NCCL, HDF5, and certificate variables. Keep
those site-specific values in `launch/sites/<site>.yaml`, not in a training
config.

## Common failures

### `uv run` cannot find `pimm`

Run from the checkout or select it explicitly:

```bash
uv run --project /path/to/particle-imaging-models pimm --help
```

### A native module cannot be imported

Confirm the supported platform and that the locked default group was installed:

```bash
uname -srm
uv sync --locked
uv run python -c "import torch, spconv, pointops, torch_scatter; print(torch.__version__, torch.version.cuda)"
```

Do not repair one native wheel in isolation; it must match Python, PyTorch, and
CUDA. Re-sync the lockfile or use the release container.

### CUDA is unavailable

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

The bundled launcher dry run works without a visible GPU. Training does not.

More symptoms are indexed in {doc}`Troubleshooting
<../operations/troubleshooting>`.

## Next

Continue with the {doc}`first experiment <quickstart>`, which downloads a few
hundred liquid argon time projection chamber (LArTPC) images from
[PILArNet-M-mini](https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M-mini).
