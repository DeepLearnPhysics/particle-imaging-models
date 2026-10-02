# Install

pimm runs from a checkout of its repository, with every dependency pinned by `uv.lock`.

| Need | Version |
|---|---|
| Operating system | Linux x86-64 to train; other platforms can install the launcher only |
| Python | 3.10, installed by uv |
| GPU | NVIDIA with compute capability 7.0–9.0 and a driver that supports CUDA 12.6 |
| Package manager | [uv](https://docs.astral.sh/uv/) |

PyTorch 2.10.0 and the CUDA 12.6 runtime come from the lockfile. The native operators (spconv, pointops, flash-attn and others) are prebuilt wheels, so you don't need a CUDA toolkit or a compiler.

## Install with one command

```bash
curl -sSL https://raw.githubusercontent.com/DeepLearnPhysics/particle-imaging-models/main/install.sh | bash
cd particle-imaging-models
```

The script installs uv if it's missing, clones the repository into `particle-imaging-models/`, runs `uv sync --locked`, and checks that PyTorch and the native operators import. Run pimm commands through `uv run`; there's no environment to activate.

Check the installation:

```bash
uv run pimm --help
uv run pimm launch --train.config tests/tiny_semseg --resources.nproc-per-node 1 --dry-run
```

The dry run prints the `torchrun` command it would run and exits. It doesn't read data or start training.

## Install by hand

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models
uv sync --locked
```

`--locked` installs exactly what `uv.lock` pins. The native wheels are built for the pinned PyTorch and CUDA versions, so change them only through the lockfile.

## GPUs

| GPU | Compute capability | Flash Attention | Mixed precision |
|---|---:|---|---|
| V100 | 7.0 | off | FP16 |
| RTX 20xx | 7.5 | off | FP16 |
| A100 | 8.0 | on | BF16 |
| RTX 30xx | 8.6 | on | BF16 |
| RTX 40xx | 8.9 | on | BF16 |
| L40S | 8.9 | off | BF16 |
| H100, H200 | 9.0 | on | BF16 |

On a GPU without BF16 or Flash Attention, override the recipe. Training-config overrides go after a bare `--`:

```bash
uv run pimm launch --train.config <recipe> -- enable_amp=True amp_dtype=float16 model.backbone.enable_flash=False
```

Panda detector recipes have a second attention stack, so also pass `model.enable_flash=False`. Other recipes can place the field elsewhere; search the recipe for `enable_flash`.

Panda models need CUDA because their backbone uses spconv. PoLAr-MAE inference also runs on CPU.

## Containers

The image `ghcr.io/deeplearnphysics/pimm:main` holds the locked environment in `/opt/pimm/.venv` but not the pimm source. Run it from your checkout.

With Apptainer, which binds the current directory by default:

```bash
apptainer pull pimm.sif docker://ghcr.io/deeplearnphysics/pimm:main
apptainer exec --nv pimm.sif pimm launch --train.config tests/tiny_semseg --dry-run
```

With Docker:

```bash
docker run --rm --gpus all -v "$PWD:$PWD" -w "$PWD" ghcr.io/deeplearnphysics/pimm:main \
  pimm launch --train.config tests/tiny_semseg --dry-run
```

On NERSC Perlmutter, use `ghcr.io/deeplearnphysics/pimm-nersc`. When a site profile names an image, `pimm launch` and `pimm submit` mount the checkout inside it (at `/opt/pimm/src` by default); see [Scale up](../training/scale.md).

## Launcher only

A login node that only submits jobs can skip the training dependencies:

```bash
./install.sh --launcher-only
```

This runs `uv sync --locked --no-default-groups`. The result renders and submits jobs but can't train. It still installs PyTorch, because `pimm-data`, one of pimm's base dependencies, requires it.

## Settings

To keep settings with the checkout, put shell assignments in a `.env` file at the repository root; `scripts/train.sh` sources it before training. [Environment variables](../reference/environment.md) lists every variable pimm reads.

## If the installation fails

`uv run` can't find `pimm`
: Run from the checkout, or point uv at it: `uv run --project /path/to/particle-imaging-models pimm --help`.

A native module doesn't import
: Check the platform with `uname -srm`, run `uv sync --locked` again, then `uv run python -c "import torch, spconv, pointops; print(torch.__version__, torch.version.cuda)"`.

CUDA isn't available
: Run `nvidia-smi`, then `uv run python -c "import torch; print(torch.cuda.is_available())"`. Dry runs work without a GPU; training doesn't.
