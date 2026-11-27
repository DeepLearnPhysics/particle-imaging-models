# Install

You run pimm from a checkout. The lower bounds are in `pyproject.toml`, and a reproducible resolution is in `uv.lock`.

| Need | Version |
|---|---|
| Operating system | Linux x86-64 to train; on other platforms, you can install only the launcher |
| Python | 3.10.12+, 3.11.4+ or 3.12 (the range supported by exex) |
| PyTorch | >=2.4 |
| Triton | >=3.0 |
| GPU | NVIDIA with a driver compatible with the installed PyTorch CUDA runtime |
| Package manager | [uv](https://docs.astral.sh/uv/) |

pimm's package dependencies are the core model, data and engine packages, so if you `pip install` into an existing environment, you keep its PyTorch (see below). spconv is required: its GPU distributions are named `spconv-cuXXX`, such as `spconv-cu126` for CUDA 12.6. The dependency in this checkout is `spconv-cu126`; if you change CUDA, choose the matching distribution. The CUDA runtime bundled with PyTorch and the GPU driver must also be compatible.

## Install with one command

```bash
curl -sSL https://raw.githubusercontent.com/DeepLearnPhysics/particle-imaging-models/main/install.sh | bash
cd particle-imaging-models
```

The script has four steps: install uv if it's missing, clone the repository into `particle-imaging-models/`, run `uv sync --locked`, and check that pimm, PyTorch and spconv can be imported. Run pimm commands through `uv run`; there's no environment to activate.

Check the installation:

```bash
uv run pimm --help
uv run pimm launch --train.config tests/tiny_semseg --resources.nproc-per-node 1 --dry-run
```

The second command is a dry run. You see the `torchrun` command that would be run, but no data is read and no training is started.

## Install by hand

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models
uv sync --locked
```

With `--locked`, you get the recorded resolution: PyTorch 2.10 with CUDA 12.6, plus a prebuilt `torch-scatter` wheel for it. pimm itself has only lower bounds (`torch>=2.4`); to use another PyTorch, install pimm with pip as below.

## Install with pip

You can install pimm with pip into a new environment or into one that already has PyTorch 2.4 or newer, with Python 3.10.12 to 3.12.

1. Create and activate an environment, unless you already have one:

   ```bash
   python3 -m venv pimm-env
   source pimm-env/bin/activate
   ```

2. Install PyTorch for your CUDA version, unless it is already installed. For example, for CUDA 12.8:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```

3. Install `torch-scatter` from the [PyG wheel index](https://data.pyg.org/whl/) for your PyTorch and CUDA versions. PyPI has only its source, so without the index you'd have to compile it. For PyTorch 2.8 with CUDA 12.8:

   ```bash
   python -c "import torch; print(torch.__version__)"  # e.g. 2.8.0+cu128
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
   ```

4. Install pimm with its remaining dependencies. The package is named `particle-imaging-models` and imported as `pimm`:

   ```bash
   pip install particle-imaging-models
   ```

   The recipes and launcher settings are not in the package, so `pimm launch` and `pimm submit` run from a clone. To train, install pimm from a clone instead:

   ```bash
   git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
   cd particle-imaging-models
   pip install -e .
   ```

5. Check the installation:

   ```bash
   python -c "import torch, spconv.pytorch, torch_scatter, pimm.models; print('ok', torch.__version__)"
   pimm --help
   ```

Install extras the same way, for example `pip install "particle-imaging-models[tensorboard]"`, or `pip install -e ".[tensorboard]"` in a clone. For FlashAttention, install a `flash-attn` wheel built for your PyTorch and CUDA versions, for example from [these prebuilt wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels), or use PyTorch's attention with `attention_backend=torch`. The dependency `spconv-cu126` is compatible with CUDA 12.x builds of PyTorch; for another CUDA major version, install the matching `spconv-cuXXX` first.

## Optional operators and writers

| Extra | Used by |
|---|---|
| `cluster` | `DINOEnhancedSegmentor` nearest-neighbor lookup |
| `tensorboard` | TensorBoard event writers; without this extra, metrics are still logged |
| `nvml` | System GPU resource monitoring |
| `rich`, `matplotlib`, `termcolor` | Formatted tables, figures and colored logs |

For example, run `uv sync --locked --extra tensorboard` to enable TensorBoard. FlashAttention is not an extra: `uv sync` installs it by default, as the `flash-attn` dependency group, and `--no-group flash-attn` leaves it out. Add `--group eval` for the parquet evaluation scripts.

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

On a GPU without BF16 or Flash Attention, override the recipe. Put training-config overrides after a bare `--`:

```bash
uv run pimm launch --train.config <recipe> -- enable_amp=True amp_dtype=float16 attention_backend=torch
```

With `attention_backend=torch`, PyTorch's attention is used in every attention layer of a Panda V2 run, in the encoder and in the detector. You choose the attention for each run rather than for each model, so you can, for example, fine-tune on L40S GPUs a backbone that was pretrained with FlashAttention on A100s. To load a Panda V2 model on such a GPU, call `pimm.from_pretrained(..., attention_backend="torch")`.

For Point Transformer V3 and the Panda V1 detector, FlashAttention is set with an `enable_flash` argument instead. In Point Transformer V3 recipes, set `model.backbone.enable_flash=False`; Panda V1 detector recipes also need `model.enable_flash=False`. In other recipes, the field can be elsewhere; search the recipe for `enable_flash`.

Panda models need CUDA because their backbone requires spconv. You can also run PoLAr-MAE inference on CPU.

## Containers

The image `ghcr.io/deeplearnphysics/pimm:main` contains the locked environment, in `/opt/pimm/.venv`, but not the pimm source. Run it from your checkout.

With Apptainer, where the current directory is bound by default:

```bash
apptainer pull pimm.sif docker://ghcr.io/deeplearnphysics/pimm:main
apptainer exec --nv pimm.sif pimm launch --train.config tests/tiny_semseg --dry-run
```

With Docker:

```bash
docker run --rm --gpus all -v "$PWD:$PWD" -w "$PWD" ghcr.io/deeplearnphysics/pimm:main \
  pimm launch --train.config tests/tiny_semseg --dry-run
```

On NERSC Perlmutter, use `ghcr.io/deeplearnphysics/pimm-nersc`. If you name an image in a site profile, the checkout is mounted inside it (at `/opt/pimm/src` by default) when you run `pimm launch` or `pimm submit`; see [Scale up](../training/scale.md).

## Launcher only

On a login node that you use only to submit jobs, you can skip the training dependencies:

```bash
./install.sh --launcher-only
```

You get the locked `launcher` group and pimm itself, without the training dependencies. Run commands with `uv run --no-sync pimm`; with a plain `uv run`, you would also install the training environment. You can then render and submit jobs, but not train.

## Settings

To keep settings with the checkout, put shell assignments in a `.env` file at the repository root; the file is sourced in `scripts/train.sh` before training. See [Environment variables](../reference/environment.md) for every pimm environment variable.

## If the installation fails

`pimm` isn't found under `uv run`
: Run from the checkout, or point uv at it: `uv run --project /path/to/particle-imaging-models pimm --help`.

A native module can't be imported
: Check `uv run python -c "import torch, spconv; print(torch.__version__, torch.version.cuda)"`, then install the optional operator for that PyTorch/CUDA pair. Source builds need the CUDA development toolkit and a compatible compiler.

CUDA isn't available
: Run `nvidia-smi`, then `uv run python -c "import torch; print(torch.cuda.is_available())"`. You can do a dry run without a GPU, but not training.
