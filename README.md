<div align="center">

<img src="https://raw.githubusercontent.com/DeepLearnPhysics/particle-imaging-models/main/assets/logo.svg" alt="pimm logo" height="72">

# particle imaging models

[Documentation](https://deeplearnphysics.org/particle-imaging-models/stable/) ·
[First run](https://deeplearnphysics.org/particle-imaging-models/stable/get-started/first-run.html) ·
[Models](https://deeplearnphysics.org/particle-imaging-models/stable/models/index.html) ·
[Recipe catalog](https://deeplearnphysics.org/particle-imaging-models/stable/reference/recipes.html)

</div>



## What's new

### 11-27-2026

- Panda V2: encoder, self-distillation pretraining and detector (`panda-{encoder,ssl,detector}-v2m1`), with recipes for
  LArTPC, sPHENIX, water Cherenkov and NEXT in `configs/panda/v2`.
- New [`pimm.ops`](https://github.com/DeepLearnPhysics/particle-imaging-models/blob/main/pimm/ops/README.md) submodule of accelerated operations.
- You can now run Panda V2 on any CUDA GPU: add `attention_backend=torch` to a run's overrides to use PyTorch's attention
  where FlashAttention can't run.
- pimm is on PyPI: run `pip install particle-imaging-models` in an environment with PyTorch 2.4 or newer.
- Panda V1 detector renamed `panda-detector-v1m1`; its recipes moved to `configs/panda/v1`.
- Removed Point Transformer v1 and v2, PointGroup and Volt.

### 04-07-2026

- PoLAr-MAE ([Young et al., 2025](https://arxiv.org/abs/2502.02558)), masked point modeling for liquid argon TPC data,
was added with its pretraining and semantic segmentation recipes.

### 12-12-2025

- First release of pimm, with Panda ([Young et al., 2025](https://arxiv.org/abs/2512.01324)): self-distillation  
pretraining on PILArNet-M with a Point Transformer V3 backbone, semantic segmentation, and a detector for particles  
and interactions.

## Introduction

pimm (particle imaging models) is a library for pretraining, fine-tuning and running foundation models on sparse point clouds
from particle-imaging detectors. You can import its models and operations into your own code, and the same package
includes a complete training framework, with a launcher for a workstation or a Slurm cluster.

## Install

On Linux x86-64 with an NVIDIA GPU:

```bash
curl -sSL https://raw.githubusercontent.com/DeepLearnPhysics/particle-imaging-models/main/install.sh | bash
cd particle-imaging-models
uv run pimm --help
```

With the install script, you get [uv](https://docs.astral.sh/uv/) if it is missing, a clone of this repository, and a
locked training environment (PyTorch 2.10 with CUDA 12.6). pimm's own requirements are only lower bounds (Python
3.10.12 or newer, PyTorch 2.4 or newer), so you can also install pimm with `pip` into an environment you already have.
spconv is required, and its distribution must match your CUDA runtime (`spconv-cuXXX`; `spconv-cu126` is selected
here). Compiled operators, TensorBoard and plotting tools are optional extras. FlashAttention is installed in the
locked environment and used by default in Panda V2. On a GPU older than Ampere, or to use PyTorch's attention instead,
set `attention_backend=torch` for a run or pass `attention_backend="torch"` to `pimm.from_pretrained`. Run pimm commands through `uv run`; there is no environment to
activate.

To install with pip instead, into an environment that has PyTorch 2.4 or newer:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-<version>+<cuda>.html  # e.g. torch-2.8.0+cu128
pip install particle-imaging-models
```

The package is imported as `pimm`. The recipes and launcher settings are not in it, so to train, run `pip install -e .` in a
clone of this repository instead.

Or run pimm in a container. The image `ghcr.io/deeplearnphysics/pimm:main` contains the locked environment but not
the pimm source, so run it from your checkout:

```bash
apptainer exec --nv docker://ghcr.io/deeplearnphysics/pimm:main pimm --help                       # Apptainer
docker run --rm --gpus all -v "$PWD:$PWD" -w "$PWD" ghcr.io/deeplearnphysics/pimm:main pimm --help  # Docker
```

See [Install](https://deeplearnphysics.org/particle-imaging-models/stable/get-started/install.html) for step-by-step
installation with uv or pip, containers, launcher-only hosts, and the GPUs on which FlashAttention and BF16 are
supported.

## Run a released model

```python
import pimm

model = pimm.from_pretrained("DeepLearnPhysics/Panda-Semantic", device="cuda")  # Hugging Face
model = pimm.from_pretrained("artifacts/my-model", device="cuda")               # local export
model = pimm.from_pretrained("exp/panda/v1/semseg/my-run/model/last", device="cuda")  # run checkpoint
```

Call `pimm.from_pretrained` to rebuild the model, load its weights and get it back in eval mode. You can give a Hugging
Face repository, a directory made with `pimm export`, or a checkpoint from one of your runs; for a checkpoint, the
architecture comes from the run's config. Prepare a model's inputs with the same transforms as the recipe it was
trained with. To label a full event, see
[Run a released model](https://deeplearnphysics.org/particle-imaging-models/stable/models/run.html); for the released
models, see [Models](https://deeplearnphysics.org/particle-imaging-models/stable/models/index.html).

## Minimal training example

Download PILArNet-M-mini, 120 events from the PILArNet-M dataset, from Hugging Face:

```bash
uv run hf download DeepLearnPhysics/PILArNet-M-mini \
  --repo-type dataset \
  --local-dir data/PILArNet-M-mini
```

Train a tiny semantic-segmentation model on one GPU of this machine:

```bash
uv run pimm launch \
  --train.config tests/tiny_semseg \
  --resources.nproc-per-node 1 \
  -- \
  data.train.data_root="$PWD/data/PILArNet-M-mini" \
  data.val.data_root="$PWD/data/PILArNet-M-mini"
```

The flags before the bare `--` are launcher settings, and the `key=value` pairs after it are overrides for the
training config. You'll find the run's config, log and checkpoints under `exp/tests/`. For what each saved file is, see
[First run](https://deeplearnphysics.org/particle-imaging-models/stable/get-started/first-run.html); to list every
recipe, run `uv run pimm ls`.

To run the same job on a Slurm cluster, use `pimm submit` with the same flags plus your account and partition. Add
`--dry-run` to print the job script without submitting it:

```bash
uv run pimm submit \
  --site slurm \
  --resources.account <account> \
  --resources.partition <gpu-partition> \
  --resources.nproc-per-node 1 \
  --resources.time 00:30:00 \
  --train.config tests/tiny_semseg \
  --dry-run \
  -- \
  data.train.data_root="$PWD/data/PILArNet-M-mini" \
  data.val.data_root="$PWD/data/PILArNet-M-mini"
```

Remove `--dry-run` to submit the job. You get your prompt back as soon as the job is queued, and you'll find the
training log in the same run directory. Without `--site`, you get the generic `slurm` profile; give your account and
partition with flags or in a site file. To write a site profile for your cluster or run on several GPUs or nodes, see
[Scale up](https://deeplearnphysics.org/particle-imaging-models/stable/training/scale.html).

To train on another detector, write its events as `.npy` files or add a reader; see
[Bring your own data](https://deeplearnphysics.org/particle-imaging-models/stable/data/your-data.html).

## What's in pimm

| Area | Contents |
|---|---|
| Models | Panda V1, Panda V2 (self-distillation) and PoLAr-MAE (masked point modeling) pretraining; semantic segmentation, the Panda detectors and event classification; Point Transformer V3, SparseUNet, LitePT and the Panda V2 encoder as backbones |
| Data | PILArNet-M and sPHENIX (HDF5), WAND water Cherenkov, NEXT, JAXTPC, LUCiD, and a generic `.npy` reader |
| Operations | [`pimm.ops`](https://github.com/DeepLearnPhysics/particle-imaging-models/blob/main/pimm/ops/README.md): accelerated attention, distillation loss, sparse convolution and serialization |
| Training | `torchrun` with DDP, an experimental FSDP2 path, Slurm through Submitit |
| Models out | Portable exports, loaded with `pimm.from_pretrained` from disk or Hugging Face |

pimm is research software; its APIs and recipes can change between versions.

## Documentation

| To | Read |
|---|---|
| see how a config becomes a training run | [How pimm works](https://deeplearnphysics.org/particle-imaging-models/stable/get-started/how-pimm-works.html) |
| change a recipe or override it on the command line | [Configs and overrides](https://deeplearnphysics.org/particle-imaging-models/stable/training/configs.html) |
| fine-tune a released model | [Fine-tune](https://deeplearnphysics.org/particle-imaging-models/stable/models/fine-tune.html) |
| resume a run or find its checkpoints | [Checkpoints and resume](https://deeplearnphysics.org/particle-imaging-models/stable/training/checkpoints.html) |
| export weights or publish them on Hugging Face | [Export and publish](https://deeplearnphysics.org/particle-imaging-models/stable/models/export.html) |
| add a model, dataset, transform, loss or hook | [Add a component](https://deeplearnphysics.org/particle-imaging-models/stable/develop/add-component.html) |
| fix a failed run | [Troubleshooting](https://deeplearnphysics.org/particle-imaging-models/stable/training/troubleshooting.html) |
| look up a command, recipe, registered name or environment variable | [Reference](https://deeplearnphysics.org/particle-imaging-models/stable/reference/index.html) |

## Contributing

See [Contributing](https://deeplearnphysics.org/particle-imaging-models/stable/develop/contributing.html).

pimm is built on [Pointcept](https://github.com/Pointcept/Pointcept),
[torchtitan](https://github.com/pytorch/torchtitan) and [TorchRL](https://github.com/pytorch/rl), and it is
distributed under the [MIT License](https://github.com/DeepLearnPhysics/particle-imaging-models/blob/main/LICENSE).

## Citation

If you use pimm, please cite the repository and the papers for the models you use (their BibTeX entries are on
the [Panda](https://deeplearnphysics.org/particle-imaging-models/stable/models/panda.html) and
[PoLAr-MAE](https://deeplearnphysics.org/particle-imaging-models/stable/models/polarmae.html) pages):

```bibtex
@misc{pimm,
  title        = {pimm: particle imaging models},
  author       = {Samuel Young},
  year         = {2025},
  howpublished = {\url{https://github.com/DeepLearnPhysics/particle-imaging-models}},
  note         = {Version 0.6.0}
}
```
