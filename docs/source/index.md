---
sd_hide_title: true
---

# pimm - Particle Imaging Models

:::{div} pimm-hero

```{image} _static/logo.svg
:alt: pimm logo
:class: pimm-hero-logo
:width: 96px
```

# particle imaging models (pimm)

[Foundation-model research for particle-imaging detectors.]{.pimm-tagline}

```{button-ref} getting_started/overview
:ref-type: myst
:color: primary
:class: sd-px-4 sd-fs-6
Get started
```
```{button-link} https://github.com/DeepLearnPhysics/particle-imaging-models
:color: secondary
:outline:
:class: sd-px-4 sd-fs-6
View on GitHub
```

:::

---

pimm is a framework for pre-training and fine-tuning point cloud foundation models for sparse data from high energy physics experiments. It is designed to be easy to use and hackable, yet capable of scaling to 128+ GPUs. It provides:

1. Pre-training and fine-tuning recipes for foundation model development.
2. Native implementations and uses of modern deep learning methods that allow you to scale: [DDP](), [FSDP2](), [flash-attention]()
3. One-line multi-node deployment with SLURM (and HTCondor -- experimental).
4. Hackable, modular, and extensible by design.
5. One-line loading and saving of models from HuggingFace
6. End-to-end pre-training, fine-tuning, and evaluation.

## Setup



First, clone the repository.

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models
```

Then get pimm running through a container or a local uv environment.
Each path runs pimm from your clone, so your checkout is the pimm source.

::::{tab-set}

:::{tab-item} Local (uv)
```bash
# requires Linux x86_64 and an NVIDIA driver (training packages are prebuilt wheels)
./install.sh
uv run pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```
:::

:::{tab-item} Singularity / Apptainer (recommended)
```bash
apptainer pull /path/to/pimm.sif docker://youngsm/pimm:pytorch2.10.0-cuda12.6

# run a command directly:
apptainer run --nv /path/to/pimm.sif \
  pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask

# ...or open a shell, then run pimm inside:
apptainer run --nv /path/to/pimm.sif
pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```
:::

:::{tab-item} Docker
```bash
# run a command directly:
docker run --rm --gpus all -v "$PWD:$PWD" -w "$PWD" youngsm/pimm:pytorch2.10.0-cuda12.6 \
  pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask

# ...or open a shell, then run pimm inside:
docker run --rm -it --gpus all -v "$PWD:$PWD" -w "$PWD" youngsm/pimm:pytorch2.10.0-cuda12.6 bash
pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```
:::


::::

The command above trains on a single GPU; add `--resources.nproc-per-node 4` for four.
Full details, image tags, and verification: {doc}`getting_started/installation`.
For a guided walk-through of your first run, follow the {doc}`getting_started/quickstart`.

## What's inside

- {doc}`Quick start <getting_started/index>` - install from source or a
  container, run your first job locally, and learn the three core ideas in
  pimm: packed tensors, registries, and configs.
- {doc}`Distributed training <distributed/index>` - one launch path for
  single-GPU, multi-GPU, and multi-node. DDP and FSDP2, deterministic
  checkpointing, exact mid-epoch resume across world-size changes.
- {doc}`Training on a cluster <hpc/index>` - interactive vs batch submission,
  site profiles for your cluster, QOS, requeue chaining, environment variables, job
  monitoring, and resuming.
- {doc}`Checkpoints <checkpoints/index>` - one atomic, reshardable checkpoint
  format for every parallelism. Save-frequency hooks, manual export, and automatic
  Hugging Face upload.
- {doc}`Research ecosystem <research_ecosystem/index>` - load any export with
  `pimm.from_pretrained` and fine-tune from the Hub, then contribute your own
  models, hooks, datasets, and transforms to pimm.
- {doc}`Datasets & transforms <datasets/index>` - the packed point-cloud format, PILArNet-M, multimodal LArTPC/Water-Cherenkov readers, and transform
  pipelines.
- {doc}`Hooks <hooks/index>` - the lifecycle hook system (logging, diagnostics,
  evaluators, checkpoint savers).
- {doc}`Evaluation <evaluation/index>` - in-loop evaluators, probe suites for
  SSL, and final testing with `test.sh`.
- {doc}`Core concepts <getting_started/concepts>` - what pimm does differently
  if you are coming from another Pointcept-style codebase.

## Integrated works

- **Backbones** - [PTv3](https://arxiv.org/abs/2312.10035),
  [PTv2](https://arxiv.org/abs/2210.05666),
  [PTv1](https://arxiv.org/abs/2012.09164), and SparseUNet (SpConv/Minkowski).
- **Pre-training** - [Sonata](https://arxiv.org/abs/2503.16429) /
  [Panda](https://arxiv.org/abs/2512.01324) discriminative SSL and
  [PoLAr-MAE](https://arxiv.org/abs/2502.02558) masked autoencoding.
- **Segmentation** - semantic segmentation, PointGroup, and the
  [Panda Detector](https://arxiv.org/abs/2512.01324) panoptic model.
- **Datasets** - [PILArNet-M](https://arxiv.org/abs/2502.02558), multimodal
  JAXTPC, Water-Cherenkov (LUCiD).

---

## Acknowledgements

* pimm's name is inspired by the pytorch-imaging-models library, which is called timm.
* pimm's codebase was initially forked from [Pointcept](https://github.com/Pointcept/Pointcept), and the modified with influences from [torchtitan](https://github.com/pytorch/torchtitan), [mmcv](https://github.com/open-mmlab/mmcv), [torchrl](https://github.com/pytorch/rl), and [Megatron-LM](https://github.com/nvidia/megatron-lm).
* The documentation design is largely based off PyTorch's own docs theme [`pytorch_sphinx_theme2`](https://github.com/pytorch/pytorch_sphinx_theme/tree/pytorch_sphinx_theme2). 

```{toctree}
:hidden:
:maxdepth: 2

Getting started <getting_started/index>
Tutorials <tutorials/index>
Research ecosystem <research_ecosystem/index>
Datasets <datasets/index>
Configurations <configuration/index>
Hooks <hooks/index>
Checkpoints <checkpoints/index>
Model evaluation <evaluation/index>
Distributed training <distributed/index>
Training on a cluster <hpc/index>
API reference <api/index>
```
