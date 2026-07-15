<div align="center">

<img src="assets/logo.svg" alt="pimm logo" height="72">

# particle imaging models (pimm)

Foundation-model research for particle-imaging detectors.

[Documentation](https://deeplearnphysics.org/particle-imaging-models/stable/) ·
[Quickstart](https://deeplearnphysics.org/particle-imaging-models/stable/getting_started/quickstart.html) ·
[Models](https://deeplearnphysics.org/particle-imaging-models/stable/models/index.html) ·
[Python API](https://deeplearnphysics.org/particle-imaging-models/stable/api/index.html)

</div>

pimm is a PyTorch research toolkit for variable-length, three-dimensional point
clouds from particle-imaging detectors. It brings model families, datasets,
training and evaluation loops, distributed launchers, and portable pretrained
exports into one reproducible experiment system.

It is designed for both researchers running large distributed studies and new
students exploring released models on a single machine. The current scope is
3D sparse point clouds; 2D detector images and waveforms are planned.

## Install

Linux x86-64 users can install the locked training environment in one command:

```bash
curl -sSL https://raw.githubusercontent.com/DeepLearnPhysics/particle-imaging-models/main/install.sh | bash
cd particle-imaging-models
```

The installer sets up `uv`, clones the repository, installs the lockfile, and
checks the native operators. No environment activation is needed; run project
commands through `uv run`.

```bash
uv run pimm launch \
  --train.config tests/tiny_semseg \
  --resources.nproc-per-node 1 \
  --dry-run
```

See the [installation guide](https://deeplearnphysics.org/particle-imaging-models/stable/getting_started/installation.html)
for the manual install, containers, launcher-only hosts, environment variables,
and the GPU compatibility table.

## Run a released model

[`pimm.from_pretrained`](https://deeplearnphysics.org/particle-imaging-models/stable/api/generated/pimm.from_pretrained.html)
supports local exports and Hugging Face repositories. Inference can run on CPU
when the selected architecture and operators support it; PoLAr-MAE does.

```python
import torch
import pimm

device = "cpu"  # use "cuda" when available
model = pimm.from_pretrained(
    "DeepLearnPhysics/PoLAr-MAE-Semantic",
    device=device,
)  # weights use the Hugging Face cache, configurable with HF_HUB_CACHE

input_dict = {
    "coord": coord.to(device),    # (N, 3), transformed coordinates
    "feat": feat.to(device),      # (N, C), transformed point features
    "offset": offset.to(device),  # (B,), cumulative points per event
}

with torch.inference_mode():
    output = model(input_dict)

labels = output["seg_logits"].argmax(-1)  # (N,)
```

Preprocessing is part of a model's scientific contract. Follow the
[pretrained-model guide](https://deeplearnphysics.org/particle-imaging-models/stable/models/pretrained.html)
for complete Panda and PoLAr-MAE transforms, packed batching, output schemas,
fine-tuning, and CPU/GPU constraints.

## Start an experiment

The [first experiment](https://deeplearnphysics.org/particle-imaging-models/stable/getting_started/quickstart.html)
downloads the small public
[PILArNet-M-mini](https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M-mini)
dataset and trains a tiny semantic-segmentation model. A normal local run uses
the same launcher with a research config:

```bash
uv run pimm launch \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  --resources.nproc-per-node 1
```

Everything after a bare `--` overrides the Python training config:

```bash
uv run pimm launch \
  --train.config tests/tiny_semseg \
  --resources.nproc-per-node 1 \
  -- epoch=2 batch_size=4 use_wandb=False
```

For Slurm, use `pimm submit`; for portable weights and Hub publication, use
`pimm export`. Each command has a complete reference in the
[CLI guide](https://deeplearnphysics.org/particle-imaging-models/stable/reference/cli.html).

## Included research

| Area | Implementations |
|---|---|
| Sparse backbones | Point Transformer v1/v2/v3, SparseUNet, LitePT |
| Representation learning | Panda/Sonata, PoLAr-MAE, LeJEPA, Volt-MAE |
| Downstream tasks | semantic segmentation, PointGroup, Panda Detector |
| Data | PILArNet-M v1/v2 and custom packed point-cloud datasets |
| Scale | local `torchrun`, DDP, experimental FSDP2, Slurm via Submitit |
| Portability | structured resume checkpoints, plain weights, Hugging Face exports |

Released checkpoints and their exact output contracts are listed in the
[model guide](https://deeplearnphysics.org/particle-imaging-models/stable/models/index.html).
The interactive [Explore Panda](https://deeplearnphysics.org/particle-imaging-models/stable/tutorials/explore_panda.html)
and [Explore PoLAr-MAE](https://deeplearnphysics.org/particle-imaging-models/stable/tutorials/explore_polarmae.html)
tutorials use real PILArNet-M-mini events and provide runnable Python notebook
sources for regenerating their figures.

## Hardware

The prebuilt CUDA stack targets NVIDIA compute capabilities 7.0–9.0:

- V100 and RTX 20xx: disable Flash Attention and use FP16 or full precision;
- A100, RTX 30xx/40xx, and H100/H200: Flash Attention and BF16 are supported;
- L40S: disable Flash Attention; BF16 is supported.

Panda's released PTv3 models currently require CUDA because they use `spconv`.
Released PoLAr-MAE inference also runs on CPU, although CUDA is faster. Consult
the [compatibility table](https://deeplearnphysics.org/particle-imaging-models/stable/getting_started/installation.html#supported-gpus)
before starting a long run.

## Documentation

| Question | Start here |
|---|---|
| How are events represented? | [Data conventions](https://deeplearnphysics.org/particle-imaging-models/stable/data/conventions.html) |
| How do configs and overrides work? | [Configuration](https://deeplearnphysics.org/particle-imaging-models/stable/operations/configuration.html) |
| How do I train or fine-tune? | [Training](https://deeplearnphysics.org/particle-imaging-models/stable/workflows/train.html) · [Fine-tuning](https://deeplearnphysics.org/particle-imaging-models/stable/workflows/fine_tune.html) |
| What exactly is saved? | [Checkpoints and resume](https://deeplearnphysics.org/particle-imaging-models/stable/operations/checkpoints.html) |
| How do I use multiple GPUs or Slurm? | [Distributed training](https://deeplearnphysics.org/particle-imaging-models/stable/workflows/distributed.html) · [Slurm](https://deeplearnphysics.org/particle-imaging-models/stable/workflows/slurm.html) |
| How do I add a model, loss, dataset, transform, or hook? | [Extending pimm](https://deeplearnphysics.org/particle-imaging-models/stable/extend/index.html) |
| Something failed—what should I inspect? | [Troubleshooting](https://deeplearnphysics.org/particle-imaging-models/stable/operations/troubleshooting.html) |

## Contributing and citation

Start with the [contributor guide](https://deeplearnphysics.org/particle-imaging-models/stable/extend/contributing.html)
and open an issue before a large architectural change. Scientific results should
record the full pimm commit, resolved config, data revision and transforms,
checkpoint revision, and evaluation protocol. The
[citation guide](https://deeplearnphysics.org/particle-imaging-models/stable/project/citation.html)
lists the software, model, backbone, and dataset records to preserve.

pimm builds on [Pointcept](https://github.com/Pointcept/Pointcept),
[torchtitan](https://github.com/pytorch/torchtitan), and
[TorchRL](https://github.com/pytorch/rl). It is distributed under the
[MIT License](LICENSE).
