<div align="center">

<img src="assets/logo.svg" alt="pimm logo" height="72">

# pimm

[Documentation](https://deeplearnphysics.org/particle-imaging-models/stable/) ·
[First run](https://deeplearnphysics.org/particle-imaging-models/stable/get-started/first-run.html) ·
[Models](https://deeplearnphysics.org/particle-imaging-models/stable/models/index.html) ·
[Recipe catalog](https://deeplearnphysics.org/particle-imaging-models/stable/reference/recipes.html)

</div>

pimm (particle imaging models) trains, fine-tunes and runs foundation models on point clouds from
particle-imaging detectors. It contains the Panda and PoLAr-MAE models, readers for liquid argon TPC
and water Cherenkov simulations, and one launcher for a workstation or a Slurm cluster.

## Install

On Linux x86-64 with an NVIDIA GPU:

```bash
curl -sSL https://raw.githubusercontent.com/DeepLearnPhysics/particle-imaging-models/main/install.sh | bash
cd particle-imaging-models
uv run pimm --help
```

The script installs [uv](https://docs.astral.sh/uv/) if it's missing, clones this repository and
installs the locked environment: Python 3.10, PyTorch 2.10.0, CUDA 12.6 and prebuilt native
operators, so you don't need a CUDA toolkit or a compiler. Run pimm through `uv run`; there's no
environment to activate. [Install](https://deeplearnphysics.org/particle-imaging-models/stable/get-started/install.html)
covers installing by hand, containers, launcher-only hosts and which
[GPUs](https://deeplearnphysics.org/particle-imaging-models/stable/get-started/install.html#gpus)
support Flash Attention and BF16.

## Run a released model

```python
import pimm

model = pimm.from_pretrained("DeepLearnPhysics/Panda-Semantic", device="cuda")
```

`pimm.from_pretrained` downloads the export from Hugging Face, rebuilds the model and returns it in
eval mode. A model expects the same transforms as the recipe that trained it;
[Run a released model](https://deeplearnphysics.org/particle-imaging-models/stable/models/run.html)
labels a full event, and [Models](https://deeplearnphysics.org/particle-imaging-models/stable/models/index.html)
lists the six releases. Panda needs CUDA; PoLAr-MAE also runs on CPU.

## Train on PILArNet-M

Download the 120-event mini dataset and train a tiny semantic-segmentation model on one GPU:

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('DeepLearnPhysics/PILArNet-M-mini', repo_type='dataset', local_dir='data/PILArNet-M-mini')"

uv run pimm launch --resources.nproc-per-node 1 --train.config tests/tiny_semseg -- \
  data.train.data_root="$PWD/data/PILArNet-M-mini" \
  data.val.data_root="$PWD/data/PILArNet-M-mini"
```

Flags before the bare `--` configure the launcher; `key=value` pairs after it override the training
config. The run writes its config, log and checkpoints under `exp/tests/`.
[First run](https://deeplearnphysics.org/particle-imaging-models/stable/get-started/first-run.html)
walks through what it saves, `uv run pimm ls` lists every recipe, and
[Scale up](https://deeplearnphysics.org/particle-imaging-models/stable/training/scale.html) runs the
same recipe on more GPUs or on Slurm with `pimm submit`.

To train on another detector, write its events as `.npy` files or add a reader; see
[Bring your own data](https://deeplearnphysics.org/particle-imaging-models/stable/data/your-data.html).

## What's in pimm

| Area | Contents |
|---|---|
| Models | Panda (self-distillation) and PoLAr-MAE (masked point modeling) pretraining; semantic segmentation, the Panda detector and PointGroup task models; Point Transformer v1–v3, SparseUNet, LitePT and Volt backbones |
| Data | PILArNet-M (HDF5 and Parquet), JAXTPC, LUCiD water Cherenkov, and a generic `.npy` reader |
| Training | `torchrun` with DDP, an experimental FSDP2 path, Slurm through Submitit, and an experimental exex execution path |
| Models out | Portable exports that `pimm.from_pretrained` loads from disk or Hugging Face |

pimm is research software; its APIs and recipes change between versions.

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

## Contributing and citing

See [Contributing](https://deeplearnphysics.org/particle-imaging-models/stable/develop/contributing.html).
To cite pimm, cite the repository, the version you used and the papers behind the models and data;
[Citing pimm](https://deeplearnphysics.org/particle-imaging-models/stable/reference/cite.html) has
the BibTeX entries.

pimm builds on [Pointcept](https://github.com/Pointcept/Pointcept),
[torchtitan](https://github.com/pytorch/torchtitan) and [TorchRL](https://github.com/pytorch/rl). It
is distributed under the [MIT License](LICENSE).
