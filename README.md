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

model = pimm.from_pretrained("DeepLearnPhysics/Panda-Semantic", device="cuda")  # Hugging Face
model = pimm.from_pretrained("artifacts/my-model", device="cuda")               # local export
model = pimm.from_pretrained("exp/panda/semseg/my-run/model/last", device="cuda")  # run checkpoint
```

`pimm.from_pretrained` rebuilds the model, loads its weights and returns it in eval mode. It takes a
Hugging Face repository, a directory written by `pimm export`, or a checkpoint from one of your
runs, whose architecture it reads from the run's config. A model expects the same transforms as the
recipe that trained it;
[Run a released model](https://deeplearnphysics.org/particle-imaging-models/stable/models/run.html)
labels a full event, and [Models](https://deeplearnphysics.org/particle-imaging-models/stable/models/index.html)
lists the six releases. Panda needs CUDA; PoLAr-MAE also runs on CPU.

## Train on PILArNet-M

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

Flags before the bare `--` configure the launcher; `key=value` pairs after it override the training
config. The run writes its config, log and checkpoints under `exp/tests/`.
[First run](https://deeplearnphysics.org/particle-imaging-models/stable/get-started/first-run.html)
walks through what it saves, and `uv run pimm ls` lists every recipe.

To run the same job on a Slurm cluster, use `pimm submit` with the same flags plus your account and
partition. `--dry-run` prints the job script without submitting it:

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

Remove `--dry-run` to submit. The command returns once the job is queued, and the training log goes
to the same run directory. Without `--site`, `pimm submit` uses SLAC's S3DF settings.
[Scale up](https://deeplearnphysics.org/particle-imaging-models/stable/training/scale.html) shows how
to write a site profile for your cluster and how to run on several GPUs or nodes.

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
