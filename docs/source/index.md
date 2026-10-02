<div class="hero-mark" aria-hidden="true"></div>

# pimm

pimm (particle imaging models) is a library for training, fine-tuning and running foundation models on point clouds from particle-imaging detectors. It contains the Panda and PoLAr-MAE models, readers for liquid argon TPC and water Cherenkov simulations, and one launcher for a workstation or a Slurm cluster.

```bash
curl -sSL https://raw.githubusercontent.com/DeepLearnPhysics/particle-imaging-models/main/install.sh | bash
```

## Where to start

[Run a released model](models/run.md)
: Load Panda or PoLAr-MAE from Hugging Face and label the points of an event. No training needed.

[Train on PILArNet-M](get-started/first-run.md)
: Install, train a tiny model on 100 events, then move to the full recipes.

[Bring your detector's data](data/your-data.md)
: Write your events as `.npy` files or add a reader, then train any pimm model on them.

## What's in pimm

| Area | Contents |
|---|---|
| Models | Panda (self-distillation) and PoLAr-MAE (masked point modeling) pretraining; semantic segmentation, the Panda detector and PointGroup task models; Point Transformer v1–v3, SparseUNet, LitePT and Volt backbones |
| Data | PILArNet-M (HDF5 and Parquet), JAXTPC, LUCiD water Cherenkov, and a generic `.npy` reader |
| Training | `torchrun` with DDP, an experimental FSDP2 path, Slurm through Submitit, and an experimental exex execution path |
| Models out | Portable exports that `pimm.from_pretrained` loads from disk or Hugging Face |

This site documents pimm {{ version }} on the `main` branch. pimm is research software; its APIs and recipes change between versions.

```{toctree}
:hidden:

Get started <get-started/index>
Models <models/index>
Data <data/index>
Training <training/index>
Develop <develop/index>
Reference <reference/index>
```
