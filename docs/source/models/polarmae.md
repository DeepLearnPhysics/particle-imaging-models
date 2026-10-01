# PoLAr-MAE

PoLAr-MAE pretrains a transformer on LArTPC point clouds by masked point modeling: it hides groups of points and learns to reconstruct their positions and energies. Paper: [arXiv:2502.02558](https://arxiv.org/abs/2502.02558).

## Released checkpoints

| Checkpoint | Model | Output |
|---|---|---|
| `PoLAr-MAE-Pretrain` | `PoLAr-MAE` | `loss`, `chamfer_loss`, `energy_loss`, `mean_points`, `mean_groups` |
| `PoLAr-MAE-Semantic` | `PoLArMAE-SemSeg` | `seg_logits`, shape (N, 4) |

Semantic classes, in output order: shower, track, Michel, delta. Both models run on CPU as well as CUDA.

## Input pipeline

The semantic model centers and scales coordinates itself. Its recipe log-scales energy, with a higher floor than Panda, and doesn't grid-sample:

```python
from pimm.datasets.transform import Compose

polarmae_semantic_transform = Compose([
    dict(type="LogTransform", min_val=0.13, max_val=20.0),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord",), feat_keys=("coord", "energy")),
])
```

The pretraining recipe normalizes coordinates before the model, with `NormalizeCoord(center=[384, 384, 384], scale=665.1076)` and `LogTransform(min_val=0.01, max_val=20.0)`.

## Recipes

| Recipe | Trains |
|---|---|
| `polarmae/pretrain-polarmae-pilarnet` | PoLAr-MAE, from scratch, with masked point modeling |
| `polarmae/semseg/semseg-polarmae-pilarnet-fft` | the whole semantic model |
| `polarmae/semseg/semseg-polarmae-pilarnet-peft` | the semantic head only (`freeze_encoder=True`) |
| `polarmae/semseg/semseg-polarmae-pilarnet-fft-reproduce` | one epoch, set up for the released semantic checkpoint |

## Cite

```bibtex
@misc{young2025particletrajectoryrepresentationlearning,
  title         = {Particle Trajectory Representation Learning with Masked Point Modeling},
  author        = {Sam Young and Yeon-jae Jwa and Kazuhiro Terao},
  year          = {2025},
  eprint        = {2502.02558},
  archivePrefix = {arXiv},
  primaryClass  = {hep-ex},
  doi           = {10.48550/arXiv.2502.02558}
}
```
