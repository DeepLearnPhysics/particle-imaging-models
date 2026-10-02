# Panda

Panda is a sparse hierarchical point encoder for LArTPC events, pretrained on unlabeled data with a multi-view, prototype-based self-distillation objective. Its backbone is `PT-v3m2`, a Point Transformer V3 variant; its pretraining model is `Sonata-v1m1`, pimm's implementation of Sonata. Task heads on the encoder label points, find particles and group interactions. Paper: [arXiv:2512.01324](https://arxiv.org/abs/2512.01324).

## Released checkpoints

| Checkpoint | Model | Output |
|---|---|---|
| `Panda-Base` | `PT-v3m2` encoder | a `Point`; `point.feat` holds per-point features |
| `Panda-Semantic` | `DefaultSegmentorV2` on `PT-v3m2` | `seg_logits`, shape (N, 5) |
| `Panda-Particle` | Panda detector on `PT-v3m2` | query masks and classes; `model.postprocess()` turns them into per-point particles |
| `Panda-Interaction` | Panda detector on `PT-v3m2` | query masks; `model.postprocess()` turns them into per-point interactions |

Semantic classes, in output order: shower, track, Michel, delta, low-energy deposit.

Particle classes, in output order: photon, electron, muon, pion, proton. Points that belong to no particle, such as low-energy deposits, get instance `-1`.

## Input pipeline

The Panda recipes apply these transforms at inference:

```python
from pimm.datasets.transform import Compose

panda_transform = Compose([
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="LogTransform", min_val=0.01, max_val=20.0, keys=("energy",)),
    dict(type="GridSample", grid_size=0.001, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord"), feat_keys=("coord", "energy")),
])
```

- `NormalizeCoord` maps the PILArNet-M volume, a cube of side 768 centered at 384, into the unit ball.
- `LogTransform` maps energy onto [-1, 1] on a log scale between 0.01 and 20.
- `GridSample` keeps one point per occupied cell of a 0.001 grid and adds integer `grid_coord`. With `mode="train"` it picks a random point in each cell; seed NumPy to make the choice repeatable. Predictions refer to the points it keeps.
- `Collect` builds `feat` from coordinates then energy, so the backbone has `in_channels=4`.

## Recipes

| Recipe | Trained |
|---|---|
| `panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask` | the Panda encoder, from scratch, with self-distillation |
| `panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8` | the same with the `PT-v3m8` backbone |
| `panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-lin` | semantic head on a frozen encoder |
| `panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-dec` | decoder and head on a frozen encoder |
| `panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft` | the whole semantic model |
| `panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-scratch` | the whole semantic model; run it without weights |
| `panda/panseg/detector-v5-pt-v3m2-ft-pid-{dec,fft,scratch}` | the particle detector |
| `panda/panseg/detector-v5-pt-v3m2-ft-vtx-{dec,fft,scratch}` | the interaction detector |
| `panda/panseg/detector-v5-pt-v3m2-ft-{pid,vtx}-fft-detector` | the whole detector, starting from the released Particle or Interaction weights pinned in the recipe |

To fine-tune the semantic and detector recipes from `Panda-Base`, pass `--train.weight hf://DeepLearnPhysics/Panda-Base`; see [Fine-tune](fine-tune.md). The detector recipes build `detector-v5`. The repository also registers `detector-v3m2`, `detector-v4` and `detector-v5m2`.

## Cite

```bibtex
@misc{young2025pandaselfdistillationreusablesensorlevel,
  title         = {Panda: Self-distillation of Reusable Sensor-level Representations for High Energy Physics},
  author        = {Samuel Young and Kazuhiro Terao},
  year          = {2025},
  eprint        = {2512.01324},
  archivePrefix = {arXiv},
  primaryClass  = {hep-ex},
  url           = {https://arxiv.org/abs/2512.01324}
}
```

Panda builds on Sonata and Point Transformer V3; their entries are on [Citing pimm](../reference/cite.md).
