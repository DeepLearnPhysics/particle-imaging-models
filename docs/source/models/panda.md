# Panda

Panda is a sparse hierarchical point encoder for LArTPC events, pretrained on unlabeled data with a multi-view, prototype-based self-distillation objective. Its backbone is `PT-v3m2`, a Point Transformer V3 variant; its pretraining model is `Sonata-v1m1`, pimm's implementation of Sonata. Add task heads to the encoder to label points, find particles and group interactions. Paper: [arXiv:2512.01324](https://arxiv.org/abs/2512.01324).

## Panda V1 vs V2

| | Panda V1 | Panda V2 |
|---|---|---|
| Encoder | `PT-v3m2` | `panda-encoder-v2m1` |
| Pretraining | `Sonata-v1m1` | `panda-ssl-v2m1` |
| Detector | `panda-detector-v1m1` | `panda-detector-v2m1` |
| Decoding preset | Original V1 decoding | `panda_v2` by default; `panda_v1` and `wc` available |
| Decoder initialization | Random initialization | Synthetic `panda-detector-v2m1` checkpoint via `CheckpointLoader` |
| Configs folder | `configs/panda/v1/` | `configs/panda/v2/` |

## Panda V2 encoder and pretraining

As in PTv3, `panda-encoder-v2m1` returns its coarsest stage as a `Point`. With `point.upcast()`, the four stages' features are joined at each input point (1170 channels by default); `point.upcast(levels)` stops after that many upcasts. The encoder's `conv_stages` (2) sets how many of the four stages are sparse convolutions, `strides` (2, 1, 4) the pooling between stages, and `window_size` (1024) the number of serialized points in each attention window.

In `panda-ssl-v2m1`, `up_cast_level` sets where the losses are computed: 3 at the input points (the LArTPC, sPHENIX and NEXT recipes, with 1170-channel heads) and 2 at the second stage's sites (water Cherenkov, with 1116-channel heads). Training batches must be prepared by the `SonataPreparation` transform after `Collect`: masks are drawn and nearest-neighbour matches are found at the loss level in the data workers. The transform is given the model's config, `dict(type="SonataPreparation", model=detector_model)` in each pretraining recipe, and reads from it the encoder's `strides` and the model's `observable`, `up_cast_level`, constant `mask_size`, `mask_ratio` and `mask_jitter`, and match radii `match_max_r` and `observable_match_max_r` (which defaults to `match_max_r`); omitted values take the classes' defaults. A batch prepared with other settings than the model's is rejected with an error. With `observable=True`, masked points are removed from the student's views; with `observable=False`, they keep their place with the encoder's mask token (`mask_token=True`) and coordinates jittered by `mask_jitter`. The teacher temperature rises linearly from `teacher_temp_start` to `teacher_temp_base` over the first `teacher_temp_warmup_ratio` of training and then stays constant; the EMA momentum follows a cosine from `momentum_base` to `momentum_final`, raised to the power `ema_power`.

## Released checkpoints

| Checkpoint | Model | Output |
|---|---|---|
| `Panda-Base` | `PT-v3m2` encoder | a `Point`; `point.feat` contains per-point features |
| `Panda-Semantic` | `DefaultSegmentorV2` on `PT-v3m2` | `seg_logits`, shape (N, 5) |
| `Panda-Particle` | Panda detector on `PT-v3m2` | query masks and classes; call `model.postprocess()` for per-point particles |
| `Panda-Interaction` | Panda detector on `PT-v3m2` | query masks; call `model.postprocess()` for per-point interactions |

Semantic classes, in output order: shower, track, Michel, delta, low-energy deposit.

Particle classes, in output order: photon, electron, muon, pion, proton. Points that belong to no particle, such as low-energy deposits, have instance `-1`.

## Input pipeline

The Panda recipes' transforms at inference are:

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

- `NormalizeCoord`: the PILArNet-M volume, a cube of side 768 centered at 384, is mapped into the unit ball.
- `LogTransform`: energy is mapped onto [-1, 1] on a log scale between 0.01 and 20.
- `GridSample`: one point per occupied cell of a 0.001 grid is kept, and integer `grid_coord` is added. With `mode="train"`, the point kept in each cell is random; seed NumPy to make the choice repeatable. The predictions of Panda Particle and Panda Interaction are for the kept points. The predictions of Panda Semantic are for every original point; for these, pass `return_inverse=True` and collect `inverse`, as in [Run a released model](run.md#label-one-event).
- `Collect`: `feat` is the coordinates followed by energy, so the backbone has `in_channels=4`.

## Recipes

| Recipe | Trained |
|---|---|
| `panda/v1/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask` | the Panda encoder, from scratch, with self-distillation |
| `panda/v1/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8` | the same with the `PT-v3m8` backbone |
| `panda/v1/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-lin` | semantic head on a frozen encoder |
| `panda/v1/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-dec` | decoder and head on a frozen encoder |
| `panda/v1/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft` | the whole semantic model |
| `panda/v1/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-scratch` | the whole semantic model; run it without weights |
| `panda/v1/panseg/pt-v3m2-ft-pid-{dec,fft,scratch}` | the particle detector |
| `panda/v1/panseg/pt-v3m2-ft-vtx-{dec,fft,scratch}` | the interaction detector |
| `panda/v1/panseg/pt-v3m2-ft-{pid,vtx}-fft-detector` | the whole detector, starting from the released Particle or Interaction weights pinned in the recipe |
| `panda/v2/{pilarnet-m,tpcpp-10m,wand-sk-like,next-aiss}/pretrain-panda-v2m1-0-base` | the V2 encoder, from scratch, with `panda-ssl-v2m1` self-distillation on that dataset |
| `panda/v2/sphere-prior/pretrain-panda-v2m1-0-decoder` | the `panda-detector-v2m1` decoder without an encoder, on synthetic sphere-prior events |
| `panda/v2/{pilarnet-m,tpcpp-10m,wand-sk-like}/instance-panda-v2m1-0a-lora` | the detector's decoder, starting from the decoder core in `hf://DeepLearnPhysics/Panda-V2-Decoder`, and LoRA adapters on the encoder |
| `panda/v2/{pilarnet-m,tpcpp-10m,wand-sk-like}/instance-panda-v2m1-0b-frozen` | the detector's decoder on a frozen encoder |
| `panda/v2/{pilarnet-m,tpcpp-10m,wand-sk-like}/instance-panda-v2m1-0c-scratch` | the whole detector, from a randomly initialized encoder |
| `panda/v2/{pilarnet-m,tpcpp-10m,wand-sk-like}/instance-panda-v2m1-1{a,b}-transfer-{lora,frozen}` | as `0a` and `0b`, on the encoder pretrained on the dataset named by `source`; the encoder's stem is also trained and its batch-norm statistics are updated |
| `panda/v2/{pilarnet-m,tpcpp-10m}/semseg-panda-v2m1-0a-lora` | MLP semantic heads and LoRA adapters on the encoder: 8 joint semantic and particle classes on `pilarnet-m`, particle and noise classes on `tpcpp-10m` |
| `panda/v2/{pilarnet-m,tpcpp-10m}/semseg-panda-v2m1-0b-frozen` | linear and MLP semantic heads on a frozen encoder: 5 semantic classes on `pilarnet-m`, particle and noise classes on `tpcpp-10m` |
| `panda/v2/{pilarnet-m,tpcpp-10m}/semseg-panda-v2m1-0c-scratch` | the whole semantic model, from a randomly initialized encoder |
| `panda/v2/wand-sk-like/kinematics-panda-v2m1-0b-frozen` | linear and MLP probes of the vertex, direction and momentum of single electrons and muons, on a frozen encoder |
| `panda/v2/next-aiss/cls-panda-v2m1-0a-lora` | a signal-versus-background event classification head and LoRA adapters on the encoder |
| `panda/v2/next-aiss/cls-panda-v2m1-1{a,b}-transfer-{lora,frozen}` | the classification head, the encoder's stem and, in `1a`, LoRA adapters, on the encoder pretrained on the dataset named by `source`; batch-norm statistics are updated |
| `panda/v2/next-aiss/cls-sparse-resnet-0-base` | a sparse ResNet event classifier, from scratch, as a baseline |

To fine-tune the semantic and detector recipes from `Panda-Base`, pass `--train.weight hf://DeepLearnPhysics/Panda-Base`; see [Fine-tune](fine-tune.md). The model in the detector recipes is `panda-detector-v1m1`, with the same parameter layout as the released V1 state dictionaries.

Each V2 recipe that trains on labels sets its number of labeled training events in `num_train_events` at the top of the file, and its schedule length and evaluation interval follow from that number. The V2 recipes that start from a pretrained encoder load it from `weights/<dataset>.pth`. The `tpcpp-10m` recipes read the TPCpp-10M release from `data/tpcpp-10m`: download it and, before pretraining, run `python -m pimm.datasets.preprocessing.tpcpp10m data/tpcpp-10m` once, as described in [Supported detectors](../data/datasets.md#tpcpp-10m).

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

Panda's pretraining is based on Sonata ([Wu et al., 2025](https://arxiv.org/abs/2503.16429)), and its V1 backbone is Point Transformer V3 ([Wu et al., 2024](https://arxiv.org/abs/2312.10035)).
