# Panda V2

Here is the training code of the paper **_Panda Diplomacy: Sensor-level Representation Learning across High-Energy and Nuclear Physics_**. With one self-distillation recipe, we pretrain a sparse point encoder on each of four detectors (a liquid argon TPC, an sPHENIX-like TPC, a water Cherenkov detector and the NEXT high-pressure xenon TPC), and you can then probe or fine-tune each encoder for instance segmentation, semantic segmentation and event classification.

*If you only want to run our pretrained models, we recommend starting with the standalone inference repo [Panda](https://github.com/DeepLearnPhysics/Panda), a small library for loading Panda into your own project. This page is for you if you want to reproduce our pretraining and fine-tuning in pimm.*

[ **Pretrain** ] [ **Panda V2** ] - [ [Inference repo](https://github.com/DeepLearnPhysics/Panda) ] [ [Weights](https://huggingface.co/DeepLearnPhysics) ] [ [Docs](../../../docs/source/models/panda.md) ] [ [Bib](#citation) ]

<div align='left'>
<img src="https://raw.githubusercontent.com/DeepLearnPhysics/Panda/main/assets/teaser_v2.png" alt="teaser" width="800" />
</div>

We register three models: the encoder `panda-encoder-v2m1` (`panda_encoder_v2m1.py`), its self-distillation pretraining `panda-ssl-v2m1` (`panda_ssl_v2m1.py`) and the query-based detector `panda-detector-v2m1` (`panda_detector_v2m1.py`, with its decoder layers, matcher and postprocessing in `detector_utils/`).

## Quick start

#### 1. Installation
Follow [Install](../../../docs/source/get-started/install.md), installing pimm from a clone, since the recipes live in `configs/`.

#### 2. Datasets
We train on four datasets from the Hugging Face Hub, and next to each we give the `data_root` the recipes expect:
- PILArNet-M, liquid argon TPC - [ [DeepLearnPhysics/PILArNet-M](https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M/tree/v3_extra) ], revision `v3_extra` in `data/pilarnet-m` (for pretraining, revision `v2` in `data/pilarnet-m-v2`)
- TPCpp-10M, sPHENIX-like TPC - [ [DeepLearnPhysics/tpcpp-10m](https://huggingface.co/datasets/DeepLearnPhysics/tpcpp-10m) ], in `data/tpcpp-10m`
- WAND SK-like, water Cherenkov - [ [DeepLearnPhysics/wand-sk-like](https://huggingface.co/datasets/DeepLearnPhysics/wand-sk-like) ], in `data/wand-sk-like`
- NEXT-AISS, high-pressure xenon TPC - [ [DeepLearnPhysics/next-aiss](https://huggingface.co/datasets/DeepLearnPhysics/next-aiss) ], in `data/next-aiss`

For downloading them, and for the one-time preprocessing of TPCpp-10M before pretraining, see [Supported detectors](../../../docs/source/data/datasets.md).

#### 3. Pretraining
We pretrain each encoder for 20 epochs at batch 96 on four GPUs, with 24 events per GPU:
```bash
uv run pimm launch --train.config panda/v2/pilarnet-m/pretrain-panda-v2m1-0-base --resources.nproc-per-node 4
uv run pimm launch --train.config panda/v2/tpcpp-10m/pretrain-panda-v2m1-0-base --resources.nproc-per-node 4
uv run pimm launch --train.config panda/v2/wand-sk-like/pretrain-panda-v2m1-0-base --resources.nproc-per-node 4
uv run pimm launch --train.config panda/v2/next-aiss/pretrain-panda-v2m1-0-base --resources.nproc-per-node 4

# (optional) the detector's decoder, without an encoder, on synthetic sphere-prior events;
# the instance recipes start from ours, hf://DeepLearnPhysics/Panda-V2-Decoder
uv run pimm launch --train.config panda/v2/sphere-prior/pretrain-panda-v2m1-0-decoder
```

#### 4. Probing and fine-tuning
In the fine-tuning recipes, we load the pretrained encoder, a state dict of `panda-encoder-v2m1` alone, from `weights/<dataset>.pth`. To start from our released encoders instead, pass `--train.weight`:
```bash
# instance segmentation, LoRA adapters on the encoder
uv run pimm launch --train.config panda/v2/pilarnet-m/instance-panda-v2m1-0a-lora --train.weight hf://DeepLearnPhysics/panda-v2.pilarnet-m
# semantic segmentation on a frozen encoder
uv run pimm launch --train.config panda/v2/tpcpp-10m/semseg-panda-v2m1-0b-frozen --train.weight hf://DeepLearnPhysics/panda-v2.tpcpp-10m
# event classification, LoRA adapters on the encoder
uv run pimm launch --train.config panda/v2/next-aiss/cls-panda-v2m1-0a-lora --train.weight hf://DeepLearnPhysics/panda-v2.next-aiss
```
The recipes are named `<task>-panda-v2m1-<variant>` and sit in `configs/panda/v2/<dataset>/`:

| Variant | Trained |
|---|---|
| `0a-lora` | the head and LoRA adapters on the encoder |
| `0b-frozen` | the head, on a frozen encoder |
| `0c-scratch` | the whole model, from a randomly initialized encoder |
| `1a-transfer-lora`, `1b-transfer-frozen` | as `0a` and `0b`, on the encoder pretrained on the dataset named by `source`; the encoder's stem is also trained and its batch-norm statistics are updated |

Instance segmentation (`instance`) has all five variants on `pilarnet-m`, `tpcpp-10m` and `wand-sk-like`, and semantic segmentation (`semseg`) has `0a` to `0c` on `pilarnet-m` and `tpcpp-10m`. On `next-aiss`, event classification (`cls`) has `0a`, `1a` and `1b`, with `cls-sparse-resnet-0-base` as a sparse ResNet baseline trained from scratch, and on `wand-sk-like`, we probe the vertex, direction and momentum of single electrons and muons with `kinematics-panda-v2m1-0b-frozen`. You set the label budget in `num_train_events` (100, 1000, 10000, 100000 or the full split) at the top of each recipe, and in the transfer recipes also the pretrained encoder in `source`. The schedule and evaluation interval are computed from `num_train_events` in the file, so you change it there rather than after `--`.

#### 5. Pretrained weights on the Hub
- Encoders: `DeepLearnPhysics/panda-v2.<dataset>` for `pilarnet-m`, `tpcpp-10m`, `wand-sk-like` and `next-aiss`. On `main` you find the encoder at the end of pretraining, and under the tags `01-of-40` to `40-of-40` the encoder every 5,208 steps; you load a tag with `hf://DeepLearnPhysics/panda-v2.pilarnet-m@11-of-40`.
- Fine-tuned with LoRA on the full labeled training set: `DeepLearnPhysics/panda-v2-instance.<dataset>` for `pilarnet-m`, `tpcpp-10m` and `wand-sk-like`, `DeepLearnPhysics/panda-v2-semseg.<dataset>` for `pilarnet-m` and `tpcpp-10m`, and `DeepLearnPhysics/panda-v2-cls.next-aiss`.
- Across detectors: `DeepLearnPhysics/panda-v2-instance.<source>_ft_<target>-1k`, the encoder pretrained on `<source>` and fine-tuned for instance segmentation on 1,000 labeled events of `<target>`, for every source and each of `pilarnet-m`, `tpcpp-10m` and `wand-sk-like` as target.

You can load any of them in Python with `pimm.from_pretrained("DeepLearnPhysics/panda-v2.pilarnet-m")`; see [Run a released model](../../../docs/source/models/run.md).

## Citation
If you find Panda V2 useful in your research, please cite our paper:
```bib
@misc{young2026pandadiplomacy,
      title={Panda Diplomacy: Sensor-level Representation Learning across High-Energy and Nuclear Physics},
      author={Samuel Young and C\'esar Jes\'us-Valls and Omar A. Alterkait and Gregor Krzmanc and Kazuhiro Terao},
      year={2026},
}
```
