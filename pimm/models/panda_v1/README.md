# Panda

Here is the training code of the paper **_Panda: Self-distillation of Reusable Sensor-level Representations for High Energy Physics_**. We pretrain a Point Transformer V3 encoder on unlabeled liquid argon TPC events with Sonata-style self-distillation, and you can then fine-tune it to label points, find particles and group interactions.

*If you only want to run our pretrained models, we recommend starting with the standalone inference repo [Panda](https://github.com/DeepLearnPhysics/Panda), a small library for loading Panda into your own project. This page is for you if you want to reproduce our pretraining and fine-tuning in pimm.*

[ **Pretrain** ] [ **Panda** ] - [ [Paper](https://arxiv.org/abs/2512.01324) ] [ [Dataset](https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M) ] [ [Project Site](https://youngsm.com/panda) ] [ [Inference repo](https://github.com/DeepLearnPhysics/Panda) ] [ [Bib](#citation) ]

<div align='left'>
<img src="https://youngsm.com/assets/img/panda/teaser_full.png" alt="teaser" width="800" />
</div>

Here you find the Panda detector `panda-detector-v1m1` (`panda_detector_v1m1.py`, with its decoder layers, matcher and postprocessing in `detector_utils/`). For pretraining, we use pimm's Sonata, `Sonata-v1m1` in [`pimm/models/sonata/`](../sonata/), with the `PT-v3m2` backbone from [`pimm/models/point_transformer_v3/`](../point_transformer_v3/).

## Quick start

#### 1. Installation
Follow [Install](../../../docs/source/get-started/install.md), installing pimm from a clone, since the recipes live in `configs/`.

#### 2. Dataset
We train on [PILArNet-M](https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M). We run the detector recipes on revision `v3`. We wrote the pretraining and semantic recipes for revision `v1`, which pimm no longer reads, so you run them on `v3` by adding `-- data.train.revision=v3 data.val.revision=v3` (and `data.test.revision=v3` for the semantic recipes) to the commands below. For downloading each revision, see [Supported detectors](../../../docs/source/data/datasets.md#pilarnet-m).

#### 3. Pretraining
```bash
uv run pimm launch --train.config panda/v1/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask -- data.train.revision=v3 data.val.revision=v3

# (or) the same with the PT-v3m8 backbone, already on v3
uv run pimm launch --train.config panda/v1/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask-v3m8
```

#### 4. Probing and fine-tuning
Pass the encoder with `--train.weight`, either ours or the checkpoint of your own pretraining run:
```bash
# semantic segmentation: -lin (head), -dec (decoder and head), -fft (everything)
uv run pimm launch --train.config panda/v1/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft --train.weight hf://DeepLearnPhysics/Panda-Base \
  -- data.train.revision=v3 data.val.revision=v3 data.test.revision=v3

# particles (pid) and interactions (vtx): -dec (decoder and heads), -fft (everything)
uv run pimm launch --train.config panda/v1/panseg/pt-v3m2-ft-pid-fft --train.weight hf://DeepLearnPhysics/Panda-Base
uv run pimm launch --train.config panda/v1/panseg/pt-v3m2-ft-vtx-fft --train.weight hf://DeepLearnPhysics/Panda-Base

# particles and interactions with one detector
uv run pimm launch --train.config panda/v1/panseg/pt-v3m2-ft-joint-fft --train.weight hf://DeepLearnPhysics/Panda-Base
```
The `-scratch` recipes (`semseg-pt-v3m2-pilarnet-ft-5cls-scratch`, `pt-v3m2-ft-pid-scratch`, `pt-v3m2-ft-vtx-scratch`) train the whole model from a random initialization, so you run them without `--train.weight`. The `pt-v3m2-ft-{pid,vtx}-fft-detector` recipes train the whole detector further, starting from our released Particle or Interaction weights pinned in the recipe.

#### 5. Pretrained weights on the Hub
In each repo you find `model.safetensors` and `config.json`:
- [`DeepLearnPhysics/Panda-Base`](https://huggingface.co/DeepLearnPhysics/Panda-Base): the pretrained `PT-v3m2` encoder
- [`DeepLearnPhysics/Panda-Semantic`](https://huggingface.co/DeepLearnPhysics/Panda-Semantic): `DefaultSegmentorV2` with five semantic classes
- [`DeepLearnPhysics/Panda-Particle`](https://huggingface.co/DeepLearnPhysics/Panda-Particle): the Panda detector for particles
- [`DeepLearnPhysics/Panda-Interaction`](https://huggingface.co/DeepLearnPhysics/Panda-Interaction): the Panda detector for interactions

You can load any of them in Python with `pimm.from_pretrained("DeepLearnPhysics/Panda-Base")`; see [Run a released model](../../../docs/source/models/run.md).

## Citation
If you find Panda useful in your research, please cite our paper:
```bib
@misc{young2025pandaselfdistillationreusablesensorlevel,
      title={Panda: Self-distillation of Reusable Sensor-level Representations for High Energy Physics},
      author={Samuel Young and Kazuhiro Terao},
      year={2025},
      eprint={2512.01324},
      archivePrefix={arXiv},
      primaryClass={hep-ex},
      url={https://arxiv.org/abs/2512.01324},
}
```
