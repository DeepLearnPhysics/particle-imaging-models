# Citing pimm

pimm has no `CITATION.cff` or DOI. Cite the repository, the version you used, and the papers behind the models and data you used.

```text
pimm (particle imaging models), https://github.com/DeepLearnPhysics/particle-imaging-models
```

Print the installed version with `uv run python -c "import importlib.metadata as m; print(m.version('pimm'))"`.

## Models

Panda:

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

PoLAr-MAE:

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

Sonata, which Panda's pretraining builds on:

```bibtex
@inproceedings{wu2025sonata,
  title     = {Sonata: Self-Supervised Learning of Reliable Point Representations},
  author    = {Wu, Xiaoyang and DeTone, Daniel and Frost, Duncan and Shen, Tianwei and Xie, Chris and Yang, Nan and Engel, Jakob and Newcombe, Richard and Zhao, Hengshuang and Straub, Julian},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {22193--22204},
  year      = {2025}
}
```

Point Transformer V3, the `PT-v3` backbones:

```bibtex
@inproceedings{wu2024ptv3,
  title     = {Point Transformer V3: Simpler, Faster, Stronger},
  author    = {Wu, Xiaoyang and Jiang, Li and Wang, Peng-Shuai and Liu, Zhijian and Liu, Xihui and Qiao, Yu and Ouyang, Wanli and He, Tong and Zhao, Hengshuang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```

## Data

PILArNet, the dataset family PILArNet-M belongs to:

```bibtex
@article{adams2020pilarnet,
  title   = {PILArNet: Public Dataset for Particle Imaging Liquid Argon Detectors in High Energy Physics},
  author  = {Adams, Corey and Terao, Kazuhiro and Wongjirad, Taritree},
  journal = {arXiv preprint arXiv:2006.01993},
  year    = {2020}
}
```

The [PILArNet-M dataset card](https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M) lists PILArNet, PoLAr-MAE and Panda as its references.

pimm builds on [Pointcept](https://github.com/Pointcept/Pointcept), [torchtitan](https://github.com/pytorch/torchtitan) and [TorchRL](https://github.com/pytorch/rl), and is released under the MIT License.
