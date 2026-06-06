"""Dataset registry and config builder.

All dataset classes that can be constructed from config dictionaries register
with ``DATASETS``. Configs are passed through ``Registry.build`` unchanged, so
dataset constructors own validation of modality- or format-specific options.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pimm.utils.registry import Registry

DATASETS = Registry("datasets")


def build_dataset(cfg):
    return DATASETS.build(cfg)
