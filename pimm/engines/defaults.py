"""
Default configuration and process setup for training and evaluation.

This module parses CLI/config inputs, records resolved run artifacts, and
derives per-rank dataloader settings from the distributed world size.

modified from detectron2(https://github.com/facebookresearch/detectron2)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import argparse
import json
import multiprocessing as mp
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone

import pimm.utils.comm as comm
from pimm.engines._train_utils import (
    _apply_hook_overrides,
    _apply_hook_overrides_from_dict,
    _save_config_artifacts,
    _split_hook_type_options,
)
from pimm.utils.config import Config, DictAction
from pimm.utils.env import get_random_seed, set_seed


def default_argument_parser(epilog=None):
    """Create the standard training CLI parser."""
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
    Examples:
    Run on single machine:
        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
    Change some config options:
        $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
    Run on multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        # default="tcp://127.0.0.1:{}".format(port),
        default="auto",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    return parser



def default_config_parser(file_path, options, *, save_artifacts=True):
    """Load a config, apply CLI/hook overrides, and prepare save paths."""
    # config name protocol: dataset_name/model_name-exp_name
    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1 :]))

    # Apply hook overrides from config file (hooks_override dict)
    if hasattr(cfg, 'hooks_override'):
        _apply_hook_overrides_from_dict(cfg, cfg.hooks_override)
        # Remove hooks_override from config after processing (it's not a real config key)
        try:
            delattr(cfg, 'hooks_override')
        except AttributeError:
            # Config objects may not support delattr; try dict-style deletion
            if 'hooks_override' in cfg:
                del cfg._cfg_dict['hooks_override']

    if options is not None:
        hook_options, merge_options = _split_hook_type_options(options)
        if merge_options:
            cfg.merge_from_dict(merge_options)
        object.__setattr__(
            cfg,
            "_cli_options",
            set(options.keys()),
        )
        _apply_hook_overrides(cfg, hook_options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()
    object.__setattr__(cfg, "_config_file", file_path)

    model_path = os.path.join(cfg.save_path, "model")
    try:
        os.makedirs(model_path, exist_ok=True)
    except FileExistsError:
        pass

    if not cfg.resume and save_artifacts:
        _save_config_artifacts(cfg, file_path, options)
    
    return cfg


def default_setup(cfg):
    """Derive per-rank settings and seed this process."""
    # Batch-size configs are global totals and are divided across ranks here.
    world_size = comm.get_world_size()
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    assert cfg.batch_size % world_size == 0
    assert cfg.batch_size_val is None or cfg.batch_size_val % world_size == 0
    assert cfg.batch_size_test is None or cfg.batch_size_test % world_size == 0
    cfg.batch_size_per_gpu = cfg.batch_size // world_size
    cfg.batch_size_val_per_gpu = (
        cfg.batch_size_val // world_size if cfg.batch_size_val is not None else 1
    )
    cfg.batch_size_test_per_gpu = (
        cfg.batch_size_test // world_size if cfg.batch_size_test is not None else 1
    )
    # Offset the seed by rank and worker allocation to avoid duplicate streams.
    rank = comm.get_rank()
    seed = None if cfg.seed is None else cfg.seed + rank * cfg.num_worker_per_gpu
    set_seed(seed, deterministic=cfg.deterministic)
    return cfg
