"""
Default training/testing logic

modified from detectron2(https://github.com/facebookresearch/detectron2)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import json
import argparse
import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import pimm.utils.comm as comm
from pimm.utils.env import get_random_seed, set_seed
from pimm.utils.config import Config, DictAction


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.
    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if comm.get_world_size() == 1:
        return model
    
    # kwargs['find_unused_parameters'] = True
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
        if "output_device" not in kwargs:
            kwargs["output_device"] = [comm.get_local_rank()]
    # try enabling static_graph to avoid per-iteration bucket rebuilds when the set of used params is static
    try:
        ddp = DistributedDataParallel(model, static_graph=True, **kwargs)
    except TypeError:
        ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = None if seed is None else num_workers * rank + worker_id + seed
    set_seed(worker_seed)


def default_argument_parser(epilog=None):
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


def _set_nested(d, key_path, value):
    """Set nested dict value: 'a.b.c' -> d['a']['b']['c'] = value"""
    keys = key_path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _apply_hook_overrides_from_dict(cfg, override_dict):
    """
    Apply hook parameter overrides from a dict.
    
    Args:
        cfg: Config object
        override_dict: Dict mapping hook type to parameter dict.
                      Example: {"WandbNamer": {"extra": "fft"}, "InstanceSegmentationEvaluator": {"every_n_steps": 500}}
    """
    if not hasattr(cfg, 'hooks') or not override_dict:
        return
    
    for hook_type, params in override_dict.items():
        if not isinstance(params, dict):
            continue
        
        # find and update matching hook(s)
        for hook_cfg in cfg.hooks:
            if hook_cfg.get('type') == hook_type:
                for param_path, val in params.items():
                    _set_nested(hook_cfg, param_path, val)


def _apply_hook_overrides(cfg, options):
    """
    Process --options hooks.HookType.param=value and apply to matching hooks.
    
    Example: --options hooks.PretrainEvaluator.every_n_steps=500
    """
    if not hasattr(cfg, 'hooks') or not options:
        return
    
    for key, val in options.items():
        if not key.startswith('hooks.'):
            continue
        
        # parse: hooks.HookType.param.subparam -> (HookType, param.subparam)
        parts = key.split('.', 2)
        if len(parts) < 3:
            continue
        
        hook_type = parts[1]
        param_path = parts[2]
        
        # find and update matching hook(s)
        for hook_cfg in cfg.hooks:
            if hook_cfg.get('type') == hook_type:
                _set_nested(hook_cfg, param_path, val)


def default_config_parser(file_path, options):
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
        cfg.merge_from_dict(options)
        cfg._cli_options = set(options.keys())  # track CLI-provided keys for ConfigModifier
        _apply_hook_overrides(cfg, options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    model_path = os.path.join(cfg.save_path, "model")
    try:
        os.makedirs(model_path, exist_ok=True)
    except FileExistsError:
        pass

    if not cfg.resume:
        # Save config as Python file (for backward compatibility)
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
        
        # Also save as JSON for easy loading with OmegaConf
        # Convert config to dict - Config uses _cfg_dict internally
        cfg_dict = cfg._cfg_dict.to_dict() if hasattr(cfg, '_cfg_dict') else dict(cfg)
        json_path = os.path.join(cfg.save_path, "model.json")
        with open(json_path, 'w') as f:
            json.dump(cfg_dict, f, indent=2, default=str)
    
    return cfg


def default_setup(cfg):
    # scalar by world size
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
    # settle random seed
    rank = comm.get_rank()
    seed = None if cfg.seed is None else cfg.seed + rank * cfg.num_worker_per_gpu
    set_seed(seed, deterministic=cfg.deterministic)
    return cfg
