"""
Optimizer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
from pimm.utils.logger import get_root_logger
from pimm.utils.registry import Registry

OPTIMIZERS = Registry("optimizers")


OPTIMIZERS.register_module(module=torch.optim.SGD, name="SGD")
OPTIMIZERS.register_module(module=torch.optim.Adam, name="Adam")
OPTIMIZERS.register_module(module=torch.optim.AdamW, name="AdamW")


def build_optimizer(cfg, model, param_dicts=None):
    """Build a PyTorch optimizer from a registry config.

    Args:
        cfg: Optimizer config with a registered ``type`` and optimizer kwargs.
        model (torch.nn.Module): Model whose parameters should be optimized.
        param_dicts (list | None): Optional parameter group specs selected by
            substring matches against parameter names.
    """
    if param_dicts is None:
        cfg.params = model.parameters()
    else:
        # Group 0 is the default group; following groups match configured
        # keywords and may override lr, momentum, or weight decay.
        cfg.params = [dict(names=[], params=[], lr=cfg.lr)]
        for i in range(len(param_dicts)):
            param_group = dict(names=[], params=[])
            if "lr" in param_dicts[i].keys():
                param_group["lr"] = param_dicts[i].lr
            if "momentum" in param_dicts[i].keys():
                param_group["momentum"] = param_dicts[i].momentum
            if "weight_decay" in param_dicts[i].keys():
                param_group["weight_decay"] = param_dicts[i].weight_decay
            cfg.params.append(param_group)

        for n, p in model.named_parameters():
            flag = False
            for i in range(len(param_dicts)):
                if param_dicts[i].keyword in n:
                    cfg.params[i + 1]["names"].append(n)
                    cfg.params[i + 1]["params"].append(p)
                    flag = True
                    break
            if not flag:
                cfg.params[0]["names"].append(n)
                cfg.params[0]["params"].append(p)

        logger = get_root_logger()
        for i in range(len(cfg.params)):
            param_names = cfg.params[i].pop("names")
            message = ""
            for key in cfg.params[i].keys():
                if key != "params":
                    message += f" {key}: {cfg.params[i][key]};"
            if param_names:
                logger.info(f"Params Group {i+1} -{message} Params: [{param_names[0]}, ...].")
            else:
                logger.info(f"Params Group {i+1} -{message} Params: [empty group].")
    return OPTIMIZERS.build(cfg=cfg)
