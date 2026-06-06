# Copyright (c) Facebook, Inc. and its affiliates.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
Modified from detectron2(https://github.com/facebookresearch/detectron2)

Copyright (c) Xiaoyang Wu (xiaoyang.wu@connect.hku.hk). All Rights Reserved.
Please cite our work if you use any part of the code.
"""

import functools
import os
import logging
import numpy as np
import torch
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set by setup_distributed() when running under torchrun or Slurm.
"""


def _parse_tasks_per_node(value: str, default_nodes: int):
    """Parse SLURM *_TASKS_PER_NODE strings into one task count per node."""
    if not value:
        return [1] * max(1, default_nodes)
    value = value.strip()
    parts = [p.strip() for p in value.split(',') if p.strip()]
    result = []
    for p in parts:
        if '(x' in p:
            # format N(xK)
            try:
                n_str, rep = p.split('(x')
                n = int(n_str)
                k = int(rep.rstrip(')'))
                result.extend([n] * k)
            except Exception:
                continue
        else:
            try:
                result.append(int(p))
            except Exception:
                continue
    if not result:
        result = [1] * max(1, default_nodes)
    # if a single number and multiple nodes, replicate
    if len(result) == 1 and default_nodes > 1:
        result = [result[0]] * default_nodes
    return result


def get_slurm_env():
    """Extract the SLURM fields needed to initialize torch.distributed."""
    nnodes = int(os.environ.get('SLURM_NNODES', 1))
    tpn_str = (
        os.environ.get('SLURM_TASKS_PER_NODE')
        or os.environ.get('SLURM_STEP_TASKS_PER_NODE')
        or os.environ.get('SLURM_NTASKS_PER_NODE')
        or ''
    )
    tasks_per_node_list = _parse_tasks_per_node(tpn_str, nnodes)
    try:
        ntasks_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE', ''))
    except Exception:
        ntasks_per_node = tasks_per_node_list[0] if tasks_per_node_list else 1

    return {
        'job_id': os.environ.get('SLURM_JOB_ID'),
        'ntasks': int(os.environ.get('SLURM_NTASKS', 1)),
        'ntasks_per_node': ntasks_per_node,
        'tasks_per_node_list': tasks_per_node_list,
        'nnodes': nnodes,
        'node_id': int(os.environ.get('SLURM_NODEID', 0)),
        'proc_id': int(os.environ.get('SLURM_PROCID', 0)),
        'local_id': int(os.environ.get('SLURM_LOCALID', 0)),
        'nodelist': os.environ.get('SLURM_NODELIST', '127.0.0.1'),
    }


def setup_distributed():
    """Initialize torch.distributed from torchrun or SLURM environment."""
    logger = logging.getLogger(__name__)
    if "SLURM_PROCID" not in os.environ and "RANK" not in os.environ:
        logger.info("No distributed environment found")
        return

    # Prefer torchrun's explicit rank env, then fall back to SLURM variables.
    slurm_env = get_slurm_env() if "SLURM_PROCID" in os.environ else None
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', world_size))
        tasks_per_node_list = [local_world_size]
        node_id = int(os.environ.get('GROUP_RANK', os.environ.get('SLURM_NODEID', 0)))
    elif slurm_env is not None:
        world_size = slurm_env['ntasks']
        rank = slurm_env['proc_id']
        local_rank = slurm_env['local_id']
        tasks_per_node_list = slurm_env.get('tasks_per_node_list', [world_size])
        node_id = slurm_env['node_id']
    else:
        world_size = 1
        rank = 0
        local_rank = 0
        tasks_per_node_list = [1]
        node_id = 0
    
    # Set CUDA device while respecting CUDA_VISIBLE_DEVICES.
    num_visible = torch.cuda.device_count()
    if num_visible == 0:
        raise RuntimeError("no CUDA devices available")
    device_index = local_rank if num_visible > 1 else 0
    if device_index >= num_visible:
        device_index = device_index % num_visible
    torch.cuda.set_device(device_index)
    
    # Get master address from SLURM_NODELIST when torchrun/env did not set it.
    if 'MASTER_ADDR' not in os.environ:
        if slurm_env is not None:
            nodelist = slurm_env['nodelist']
            if '[' in nodelist:
                base = nodelist.split('[')[0]
                indices = nodelist.split('[')[1].split(']')[0]
                first_index = indices.split(',')[0].split('-')[0]
                master_addr = f"{base}{first_index}"
            elif ',' in nodelist:
                master_addr = nodelist.split(',')[0]
            else:
                master_addr = nodelist
        else:
            master_addr = '127.0.0.1'
        os.environ['MASTER_ADDR'] = master_addr
    
    # Derive a stable port for SLURM jobs so all ranks rendezvous together.
    if 'MASTER_PORT' not in os.environ:
        if slurm_env is not None:
            job_id = slurm_env['job_id']
            if job_id and len(job_id) >= 4:
                port = 20000 + int(job_id[-4:]) % 10000
            else:
                port = 29500
        else:
            port = 29500
        os.environ['MASTER_PORT'] = str(port)
        
    logger.info("Initializing distributed training:")

    logger.info(f"  - World size: {world_size}")
    logger.info(f"  - Rank: {rank}")
    logger.info(f"  - Local rank: {local_rank}")
    logger.info(f"  - Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

    # Initialize the global NCCL process group.
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
    )
    
    # Build a same-node group for local rank and local size queries.
    global _LOCAL_PROCESS_GROUP
    _LOCAL_PROCESS_GROUP = None
    offset = 0
    for i, tpn in enumerate(tasks_per_node_list):
        if offset >= world_size:
            break
        ranks_on_node = list(range(offset, min(offset + tpn, world_size)))
        if ranks_on_node:
            pg = dist.new_group(ranks_on_node)
            if i == node_id:
                _LOCAL_PROCESS_GROUP = pg
        offset += tpn
        
    # Fallback if logic didn't set it (e.g. strange task/node mismatch)
    if _LOCAL_PROCESS_GROUP is None:
         _LOCAL_PROCESS_GROUP = dist.new_group([rank])

    return True


def get_world_size() -> int:
    """Return the initialized distributed world size, or 1."""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Return the initialized distributed rank, or 0."""
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert (
        _LOCAL_PROCESS_GROUP is not None
    ), "Local process group is not created! Please initialize distributed state first."
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    """Return whether this process is global rank zero."""
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = (
            _get_global_gloo_group()
        )  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.
    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum
    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def cleanup_distributed():
    """Destroy the distributed process group if initialized."""
    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return

    if get_world_size() > 1:
        try:
            synchronize()
        except Exception:
            pass
    try:
        torch.distributed.destroy_process_group()
    except Exception:
        pass
