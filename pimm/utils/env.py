"""
Environment Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from datetime import datetime


def get_random_seed():
    """Return a process-local random seed mixed from time and OS entropy."""
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    return seed


def set_seed(seed=None, deterministic=False):
    """Seed Python, NumPy, and PyTorch RNGs for reproducible experiments.

    Args:
        seed (int | None): Seed to use. If None, a random seed is generated.
        deterministic (bool): If True, request deterministic CUDA algorithms
            and disable TF32 paths where PyTorch exposes the controls.
    """
    if seed is None:
        seed = get_random_seed()

    # Seed every RNG family used by the training stack.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    if deterministic:
        # Required by CUDA for deterministic cublas kernels in recent PyTorch.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
