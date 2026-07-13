"""Build script for the ``serialize_cuda`` point-serialization extension.

Vendored from https://github.com/ChristianSchott/point_serialization_cuda
(commit ceebfd14814fb19d11bc622eb1375013362f6327). The upstream ``setup.py``
hard-coded a single CUDA architecture (``sm_86``); this version instead
auto-detects the visible GPUs and otherwise honors ``TORCH_CUDA_ARCH_LIST``,
matching the other CUDA extensions under ``libs/`` so the container and uv
builds target every supported arch.
"""

import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_cuda_arch_flags():
    cuda_flags = []
    if torch.cuda.is_available():
        cuda_arch = set()
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            cuda_arch.add(f"{capability[0]}{capability[1]}")
        for arch in sorted(cuda_arch):
            cuda_flags.extend(
                [
                    f"-gencode=arch=compute_{arch},code=sm_{arch}",
                    f"-gencode=arch=compute_{arch},code=compute_{arch}",
                ]
            )
    elif not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        raise RuntimeError(
            "No visible CUDA device and TORCH_CUDA_ARCH_LIST is unset. "
            "Run the build on a GPU node or set TORCH_CUDA_ARCH_LIST, e.g. 8.0."
        )
    # When no explicit flags are returned, torch's BuildExtension falls back to
    # TORCH_CUDA_ARCH_LIST on its own.
    return cuda_flags


CXX_FLAGS = ["-O3"]
NVCC_FLAGS = get_cuda_arch_flags() + ["-O3", "--use_fast_math"]

setup(
    name="serialize_cuda",
    version="0.1.0+cu126torch2.10",
    ext_modules=[
        CUDAExtension(
            name="serialize_cuda",
            sources=[
                "bindings.cpp",
                "hilbert_cuda.cu",
                "hilbert_cuda_approx.cu",
                "morton_cuda.cu",
            ],
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": NVCC_FLAGS,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
