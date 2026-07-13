from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

CUDA_FLAGS = []
if torch.cuda.is_available():
    CUDA_ARCH = []
    for i in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(i)
        CUDA_ARCH.append(f"{capability[0]}.{capability[1]}")
    
    for arch in set(CUDA_ARCH):
        arch_num = arch.replace('.', '')
        CUDA_FLAGS.extend([
            f"-gencode=arch=compute_{arch_num},code=sm_{arch_num}",
            f"-gencode=arch=compute_{arch_num},code=compute_{arch_num}"
        ])

CXX_FLAGS = ["-O3", "-ffast-math", "-ftree-vectorize"]
NVCC_FLAGS = ["-O3", "--use_fast_math"]

setup(
    name="cnms",
    version="2.0.0+cu126torch2.10",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="cnms._ext",
            sources=[
                "csrc/greedy_reduction_packed.cpp",
                "csrc/greedy_reduction_packed_cuda.cu",
                "csrc/greedy_reduction_packed_cpu.cpp",
                "csrc/greedy_reduction_padded_cuda.cu",
                "csrc/greedy_reduction_padded_cpu.cpp",
            ],
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": CUDA_FLAGS + NVCC_FLAGS,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
