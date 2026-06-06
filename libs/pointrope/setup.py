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
    return cuda_flags


CXX_FLAGS = ["-O3"]
NVCC_FLAGS = get_cuda_arch_flags() + [
    "-O3",
    "--ptxas-options=-v",
    "--use_fast_math",
]

setup(
    name="pointrope",
    version="0.1.0",
    packages=["pointrope"],
    package_dir={"pointrope": "."},
    ext_modules=[
        CUDAExtension(
            name="pointrope._C",
            sources=[
                "pointrope.cpp",
                "kernels.cu",
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
