# syntax=docker/dockerfile:1

ARG TORCH_VERSION=2.5.0
ARG CUDA_VERSION=12.4
ARG CUDNN_VERSION=9
FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

# Re-declare ARGs after FROM (pre-FROM ARGs are not inherited)
ARG TORCH_VERSION=2.5.0
ARG CUDA_VERSION=12.4
ARG CUDA_VERSION_NO_DOT=124
ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

# Prevent user pip packages (~/.local/lib/python*/) from shadowing
# container packages when home directories are bind-mounted (Apptainer).
ENV PYTHONNOUSERSITE=1

# Fix nvidia-key error
RUN rm -f /etc/apt/sources.list.d/*.list

# System packages
RUN export DEBIAN_FRONTEND=noninteractive \
  && apt-get update --no-install-recommends \
  && apt-get install -y --no-install-recommends \
    git wget tmux vim zsh build-essential cmake ninja-build \
    libopenblas-dev libsparsehash-dev \
  && apt-get autoremove -y && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

# Conda packages
RUN conda install -c conda-forge -y \
    h5py pyyaml tensorboard tensorboardx wandb yapf addict \
    einops scipy plyfile termcolor matplotlib black open3d jupyterlab \
  && conda clean -afy

# Pip packages
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir \
    scikit-learn netifaces nvidia-ml-py rich safetensors pandas psutil jupyterlab \
    peft timm huggingface_hub ftfy regex tqdm torchdata submitit tyro \
    torch-geometric \
  && pip install --no-cache-dir \
    torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION_NO_DOT}.html \
  && pip install --no-cache-dir spconv-cu${CUDA_VERSION_NO_DOT} \
  && pip install --no-cache-dir git+https://github.com/octree-nn/ocnn-pytorch.git \
  && pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# FlashAttention2
RUN MAX_JOBS=1 pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# CUDA extensions from build context
COPY libs/ /tmp/libs/
RUN MAX_JOBS=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    pip install --no-cache-dir /tmp/libs/pointrope -v --no-build-isolation \
 && MAX_JOBS=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    pip install --no-cache-dir /tmp/libs/pointops -v --no-build-isolation \
 && MAX_JOBS=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    pip install --no-cache-dir /tmp/libs/pointgroup_ops -v --no-build-isolation \
 && MAX_JOBS=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    pip install --no-cache-dir /tmp/libs/cnms -v --no-build-isolation \
 && MAX_JOBS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    pip install --no-cache-dir /tmp/libs/pytorch3d_ops -v --no-build-isolation \
 && rm -rf /tmp/libs

# Install pimm itself as editable source inside the image.
COPY . /opt/pimm/src
WORKDIR /opt/pimm/src
RUN pip install --no-cache-dir -e .

# Entrypoint neutralizes conda/mamba init blocks in the user's ~/.bashrc,
# which would otherwise hang when Apptainer bind-mounts the home directory.
COPY docker/entrypoint.sh /opt/pimm/entrypoint.sh
RUN chmod +x /opt/pimm/entrypoint.sh
ENTRYPOINT ["/opt/pimm/entrypoint.sh"]
CMD ["bash"]
