# syntax=docker/dockerfile:1

ARG TORCH_VERSION=2.10.0
ARG CUDA_VERSION=12.6
ARG CUDNN_VERSION=9
FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
ENV CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    MAX_JOBS=1 \
    PATH=/opt/pimm/.venv/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/opt/pimm/.venv

RUN rm -f /etc/apt/sources.list.d/*.list
COPY .github/docker/common /opt/pimm/.github/docker/common
RUN bash /opt/pimm/.github/docker/common/install_system.sh

# the image ships only the locked environment: no pimm source is baked in,
# so code always comes from a bound checkout (entrypoint puts cwd on PYTHONPATH)
WORKDIR /opt/pimm
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

COPY .github/docker/pimm /usr/local/bin/pimm
COPY .github/docker/entrypoint.sh /opt/pimm/entrypoint.sh
RUN chmod +x /usr/local/bin/pimm /opt/pimm/entrypoint.sh

# default mount point for containerized jobs that bind the checkout
WORKDIR /opt/pimm/src
ENTRYPOINT ["/opt/pimm/entrypoint.sh"]
CMD ["bash"]
