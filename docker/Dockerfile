# Reusable CUDA + PyTorch build environment for Linux projects.
# Default values target the current project's validated toolchain, but can be
# overridden at build time with --build-arg.

ARG BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TZ=UTC
ARG APT_MIRROR=
ARG PIP_INDEX_URL=
ARG PIP_TRUSTED_HOST=
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_PACKAGES="torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1"
ARG INSTALL_TORCHCODEC=1
ARG TORCHCODEC_PACKAGE="torchcodec==0.9.1"
ARG PYTHON_BUILD_TOOLS="pip==25.3 setuptools==80.9.0 wheel==0.45.1 build==1.3.0"
ARG TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;8.9;9.0;12.0"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=${TZ} \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Optional Ubuntu mirror override for regions where the default archive is slow.
RUN if [[ -n "${APT_MIRROR}" ]]; then \
      sed -i "s|http://archive.ubuntu.com|${APT_MIRROR}|g" /etc/apt/sources.list && \
      sed -i "s|http://security.ubuntu.com|${APT_MIRROR}|g" /etc/apt/sources.list; \
    fi

# Common native build and multimedia dependencies. The FFmpeg-related packages
# keep this image usable for torchcodec-based projects, while still remaining
# broadly useful as a CUDA/PyTorch build environment.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    ninja-build \
    clang \
    gdb \
    git \
    wget \
    curl \
    unzip \
    zip \
    pkg-config \
    file \
    less \
    jq \
    rsync \
    tree \
    tmux \
    htop \
    ripgrep \
    vim \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    libsndfile1 \
    libsox-dev \
    sox \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY docker/requirements.txt /tmp/quant-gru-requirements.txt

RUN if [[ -n "${PIP_INDEX_URL}" ]]; then \
      python3 -m pip config set global.index-url "${PIP_INDEX_URL}"; \
    fi && \
    if [[ -n "${PIP_TRUSTED_HOST}" ]]; then \
      python3 -m pip config set global.trusted-host "${PIP_TRUSTED_HOST}"; \
    fi

RUN python3 -m pip install --upgrade ${PYTHON_BUILD_TOOLS} && \
    python3 -m pip install --no-cache-dir ${TORCH_PACKAGES} --index-url ${PYTORCH_INDEX_URL} && \
    if [[ "${INSTALL_TORCHCODEC}" == "1" ]]; then \
        python3 -m pip install --no-cache-dir ${TORCHCODEC_PACKAGE} --index-url ${PYTORCH_INDEX_URL}; \
    fi && \
    python3 -m pip install --no-cache-dir -r /tmp/quant-gru-requirements.txt && \
    python3 -m pip check

WORKDIR /workspace

RUN python3 - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
try:
    import torchaudio
    print(f"torchaudio: {torchaudio.__version__}")
except Exception as exc:
    print(f"torchaudio import skipped: {exc}")
PY

RUN if [[ "${INSTALL_TORCHCODEC}" == "1" ]]; then \
      python3 -c "from torchcodec.decoders import AudioDecoder; print('torchcodec: OK')"; \
    fi

CMD ["bash"]
