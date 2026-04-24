# =============================================================================
# Dockerfile — Foundation-1 Training Container (DNL fork)
# =============================================================================
# Base:  NVIDIA PyTorch 24.04 (CUDA 12.4.1, PyTorch 2.3, Python 3.10, Ubuntu 22.04)
# Driver requirement: >= 525.85 (Vertex AI A100 VMs ship driver 525.x — compatible)
# DO NOT upgrade this base image past 24.04 until Vertex AI A100 VMs ship driver >= 550
# (required for CUDA 12.6+).  See: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/
# Source: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
#
# SIZE OPTIMISATION NOTES
# -----------------------
# The NVIDIA base image is ~18 GB uncompressed.  Cloud Shell has a ~50 GB home
# directory; Docker stores overlay snapshots there, so the total uncompressed
# image must stay well under ~45 GB to leave room for the export step.
#
# Techniques used here:
#   1. Remove large unused packages from the base image in the very first layer
#      (apex, transformer-engine, nsight tools, CUDA samples) — saves ~3-4 GB.
#   2. Merge all pip install steps into ONE RUN command — avoids duplicate layer blobs.
#      All installs use --no-cache-dir so no cache is written to any layer.
#   3. Merge small housekeeping RUN steps (fuse.conf, chmod, git-lfs) into
#      adjacent layers to reduce total snapshot count.
#   4. Use --no-cache-dir on every pip call and clean apt lists after every
#      apt-get block.
#
# Build & push (Artifact Registry — recommended):
#   gcloud auth configure-docker europe-west4-docker.pkg.dev
#   docker build -t europe-west4-docker.pkg.dev/YOUR_PROJECT/dnl/f1-training:latest .
#   docker push europe-west4-docker.pkg.dev/YOUR_PROJECT/dnl/f1-training:latest
#
# Run locally (with GPU):
#   docker run --gpus all --rm -it \
#     -e GCS_BUCKET=your-bucket \
#     -e STEP=train \
#     europe-west4-docker.pkg.dev/YOUR_PROJECT/dnl/f1-training:latest
# =============================================================================

# CUDA 12.4.1 — requires driver >= 525.85
# Vertex AI A100 VMs (europe-west4) ship driver 525.x (CUDA 12.3 runtime),
# which satisfies the 525.85 minimum for this image.
# Do NOT bump past 24.04 until Vertex AI upgrades its A100 driver to >= 550.
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# ---------------------------------------------------------------------------
# 0. Strip large unused packages from the base image  (~3-4 GB saved)
# ---------------------------------------------------------------------------
# The NVIDIA base image bundles several packages that this pipeline does not
# use: apex (custom CUDA kernels for NLP), transformer-engine (FP8 training),
# nsight-systems / nsight-compute (profiling GUIs), and the full CUDA samples.
# Removing them in the very first layer keeps the overlay diff small.
RUN pip uninstall -y \
        apex \
        transformer-engine \
        pynvml \
    2>/dev/null || true \
    && apt-get purge -y --auto-remove \
        nsight-systems-* \
        nsight-compute-* \
        cuda-samples-* \
    2>/dev/null || true \
    && rm -rf \
        /usr/local/cuda/samples \
        /usr/local/cuda/extras \
        /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 1. System packages + FUSE config
# ---------------------------------------------------------------------------
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        curl \
        wget \
        unzip \
        zip \
        ca-certificates \
        gnupg \
        lsb-release \
        software-properties-common \
        build-essential \
        pkg-config \
        cmake \
        ninja-build \
        libssl-dev \
        libffi-dev \
        zlib1g-dev \
        libbz2-dev \
        libsndfile1 \
        libsndfile1-dev \
        libsox-fmt-all \
        sox \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswresample-dev \
        libswscale-dev \
        libportaudio2 \
        portaudio19-dev \
        libasound2-dev \
        libflac-dev \
        libvorbis-dev \
        libopus-dev \
        libmp3lame-dev \
        libmpg123-dev \
        fuse \
        htop \
        tmux \
        vim \
        jq \
        rsync \
        pv \
    && echo "user_allow_other" >> /etc/fuse.conf \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Google Cloud SDK (gcloud, gsutil) + gcsfuse
# ---------------------------------------------------------------------------
# Key format note (2024+):
#   Both the Cloud SDK and gcsfuse source lines must use 'signed-by=' pointing
#   to a plain ASCII-armored .asc key file (not a dearmored .gpg binary).
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | tee /usr/share/keyrings/cloud.google.asc > /dev/null \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/cloud.google.asc] \
       https://packages.cloud.google.com/apt cloud-sdk main" \
       | tee /etc/apt/sources.list.d/google-cloud-sdk.list \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] \
       https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -cs) main" \
       | tee /etc/apt/sources.list.d/gcsfuse.list \
    && apt-get update -qq \
    && apt-get install -y --no-install-recommends \
       google-cloud-cli \
       gcsfuse \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 3. Python packages  (single RUN layer)
# ---------------------------------------------------------------------------
# All pip installs are merged into one RUN command so Docker creates only one
# overlay diff instead of four.  Every install uses --no-cache-dir so nothing
# is written to the pip cache.  We do NOT call 'pip cache purge' because Cloud
# Build disables the pip cache globally (PIP_NO_CACHE_DIR=1) and the purge
# command exits with code 1 when the cache is disabled.
#
# Install order:
#   a) pip/setuptools/wheel upgrade
#   b) GCS + experiment tracking (light deps, install first)
#   c) requirements/base.txt  (audio libs, transformers, diffusion utils)
#   d) requirements/encode.txt (pytorch-lightning, torchmetrics)
#   e) requirements/train.txt  (deepspeed, bitsandbytes)
#   f) torch==2.6.0 + torchaudio==2.6.0 + torchvision==0.21.0 pinned to cu124.
#      The NGC PyPI index now serves torch 2.11+cu130 (CUDA 13.0 nightly),
#      which requires driver >= 570.  Vertex AI A100 VMs ship driver 525.x
#      (CUDA 12.3), so we override with the stable cu124 wheel instead.

COPY requirements/ /workspace/requirements/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    \
    && pip install --no-cache-dir \
        "google-cloud-storage>=2.14.0,<3.0" \
        "clearml>=1.14.0" \
        "wandb>=0.15.4,<0.16.0" \
        "huggingface_hub>=0.20.0" \
        "safetensors>=0.4.0" \
    \
    && pip install --no-cache-dir -r /workspace/requirements/base.txt \
    && pip install --no-cache-dir -r /workspace/requirements/encode.txt \
    && pip install --no-cache-dir -r /workspace/requirements/train.txt \
    \
    # Pin torch/torchaudio to stable CUDA 12.4 builds.
    # The NGC PyPI index (pypi.ngc.nvidia.com) now serves torch 2.11+cu130
    # (CUDA 13.0 nightly), which requires driver >= 570 — not available on
    # Vertex AI A100 VMs.  We override with the stable cu124 wheel from
    # download.pytorch.org, which needs driver >= 525.60 (satisfied).
    && pip install --no-cache-dir \
        "torch==2.6.0" \
        "torchaudio==2.6.0" \
        "torchvision==0.21.0" \
        --index-url https://download.pytorch.org/whl/cu124 \
    \
    && find /usr/local/lib/python3.10 -name '*.pyc' -delete \
    && find /usr/local/lib/python3.10 -name '__pycache__' -type d -empty -delete

# ---------------------------------------------------------------------------
# 4. Copy repository and install in editable mode
# ---------------------------------------------------------------------------
COPY . /workspace/

RUN pip install --no-cache-dir -e /workspace/ --no-deps \
        || echo "WARNING: editable install failed, continuing (deps already installed above)" \
    && chmod +x /workspace/scripts/*.sh \
    && git lfs install --system 2>/dev/null || git lfs install

# ---------------------------------------------------------------------------
# 5. Verify installation
# ---------------------------------------------------------------------------
# NOTE: All Python statements are on a single logical line (joined with ';')
# so that Docker's parser never sees 'import' at the start of a new line.
# Heredoc syntax (<<'EOF') is NOT used here because Cloud Build runs an older
# Docker daemon that does not support BuildKit heredocs and would misparse the
# indented 'import' lines as unknown Dockerfile instructions.
RUN python3 -c "\
import torch, torchaudio; \
print('PyTorch:    ' + torch.__version__); \
print('torchaudio: ' + torchaudio.__version__); \
print('CUDA built: ' + str(torch.version.cuda)); \
import google.cloud.storage; print('google-cloud-storage: OK'); \
import soundfile; print('soundfile: OK'); \
import librosa; print('librosa: OK'); \
import pytorch_lightning as pl; print('pytorch-lightning: ' + pl.__version__); \
import clearml; print('clearml: ' + clearml.__version__) \
"

# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------
# STEP env var controls which pipeline stage runs:
#   STEP=train   → runs train.py via gcp_train_f1_3s.sh (default)
#   STEP=encode  → runs pre_encode.py via gcp_train_f1_3s.sh (encode-only mode)
#   STEP=smoke   → runs smoke_test_gcs.py
ENV STEP=train

CMD ["bash", "-c", "bash /workspace/scripts/gcp_train_f1_3s.sh"]
