# =============================================================================
# Dockerfile — Foundation-1 Training Container (DNL fork)
# =============================================================================
# Base:  NVIDIA PyTorch 24.01 (CUDA 12.3.2, PyTorch 2.2, Python 3.10, Ubuntu 22.04)
# Source: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
#
# This image is used for BOTH pre-encoding (pre_encode.py) and training
# (train.py). The STEP env var controls which script runs at container start.
#
# Build & push (Artifact Registry — recommended):
#   gcloud auth configure-docker europe-west4-docker.pkg.dev
#   docker build -t europe-west4-docker.pkg.dev/YOUR_PROJECT/dnl/f1-training:latest .
#   docker push europe-west4-docker.pkg.dev/YOUR_PROJECT/dnl/f1-training:latest
#
# Build & push (Container Registry — legacy):
#   docker build -t gcr.io/YOUR_PROJECT_ID/dnl-f1-training:latest .
#   docker push gcr.io/YOUR_PROJECT_ID/dnl-f1-training:latest
#
# Run locally (with GPU):
#   docker run --gpus all --rm -it \
#     -e GCS_BUCKET=your-bucket \
#     -e STEP=train \
#     europe-west4-docker.pkg.dev/YOUR_PROJECT/dnl/f1-training:latest
# =============================================================================

FROM nvcr.io/nvidia/pytorch:24.01-py3

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
# NOTE: The NVIDIA PyTorch base image ships Ubuntu 22.04 with a minimal set
# of packages. git, curl, and many audio libs are NOT pre-installed.
# This block installs everything needed by the full pipeline.
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    # Core utilities (NOT pre-installed on the base image)
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
    # Python build dependencies
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libbz2-dev \
    # Audio system libraries
    # Required by: soundfile, librosa, pydub, pedalboard, torchaudio
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
    # FUSE (required by gcsfuse)
    fuse \
    # Useful utilities
    htop \
    tmux \
    vim \
    jq \
    rsync \
    pv \
    && rm -rf /var/lib/apt/lists/*

# Allow non-root users to mount FUSE filesystems
RUN echo "user_allow_other" >> /etc/fuse.conf

# ---------------------------------------------------------------------------
# 2. Google Cloud SDK (gcloud, gsutil) + gcsfuse
# ---------------------------------------------------------------------------
# Key format note (2024+):
#   The Google Cloud apt repos require the key saved as a plain ASCII-armored
#   .asc file (via 'tee'), NOT a dearmored .gpg binary.  Both the Cloud SDK
#   and gcsfuse source lines must reference it with 'signed-by='.  Using
#   'gpg --dearmor' and omitting 'signed-by=' from the gcsfuse line causes:
#     W: GPG error: ... NO_PUBKEY C0BA5CE6DC6315A3
#     E: The repository ... is not signed.
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
# 3. Python packages
# ---------------------------------------------------------------------------
# The base image already provides: torch 2.2, torchaudio 2.2, CUDA 12.3 toolkit.
# We upgrade pip and install in dependency order using the requirements/ files.

RUN pip install --upgrade pip setuptools wheel

# GCS + experiment tracking (no heavy transitive deps — install first)
RUN pip install --no-cache-dir \
    "google-cloud-storage>=2.14.0,<3.0" \
    "clearml>=1.14.0" \
    "wandb>=0.15.4,<0.16.0" \
    "huggingface_hub>=0.20.0" \
    "safetensors>=0.4.0"

# Copy requirements files before the full repo (better Docker layer caching)
COPY requirements/ /workspace/requirements/

# Install base + encode + train requirements
# base.txt: shared deps (transformers, librosa, auraloss, etc.)
# encode.txt: pytorch-lightning, torchmetrics
# train.txt: deepspeed, bitsandbytes, pytorch-lightning
RUN pip install --no-cache-dir -r /workspace/requirements/base.txt \
    && pip install --no-cache-dir -r /workspace/requirements/encode.txt \
    && pip install --no-cache-dir -r /workspace/requirements/train.txt

# ---------------------------------------------------------------------------
# 4. Copy repository and install in editable mode
# ---------------------------------------------------------------------------
COPY . /workspace/

# Install the package itself without reinstalling already-present deps
RUN pip install --no-cache-dir -e /workspace/ --no-deps \
    || echo "WARNING: editable install failed, continuing (deps already installed above)"

# Make all scripts executable
RUN chmod +x /workspace/scripts/*.sh

# git-lfs init
RUN git lfs install --system 2>/dev/null || git lfs install

# ---------------------------------------------------------------------------
# 5. Align torchvision with the installed torch version
# ---------------------------------------------------------------------------
# The NVIDIA base image ships torchvision built against its bundled torch 2.2.
# After upgrading torch (via requirements/), the old torchvision crashes on
# import with:
#   RuntimeError: operator torchvision::nms does not exist
# because its compiled C++ extension references ops that no longer exist in
# the new torch dispatcher. Upgrading torchvision here ensures it is rebuilt
# against the current torch ABI. The --extra-index-url is needed so pip can
# find CUDA-enabled torchvision wheels from the PyTorch index.
RUN pip install --no-cache-dir --upgrade \
    torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# 6. Verify installation
# ---------------------------------------------------------------------------
# NOTE: Use a heredoc (<<'EOF') so that the Python source lines are NOT
# parsed by Docker as Dockerfile instructions. A plain RUN python -c "..."
# with indented continuation lines causes Docker to misparse the indented
# 'import' keyword as an unknown instruction.
RUN python3 << 'EOF'
import torch, torchaudio
print(f'PyTorch:    {torch.__version__}')
print(f'torchaudio: {torchaudio.__version__}')
print(f'CUDA built: {torch.version.cuda}')
import google.cloud.storage; print('google-cloud-storage: OK')
import soundfile; print('soundfile: OK')
import librosa; print('librosa: OK')
import pytorch_lightning as pl; print(f'pytorch-lightning: {pl.__version__}')
import clearml; print(f'clearml: {clearml.__version__}')
EOF

# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------
# STEP env var controls which pipeline stage runs:
#   STEP=train   → runs train.py via gcp_train_f1_3s.sh (default)
#   STEP=encode  → runs pre_encode.py via gcp_train_f1_3s.sh (encode-only mode)
#   STEP=smoke   → runs smoke_test_gcs.py
ENV STEP=train

CMD ["bash", "-c", "bash /workspace/scripts/gcp_train_f1_3s.sh"]
