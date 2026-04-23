#!/usr/bin/env bash
# =============================================================================
# bootstrap_vertex_inline.sh
# =============================================================================
# This script is embedded inline in vertex_job_*.yaml container commands.
# It bootstraps a bare Vertex AI DLVM container (which ships without git,
# audio libs, gcsfuse, etc.) before running the actual training code.
#
# It is NOT meant to be run directly — it is sourced/inlined by the YAML
# container command blocks. See scripts/vertex_job_f1_3s.yaml for usage.
#
# Idempotent: safe to re-run if the job is preempted and restarted.
# =============================================================================
set -euo pipefail

log() { echo -e "\n\033[1;32m[vertex-bootstrap] $*\033[0m"; }

log "=== Stage 0: System packages ==="
apt-get update -qq
apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    gnupg \
    lsb-release \
    software-properties-common \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    libsox-fmt-all \
    sox \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libportaudio2 \
    libasound2-dev \
    libflac-dev \
    libvorbis-dev \
    libopus-dev \
    libmp3lame-dev \
    fuse \
    jq \
    rsync \
    pv
rm -rf /var/lib/apt/lists/*

log "=== Stage 1: Google Cloud SDK + gcsfuse ==="
if ! command -v gcloud &>/dev/null; then
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/cloud.google.gpg] \
        https://packages.cloud.google.com/apt cloud-sdk main" \
        | tee /etc/apt/sources.list.d/google-cloud-sdk.list
    echo "deb https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -cs) main" \
        | tee /etc/apt/sources.list.d/gcsfuse.list
    apt-get update -qq
    apt-get install -y --no-install-recommends google-cloud-cli gcsfuse
    rm -rf /var/lib/apt/lists/*
    echo "user_allow_other" >> /etc/fuse.conf
else
    log "gcloud already available, skipping SDK install"
fi

log "=== Stage 2: Python packages ==="
pip install --upgrade pip setuptools wheel --quiet
pip install --no-cache-dir \
    "google-cloud-storage>=2.14.0,<3.0" \
    "clearml>=1.14.0" \
    "wandb>=0.15.4,<0.16.0" \
    "huggingface_hub>=0.20.0" \
    "safetensors>=0.4.0" \
    --quiet

log "=== Stage 3: Clone repository ==="
REPO_DIR="/workspace/dnl-f1-training"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone https://github.com/arsenylosev/dnl-f1-training.git "$REPO_DIR"
else
    log "Repo already cloned, pulling latest..."
    cd "$REPO_DIR" && git pull --ff-only
fi
cd "$REPO_DIR"

log "=== Stage 4: Install Python dependencies ==="
pip install --no-cache-dir -r requirements/base.txt --quiet
pip install --no-cache-dir -r requirements/encode.txt --quiet
pip install --no-cache-dir -r requirements/train.txt --quiet
pip install --no-cache-dir -e . --no-deps --quiet \
    || log "WARNING: editable install failed, continuing"

log "=== Bootstrap complete. Working directory: $(pwd) ==="
