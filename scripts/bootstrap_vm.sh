#!/usr/bin/env bash
# =============================================================================
# bootstrap_vm.sh — Full system bootstrap for a bare GCE VM
# =============================================================================
# Run this ONCE on a freshly created GCE VM before any other script.
# Tested on: Debian 11 (Bullseye), Debian 12 (Bookworm), Ubuntu 20.04/22.04
# GPU VMs: a2-highgpu-*, a3-highgpu-*, n1-standard-* + T4
#
# What it installs:
#   - System packages: git, curl, wget, unzip, build tools, audio libs,
#     fuse, gcsfuse, Google Cloud SDK (gcloud + gsutil)
#   - Python 3.10 (via deadsnakes PPA on Ubuntu, or system on Debian 11+)
#   - pip, virtualenv
#   - CUDA drivers (optional, skip if using a DLVM image)
#
# Usage:
#   chmod +x scripts/bootstrap_vm.sh
#   sudo bash scripts/bootstrap_vm.sh [--skip-cuda] [--python-version 3.10]
#
# After this script completes, run:
#   bash scripts/setup_python_env.sh
# =============================================================================
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SKIP_CUDA=false
INSTALL_DIR="/opt/dnl"

# Parse flags
for arg in "$@"; do
  case $arg in
    --skip-cuda) SKIP_CUDA=true ;;
    --python-version) PYTHON_VERSION="$2"; shift ;;
  esac
done

log() { echo -e "\n\033[1;32m[bootstrap] $*\033[0m"; }
err() { echo -e "\033[1;31m[bootstrap ERROR] $*\033[0m" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Detect OS
# ---------------------------------------------------------------------------
if [ -f /etc/os-release ]; then
  . /etc/os-release
  OS_ID="$ID"
  OS_VERSION="$VERSION_ID"
else
  err "Cannot detect OS. /etc/os-release not found."
fi
log "Detected OS: $OS_ID $OS_VERSION"

# ---------------------------------------------------------------------------
# 1. Core system packages
# ---------------------------------------------------------------------------
log "Installing core system packages..."
apt-get update -qq

apt-get install -y --no-install-recommends \
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
  libxml2-dev \
  libxslt1-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  libncurses5-dev \
  libncursesw5-dev \
  xz-utils \
  tk-dev \
  liblzma-dev \
  uuid-dev \
  htop \
  tmux \
  screen \
  vim \
  nano \
  jq \
  rsync \
  pv

# ---------------------------------------------------------------------------
# 2. Audio system libraries (required by librosa, soundfile, pydub, pedalboard)
# ---------------------------------------------------------------------------
log "Installing audio system libraries..."
apt-get install -y --no-install-recommends \
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
  libmpg123-dev

# ---------------------------------------------------------------------------
# 3. FUSE (required by gcsfuse)
# ---------------------------------------------------------------------------
log "Installing FUSE..."
apt-get install -y --no-install-recommends fuse

# Allow non-root users to mount FUSE filesystems
if ! grep -q "^user_allow_other" /etc/fuse.conf 2>/dev/null; then
  echo "user_allow_other" >> /etc/fuse.conf
fi

# ---------------------------------------------------------------------------
# 4. Google Cloud SDK (gcloud, gsutil, bq)
# ---------------------------------------------------------------------------
log "Installing Google Cloud SDK..."
if ! command -v gcloud &>/dev/null; then
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/cloud.google.gpg] \
    https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee /etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update -qq
  apt-get install -y google-cloud-cli
else
  log "gcloud already installed: $(gcloud version --format='value(Google Cloud SDK)' 2>/dev/null)"
fi

# ---------------------------------------------------------------------------
# 5. gcsfuse
# ---------------------------------------------------------------------------
log "Installing gcsfuse..."
if ! command -v gcsfuse &>/dev/null; then
  GCSFUSE_REPO="gcsfuse-$(lsb_release -cs)"
  echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list
  # Key already added in step 4 above
  apt-get update -qq
  apt-get install -y gcsfuse
else
  log "gcsfuse already installed: $(gcsfuse --version 2>&1 | head -1)"
fi

# ---------------------------------------------------------------------------
# 6. Python 3.10
# ---------------------------------------------------------------------------
log "Installing Python ${PYTHON_VERSION}..."

PYTHON_BIN="python${PYTHON_VERSION}"

if ! command -v "$PYTHON_BIN" &>/dev/null; then
  if [ "$OS_ID" = "ubuntu" ]; then
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y \
      "python${PYTHON_VERSION}" \
      "python${PYTHON_VERSION}-dev" \
      "python${PYTHON_VERSION}-venv" \
      "python${PYTHON_VERSION}-distutils"
  elif [ "$OS_ID" = "debian" ]; then
    # Debian 11+ ships Python 3.9/3.11; build 3.10 from source if needed
    if apt-cache show "python${PYTHON_VERSION}" &>/dev/null; then
      apt-get install -y "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-dev" "python${PYTHON_VERSION}-venv"
    else
      log "Building Python ${PYTHON_VERSION} from source (Debian)..."
      cd /tmp
      wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}.15/Python-${PYTHON_VERSION}.15.tgz"
      tar xzf "Python-${PYTHON_VERSION}.15.tgz"
      cd "Python-${PYTHON_VERSION}.15"
      ./configure --enable-optimizations --with-ensurepip=install --prefix=/usr/local
      make -j"$(nproc)"
      make altinstall
      cd /tmp && rm -rf "Python-${PYTHON_VERSION}.15"*
    fi
  fi
else
  log "Python ${PYTHON_VERSION} already available: $($PYTHON_BIN --version)"
fi

# Install / upgrade pip
log "Upgrading pip..."
"$PYTHON_BIN" -m ensurepip --upgrade 2>/dev/null || true
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 7. CUDA drivers (skip if using DLVM image or --skip-cuda flag)
# ---------------------------------------------------------------------------
if [ "$SKIP_CUDA" = "false" ]; then
  if ! command -v nvidia-smi &>/dev/null; then
    log "Installing CUDA 12.3 drivers..."
    # For Debian/Ubuntu: use the NVIDIA CUDA network repo
    CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos"
    if [ "$OS_ID" = "ubuntu" ] && [ "$OS_VERSION" = "22.04" ]; then
      CUDA_REPO="${CUDA_KEYRING_URL}/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    elif [ "$OS_ID" = "ubuntu" ] && [ "$OS_VERSION" = "20.04" ]; then
      CUDA_REPO="${CUDA_KEYRING_URL}/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb"
    elif [ "$OS_ID" = "debian" ] && [ "$OS_VERSION" = "11" ]; then
      CUDA_REPO="${CUDA_KEYRING_URL}/debian11/x86_64/cuda-keyring_1.1-1_all.deb"
    elif [ "$OS_ID" = "debian" ] && [ "$OS_VERSION" = "12" ]; then
      CUDA_REPO="${CUDA_KEYRING_URL}/debian12/x86_64/cuda-keyring_1.1-1_all.deb"
    else
      log "WARNING: Unknown OS ${OS_ID} ${OS_VERSION}. Skipping CUDA driver install."
      CUDA_REPO=""
    fi

    if [ -n "$CUDA_REPO" ]; then
      wget -q "$CUDA_REPO" -O /tmp/cuda-keyring.deb
      dpkg -i /tmp/cuda-keyring.deb
      apt-get update -qq
      apt-get install -y cuda-drivers
      log "CUDA drivers installed. A reboot may be required before nvidia-smi works."
    fi
  else
    log "CUDA drivers already present: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  fi
else
  log "Skipping CUDA driver install (--skip-cuda flag set or DLVM image detected)."
fi

# ---------------------------------------------------------------------------
# 8. git-lfs init
# ---------------------------------------------------------------------------
log "Initialising git-lfs..."
git lfs install --system 2>/dev/null || git lfs install

# ---------------------------------------------------------------------------
# 9. Verify
# ---------------------------------------------------------------------------
log "=== Bootstrap complete. Verification ==="
echo "git:      $(git --version)"
echo "python:   $($PYTHON_BIN --version)"
echo "pip:      $($PYTHON_BIN -m pip --version)"
echo "gcloud:   $(gcloud version --format='value(Google Cloud SDK)' 2>/dev/null || echo 'not found')"
echo "gsutil:   $(gsutil version 2>/dev/null | head -1 || echo 'not found')"
echo "gcsfuse:  $(gcsfuse --version 2>&1 | head -1 || echo 'not found')"
echo "ffmpeg:   $(ffmpeg -version 2>&1 | head -1 || echo 'not found')"
echo "sox:      $(sox --version 2>&1 || echo 'not found')"
echo "nvidia-smi: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not available')"

log "Next step: bash scripts/setup_python_env.sh"
