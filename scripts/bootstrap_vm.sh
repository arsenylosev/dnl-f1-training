#!/usr/bin/env bash
# =============================================================================
# bootstrap_vm.sh — Full system bootstrap for a bare GCE VM
# =============================================================================
# Run this ONCE on a freshly created GCE VM before any other script.
#
# Tested on:
#   - Debian 12 (Bookworm) — default GCE image, Python 3.11 native ✓
#   - Debian 11 (Bullseye) — Python 3.9 native, 3.11 via source ✓
#   - Ubuntu 22.04 LTS     — Python 3.10/3.11 native ✓
#   - Ubuntu 20.04 LTS     — Python 3.11 via deadsnakes PPA ✓
#
# Python version strategy:
#   Debian 12 (Bookworm): python3.11 is the system default — use it directly.
#     python3.10-dev and python3.10-venv are NOT in Bookworm repos.
#   Ubuntu 22.04: python3.11 available natively (python3.10 also available).
#   Ubuntu 20.04: python3.11 via deadsnakes PPA.
#   Default: 3.11 for all distros (compatible with PyTorch 2.1+ and stable-audio-tools).
#
# Usage:
#   chmod +x scripts/bootstrap_vm.sh
#   sudo bash scripts/bootstrap_vm.sh [--skip-cuda] [--python-version 3.11]
#
# After this script completes, run:
#   bash scripts/setup_python_env.sh [--step <smoke|augment|encode|train|eval>]
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults — Python 3.11 works on all supported distros without extra repos
# ---------------------------------------------------------------------------
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
SKIP_CUDA=false
INSTALL_DIR="/opt/dnl"

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-cuda) SKIP_CUDA=true; shift ;;
    --python-version) PYTHON_VERSION="$2"; shift 2 ;;
    *) shift ;;
  esac
done

log()  { echo -e "\n\033[1;32m[bootstrap] $*\033[0m"; }
warn() { echo -e "\033[1;33m[bootstrap WARN] $*\033[0m"; }
err()  { echo -e "\033[1;31m[bootstrap ERROR] $*\033[0m" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Detect OS
# ---------------------------------------------------------------------------
if [ -f /etc/os-release ]; then
  . /etc/os-release
  OS_ID="${ID:-unknown}"
  OS_VERSION="${VERSION_ID:-unknown}"
else
  err "Cannot detect OS. /etc/os-release not found."
fi
log "Detected OS: $OS_ID $OS_VERSION"

# Validate that we are running as root
if [ "$(id -u)" -ne 0 ]; then
  err "This script must be run as root (use: sudo bash scripts/bootstrap_vm.sh)"
fi

# ---------------------------------------------------------------------------
# 1. Core system packages
# ---------------------------------------------------------------------------
log "Updating apt and installing core system packages..."
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

# rubberband-cli: required by pyrubberband (used in augmentation pipeline)
if apt-cache show rubberband-cli &>/dev/null 2>&1; then
  apt-get install -y --no-install-recommends rubberband-cli
else
  warn "rubberband-cli not found in apt — installing from source is not required unless you run the augmentation step."
fi

# ---------------------------------------------------------------------------
# 3. FUSE (required by gcsfuse)
# ---------------------------------------------------------------------------
log "Installing FUSE..."
# libfuse2 is the package name on Debian 12; fuse2 on older distros
if apt-cache show libfuse2 &>/dev/null 2>&1; then
  apt-get install -y --no-install-recommends fuse libfuse2
else
  apt-get install -y --no-install-recommends fuse
fi

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
  log "gcloud already installed: $(gcloud version --format='value(Google Cloud SDK)' 2>/dev/null || echo 'unknown')"
fi

# ---------------------------------------------------------------------------
# 5. gcsfuse
# ---------------------------------------------------------------------------
log "Installing gcsfuse..."
if ! command -v gcsfuse &>/dev/null; then
  GCSFUSE_CODENAME="$(lsb_release -cs)"
  # gcsfuse repo uses the same codename as the distro (bookworm, bullseye, jammy, focal)
  echo "deb https://packages.cloud.google.com/apt gcsfuse-${GCSFUSE_CODENAME} main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list
  apt-get update -qq
  apt-get install -y gcsfuse
else
  log "gcsfuse already installed: $(gcsfuse --version 2>&1 | head -1)"
fi

# ---------------------------------------------------------------------------
# 6. Python — version matrix per distro
# ---------------------------------------------------------------------------
log "Installing Python ${PYTHON_VERSION}..."

PYTHON_BIN="python${PYTHON_VERSION}"

install_python_from_deadsnakes() {
  # deadsnakes PPA is Ubuntu-only
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -qq
  apt-get install -y \
    "python${PYTHON_VERSION}" \
    "python${PYTHON_VERSION}-dev" \
    "python${PYTHON_VERSION}-venv"
  # distutils was removed from Python 3.12+; install separately if available
  apt-get install -y "python${PYTHON_VERSION}-distutils" 2>/dev/null || true
}

install_python_native() {
  # Install the exact package names that exist on this distro
  apt-get install -y "python${PYTHON_VERSION}" || err "python${PYTHON_VERSION} not found in apt repos for ${OS_ID} ${OS_VERSION}. Try --python-version 3.11"
  apt-get install -y "python${PYTHON_VERSION}-dev" || err "python${PYTHON_VERSION}-dev not found. Try --python-version 3.11"
  # venv: try the versioned package first, fall back to python3-venv
  apt-get install -y "python${PYTHON_VERSION}-venv" 2>/dev/null \
    || apt-get install -y python3-venv \
    || warn "python venv not installed — virtual environments may not work"
}

if command -v "$PYTHON_BIN" &>/dev/null; then
  log "Python ${PYTHON_VERSION} already available: $($PYTHON_BIN --version)"
else
  case "$OS_ID" in
    ubuntu)
      case "$OS_VERSION" in
        22.04|24.04)
          # Python 3.11 is native on 22.04+; 3.10 is also native on 22.04
          if apt-cache show "python${PYTHON_VERSION}-dev" &>/dev/null 2>&1; then
            install_python_native
          else
            log "python${PYTHON_VERSION}-dev not in native repos — using deadsnakes PPA..."
            install_python_from_deadsnakes
          fi
          ;;
        20.04)
          # Ubuntu 20.04 only has Python 3.8 natively; use deadsnakes for 3.11
          log "Ubuntu 20.04: using deadsnakes PPA for Python ${PYTHON_VERSION}..."
          install_python_from_deadsnakes
          ;;
        *)
          warn "Unknown Ubuntu version ${OS_VERSION}. Attempting native install..."
          install_python_native
          ;;
      esac
      ;;
    debian)
      case "$OS_VERSION" in
        12)
          # Debian 12 (Bookworm): python3.11 and python3.12 are native.
          # python3.10-dev and python3.10-venv do NOT exist in Bookworm repos.
          if [ "$PYTHON_VERSION" = "3.10" ]; then
            warn "Python 3.10 is not available in Debian 12 (Bookworm) apt repos."
            warn "Switching to Python 3.11 (fully compatible with stable-audio-tools and PyTorch 2.x)."
            PYTHON_VERSION="3.11"
            PYTHON_BIN="python3.11"
          fi
          install_python_native
          ;;
        11)
          # Debian 11 (Bullseye): python3.9 native, python3.11 available
          if apt-cache show "python${PYTHON_VERSION}-dev" &>/dev/null 2>&1; then
            install_python_native
          else
            warn "python${PYTHON_VERSION} not in Bullseye repos. Switching to python3.11..."
            PYTHON_VERSION="3.11"
            PYTHON_BIN="python3.11"
            # python3.11 is in Bullseye backports
            echo "deb http://deb.debian.org/debian bullseye-backports main" \
              | tee /etc/apt/sources.list.d/bullseye-backports.list
            apt-get update -qq
            apt-get install -y -t bullseye-backports python3.11 python3.11-dev python3.11-venv
          fi
          ;;
        *)
          warn "Unknown Debian version ${OS_VERSION}. Attempting native install..."
          install_python_native
          ;;
      esac
      ;;
    *)
      warn "Unknown OS ${OS_ID}. Attempting native install..."
      install_python_native
      ;;
  esac
fi

# ---------------------------------------------------------------------------
# 7. pip and virtualenv
# ---------------------------------------------------------------------------
log "Upgrading pip for Python ${PYTHON_VERSION}..."
"$PYTHON_BIN" -m ensurepip --upgrade 2>/dev/null || true
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 8. CUDA drivers (skip if using DLVM image or --skip-cuda flag)
# ---------------------------------------------------------------------------
if [ "$SKIP_CUDA" = "false" ]; then
  if ! command -v nvidia-smi &>/dev/null; then
    log "Installing CUDA 12.3 drivers..."
    CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos"
    if   [ "$OS_ID" = "ubuntu" ] && [ "$OS_VERSION" = "22.04" ]; then
      CUDA_REPO="${CUDA_KEYRING_URL}/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    elif [ "$OS_ID" = "ubuntu" ] && [ "$OS_VERSION" = "20.04" ]; then
      CUDA_REPO="${CUDA_KEYRING_URL}/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb"
    elif [ "$OS_ID" = "debian" ] && [ "$OS_VERSION" = "11" ]; then
      CUDA_REPO="${CUDA_KEYRING_URL}/debian11/x86_64/cuda-keyring_1.1-1_all.deb"
    elif [ "$OS_ID" = "debian" ] && [ "$OS_VERSION" = "12" ]; then
      CUDA_REPO="${CUDA_KEYRING_URL}/debian12/x86_64/cuda-keyring_1.1-1_all.deb"
    else
      warn "Unknown OS ${OS_ID} ${OS_VERSION}. Skipping CUDA driver install."
      CUDA_REPO=""
    fi

    if [ -n "${CUDA_REPO:-}" ]; then
      wget -q "$CUDA_REPO" -O /tmp/cuda-keyring.deb
      dpkg -i /tmp/cuda-keyring.deb
      apt-get update -qq
      apt-get install -y cuda-drivers
      log "CUDA drivers installed. A REBOOT IS REQUIRED before nvidia-smi will work."
      log "After reboot, re-run: bash scripts/setup_python_env.sh --step <name>"
    fi
  else
    log "CUDA drivers already present: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  fi
else
  log "Skipping CUDA driver install (--skip-cuda flag set)."
fi

# ---------------------------------------------------------------------------
# 9. git-lfs init
# ---------------------------------------------------------------------------
log "Initialising git-lfs..."
git lfs install --system 2>/dev/null || git lfs install

# ---------------------------------------------------------------------------
# 10. Create install directory and hand ownership to the invoking user
# ---------------------------------------------------------------------------
# bootstrap_vm.sh must run as root (sudo) so that apt-get and CUDA driver
# installation work.  However, ALL subsequent scripts (setup_python_env.sh,
# smoke_test_gcs.py, train.py) must run as the *login user* — not root.
#
# If we leave /opt/dnl owned by root, a non-root pip install into the venv
# will fail with OSError: [Errno 13] Permission denied.  'sudo pip' does not
# help because GCE VMs use a restricted secure_path that does not include
# the venv's bin/ directory, so sudo cannot find 'pip'.
#
# Solution: after creating the directory, chown it to $SUDO_USER (the user
# who invoked sudo).  If the script was run directly as root (no SUDO_USER),
# we skip the chown so root-only environments still work.
mkdir -p "$INSTALL_DIR"
if [ -n "${SUDO_USER:-}" ]; then
  chown -R "${SUDO_USER}:${SUDO_USER}" "$INSTALL_DIR"
  log "Created install directory: $INSTALL_DIR (owned by ${SUDO_USER})"
else
  log "Created install directory: $INSTALL_DIR (owned by root — running as root directly)"
  warn "If you plan to run setup_python_env.sh as a non-root user, run:"
  warn "  sudo chown -R \$USER:\$USER $INSTALL_DIR"
fi

# ---------------------------------------------------------------------------
# 11. Verify
# ---------------------------------------------------------------------------
log "=== Bootstrap complete. Verification ==="
echo "OS:       $OS_ID $OS_VERSION"
echo "git:      $(git --version)"
echo "python:   $($PYTHON_BIN --version)"
echo "pip:      $($PYTHON_BIN -m pip --version)"
echo "gcloud:   $(gcloud version --format='value(Google Cloud SDK)' 2>/dev/null || echo 'not found')"
echo "gsutil:   $(gsutil version 2>/dev/null | head -1 || echo 'not found')"
echo "gcsfuse:  $(gcsfuse --version 2>&1 | head -1 || echo 'not found')"
echo "ffmpeg:   $(ffmpeg -version 2>&1 | head -1 || echo 'not found')"
echo "sox:      $(sox --version 2>&1 || echo 'not found')"
echo "nvidia-smi: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not available (reboot may be needed)')"
echo ""
log "Next step: bash scripts/setup_python_env.sh --step <smoke|augment|encode|train|eval>"
