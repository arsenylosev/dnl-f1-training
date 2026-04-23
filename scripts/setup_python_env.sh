#!/usr/bin/env bash
# =============================================================================
# setup_python_env.sh — Python virtual environment setup
# =============================================================================
# Run AFTER bootstrap_vm.sh. Creates a virtualenv at /opt/dnl/venv and
# installs all Python dependencies in the correct order.
#
# Usage:
#   bash scripts/setup_python_env.sh [--step smoke|augment|encode|train|eval|all]
#
# Steps:
#   smoke    — minimal deps for smoke_test_gcs.py (fast, ~2 min)
#   augment  — deps for data augmentation pipeline (no GPU needed)
#   encode   — deps for pre_encode.py (GPU, VAE encoding)
#   train    — deps for train.py (GPU, full training)
#   eval     — deps for evaluation / Gradio demo
#   all      — install everything (default, ~10 min on fast connection)
# =============================================================================
set -euo pipefail

# Python 3.11 is the default: it is native on Debian 12 (Bookworm) and Ubuntu 22.04.
# scipy==1.8.1 (the original pin) has no pre-built wheel for Python 3.11.
# setup.py has been patched to scipy>=1.8.1,<1.14 which resolves this.
# If you need Python 3.10 for any reason, set: PYTHON_VERSION=3.10 before running.
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTHON_BIN="python${PYTHON_VERSION}"
VENV_DIR="${VENV_DIR:-/opt/dnl/venv}"
STEP="${1:-all}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { echo -e "\n\033[1;32m[setup_env] $*\033[0m"; }

# Parse --step flag
for i in "$@"; do
  case $i in
    --step=*) STEP="${i#*=}" ;;
    --step)   STEP="$2"; shift ;;
  esac
done

# ---------------------------------------------------------------------------
# 1. Create virtualenv
# ---------------------------------------------------------------------------
log "Creating virtualenv at ${VENV_DIR}..."

# Ensure the parent directory exists and is writable by the current user.
# bootstrap_vm.sh (run as sudo) creates /opt/dnl and chowns it to $SUDO_USER.
# If that chown did not happen (e.g. bootstrap was run directly as root),
# the directory will be owned by root and pip installs will fail with
# OSError: [Errno 13] Permission denied.  Detect this early and abort with
# a clear message instead of a cryptic pip error deep in the install.
VENV_PARENT="$(dirname "$VENV_DIR")"
mkdir -p "$VENV_PARENT"
if [ ! -w "$VENV_PARENT" ]; then
  echo "ERROR: $VENV_PARENT is not writable by $(whoami)." >&2
  echo "       Fix with: sudo chown -R \$USER:\$USER $VENV_PARENT" >&2
  echo "       Then re-run this script." >&2
  exit 1
fi

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

log "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 2. Install PyTorch
# ---------------------------------------------------------------------------
# smoke step: CPU-only torch + torchaudio.  The smoke test only decodes WAV
# files and reads GCS objects — it does not need CUDA at all.  Installing
# the CPU wheel avoids the need for CUDA drivers on the smoke VM and removes
# the NVIDIA package OSError that occurs when pip tries to install CUDA libs
# without the matching kernel driver present.
if [[ "$STEP" == "smoke" ]]; then
  log "Installing PyTorch 2.2 + torchaudio (CPU-only, smoke step)..."
  pip install --upgrade \
    torch==2.2.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu
fi

# GPU steps: CUDA 12.1 build — compatible with CUDA 12.3 drivers on GCE VMs
if [[ "$STEP" == "encode" || "$STEP" == "train" || "$STEP" == "eval" || "$STEP" == "all" ]]; then
  log "Installing PyTorch 2.2 + torchaudio (CUDA 12.1)..."
  pip install --upgrade \
    torch==2.2.2+cu121 \
    torchaudio==2.2.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
fi

# ---------------------------------------------------------------------------
# 3. Step-specific requirements
# ---------------------------------------------------------------------------
install_requirements() {
  local req_file="${REPO_DIR}/requirements/${1}.txt"
  if [ -f "$req_file" ]; then
    log "Installing from ${req_file}..."
    pip install -r "$req_file"
  else
    log "WARNING: ${req_file} not found, skipping."
  fi
}

case "$STEP" in
  smoke)
    install_requirements "smoke"
    ;;
  augment)
    install_requirements "augment"
    ;;
  encode)
    install_requirements "base"
    install_requirements "encode"
    ;;
  train)
    install_requirements "base"
    install_requirements "train"
    ;;
  eval)
    install_requirements "base"
    install_requirements "eval"
    ;;
  all)
    install_requirements "base"
    install_requirements "smoke"
    install_requirements "augment"
    install_requirements "encode"
    install_requirements "train"
    install_requirements "eval"
    ;;
  *)
    echo "Unknown step: $STEP. Use: smoke|augment|encode|train|eval|all"
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# 4. Install the repo itself in editable mode
# ---------------------------------------------------------------------------
if [[ "$STEP" == "encode" || "$STEP" == "train" || "$STEP" == "eval" || "$STEP" == "all" ]]; then
  log "Installing stable-audio-tools (editable)..."
  pip install -e "${REPO_DIR}" --no-deps
fi

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------
log "=== Environment setup complete ==="
echo "Virtualenv: ${VENV_DIR}"
echo "Python:     $(python --version)"
echo "pip:        $(pip --version)"

if [[ "$STEP" == "encode" || "$STEP" == "train" || "$STEP" == "all" ]]; then
  python -c "
import torch, torchaudio
print(f'PyTorch:    {torch.__version__}')
print(f'torchaudio: {torchaudio.__version__}')
print(f'CUDA:       {torch.cuda.is_available()} (device count: {torch.cuda.device_count()})')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory // 1024**3} GB)')
"
fi

log "Activate with: source ${VENV_DIR}/bin/activate"
log "Next step (encode): python pre_encode.py --help"
log "Next step (train):  python train.py --help"
