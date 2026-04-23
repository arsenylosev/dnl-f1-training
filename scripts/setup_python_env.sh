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

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
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
mkdir -p "$(dirname "$VENV_DIR")"
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

log "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 2. Install PyTorch (CUDA 12.1 build — compatible with CUDA 12.3 drivers)
# ---------------------------------------------------------------------------
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
