#!/usr/bin/env bash
# =============================================================================
# Foundation-1 Fine-Tuning Launch Script — Google Cloud Platform
# =============================================================================
# Target:  Vertex AI custom training job or GCE VM with 2× NVIDIA A100/H100
# Data:    GCS bucket with WAV+JSON pairs in Foundation-1 format
# Model:   Fine-tuned from Foundation-1 (stabilityai/stable-audio-open-1.0 fork)
#
# Usage:
#   1. Set the variables in the CONFIGURATION section below.
#   2. Run: bash scripts/gcp_train_f1_3s.sh
#
# Prerequisites (already handled if using the Vertex AI Docker image):
#   - Python 3.10+, CUDA 12.1+, PyTorch 2.1+
#   - google-cloud-storage: pip install google-cloud-storage
#   - gcsfuse (optional, for pre-encoding step): https://cloud.google.com/storage/docs/gcsfuse-install
#   - Application Default Credentials active: gcloud auth application-default login
#
# Source: https://cloud.google.com/vertex-ai/docs/training/overview
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIGURATION — edit these before running
# ---------------------------------------------------------------------------

GCS_BUCKET="your-gcs-bucket-name"          # GCS bucket (no gs:// prefix)
GCS_CHECKPOINT_PREFIX="checkpoints/f1_3s"  # Where to save checkpoints in GCS
LOCAL_CHECKPOINT_DIR="/tmp/f1_checkpoints"  # Local checkpoint staging dir

FOUNDATION1_CKPT="models/foundation1_3s/Foundation_1.safetensors"  # Local path to downloaded F1 checkpoint
MODEL_CONFIG="models/foundation1_3s/model_config_3s.json"

# Training hyperparameters
BATCH_SIZE=16         # Per-GPU batch size. 16 fits on A100 80GB with pre-encoded latents.
NUM_GPUS=2            # Number of GPUs. Adjust to match your instance (a3-highgpu-2g = 2× H100).
PRECISION="16-mixed"  # bf16-mixed is also valid on A100/H100.
CHECKPOINT_EVERY=5000 # Save a checkpoint every N steps.
WANDB_PROJECT="foundation1-3s-finetune"

# ---------------------------------------------------------------------------
# ClearML experiment tracking (optional)
# ---------------------------------------------------------------------------
# Leave CLEARML_API_ACCESS_KEY and CLEARML_API_SECRET_KEY empty to disable
# ClearML tracking entirely. When set, every training run and pre-encoding
# run will be logged to your ClearML server automatically.
#
# Best practice: pass these as Vertex AI environment variables or store in
# GCP Secret Manager instead of hard-coding here.
#
# Get credentials from: https://app.clear.ml/settings/workspace-configuration
export CLEARML_API_HOST="${CLEARML_API_HOST:-https://api.clear.ml}"
export CLEARML_API_ACCESS_KEY="${CLEARML_API_ACCESS_KEY:-}"     # Set this
export CLEARML_API_SECRET_KEY="${CLEARML_API_SECRET_KEY:-}"     # Set this
export CLEARML_PROJECT="${CLEARML_PROJECT:-Foundation-1 / DNL}"
export CLEARML_TASK_NAME="${CLEARML_TASK_NAME:-f1-3s-finetune-$(date +%Y%m%d-%H%M)}"

# Install ClearML if not already present
pip install --quiet clearml 2>/dev/null || true

# ---------------------------------------------------------------------------
# STEP 0 — Verify environment
# ---------------------------------------------------------------------------

echo "=== Foundation-1 3s Fine-Tuning ==="
echo "GCS bucket : gs://${GCS_BUCKET}"
echo "GPUs       : ${NUM_GPUS}"
echo "Batch size : ${BATCH_SIZE}"
echo ""

python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# ---------------------------------------------------------------------------
# STEP 1 — Download Foundation-1 checkpoint from HuggingFace (if not present)
# ---------------------------------------------------------------------------

if [ ! -f "${FOUNDATION1_CKPT}" ]; then
    echo ""
    echo "=== Downloading Foundation-1 checkpoint ==="
    mkdir -p "$(dirname "${FOUNDATION1_CKPT}")"
    python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os

# Download the unwrapped .safetensors checkpoint
path = hf_hub_download(
    repo_id="RoyalCities/Foundation-1",
    filename="Foundation_1.safetensors",
    local_dir="models/foundation1_3s/",
)
print(f"Downloaded to: {path}")
PYEOF
fi

# ---------------------------------------------------------------------------
# STEP 2 — Pre-encode latents (skip if already done)
# ---------------------------------------------------------------------------
# Pre-encoding runs the frozen OOBLECK VAE once over the entire dataset and
# saves .npy latent tensors. This removes the VAE from the training loop,
# saving ~20 GB VRAM and making each training step ~4× faster.
#
# Output: /tmp/pre_encoded_3s/  (structured as rank_id/sample_id.npy + .json)
# Then upload to GCS: gsutil -m rsync -r /tmp/pre_encoded_3s gs://${GCS_BUCKET}/pre_encoded_3s/
#
# NOTE: Run this step ONCE on a CPU-heavy machine or a single-GPU VM.
#       It does NOT need to run on the same machine as training.

PRE_ENCODED_LOCAL="/tmp/pre_encoded_3s"
PRE_ENCODED_GCS="gs://${GCS_BUCKET}/pre_encoded_3s"

if [ ! -d "${PRE_ENCODED_LOCAL}" ] || [ -z "$(ls -A ${PRE_ENCODED_LOCAL} 2>/dev/null)" ]; then
    echo ""
    echo "=== Step 2: Pre-encoding latents ==="
    echo "This will take ~30–60 minutes for 250k samples on 1× A100."
    echo "Dataset config: stable_audio_tools/configs/dataset_configs/gcs_f1_3s_pre_encode.json"
    echo ""

    # Replace YOUR_GCS_BUCKET_NAME in the dataset config
    DATASET_CONFIG_TMP="/tmp/gcs_f1_3s_pre_encode.json"
    sed "s/YOUR_GCS_BUCKET_NAME/${GCS_BUCKET}/g" \
        stable_audio_tools/configs/dataset_configs/gcs_f1_3s_pre_encode.json \
        > "${DATASET_CONFIG_TMP}"

    python3 pre_encode.py \
        --model-config "${MODEL_CONFIG}" \
        --ckpt-path "${FOUNDATION1_CKPT}" \
        --model-half \
        --dataset-config "${DATASET_CONFIG_TMP}" \
        --output-path "${PRE_ENCODED_LOCAL}" \
        --sample-size 132300 \
        --batch-size 32 \
        --num-workers 8

    echo ""
    echo "=== Uploading pre-encoded latents to GCS ==="
    gsutil -m rsync -r "${PRE_ENCODED_LOCAL}" "${PRE_ENCODED_GCS}"
    echo "Pre-encoded latents uploaded to ${PRE_ENCODED_GCS}"
else
    echo "Pre-encoded latents already present at ${PRE_ENCODED_LOCAL}, skipping Step 2."
fi

# ---------------------------------------------------------------------------
# STEP 3 — Mount pre-encoded latents from GCS (gcsfuse)
# ---------------------------------------------------------------------------
# The pre_encoded dataset type expects a local path. We mount the GCS directory
# containing the .npy + .json files using gcsfuse.
#
# Alternative: gsutil -m rsync -r ${PRE_ENCODED_GCS} ${PRE_ENCODED_LOCAL}
# Use rsync if gcsfuse is not available or if you prefer a local copy.

MOUNT_POINT="/mnt/gcs/pre_encoded_3s"
mkdir -p "${MOUNT_POINT}"

if ! mountpoint -q "${MOUNT_POINT}"; then
    echo ""
    echo "=== Step 3: Mounting GCS pre-encoded latents via gcsfuse ==="
    # gcsfuse 2.0+ uses a config file for cache settings.
    # --stat-cache-ttl and --type-cache-ttl are deprecated; use metadata-cache.ttl-secs.
    mkdir -p /tmp/gcsfuse-cache
    cat > /tmp/gcsfuse-train-config.yaml << 'GCSFUSE_CFG'
metadata-cache:
  ttl-secs: 3600
  stat-cache-max-size-mb: 32
GCSFUSE_CFG
    gcsfuse \
        --only-dir "pre_encoded_3s" \
        --implicit-dirs \
        --config-file /tmp/gcsfuse-train-config.yaml \
        "${GCS_BUCKET}" "${MOUNT_POINT}"
    echo "Mounted gs://${GCS_BUCKET}/pre_encoded_3s at ${MOUNT_POINT}"
else
    echo "GCS already mounted at ${MOUNT_POINT}"
fi

# ---------------------------------------------------------------------------
# STEP 4 — Fine-tune the DiT
# ---------------------------------------------------------------------------

echo ""
echo "=== Step 4: Fine-tuning Foundation-1 DiT ==="
echo "Model config : ${MODEL_CONFIG}"
echo "Dataset      : pre_encoded_f1_3s (latent_crop_length=64)"
echo "Checkpoint   : ${FOUNDATION1_CKPT}"
echo "Save dir     : ${LOCAL_CHECKPOINT_DIR}"
echo ""

mkdir -p "${LOCAL_CHECKPOINT_DIR}"

python3 train.py \
    --model-config "${MODEL_CONFIG}" \
    --dataset-config stable_audio_tools/configs/dataset_configs/pre_encoded_f1_3s.json \
    --pretrained-ckpt-path "${FOUNDATION1_CKPT}" \
    --name "${WANDB_PROJECT}" \
    --batch-size "${BATCH_SIZE}" \
    --num-gpus "${NUM_GPUS}" \
    --precision "${PRECISION}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --save-dir "${LOCAL_CHECKPOINT_DIR}" \
    --strategy ddp \
    --num-workers 8 \
    --accum-batches 1

# ---------------------------------------------------------------------------
# STEP 5 — Upload checkpoints to GCS
# ---------------------------------------------------------------------------

echo ""
echo "=== Step 5: Uploading checkpoints to GCS ==="
gsutil -m rsync -r "${LOCAL_CHECKPOINT_DIR}" "gs://${GCS_BUCKET}/${GCS_CHECKPOINT_PREFIX}/"
echo "Checkpoints uploaded to gs://${GCS_BUCKET}/${GCS_CHECKPOINT_PREFIX}/"

echo ""
echo "=== Training complete ==="
echo "Next step: run scripts/unwrap_checkpoint.sh to produce a .safetensors file."
