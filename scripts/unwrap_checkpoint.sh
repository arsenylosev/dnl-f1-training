#!/usr/bin/env bash
# =============================================================================
# Unwrap a PyTorch Lightning checkpoint to a .safetensors file
# =============================================================================
# Usage:
#   bash scripts/unwrap_checkpoint.sh <path/to/checkpoint.ckpt> <output_name>
#
# Example:
#   bash scripts/unwrap_checkpoint.sh \
#     /tmp/f1_checkpoints/foundation1-3s-finetune/abc123/checkpoints/epoch=0-step=50000.ckpt \
#     foundation1_3s_step50k
#
# Output: models/foundation1_3s/<output_name>.safetensors
# =============================================================================

set -euo pipefail

CKPT_PATH="${1:?Usage: $0 <checkpoint.ckpt> <output_name>}"
OUTPUT_NAME="${2:?Usage: $0 <checkpoint.ckpt> <output_name>}"
OUTPUT_DIR="models/foundation1_3s"
OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}.safetensors"

mkdir -p "${OUTPUT_DIR}"

echo "Unwrapping: ${CKPT_PATH}"
echo "Output    : ${OUTPUT_PATH}"

python3 unwrap_model.py \
    --model-config models/foundation1_3s/model_config_3s.json \
    --ckpt-path "${CKPT_PATH}" \
    --name "${OUTPUT_NAME}" \
    --save-dir "${OUTPUT_DIR}"

echo "Done: ${OUTPUT_PATH}"
