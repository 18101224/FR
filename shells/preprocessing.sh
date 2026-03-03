#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_NAME="${1:?usage: preprocessing.sh <dataset_name> <input_root> <output_root> [device] [batch_size] [num_workers]}"
INPUT_ROOT="${2:?usage: preprocessing.sh <dataset_name> <input_root> <output_root> [device] [batch_size] [num_workers]}"
OUTPUT_ROOT="${3:?usage: preprocessing.sh <dataset_name> <input_root> <output_root> [device] [batch_size] [num_workers]}"
DEVICE="${4:-cuda:0}"
BATCH_SIZE="${5:-32}"
NUM_WORKERS="${6:-4}"

python "${ROOT_DIR}/preprocessing.py" \
  --dataset_name "${DATASET_NAME}" \
  --input_root "${INPUT_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"
