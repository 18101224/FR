#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

ROOT="${1:-/data/mj}"
DEVICE="${2:-cuda:0}"
BATCH_SIZE="${3:-256}"
NUM_WORKERS="${4:-4}"

FACEREC_VAL="${ROOT}/facerec_val"
EVAL_BINS="${ROOT}/eval_bins"
TINYFACE_ROOT="${ROOT}/TinyFace"
IJB_ROOT="${ROOT}/IJB_release"
IJBS_ROOT_DEFAULT="${IJB_ROOT}/IJBS"

mkdir -p "${FACEREC_VAL}" "${EVAL_BINS}"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 1; }; }
need_cmd python

echo "[1/7] Install minimal deps for eval prep"
python -m pip install -U datasets pyarrow >/dev/null

echo "[2/7] Ensure CFP-FP bin (and keep existing)"
if [ ! -f "${EVAL_BINS}/cfp_fp.bin" ]; then
  if command -v hf >/dev/null 2>&1; then
    hf download namkuner/namkuner_face_dataset \
      --repo-type dataset \
      cfp_fp.bin \
      --local-dir "${EVAL_BINS}" || true
  fi
fi

echo "[3/7] Convert verification bins -> run_v1 eval format"
python "${PROJECT_ROOT}/tools/prepare_verification_eval.py" \
  --bin_root "${EVAL_BINS}" \
  --out_root "${FACEREC_VAL}" \
  --names lfw agedb_30 cfp_fp cplfw calfw || true

echo "[4/7] TinyFace download (best-effort) + preprocess"
if [ ! -d "${TINYFACE_ROOT}" ]; then
  if command -v gdown >/dev/null 2>&1; then
    mkdir -p "${ROOT}/downloads"
    TINY_ZIP="${ROOT}/downloads/tinyface.zip"
    gdown 1xTZc7lNmWN33ECO2AKH6FycGdiqIK7W0 -O "${TINY_ZIP}" || true
    if [ -f "${TINY_ZIP}" ]; then
      mkdir -p "${TINYFACE_ROOT}"
      unzip -o "${TINY_ZIP}" -d "${TINYFACE_ROOT}" || true
    fi
  fi
fi
if [ -d "${TINYFACE_ROOT}" ]; then
  python "${PROJECT_ROOT}/tools/prepare_tinyface_eval.py" \
    --tinyface_root "${TINYFACE_ROOT}" \
    --out_path "${FACEREC_VAL}/tinyface_aligned_pad_0.1" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --align \
    --overwrite || true
else
  echo "[WARN] TinyFace root not found: ${TINYFACE_ROOT}"
fi

echo "[5/7] IJB-C download (best-effort) + preprocess"
if [ ! -d "${IJB_ROOT}" ]; then
  if command -v gdown >/dev/null 2>&1; then
    mkdir -p "${ROOT}/downloads"
    IJB_TAR="${ROOT}/downloads/ijb-testsuite.tar"
    gdown 1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o -O "${IJB_TAR}" || true
    if [ -f "${IJB_TAR}" ]; then
      mkdir -p "${IJB_ROOT}"
      tar -xf "${IJB_TAR}" -C "${IJB_ROOT}" || true
    fi
  fi
fi
if [ -d "${IJB_ROOT}" ]; then
  python "${PROJECT_ROOT}/tools/prepare_ijbc_eval.py" \
    --ijb_root "${IJB_ROOT}" \
    --subset ijbc \
    --out_root "${FACEREC_VAL}" \
    --device "${DEVICE}" \
    --align \
    --overwrite || true
else
  echo "[WARN] IJB root not found: ${IJB_ROOT}"
fi

echo "[6/7] IJB-S aligned data prep (no evaluator yet in run_v1)"
if [ -d "${IJBS_ROOT_DEFAULT}" ]; then
  python "${PROJECT_ROOT}/tools/prepare_ijbs_aligned.py" \
    --ijbs_root "${IJBS_ROOT_DEFAULT}" \
    --out_root "${ROOT}/ijbs_aligned_112" \
    --device "${DEVICE}" \
    --overwrite || true
else
  echo "[WARN] IJB-S root not found (${IJBS_ROOT_DEFAULT}). Skip IJB-S preprocessing."
fi

echo "[7/7] Final readiness check"
python "${PROJECT_ROOT}/tools/check_eval_ready.py" --root "${ROOT}"

echo "Done."
