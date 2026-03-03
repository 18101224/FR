#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/data}"
DOWNLOAD_ROOT="${DATA_ROOT}/downloads"

CASIA_DIR="${DATA_ROOT}/casia-webface"
CASIA_HF_DIR="${DATA_ROOT}/casia-webface-hf"
VGG2_DIR="${DATA_ROOT}/vgg2"

CASIA_KAGGLE_SLUG="congtri1305/casia-webface"
CASIA_HF_REPO="randomguyfromnepal/casia_web_face"
VGG2_HF_PRIMARY_REPO="ProgramComputer/VGGFace2"
VGG2_HF_FALLBACK_REPO="RichardErkhov/VGGFace2"
VGG2_HF_FILE="data/vggface2_train.tar.gz"

CASIA_ARCHIVE="${DOWNLOAD_ROOT}/casia-webface-kaggle.zip"
VGG2_ARCHIVE_DIR="${DOWNLOAD_ROOT}/vggface2-hf"
VGG2_ARCHIVE_PATH="${VGG2_ARCHIVE_DIR}/${VGG2_HF_FILE}"

mkdir -p "${DOWNLOAD_ROOT}" "${CASIA_DIR}" "${CASIA_HF_DIR}" "${VGG2_DIR}"

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

dir_has_files() {
  local dir="$1"
  [[ -n "$(find "${dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]
}

download_casia() {
  if dir_has_files "${CASIA_DIR}"; then
    echo "CASIA-WebFace already extracted: ${CASIA_DIR}"
    return
  fi

  if command -v kaggle >/dev/null 2>&1 && [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
    echo "Downloading CASIA-WebFace from Kaggle (${CASIA_KAGGLE_SLUG})"
    kaggle datasets download -d "${CASIA_KAGGLE_SLUG}" -p "${DOWNLOAD_ROOT}"
    local kaggle_zip
    kaggle_zip="$(find "${DOWNLOAD_ROOT}" -maxdepth 1 -type f -name '*.zip' -print | sort | tail -n 1)"
    if [[ -z "${kaggle_zip}" ]]; then
      echo "Kaggle download finished but no zip archive was found in ${DOWNLOAD_ROOT}" >&2
      exit 1
    fi
    cp -f "${kaggle_zip}" "${CASIA_ARCHIVE}"
    unzip -oq "${CASIA_ARCHIVE}" -d "${CASIA_DIR}"
    echo "CASIA-WebFace extracted to: ${CASIA_DIR}"
    return
  fi

  echo "Kaggle CLI credentials not found. Falling back to Hugging Face parquet mirror."
  echo "Note: HF fallback downloads parquet shards, not an extracted image-folder tree."
  if dir_has_files "${CASIA_HF_DIR}"; then
    echo "CASIA-WebFace HF mirror already downloaded: ${CASIA_HF_DIR}"
    return
  fi
  require_cmd hf
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}" \
    hf download "${CASIA_HF_REPO}" --repo-type dataset --local-dir "${CASIA_HF_DIR}"
  echo "CASIA-WebFace HF mirror downloaded to: ${CASIA_HF_DIR}"
}

download_vgg2() {
  if dir_has_files "${VGG2_DIR}"; then
    echo "VGGFace2 already extracted: ${VGG2_DIR}"
    return
  fi

  require_cmd hf
  require_cmd tar

  mkdir -p "${VGG2_ARCHIVE_DIR}"

  if [[ ! -f "${VGG2_ARCHIVE_PATH}" ]]; then
    echo "Downloading VGGFace2 from Hugging Face (${VGG2_HF_PRIMARY_REPO})"
    if ! HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}" \
      hf download "${VGG2_HF_PRIMARY_REPO}" "${VGG2_HF_FILE}" --repo-type dataset --local-dir "${VGG2_ARCHIVE_DIR}"; then
      echo "Primary VGGFace2 mirror failed. Trying fallback mirror (${VGG2_HF_FALLBACK_REPO})"
      HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}" \
        hf download "${VGG2_HF_FALLBACK_REPO}" "${VGG2_HF_FILE}" --repo-type dataset --local-dir "${VGG2_ARCHIVE_DIR}"
    fi
  else
    echo "Using existing VGGFace2 archive: ${VGG2_ARCHIVE_PATH}"
  fi

  echo "Extracting VGGFace2 archive"
  tar -xzf "${VGG2_ARCHIVE_PATH}" -C "${VGG2_DIR}"
  echo "VGGFace2 extracted to: ${VGG2_DIR}"
}

require_cmd unzip

download_casia
download_vgg2

echo "Done."
echo "CASIA-WebFace (Kaggle extract): ${CASIA_DIR}"
echo "CASIA-WebFace (HF parquet fallback): ${CASIA_HF_DIR}"
echo "VGGFace2: ${VGG2_DIR}"
