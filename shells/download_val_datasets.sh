#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-val_datasets}"
mkdir -p "$ROOT"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 1; }; }
need_cmd tar
need_cmd unzip || true  # xqlfw는 zip이라 unzip 필요
need_cmd curl || need_cmd wget

dl() {
  # dl URL OUT
  local url="$1"
  local out="$2"
  echo "[DL] $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 2 -o "$out" "$url"
  else
    wget -O "$out" "$url"
  fi
}

# -------------------------
# 1) LFW (Figshare mirrors used by scikit-learn)
#    - raw:      5976018  (lfw.tgz)
#    - funneled: 5976015  (lfw-funneled.tgz)
#    - pairs:    5976006  (pairs.txt)
# -------------------------
LFW_DIR="$ROOT/lfw"
mkdir -p "$LFW_DIR"
cd "$LFW_DIR"

# 기본은 funneled (같은 LFW지만 더 정렬된 버전)
LFW_ARCHIVE_URL="https://ndownloader.figshare.com/files/5976015"
LFW_ARCHIVE_NAME="lfw-funneled.tgz"

# raw로 받고 싶으면 주석 해제:
# LFW_ARCHIVE_URL="https://ndownloader.figshare.com/files/5976018"
# LFW_ARCHIVE_NAME="lfw.tgz"

dl "$LFW_ARCHIVE_URL" "$LFW_ARCHIVE_NAME"
dl "https://ndownloader.figshare.com/files/5976006" "pairs.txt"

echo "[EXTRACT] $LFW_ARCHIVE_NAME"
tar -xzf "$LFW_ARCHIVE_NAME"
rm -f "$LFW_ARCHIVE_NAME"

cd - >/dev/null

# -------------------------
# 2) XQLFW (대체 verification benchmark)
#    - aligned 112x112 zip + pairs protocol
# -------------------------
XQLFW_DIR="$ROOT/xqlfw"
mkdir -p "$XQLFW_DIR"
cd "$XQLFW_DIR"

# 공식 다운로드 페이지에 있는 GitHub release 링크를 그대로 사용
# (curl -L로 redirect 따라가면 됨)
dl "https://github.com/Martlgap/xqlfw/releases/download/1.0/xqlfw_aligned_112.zip" "xqlfw_aligned_112.zip"
dl "https://github.com/Martlgap/xqlfw/releases/download/1.0/xqlfw_pairs.txt" "xqlfw_pairs.txt"

echo "[EXTRACT] xqlfw_aligned_112.zip"
unzip -o "xqlfw_aligned_112.zip"
rm -f "xqlfw_aligned_112.zip"

cd - >/dev/null

echo "DONE. Root: $ROOT"
echo " - LFW:   $ROOT/lfw (pairs.txt 포함, lfw_funneled/ 또는 lfw/ 폴더 생성)"
echo " - XQLFW: $ROOT/xqlfw (aligned 112 + pairs)"