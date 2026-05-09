#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT_DIR}/.env.inference"
VENV_DIR="${ROOT_DIR}/.venv312"
IMAGE_PATH="${1:-${ROOT_DIR}/key_detection.jpeg}"
OUTPUT_DIR="${2:-${ROOT_DIR}/outputs}"
TMP_DIR=""

cleanup() {
  if [[ -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
    rm -rf "${TMP_DIR}"
  fi
}

trap cleanup EXIT

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy .env.inference.example to .env.inference and set ROBOFLOW_API_KEY." >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/inference" ]]; then
  echo "Inference CLI is not installed yet. Run ./setup_inference.sh first." >&2
  exit 1
fi

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Image not found: ${IMAGE_PATH}" >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

: "${ROBOFLOW_API_KEY:?Set ROBOFLOW_API_KEY in .env.inference first.}"

MODEL_ID="${ROBOFLOW_MODEL_ID:-keyboard-key-recognition-kw7nc/14}"
HOST="${INFERENCE_HOST:-http://localhost:9001}"

export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${ROOT_DIR}/.roboflow-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.mplconfig}"

mkdir -p "${MODEL_CACHE_DIR}" "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

TMP_DIR="$(mktemp -d)"
NORMALIZED_IMAGE_PATH="${TMP_DIR}/$(basename "${IMAGE_PATH}")"

# Normalize EXIF orientation before inference so the saved visualisation
# matches the way the source image appears in standard image viewers.
"${VENV_DIR}/bin/python" - "${IMAGE_PATH}" "${NORMALIZED_IMAGE_PATH}" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageOps

source = Path(sys.argv[1])
target = Path(sys.argv[2])

with Image.open(source) as image:
    normalized = ImageOps.exif_transpose(image)
    if normalized.mode not in ("RGB", "L"):
        normalized = normalized.convert("RGB")
    normalized.save(target)
PY

"${VENV_DIR}/bin/inference" infer \
  --input "${NORMALIZED_IMAGE_PATH}" \
  --model_id "${MODEL_ID}" \
  --host "${HOST}" \
  --output_location "${OUTPUT_DIR}" \
  --visualise
