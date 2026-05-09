#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT_DIR}/.env.inference"
VENV_DIR="${ROOT_DIR}/.venv312"

ensure_tmp_dir() {
  local dir_path="$1"

  if [[ -L "${dir_path}" ]]; then
    rm -f "${dir_path}"
  elif [[ -e "${dir_path}" && ! -d "${dir_path}" ]]; then
    echo "Existing path blocks cache directory: ${dir_path}" >&2
    exit 1
  fi

  mkdir -p "${dir_path}"
}

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy .env.inference.example to .env.inference and set ROBOFLOW_API_KEY." >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/inference" ]]; then
  echo "Inference CLI is not installed yet. Run ./setup_inference.sh first." >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

: "${ROBOFLOW_API_KEY:?Set ROBOFLOW_API_KEY in .env.inference first.}"

mkdir -p \
  "${ROOT_DIR}/.roboflow-cache" \
  "${ROOT_DIR}/.huggingface-cache" \
  "${ROOT_DIR}/.yolo-cache" \
  "${ROOT_DIR}/.rf-home" \
  "${ROOT_DIR}/.mplconfig" \
  "${ROOT_DIR}/outputs"

# The CLI binds host /tmp into the container. These paths must remain real
# directories inside /tmp so the container can write to them directly.
ensure_tmp_dir "/tmp/model-cache"
ensure_tmp_dir "/tmp/huggingface"
ensure_tmp_dir "/tmp/yolo"
ensure_tmp_dir "/tmp/matplotlib"
ensure_tmp_dir "/tmp/home"

exec "${VENV_DIR}/bin/inference" server start --port "${INFERENCE_PORT:-9001}" --use-local-images
