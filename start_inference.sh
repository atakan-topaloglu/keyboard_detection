#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT_DIR}/.env.inference"
VENV_DIR="${ROOT_DIR}/.venv312"

link_tmp_path() {
  local link_path="$1"
  local target_path="$2"

  mkdir -p "${target_path}"

  if [[ -L "${link_path}" ]]; then
    if [[ "$(readlink -f "${link_path}")" == "${target_path}" ]]; then
      return
    fi
    rm -f "${link_path}"
  elif [[ -e "${link_path}" ]]; then
    rmdir "${link_path}" 2>/dev/null || {
      echo "Existing path blocks cache redirection: ${link_path}" >&2
      exit 1
    }
  fi

  ln -s "${target_path}" "${link_path}"
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

# The CLI binds host /tmp into the container, so redirect those /tmp paths
# into persistent workspace directories before starting the server.
link_tmp_path "/tmp/model-cache" "${ROOT_DIR}/.roboflow-cache"
link_tmp_path "/tmp/huggingface" "${ROOT_DIR}/.huggingface-cache"
link_tmp_path "/tmp/yolo" "${ROOT_DIR}/.yolo-cache"
link_tmp_path "/tmp/matplotlib" "${ROOT_DIR}/.mplconfig"
link_tmp_path "/tmp/home" "${ROOT_DIR}/.rf-home"

exec "${VENV_DIR}/bin/inference" server start --port "${INFERENCE_PORT:-9001}" --use-local-images
