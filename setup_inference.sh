#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${ROOT_DIR}/.venv312"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Missing ${PYTHON_BIN}. Install Python 3.12 first." >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${ROOT_DIR}/requirements.txt"

mkdir -p \
  "${ROOT_DIR}/.roboflow-cache" \
  "${ROOT_DIR}/.huggingface-cache" \
  "${ROOT_DIR}/.yolo-cache" \
  "${ROOT_DIR}/.rf-home" \
  "${ROOT_DIR}/.mplconfig" \
  "${ROOT_DIR}/outputs"

if [[ ! -f "${ROOT_DIR}/.env.inference" ]]; then
  cp "${ROOT_DIR}/.env.inference.example" "${ROOT_DIR}/.env.inference"
  echo "Created ${ROOT_DIR}/.env.inference. Add your Roboflow API key before starting the server."
fi

echo "Inference CLI is ready in ${VENV_DIR}."
