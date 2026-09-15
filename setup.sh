#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12."'
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
if [[ -x "${VENV_DIR}/bin/python" ]]; then
    VENV_PYTHON="${VENV_DIR}/bin/python"
    ACTIVATE_PATH="${VENV_DIR}/bin/activate"
else
    VENV_PYTHON="${VENV_DIR}/Scripts/python.exe"
    ACTIVATE_PATH="${VENV_DIR}/Scripts/activate"
fi
"${VENV_PYTHON}" -m pip install --upgrade pip
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    "${VENV_PYTHON}" -m pip install 'torch==2.10.0' --index-url "${TORCH_INDEX_URL}"
fi
"${VENV_PYTHON}" -m pip install -r requirements.txt
"${VENV_PYTHON}" -c 'import torch; from model import SomnoOrbit; print("SomnoOrbit is ready. CUDA available:", torch.cuda.is_available())'
printf '\nActivate the environment: source %s\n' "${ACTIVATE_PATH}"
