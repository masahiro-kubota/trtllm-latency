#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${TRTLLM_LATENCY_REPO_DIR:-$(pwd -P)}"
if [[ -n "${TRTLLM_PYTHON:-}" ]]; then
  VENV_PY="${TRTLLM_PYTHON}"
else
  VENV_PY="${REPO_DIR}/.venv/bin/python"
fi

VENV_BIN_DIR="$(dirname "${VENV_PY}")"

if [[ ! -x "${VENV_PY}" ]]; then
  if [[ -n "${TRTLLM_PYTHON:-}" ]]; then
    echo "TRTLLM_PYTHON points to a missing interpreter: ${TRTLLM_PYTHON}" >&2
  else
    echo ".venv is missing under ${REPO_DIR}. cd into the repo root and run ./setup_uv_env.sh first." >&2
  fi
  exit 1
fi

VENV_BIN_DIR="$(cd "${VENV_BIN_DIR}" && pwd)"
VENV_DIR="$(cd "${VENV_BIN_DIR}/.." && pwd)"

UV_PY_LIB="$("${VENV_PY}" - <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.base_prefix) / "lib")
PY
)"

export HF_HOME="${HF_HOME:-${REPO_DIR}/.hf-home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-${REPO_DIR}/models}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_DIR}/artifacts}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_DIR}/.uv-cache}"
export PATH="${VENV_BIN_DIR}:${PATH}"
TRT_ROOT="${TRT_ROOT:-/usr/local/tensorrt}"
TRT_LIB_PREFIX=""
if [[ -d "${TRT_ROOT}/lib" ]]; then
  TRT_LIB_PREFIX="${TRT_ROOT}/lib:"
fi
export LD_LIBRARY_PATH="${TRT_LIB_PREFIX}${UV_PY_LIB}:${VENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export TLLM_WORKER_USE_SINGLE_PROCESS=1

mkdir -p "${HF_HOME}" "${MODEL_CACHE_ROOT}" "${ARTIFACT_ROOT}" "${UV_CACHE_DIR}"
if [[ -f "${VENV_BIN_DIR}/activate" ]]; then
  # Keep shell PATH and prompt behavior aligned with the selected interpreter.
  source "${VENV_BIN_DIR}/activate"
fi
