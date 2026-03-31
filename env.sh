#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${TRTLLM_LATENCY_REPO_DIR:-$(pwd -P)}"
VENV_PY="${REPO_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
  echo ".venv is missing under ${REPO_DIR}. cd into the repo root and run ./setup_uv_env.sh first." >&2
  exit 1
fi

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
export LD_LIBRARY_PATH="${UV_PY_LIB}:${REPO_DIR}/.venv/lib:${LD_LIBRARY_PATH:-}"
export TLLM_WORKER_USE_SINGLE_PROCESS=1

mkdir -p "${HF_HOME}" "${MODEL_CACHE_ROOT}" "${ARTIFACT_ROOT}" "${UV_CACHE_DIR}"
source "${REPO_DIR}/.venv/bin/activate"
