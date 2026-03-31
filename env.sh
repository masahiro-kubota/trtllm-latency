#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${REPO_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
  echo ".venv is missing. Run ./setup_uv_env.sh first." >&2
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
export LD_LIBRARY_PATH="${UV_PY_LIB}:${REPO_DIR}/.venv/lib:${LD_LIBRARY_PATH:-}"
export TLLM_WORKER_USE_SINGLE_PROCESS=1

mkdir -p "${HF_HOME}" "${MODEL_CACHE_ROOT}" "${ARTIFACT_ROOT}"
source "${REPO_DIR}/.venv/bin/activate"
