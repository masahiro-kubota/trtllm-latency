#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

uv venv --python 3.12 .venv
. "${REPO_DIR}/.venv/bin/activate"

uv pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.9.1 torchvision==0.24.1
uv pip install tensorrt_llm==1.2.0 huggingface_hub
