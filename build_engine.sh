#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_PATH="${MODEL_PATH:-${MODEL_CACHE_ROOT}/Qwen2.5-0.5B-Instruct}"
ENGINE_TAG="${ENGINE_TAG:-$(basename "${MODEL_PATH}")_bs1_seq127_tok127}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${ARTIFACT_ROOT}/workspaces/${ENGINE_TAG}}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-127}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-127}"

rm -rf "${WORKSPACE_DIR}"
mkdir -p "${WORKSPACE_DIR}"

trtllm-bench -m "${MODEL_NAME}" \
  --model_path "${MODEL_PATH}" \
  -w "${WORKSPACE_DIR}" \
  build \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --max_batch_size "${MAX_BATCH_SIZE}" \
  --max_num_tokens "${MAX_NUM_TOKENS}"

echo
echo "Engine dir:"
echo "${WORKSPACE_DIR}/${MODEL_NAME}/tp_1_pp_1"
