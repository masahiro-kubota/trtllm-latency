#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_PATH="${MODEL_PATH:-${MODEL_CACHE_ROOT}/Qwen2.5-0.5B-Instruct}"
QUANTIZATION="${QUANTIZATION:-}"
TARGET_INPUT_LEN="${TARGET_INPUT_LEN:-}"
TARGET_OUTPUT_LEN="${TARGET_OUTPUT_LEN:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
ENGINE_TAG_DEFAULT="$(basename "${MODEL_PATH}")_bs1_seq127_tok127"
if [[ -n "${QUANTIZATION}" ]]; then
  ENGINE_TAG_DEFAULT+="_$(echo "${QUANTIZATION}" | tr '[:upper:]' '[:lower:]')"
fi
if [[ -n "${TARGET_INPUT_LEN}" && -n "${TARGET_OUTPUT_LEN}" ]]; then
  ENGINE_TAG_DEFAULT+="_tgtin${TARGET_INPUT_LEN}_tgtout${TARGET_OUTPUT_LEN}"
fi
ENGINE_TAG="${ENGINE_TAG:-${ENGINE_TAG_DEFAULT}}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${ARTIFACT_ROOT}/workspaces/${ENGINE_TAG}}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-127}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-127}"
USE_SCHEDULER_LIMITS="${USE_SCHEDULER_LIMITS:-1}"

rm -rf "${WORKSPACE_DIR}"
mkdir -p "${WORKSPACE_DIR}"

build_cmd=(
  trtllm-bench -m "${MODEL_NAME}"
  --model_path "${MODEL_PATH}"
  -w "${WORKSPACE_DIR}"
  build
  --max_seq_len "${MAX_SEQ_LEN}"
)

if [[ "${USE_SCHEDULER_LIMITS}" == "1" ]]; then
  build_cmd+=(--max_batch_size "${MAX_BATCH_SIZE}" --max_num_tokens "${MAX_NUM_TOKENS}")
fi

if [[ -n "${QUANTIZATION}" ]]; then
  build_cmd+=(-q "${QUANTIZATION}")
fi
if [[ -n "${TARGET_INPUT_LEN}" ]]; then
  build_cmd+=(--target_input_len "${TARGET_INPUT_LEN}")
fi
if [[ -n "${TARGET_OUTPUT_LEN}" ]]; then
  build_cmd+=(--target_output_len "${TARGET_OUTPUT_LEN}")
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  build_cmd+=(--trust_remote_code true)
fi

"${build_cmd[@]}"

echo
echo "Engine dir:"
echo "${WORKSPACE_DIR}/${MODEL_NAME}/tp_1_pp_1"
