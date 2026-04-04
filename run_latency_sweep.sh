#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_PATH="${MODEL_PATH:-${MODEL_CACHE_ROOT}/Qwen2.5-0.5B-Instruct}"
ENGINE_TAG="${ENGINE_TAG:-$(basename "${MODEL_PATH}")_bs1_seq127_tok127}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${ARTIFACT_ROOT}/workspaces/${ENGINE_TAG}}"
ENGINE_DIR="${ENGINE_DIR:-${WORKSPACE_DIR}/${MODEL_NAME}/tp_1_pp_1}"
DATASET_DIR="${DATASET_DIR:-${ARTIFACT_ROOT}/datasets}"
REPORT_DIR="${REPORT_DIR:-${ARTIFACT_ROOT}/reports}"
INPUT_LENGTHS="${INPUT_LENGTHS:-8 16 32 64 87}"
OUTPUT_LEN="${OUTPUT_LEN:-40}"
NUM_REQUESTS="${NUM_REQUESTS:-20}"
WARMUP="${WARMUP:-5}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

mkdir -p "${DATASET_DIR}" "${REPORT_DIR}"

for inlen in ${INPUT_LENGTHS}; do
  dataset="${DATASET_DIR}/in${inlen}_out${OUTPUT_LEN}.json"
  report="${REPORT_DIR}/in${inlen}_out${OUTPUT_LEN}_bs1.json"

  prepare_cmd=(
    trtllm-bench -m "${MODEL_NAME}" --model_path "${MODEL_PATH}"
    prepare-dataset
  )
  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    prepare_cmd+=(--trust-remote-code)
    export TRUST_REMOTE_CODE=1
  fi
  prepare_cmd+=(
    --output "${dataset}"
    token-unif-dist --num-requests "${NUM_REQUESTS}"
    --input-min "${inlen}" --input-max "${inlen}"
    --output-min "${OUTPUT_LEN}" --output-max "${OUTPUT_LEN}"
  )
  latency_cmd=(
    trtllm-bench -m "${MODEL_NAME}" --model_path "${MODEL_PATH}"
    latency --backend tensorrt
    --engine_dir "${ENGINE_DIR}"
    --dataset "${dataset}"
    --num_requests "${NUM_REQUESTS}"
    --warmup "${WARMUP}"
    --report_json "${report}"
  )

  "${prepare_cmd[@]}" >/dev/null
  "${latency_cmd[@]}" >/dev/null

  python - <<PY
import json
from pathlib import Path

obj = json.loads(Path("${report}").read_text())
print(json.dumps({
    "input_len": ${inlen},
    "output_len": ${OUTPUT_LEN},
    "avg_request_latency_ms": obj["performance"]["avg_request_latency_ms"],
    "avg_ttft_ms": obj["streaming_metrics"]["avg_ttft_ms"],
    "avg_tpot_ms": obj["streaming_metrics"]["avg_tpot_ms"],
    "req_per_s": obj["performance"]["request_throughput_req_s"],
    "output_tok_per_s": obj["performance"]["system_output_throughput_tok_s"],
}))
PY
done
