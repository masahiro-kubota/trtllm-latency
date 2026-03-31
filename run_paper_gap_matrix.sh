#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_PATH="${MODEL_PATH:-${MODEL_CACHE_ROOT}/Qwen2.5-0.5B-Instruct}"
INPUT_LEN="${INPUT_LEN:-87}"
OUTPUT_LEN="${OUTPUT_LEN:-40}"
NUM_REQUESTS="${NUM_REQUESTS:-100}"
WARMUP="${WARMUP:-10}"
REPORT_ROOT="${REPORT_ROOT:-${ARTIFACT_ROOT}/matrix_reports}"
GREEDY_YAML="${GREEDY_YAML:-${SCRIPT_DIR}/configs/greedy.yaml}"

mkdir -p "${REPORT_ROOT}"

run_case() {
  local name="$1"
  local quant="${2:-}"
  local use_targets="$3"

  local max_seq_len=$((INPUT_LEN + OUTPUT_LEN))
  local max_num_tokens="${max_seq_len}"
  local engine_tag="$(basename "${MODEL_PATH}")_bs1_seq${max_seq_len}_tok${max_num_tokens}"
  if [[ -n "${quant}" ]]; then
    engine_tag+="_$(echo "${quant}" | tr '[:upper:]' '[:lower:]')"
  fi
  if [[ "${use_targets}" == "1" ]]; then
    engine_tag+="_tgtin${INPUT_LEN}_tgtout${OUTPUT_LEN}"
  fi

  echo "=== ${name} ==="
  local report="${REPORT_ROOT}/${name}.json"
  local iteration_log="${REPORT_ROOT}/${name}.iteration.jsonl"
  local workspace_dir="${ARTIFACT_ROOT}/workspaces/${engine_tag}"
  local engine_dir="${workspace_dir}/${MODEL_NAME}/tp_1_pp_1"
  local dataset="${ARTIFACT_ROOT}/datasets/in${INPUT_LEN}_out${OUTPUT_LEN}.json"
  local status_file="${REPORT_ROOT}/${name}.status.txt"

  rm -f "${report}" "${iteration_log}" "${status_file}"

  if ! QUANTIZATION="${quant}" \
      USE_SCHEDULER_LIMITS="$([[ "${use_targets}" == "1" ]] && echo "0" || echo "1")" \
      TARGET_INPUT_LEN="$([[ "${use_targets}" == "1" ]] && echo "${INPUT_LEN}")" \
      TARGET_OUTPUT_LEN="$([[ "${use_targets}" == "1" ]] && echo "${OUTPUT_LEN}")" \
      MAX_SEQ_LEN="${max_seq_len}" \
      MAX_NUM_TOKENS="${max_num_tokens}" \
      ENGINE_TAG="${engine_tag}" \
      MODEL_NAME="${MODEL_NAME}" \
      MODEL_PATH="${MODEL_PATH}" \
      "${SCRIPT_DIR}/build_engine.sh"; then
    echo "build_failed" > "${status_file}"
    echo "{\"case\":\"${name}\",\"status\":\"build_failed\"}"
    return 0
  fi

  trtllm-bench -m "${MODEL_NAME}" --model_path "${MODEL_PATH}" \
    prepare-dataset --output "${dataset}" \
    token-unif-dist --num-requests "${NUM_REQUESTS}" \
    --input-min "${INPUT_LEN}" --input-max "${INPUT_LEN}" \
    --output-min "${OUTPUT_LEN}" --output-max "${OUTPUT_LEN}" >/dev/null

  if ! trtllm-bench -m "${MODEL_NAME}" --model_path "${MODEL_PATH}" \
      latency --backend tensorrt \
      --engine_dir "${engine_dir}" \
      --dataset "${dataset}" \
      --num_requests "${NUM_REQUESTS}" \
      --warmup "${WARMUP}" \
      --sampler_options "${GREEDY_YAML}" \
      --iteration_log "${iteration_log}" \
      --report_json "${report}" >/dev/null; then
    echo "latency_failed" > "${status_file}"
    echo "{\"case\":\"${name}\",\"status\":\"latency_failed\"}"
    return 0
  fi

  CASE_NAME="${name}" CASE_QUANT="${quant}" REPORT_PATH="${report}" INPUT_LEN="${INPUT_LEN}" OUTPUT_LEN="${OUTPUT_LEN}" python - <<'PY'
import json
import os
from pathlib import Path

obj = json.loads(Path(os.environ["REPORT_PATH"]).read_text())
print(json.dumps({
    "case": os.environ["CASE_NAME"],
    "quantization": os.environ["CASE_QUANT"] or None,
    "input_len": int(os.environ["INPUT_LEN"]),
    "output_len": int(os.environ["OUTPUT_LEN"]),
    "avg_request_latency_ms": obj["performance"]["avg_request_latency_ms"],
    "avg_ttft_ms": obj["streaming_metrics"]["avg_ttft_ms"],
    "avg_tpot_ms": obj["streaming_metrics"]["avg_tpot_ms"],
    "request_throughput_req_s": obj["performance"]["request_throughput_req_s"],
    "system_output_throughput_tok_s": obj["performance"]["system_output_throughput_tok_s"],
}, indent=2))
PY
}

run_case "fp16_baseline" "" 0
run_case "fp16_target_hints" "" 1
run_case "fp8_baseline" "FP8" 0
run_case "fp8_target_hints" "FP8" 1
