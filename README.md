# trtllm-latency

Minimal reproduction for batch-1 TensorRT-LLM latency.

This repo uses only the official TensorRT-LLM CLI for the actual build and latency runs:

- `trtllm-bench build`
- `trtllm-bench latency --backend tensorrt`

The only runtime tweak is:

- `TLLM_WORKER_USE_SINGLE_PROCESS=1`

This is required on single-GPU TP1 runs to avoid the default proxy+MPI path and use the official single-process worker path instead.

## Setup

Default examples below use `Qwen/Qwen2.5-0.5B-Instruct`, but the scripts are parameterized with environment variables and can be reused for other HF models.

```bash
./setup_uv_env.sh
source ./env.sh

export MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
export MODEL_PATH="${MODEL_CACHE_ROOT}/Qwen2.5-0.5B-Instruct"

python ./download_hf_model.py \
  --repo-id "${MODEL_NAME}" \
  --output-dir "${MODEL_PATH}"
```

## Build latency-oriented engine

```bash
./build_engine.sh
```

This builds an engine with:

- `MAX_BATCH_SIZE` default: `1`
- `MAX_NUM_TOKENS` default: `127`
- `MAX_SEQ_LEN` default: `127`

You can override them:

```bash
MAX_BATCH_SIZE=1 MAX_NUM_TOKENS=255 MAX_SEQ_LEN=255 ./build_engine.sh
```

## Run official latency sweep

```bash
./run_latency_sweep.sh
```

Reports are written under:

- `${ARTIFACT_ROOT}/reports`

Each report is a JSON file from official `trtllm-bench latency`.

You can override sweep settings:

```bash
INPUT_LENGTHS="8 16 32 64 128" OUTPUT_LEN=40 NUM_REQUESTS=100 WARMUP=10 ./run_latency_sweep.sh
```

## Notes

- The scripts assume CUDA 13 wheels for PyTorch because `tensorrt_llm==1.2.0` was validated in that setup.
- The engine is intentionally compiled for `batch_size=1` latency, not throughput.
- Default input-length sweep is `8, 16, 32, 64, 87` with fixed output length `40`.
