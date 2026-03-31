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
cd /path/to/trtllm-latency
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

- Dependencies are managed in [`pyproject.toml`](/media/masa/ssd_data/trtllm-latency/pyproject.toml) and installed with `uv sync`.
- `torch` and `torchvision` are pinned to the CUDA 13 PyTorch index via `tool.uv.sources`.
- `UV_CACHE_DIR` defaults to `.uv-cache/` inside the repo so large wheel downloads stay on the SSD-mounted experiment directory.
- The scripts assume CUDA 13 wheels for PyTorch because `tensorrt_llm==1.2.0` was validated in that setup.
- The engine is intentionally compiled for `batch_size=1` latency, not throughput.
- Default input-length sweep is `8, 16, 32, 64, 87` with fixed output length `40`.

## Paper-gap matrix

To compare `FP16` vs `FP8` and `target_input_len/target_output_len` hints for the
paper-like `input=87`, `output=40` case:

```bash
./run_paper_gap_matrix.sh
```

Outputs are written under:

- `${ARTIFACT_ROOT}/matrix_reports`

Important status note:

- `greedy` decoding is implemented and measured via [`configs/greedy.yaml`](/media/masa/ssd_data/trtllm-latency/configs/greedy.yaml).
- A full 3-way comparison of `bf16` vs `fp8` vs `fp8+kv` is **not done yet**.
- Current measurements only cover:
  - `bf16` baseline
  - `fp8` in the official CLI path, which effectively behaves as `fp8(+kv)` in the current setup
- The unresolved gap is that the official `trtllm-bench build` surface does not cleanly expose a separate `kv_cache_quant_algo` control for this repo's current workflow, so `fp8` and `fp8+kv` have not yet been split into separate reproduced measurements.
