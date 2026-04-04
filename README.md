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

Recorded sweeps in this repo:

- [`measurements/rtx4070ti_qwen25_0p5b_official_cli/README.md`](/media/masa/ssd_data/trtllm-latency/measurements/rtx4070ti_qwen25_0p5b_official_cli/README.md)
- [`measurements/rtx4070ti_qwen25_0p5b_official_cli_longctx/README.md`](/media/masa/ssd_data/trtllm-latency/measurements/rtx4070ti_qwen25_0p5b_official_cli_longctx/README.md)
- [`measurements/rtx4070ti_qwen25_0p5b_official_cli_ultralongctx/README.md`](/media/masa/ssd_data/trtllm-latency/measurements/rtx4070ti_qwen25_0p5b_official_cli_ultralongctx/README.md)
- [`measurements/rtx6000_qwen25_0p5b_official_cli/README.md`](/media/masa/ssd_data/trtllm-latency/measurements/rtx6000_qwen25_0p5b_official_cli/README.md)

## Notes

- Dependencies are managed in [`pyproject.toml`](/media/masa/ssd_data/trtllm-latency/pyproject.toml) and installed with `uv sync`.
- `torch` and `torchvision` are pinned to the CUDA 13 PyTorch index via `tool.uv.sources`.
- `UV_CACHE_DIR` defaults to `.uv-cache/` inside the repo so large wheel downloads stay on the SSD-mounted experiment directory.
- The scripts assume CUDA 13 wheels for PyTorch because `tensorrt_llm==1.2.0` was validated in that setup.
- The engine is intentionally compiled for `batch_size=1` latency, not throughput.
- Default input-length sweep is `8, 16, 32, 64, 87` with fixed output length `40`.

## Off-box wheel workflow

If the target GPU server is slow or inconvenient for full source builds, this
repo supports carrying in a prebuilt custom `tensorrt_llm` wheel from another
machine and using that wheel venv through `TRTLLM_PYTHON`.

Key points:

- the carried artifact should be a deployment bundle, not only the wheel
- the target still needs official TensorRT runtime libraries and the matching
  TensorRT Python wheel
- `env.sh` already supports an external interpreter through `TRTLLM_PYTHON`

See:

- [`OFFBOX_WHEEL_WORKFLOW.md`](./OFFBOX_WHEEL_WORKFLOW.md)
- [`GPU_PLATFORM_NOTES.md`](./GPU_PLATFORM_NOTES.md)
- [`package_offbox_wheel_kit.py`](./package_offbox_wheel_kit.py)
- [`measurements/rtx6000_qwen35_0p8b_official_cli_longctx/STATUS_AND_REPRO.md`](./measurements/rtx6000_qwen35_0p8b_official_cli_longctx/STATUS_AND_REPRO.md)

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
- This `greedy` setting is used only to stabilize and simplify decode-latency measurement.
- It should not be treated as the default quality-oriented or paper-faithful sampling setup for downstream task performance.
- In particular, `greedy` can change reasoning behavior, handoff timing, and final task quality relative to sampled decoding.
- A full 3-way comparison of `bf16` vs `fp8` vs `fp8+kv` is **not done yet**.
- Current measurements only cover:
  - `bf16` baseline
  - `fp8` in the official CLI path, which effectively behaves as `fp8(+kv)` in the current setup
- The unresolved gap is that the official `trtllm-bench build` surface does not cleanly expose a separate `kv_cache_quant_algo` control for this repo's current workflow, so `fp8` and `fp8+kv` have not yet been split into separate reproduced measurements.

Current reproduced matrix results on `NVIDIA GeForce RTX 4070 Ti` for the
paper-like `input=87`, `output=40`, batch-1 case:

| condition | input | output | engine dtype | kv cache dtype | avg request latency (ms) | prefill-ish / TTFT (ms) | avg TPOT (ms) | decoding-ish total (ms) |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `bf16` baseline | 87 | 40 | `bfloat16` | `None` | `189.72` | `26.37` | `4.19` | `163.36` |
| `fp8(+kv)` official CLI path | 87 | 40 | `bfloat16` | `FP8` | `156.89` | `42.13` | `2.94` | `114.76` |

Source reports:

- [`artifacts/matrix_reports/fp16_baseline_rerun.json`](/media/masa/ssd_data/trtllm-latency/artifacts/matrix_reports/fp16_baseline_rerun.json)
- [`artifacts/matrix_reports/fp8_baseline_rerun.json`](/media/masa/ssd_data/trtllm-latency/artifacts/matrix_reports/fp8_baseline_rerun.json)

Here, `prefill-ish / TTFT` is the report's `avg_ttft_ms`, and
`decoding-ish total` is `(output_len - 1) * avg_tpot_ms`.
