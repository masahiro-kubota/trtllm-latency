# RTX 6000 Official CLI FP8 Long-Context Sweep

This directory stores the official TensorRT-LLM latency reports for a long-context
batch-1 sweep of `Qwen/Qwen2.5-0.5B-Instruct` measured on an
`NVIDIA RTX PRO 6000 Blackwell Server Edition`, using the official CLI `fp8` path
through `input_len=9192`.

## What Was Measured

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition` with `94.97 GiB` VRAM
- Benchmark path: official `trtllm-bench latency --backend tensorrt`
- Sweep script: `run_latency_sweep.sh`
- Input lengths: `8`, `16`, `32`, `64`, `87`, `128`, `256`, `512`, `768`, `1024`, `2048`, `4096`, `6144`, `8192`, `9192`
- Output length: fixed `40`
- Number of requests per point: `100`
- Warmup requests per point: `10`

## Engine / Runtime Conditions

- Single-GPU `tp_size=1`, `pp_size=1`
- `TLLM_WORKER_USE_SINGLE_PROCESS=1`
  This was required to avoid the default proxy+MPI executor path for the single-GPU TP1 case.
- The engine was built with:
  - `QUANTIZATION=FP8`
  - `TARGET_INPUT_LEN=9192`
  - `TARGET_OUTPUT_LEN=40`
  - `USE_SCHEDULER_LIMITS=0`
  - `max_seq_len=9233`
- TensorRT-LLM version: `1.2.0`
- Backend: TensorRT
- Engine dtype: `bfloat16`
- KV cache dtype: `FP8`
- In this repo's workflow, the official CLI `fp8` path should be treated as effectively `fp8(+kv)`.
- The report JSON still shows `engine.max_input_length=1024`, but the actual runtime accepted and measured requests through `input_len=9192` with this target-hints build.
- The CLI emitted `Failed to get device capability: SM 12.x requires CUDA >= 12.9.` on this SM 12.0 GPU, but both engine build and latency runs completed successfully.

## Summary

| input | output | avg request latency (ms) | prefill-ish / TTFT (ms) | avg TPOT (ms) | decoding-ish total (ms) |
|---|---:|---:|---:|---:|---:|
| 8 | 40 | 66.93 | 2.54 | 1.65 | 64.39 |
| 16 | 40 | 67.00 | 2.82 | 1.65 | 64.17 |
| 32 | 40 | 66.87 | 2.53 | 1.65 | 64.35 |
| 64 | 40 | 67.07 | 2.72 | 1.65 | 64.34 |
| 87 | 40 | 66.83 | 2.62 | 1.65 | 64.21 |
| 128 | 40 | 66.65 | 2.54 | 1.64 | 64.11 |
| 256 | 40 | 67.82 | 3.04 | 1.66 | 64.79 |
| 512 | 40 | 69.89 | 4.22 | 1.68 | 65.67 |
| 768 | 40 | 69.44 | 4.50 | 1.67 | 64.94 |
| 1024 | 40 | 69.85 | 5.01 | 1.66 | 64.84 |
| 2048 | 40 | 75.07 | 8.62 | 1.70 | 66.45 |
| 4096 | 40 | 85.34 | 16.57 | 1.76 | 68.77 |
| 6144 | 40 | 97.41 | 27.03 | 1.80 | 70.38 |
| 8192 | 40 | 112.98 | 40.10 | 1.87 | 72.88 |
| 9192 | 40 | 120.45 | 46.27 | 1.90 | 74.18 |

## Notes

- Short contexts stayed around `66.6-67.1 ms` through `input_len=128`.
- Total latency rose to `75.07 ms` at `2048`, `85.34 ms` at `4096`, `112.98 ms` at `8192`, and `120.45 ms` at `9192`.
- `prefill-ish / TTFT` grew from `2.54 ms` to `46.27 ms` across the sweep, while `decoding-ish total` remained much tighter at roughly `64-74 ms`.
- `prefill-ish / TTFT` is the report's `avg_ttft_ms`, and `decoding-ish total` is `(output_len - 1) * avg_tpot_ms`.
- For these runs, the official CLI reports satisfy `avg request latency = avg_ttft_ms + (output_len - 1) * avg_tpot_ms`.
- Compared with the RTX 6000 `bf16` long-context sweep, this FP8 path is consistently faster in the short and mid context range, with the gap narrowing near the longest prompts.
- Because each point uses `100` requests, this sweep is materially less noisy than the earlier `20`-request short-context runs.

## Files

- `reports/in8_out40_bs1.json`
- `reports/in16_out40_bs1.json`
- `reports/in32_out40_bs1.json`
- `reports/in64_out40_bs1.json`
- `reports/in87_out40_bs1.json`
- `reports/in128_out40_bs1.json`
- `reports/in256_out40_bs1.json`
- `reports/in512_out40_bs1.json`
- `reports/in768_out40_bs1.json`
- `reports/in1024_out40_bs1.json`
- `reports/in2048_out40_bs1.json`
- `reports/in4096_out40_bs1.json`
- `reports/in6144_out40_bs1.json`
- `reports/in8192_out40_bs1.json`
- `reports/in9192_out40_bs1.json`
