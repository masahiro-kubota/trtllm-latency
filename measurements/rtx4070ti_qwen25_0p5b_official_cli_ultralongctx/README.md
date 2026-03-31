# RTX 4070 Ti Official CLI Ultra-Long Input Sweep

This directory stores the official TensorRT-LLM latency reports for the
`Qwen/Qwen2.5-0.5B-Instruct` batch-1 sweep measured on an `NVIDIA GeForce RTX 4070 Ti`
with a larger latency-oriented engine used for inputs beyond `1024`.

## What Was Measured

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- GPU: `NVIDIA GeForce RTX 4070 Ti` with `11.59 GiB` VRAM
- Benchmark path: official `trtllm-bench latency --backend tensorrt`
- Sweep script: `run_latency_sweep.sh`
- Input lengths: `2048`, `4096`, `8192`, `9192`
- Output length: fixed `40`
- Number of requests per point: `100`
- Warmup requests per point: `10`

## Engine / Runtime Conditions

- Single-GPU `tp_size=1`, `pp_size=1`
- `TLLM_WORKER_USE_SINGLE_PROCESS=1`
  This was required to avoid the default proxy+MPI executor path for the single-GPU TP1 case.
- Engine build:
  - `dtype=bfloat16`
  - `quantization=None`
  - `kv_cache_dtype=None`
  - `max_batch_size=1`
  - `max_num_tokens=9232`
  - `max_seq_len=9232`
- Backend: TensorRT

## Summary

| input | output | avg request latency (ms) | prefill-ish / TTFT (ms) | avg TPOT (ms) | decoding-ish total (ms) |
|---|---:|---:|---:|---:|---:|
| 2048 | 40 | 217.84 | 58.28 | 4.09 | 159.56 |
| 4096 | 40 | 258.77 | 74.09 | 4.74 | 184.68 |
| 8192 | 40 | 320.46 | 154.11 | 4.27 | 166.35 |
| 9192 | 40 | 322.07 | 177.11 | 3.72 | 144.96 |

## Notes

- `prefill-ish / TTFT` is the report's `avg_ttft_ms`, and `decoding-ish total` is `(output_len - 1) * avg_tpot_ms`.
- These points were measured with the same official CLI path as the shorter sweeps; the only difference is the larger engine.
- The report JSON's `engine.max_input_length` field still shows `1024`, but the runtime successfully served `input=9192` with this engine. In this directory, successful request completion is the source of truth.
- At these lengths, `prefill-ish / TTFT` grows substantially, while the per-token decode cost stays in the same rough band as the shorter-context sweeps.

## Files

- `reports/in2048_out40_bs1.json`
- `reports/in4096_out40_bs1.json`
- `reports/in8192_out40_bs1.json`
- `reports/in9192_out40_bs1.json`
