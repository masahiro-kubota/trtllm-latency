# RTX 4070 Ti Official CLI Extended Input Sweep

This directory stores the official TensorRT-LLM latency reports for the
`Qwen/Qwen2.5-0.5B-Instruct` batch-1 sweep measured on an `NVIDIA GeForce RTX 4070 Ti`
with a larger latency-oriented engine that supports up to `input=1024`, `output=40`.

## What Was Measured

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- GPU: `NVIDIA GeForce RTX 4070 Ti` with `11.59 GiB` VRAM
- Benchmark path: official `trtllm-bench latency --backend tensorrt`
- Sweep script: `run_latency_sweep.sh`
- Input lengths: `8`, `16`, `32`, `64`, `87`, `128`, `256`, `512`, `768`, `1024`
- Output length: fixed `40`
- Number of requests per point: `20`
- Warmup requests per point: `5`

## Engine / Runtime Conditions

- Single-GPU `tp_size=1`, `pp_size=1`
- `TLLM_WORKER_USE_SINGLE_PROCESS=1`
  This was required to avoid the default proxy+MPI executor path for the single-GPU TP1 case.
- Engine build:
  - `dtype=bfloat16`
  - `quantization=None`
  - `kv_cache_dtype=None`
  - `max_batch_size=1`
  - `max_num_tokens=1064`
  - `max_seq_len=1064`
- Backend: TensorRT

## Summary

| input | output | avg request latency (ms) | prefill-ish / TTFT (ms) | avg TPOT (ms) | decoding-ish total (ms) |
|---|---:|---:|---:|---:|---:|
| 8 | 40 | 154.38 | 15.09 | 3.57 | 139.29 |
| 16 | 40 | 187.75 | 24.94 | 4.17 | 162.81 |
| 32 | 40 | 160.78 | 16.71 | 3.69 | 144.07 |
| 64 | 40 | 134.89 | 14.06 | 3.10 | 120.83 |
| 87 | 40 | 176.08 | 27.92 | 3.80 | 148.16 |
| 128 | 40 | 179.92 | 34.80 | 3.72 | 145.12 |
| 256 | 40 | 191.89 | 57.12 | 3.46 | 134.78 |
| 512 | 40 | 181.63 | 46.12 | 3.47 | 135.51 |
| 768 | 40 | 172.21 | 46.29 | 3.23 | 125.93 |
| 1024 | 40 | 153.88 | 20.66 | 3.42 | 133.22 |

## Notes

- `prefill-ish / TTFT` is the report's `avg_ttft_ms`, and `decoding-ish total` is `(output_len - 1) * avg_tpot_ms`.
- These runs use the same official CLI path as the shorter 4070 Ti sweep, but with a separate engine sized for `max_seq_len=1064`.
- Reducing or increasing input length did not produce a clean monotonic latency trend in this sweep; decode remained the dominant cost and the 20-request runs still have visible noise.

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
