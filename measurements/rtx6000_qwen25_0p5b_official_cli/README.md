# RTX 6000 Official CLI Sweep

This directory stores the official TensorRT-LLM latency reports for the
`Qwen/Qwen2.5-0.5B-Instruct` batch-1 sweep measured on an `NVIDIA RTX PRO 6000 Blackwell Server Edition`,
extended through `input_len=1024`.

## What Was Measured

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition` with `94.97 GiB` VRAM
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
- The engine was built with:
  - `max_batch_size=1`
  - `max_num_tokens=1065`
  - `max_seq_len=1065`
- `max_seq_len=1065` was chosen so `input_len=1024` with fixed `output_len=40`
  fits within one engine build.
- Backend: TensorRT
- Engine dtype: `bfloat16`
- TensorRT-LLM version: `1.2.0`
- The CLI emitted `Failed to get device capability: SM 12.x requires CUDA >= 12.9.` on this SM 12.0 GPU, but both engine build and latency runs completed successfully.

## Summary

| input | output | avg request latency (ms) | prefill-ish / TTFT (ms) | avg TPOT (ms) | decoding-ish total (ms) |
|---|---:|---:|---:|---:|---:|
| 8 | 40 | 71.18 | 4.51 | 1.71 | 66.66 |
| 16 | 40 | 73.61 | 6.99 | 1.71 | 66.62 |
| 32 | 40 | 71.47 | 4.87 | 1.71 | 66.61 |
| 64 | 40 | 71.40 | 4.79 | 1.71 | 66.61 |
| 87 | 40 | 73.72 | 6.93 | 1.71 | 66.78 |
| 128 | 40 | 71.86 | 5.11 | 1.71 | 66.76 |
| 256 | 40 | 77.45 | 7.85 | 1.78 | 69.60 |
| 512 | 40 | 75.90 | 5.81 | 1.80 | 70.09 |
| 768 | 40 | 76.63 | 6.53 | 1.80 | 70.10 |
| 1024 | 40 | 79.98 | 9.57 | 1.81 | 70.40 |

## Notes

- Average request latency stayed around `71-74 ms` through `input_len=128`, then rose into the `76-80 ms` range by `256-1024`.
- The `1024`-token point came in at `79.98 ms`, so this setup remained reasonably close to the paper-side `70ms (40 tokens)` target even at much longer prompts.
- `prefill-ish / TTFT` is the report's `avg_ttft_ms`, and `decoding-ish total` is `(output_len - 1) * avg_tpot_ms`.
- In other words, the official CLI reports satisfy `avg request latency = avg_ttft_ms + (output_len - 1) * avg_tpot_ms` for these runs.
- Across this sweep, `prefill-ish / TTFT` grew from `4.51 ms` to `9.57 ms`, while `decoding-ish total` grew from `66.66 ms` to `70.40 ms`.
- Because this GPU has about `95 GiB` VRAM and the runtime kept `kv_cache_percentage=0.9`, executor initialization reserved about `84 GiB` for paged KV cache.
- Because each point uses only `20` requests, these files are still best treated as a first-pass latency characterization rather than a publication-grade benchmark.

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
