# RTX 6000 Official CLI Long-Context Sweep

This directory stores the official TensorRT-LLM latency reports for a long-context
batch-1 sweep of `Qwen/Qwen2.5-0.5B-Instruct` measured on an
`NVIDIA RTX PRO 6000 Blackwell Server Edition`, extended through `input_len=9192`.

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
  - `max_batch_size=1`
  - `max_num_tokens=9233`
  - `max_seq_len=9233`
- `max_seq_len=9233` was chosen so `input_len=9192` with fixed `output_len=40`
  fits within one engine build.
- Backend: TensorRT
- Engine dtype: `bfloat16`
- TensorRT-LLM version: `1.2.0`
- The CLI emitted `Failed to get device capability: SM 12.x requires CUDA >= 12.9.` on this SM 12.0 GPU, but both engine build and latency runs completed successfully.

## Summary

| input | output | avg request latency (ms) | prefill-ish / TTFT (ms) | avg TPOT (ms) | decoding-ish total (ms) |
|---|---:|---:|---:|---:|---:|
| 8 | 40 | 71.39 | 4.47 | 1.72 | 66.91 |
| 16 | 40 | 71.50 | 4.92 | 1.71 | 66.58 |
| 32 | 40 | 72.00 | 5.14 | 1.71 | 66.85 |
| 64 | 40 | 71.49 | 4.83 | 1.71 | 66.66 |
| 87 | 40 | 71.88 | 5.10 | 1.71 | 66.77 |
| 128 | 40 | 72.50 | 5.36 | 1.72 | 67.13 |
| 256 | 40 | 74.98 | 4.87 | 1.80 | 70.11 |
| 512 | 40 | 76.43 | 5.85 | 1.81 | 70.58 |
| 768 | 40 | 79.20 | 8.44 | 1.81 | 70.76 |
| 1024 | 40 | 79.24 | 8.28 | 1.82 | 70.96 |
| 2048 | 40 | 84.80 | 13.22 | 1.84 | 71.59 |
| 4096 | 40 | 99.66 | 24.66 | 1.92 | 75.00 |
| 6144 | 40 | 108.53 | 32.52 | 1.95 | 76.01 |
| 8192 | 40 | 120.54 | 43.67 | 1.97 | 76.87 |
| 9192 | 40 | 120.70 | 48.60 | 1.85 | 72.10 |

## Notes

- Average request latency stayed around `71-72 ms` through `input_len=128`, rose into the `75-85 ms` range by `256-2048`, and reached about `100-121 ms` by `4096-9192`.
- `prefill-ish / TTFT` grew from `4.47 ms` to `48.60 ms` across the sweep, while `decoding-ish total` stayed much tighter, roughly `66.58-76.87 ms`.
- `prefill-ish / TTFT` is the report's `avg_ttft_ms`, and `decoding-ish total` is `(output_len - 1) * avg_tpot_ms`.
- In other words, the official CLI reports satisfy `avg request latency = avg_ttft_ms + (output_len - 1) * avg_tpot_ms` for these runs.
- The `8192` and `9192` points ended up very close in total request latency; the extra prompt cost mostly showed up in `TTFT`, while the decode-side average was noisier.
- Because this GPU has about `95 GiB` VRAM and the runtime kept `kv_cache_percentage=0.9`, executor initialization reserved about `84 GiB` for paged KV cache.
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
