# RTX 6000 Official CLI Sweep

This directory stores the original official TensorRT-LLM latency reports for the
`Qwen/Qwen2.5-0.5B-Instruct` batch-1 sweep measured on an `NVIDIA RTX PRO 6000 Blackwell Server Edition`.

## What Was Measured

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition` with `94.97 GiB` VRAM
- Benchmark path: official `trtllm-bench latency --backend tensorrt`
- Sweep script: `run_latency_sweep.sh`
- Input lengths: `8`, `16`, `32`, `64`, `87`
- Output length: fixed `40`
- Number of requests per point: `20`
- Warmup requests per point: `5`

## Engine / Runtime Conditions

- Single-GPU `tp_size=1`, `pp_size=1`
- `TLLM_WORKER_USE_SINGLE_PROCESS=1`
  This was required to avoid the default proxy+MPI executor path for the single-GPU TP1 case.
- The engine was built with:
  - `max_batch_size=1`
  - `max_num_tokens=127`
  - `max_seq_len=127`
- Backend: TensorRT
- Engine dtype: `bfloat16`
- TensorRT-LLM version: `1.2.0`
- The CLI emitted `Failed to get device capability: SM 12.x requires CUDA >= 12.9.` on this SM 12.0 GPU, but both engine build and latency runs completed successfully.

## Summary

| input | output | avg request latency (ms) | avg TTFT (ms) | avg TPOT (ms) |
|---|---:|---:|---:|---:|
| 8 | 40 | 71.52 | 5.36 | 1.70 |
| 16 | 40 | 70.01 | 3.82 | 1.70 |
| 32 | 40 | 71.47 | 5.15 | 1.70 |
| 64 | 40 | 71.46 | 5.23 | 1.70 |
| 87 | 40 | 70.99 | 4.74 | 1.70 |

## Notes

- Average request latency stayed tightly clustered around `70-72 ms` across this whole input-length sweep.
- These runs landed very close to the paper-side `70ms (40 tokens)` target for reasoning decode.
- Because this GPU has about `95 GiB` VRAM and the runtime kept `kv_cache_percentage=0.9`, executor initialization reserved about `84 GiB` for paged KV cache.
- Because each point uses only `20` requests, these files are still best treated as a first-pass latency characterization rather than a publication-grade benchmark.

## Files

- `reports/in8_out40_bs1.json`
- `reports/in16_out40_bs1.json`
- `reports/in32_out40_bs1.json`
- `reports/in64_out40_bs1.json`
- `reports/in87_out40_bs1.json`
