# RTX 4070 Ti Official CLI Sweep

This directory stores the original official TensorRT-LLM latency reports for the
`Qwen/Qwen2.5-0.5B-Instruct` batch-1 sweep measured on an `NVIDIA GeForce RTX 4070 Ti`.

## What Was Measured

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- GPU: `NVIDIA GeForce RTX 4070 Ti` with `11.59 GiB` VRAM
- Benchmark path: official `trtllm-bench latency --backend tensorrt`
- Sweep script: `run_official_latency_sweep.sh`
- Input lengths: `8`, `16`, `32`, `64`, `87`
- Output length: fixed `40`
- Number of requests per point: `20`
- Warmup requests per point: `5`

## Engine / Runtime Conditions

- Single-GPU `tp_size=1`, `pp_size=1`
- `TLLM_WORKER_USE_SINGLE_PROCESS=1`
  This was required to avoid the default proxy+MPI executor path for the single-GPU TP1 case.
- The first engine attempt used a large default `max_batch_size` and caused runtime OOM.
  The reports in this directory are all from the rebuilt latency-oriented engine with:
  - `max_batch_size=1`
  - `max_num_tokens=127`
  - `max_seq_len=127`
- Backend: TensorRT

## Summary

| input | output | avg request latency (ms) | avg TTFT (ms) | avg TPOT (ms) |
|---|---:|---:|---:|---:|
| 8 | 40 | 201.13 | 39.87 | 4.13 |
| 16 | 40 | 154.38 | 30.81 | 3.17 |
| 32 | 40 | 192.71 | 44.59 | 3.80 |
| 64 | 40 | 182.59 | 15.69 | 4.28 |
| 87 | 40 | 181.58 | 22.57 | 4.08 |

## Notes

- TensorRT-LLM was much faster than the stock Hugging Face runtime in the same environment.
- These runs still did not reach the paper-side `70ms (40 tokens)` reasoning-decoding number.
- Reducing input length did not change latency dramatically in this sweep, so decode remained the dominant cost.
- Because each point uses only `20` requests, there is some noise. These files are best treated as a first-pass latency characterization, not a publication-grade benchmark.

## Files

- `reports/in8_out40_bs1.json`
- `reports/in16_out40_bs1.json`
- `reports/in32_out40_bs1.json`
- `reports/in64_out40_bs1.json`
- `reports/in87_out40_bs1.json`
