# RTX 4070 Ti Official CLI Extended Input Sweep

This directory stores the official TensorRT-LLM latency reports for the
`Qwen/Qwen2.5-0.5B-Instruct` batch-1 sweep measured on an `NVIDIA GeForce RTX 4070 Ti`
with latency-oriented engines for inputs up to `9192`, `output=40`.

## What Was Measured

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- GPU: `NVIDIA GeForce RTX 4070 Ti` with `11.59 GiB` VRAM
- Benchmark path: official `trtllm-bench latency --backend tensorrt`
- Sweep script: `run_latency_sweep.sh`
- Input lengths in this table: `8`, `16`, `32`, `64`, `87`, `128`, `256`, `512`, `768`, `1024`
- Continuation beyond `1024`: `2048`, `4096`, `8192`, `9192`
- Output length: fixed `40`
- Number of requests per point: `100`
- Warmup requests per point: `10`

## Engine / Runtime Conditions

- Single-GPU `tp_size=1`, `pp_size=1`
- `TLLM_WORKER_USE_SINGLE_PROCESS=1`
  This was required to avoid the default proxy+MPI executor path for the single-GPU TP1 case.
- Engine build for `input <= 1024`:
  - `dtype=bfloat16`
  - `quantization=None`
  - `kv_cache_dtype=None`
  - `max_batch_size=1`
  - `max_num_tokens=1064`
  - `max_seq_len=1064`
- Engine build for `input > 1024`:
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
| 8 | 40 | 154.39 | 23.11 | 3.37 | 131.27 |
| 16 | 40 | 167.23 | 35.69 | 3.37 | 131.54 |
| 32 | 40 | 177.66 | 28.03 | 3.84 | 149.63 |
| 64 | 40 | 166.04 | 20.59 | 3.73 | 145.45 |
| 87 | 40 | 172.75 | 41.20 | 3.37 | 131.54 |
| 128 | 40 | 171.20 | 33.53 | 3.53 | 137.66 |
| 256 | 40 | 186.02 | 40.63 | 3.73 | 145.39 |
| 512 | 40 | 152.70 | 24.84 | 3.28 | 127.85 |
| 768 | 40 | 182.33 | 39.08 | 3.67 | 143.26 |
| 1024 | 40 | 186.17 | 31.42 | 3.97 | 154.75 |
| 2048 | 40 | 217.84 | 58.28 | 4.09 | 159.56 |
| 4096 | 40 | 258.77 | 74.09 | 4.74 | 184.68 |
| 8192 | 40 | 320.46 | 154.11 | 4.27 | 166.35 |
| 9192 | 40 | 322.07 | 177.11 | 3.72 | 144.96 |

## Notes

- `prefill-ish / TTFT` is the report's `avg_ttft_ms`, and `decoding-ish total` is `(output_len - 1) * avg_tpot_ms`.
- These runs use the same official CLI path as the shorter 4070 Ti sweep.
- `input <= 1024` uses the `max_seq_len=1064` engine.
- `input > 1024` uses a separate `max_seq_len=9232` engine.
- This README was updated to the 100-request rerun. The report JSONs in this directory reflect the larger request count.
- Reducing or increasing input length still does not produce a clean monotonic latency trend; decode remains the dominant cost, but the 100-request rerun removes the most obvious 20-request noise.
- For the `2048+` runs, the source reports live in the sibling directory:
  [rtx4070ti_qwen25_0p5b_official_cli_ultralongctx](/media/masa/ssd_data/trtllm-latency/measurements/rtx4070ti_qwen25_0p5b_official_cli_ultralongctx)
- In the `2048+` reports, the official JSON still prints `engine.max_input_length = 1024`, but the runtime successfully served `input=9192`. For those points, successful completion is the source of truth.

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
- `../rtx4070ti_qwen25_0p5b_official_cli_ultralongctx/reports/in2048_out40_bs1.json`
- `../rtx4070ti_qwen25_0p5b_official_cli_ultralongctx/reports/in4096_out40_bs1.json`
- `../rtx4070ti_qwen25_0p5b_official_cli_ultralongctx/reports/in8192_out40_bs1.json`
- `../rtx4070ti_qwen25_0p5b_official_cli_ultralongctx/reports/in9192_out40_bs1.json`
