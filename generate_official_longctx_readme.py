#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt_ms(value: float) -> str:
    return f"{value:.2f}"


def fmt_gib(value: float) -> str:
    return f"{value:.2f}"


def load_reports(report_dir: Path) -> list[dict]:
    reports = []
    for path in sorted(report_dir.glob("in*_out*_bs1.json")):
      obj = json.loads(path.read_text())
      obj["_path"] = path
      reports.append(obj)
    if not reports:
        raise SystemExit(f"no report JSON files found under {report_dir}")
    reports.sort(key=lambda obj: obj["request_info"]["avg_input_length"])
    return reports


def build_summary_rows(reports: list[dict]) -> list[dict]:
    rows = []
    for obj in reports:
        output_len = int(round(obj["request_info"]["avg_output_length"]))
        avg_request_latency_ms = obj["performance"]["avg_request_latency_ms"]
        avg_ttft_ms = obj["streaming_metrics"]["avg_ttft_ms"]
        avg_tpot_ms = obj["streaming_metrics"]["avg_tpot_ms"]
        decoding_total_ms = (output_len - 1) * avg_tpot_ms
        reconstructed = avg_ttft_ms + decoding_total_ms
        if abs(avg_request_latency_ms - reconstructed) > 0.25:
            raise SystemExit(
                f"latency mismatch in {obj['_path']}: "
                f"{avg_request_latency_ms} vs {reconstructed}"
            )
        rows.append(
            {
                "input_len": int(round(obj["request_info"]["avg_input_length"])),
                "output_len": output_len,
                "avg_request_latency_ms": avg_request_latency_ms,
                "avg_ttft_ms": avg_ttft_ms,
                "avg_tpot_ms": avg_tpot_ms,
                "decoding_total_ms": decoding_total_ms,
                "report_name": obj["_path"].name,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-dir", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-len", required=True, type=int)
    parser.add_argument("--num-requests", required=True, type=int)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--max-batch-size", required=True, type=int)
    parser.add_argument("--max-num-tokens", required=True, type=int)
    parser.add_argument("--max-seq-len", required=True, type=int)
    parser.add_argument("--worker-note", default="This was required to avoid the default proxy+MPI executor path for the single-GPU TP1 case.")
    parser.add_argument("--warning-note", default="")
    parser.add_argument("--extra-note", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    measurement_dir = args.measurement_dir.resolve()
    report_dir = measurement_dir / "reports"
    readme_path = measurement_dir / "README.md"

    reports = load_reports(report_dir)
    rows = build_summary_rows(reports)

    first = reports[0]
    last = reports[-1]
    machine_name = first["machine"]["name"]
    total_vram_gib = first["machine"]["memory.total"]
    engine = first["engine"]
    world_info = first["world_info"]
    kv_cache_percentage = world_info["kv_cache_percentage"]
    kv_cache_reserved_gib = total_vram_gib * kv_cache_percentage
    input_lengths = ", ".join(f"`{row['input_len']}`" for row in rows)

    table_lines = [
        "| input | output | avg request latency (ms) | prefill-ish / TTFT (ms) | avg TPOT (ms) | decoding-ish total (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            "| "
            f"{row['input_len']} | "
            f"{row['output_len']} | "
            f"{fmt_ms(row['avg_request_latency_ms'])} | "
            f"{fmt_ms(row['avg_ttft_ms'])} | "
            f"{fmt_ms(row['avg_tpot_ms'])} | "
            f"{fmt_ms(row['decoding_total_ms'])} |"
        )

    notes = [
        f"Average request latency ranged from `{fmt_ms(min(row['avg_request_latency_ms'] for row in rows))} ms` to `{fmt_ms(max(row['avg_request_latency_ms'] for row in rows))} ms` across `input_len={rows[0]['input_len']}` to `input_len={rows[-1]['input_len']}`.",
        f"`prefill-ish / TTFT` ranged from `{fmt_ms(min(row['avg_ttft_ms'] for row in rows))} ms` to `{fmt_ms(max(row['avg_ttft_ms'] for row in rows))} ms`, while `decoding-ish total` stayed between `{fmt_ms(min(row['decoding_total_ms'] for row in rows))} ms` and `{fmt_ms(max(row['decoding_total_ms'] for row in rows))} ms`.",
        f"`prefill-ish / TTFT` is the report's `avg_ttft_ms`, and `decoding-ish total` is `(output_len - 1) * avg_tpot_ms`.",
        f"In other words, the official CLI reports satisfy `avg request latency = avg_ttft_ms + (output_len - 1) * avg_tpot_ms` for these runs.",
        f"Because this GPU has about `{fmt_gib(total_vram_gib)} GiB` VRAM and the runtime kept `kv_cache_percentage={kv_cache_percentage}`, executor initialization reserved about `{fmt_gib(kv_cache_reserved_gib)} GiB` for paged KV cache.",
        f"Because each point uses `{args.num_requests}` requests, this sweep is materially less noisy than earlier short sweeps that used fewer requests.",
    ]
    notes.extend(args.extra_note)

    files = [f"- `reports/{row['report_name']}`" for row in rows]

    lines = [
        f"# {args.title}",
        "",
        f"This directory stores the official TensorRT-LLM latency reports for a long-context",
        f"batch-1 sweep of `{args.model_name}` measured on an",
        f"`{machine_name}`, extended through `input_len={rows[-1]['input_len']}`.",
        "",
        "## What Was Measured",
        "",
        f"- Model: `{args.model_name}`",
        f"- GPU: `{machine_name}` with `{fmt_gib(total_vram_gib)} GiB` VRAM",
        "- Benchmark path: official `trtllm-bench latency --backend tensorrt`",
        "- Sweep script: `run_latency_sweep.sh`",
        f"- Input lengths: {input_lengths}",
        f"- Output length: fixed `{args.output_len}`",
        f"- Number of requests per point: `{args.num_requests}`",
        f"- Warmup requests per point: `{args.warmup}`",
        "",
        "## Engine / Runtime Conditions",
        "",
        f"- Single-GPU `tp_size={world_info['tp_size']}`, `pp_size={world_info['pp_size']}`",
        "- `TLLM_WORKER_USE_SINGLE_PROCESS=1`",
        f"  {args.worker_note}",
        "- The engine was built with:",
        f"  - `max_batch_size={args.max_batch_size}`",
        f"  - `max_num_tokens={args.max_num_tokens}`",
        f"  - `max_seq_len={args.max_seq_len}`",
        f"- `max_seq_len={args.max_seq_len}` was chosen so `input_len={rows[-1]['input_len']}` with fixed `output_len={args.output_len}`",
        "  fits within one engine build.",
        f"- Backend: {engine['backend'] if engine['backend'] != 'TRT' else 'TensorRT'}",
        f"- Engine dtype: `{engine['dtype']}`",
        f"- TensorRT-LLM version: `{engine['version']}`",
    ]
    if args.warning_note:
        lines.append(f"- {args.warning_note}")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            *table_lines,
            "",
            "## Notes",
            "",
            *[f"- {note}" for note in notes],
            "",
            "## Files",
            "",
            *files,
            "",
        ]
    )

    readme_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
