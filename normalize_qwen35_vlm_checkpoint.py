#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


VLM_ARCHS = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}


def normalize_qwen35_config(config_dict: dict) -> dict:
    architectures = config_dict.get("architectures") or []
    if architectures and architectures[0] in VLM_ARCHS:
        text_config = dict(config_dict.get("text_config") or {})
    else:
        text_config = dict(config_dict)
    if not text_config:
        raise ValueError("Qwen3.5 config is missing a usable text_config")

    if "quantization_config" not in text_config and "quantization_config" in config_dict:
        text_config["quantization_config"] = dict(config_dict["quantization_config"])

    rope_parameters = dict(text_config.pop("rope_parameters", {}) or {})
    rope_scaling = dict(text_config.get("rope_scaling") or {})
    if rope_parameters:
        rope_theta = rope_parameters.pop("rope_theta", None)
        if rope_theta is not None:
            text_config.setdefault("rope_theta", rope_theta)
        partial_rotary_factor = rope_parameters.pop("partial_rotary_factor", None)
        if partial_rotary_factor is not None:
            text_config.setdefault("partial_rotary_factor", partial_rotary_factor)
        if rope_parameters:
            rope_scaling = rope_parameters | rope_scaling
    if rope_scaling:
        has_mrope = "mrope_section" in rope_scaling or rope_scaling.get("mrope_interleaved", False)
        if has_mrope:
            rope_scaling["type"] = "mrope"
            rope_scaling.pop("rope_type", None)
        elif "type" not in rope_scaling and "rope_type" in rope_scaling:
            rope_scaling["type"] = rope_scaling.pop("rope_type")
        text_config["rope_scaling"] = rope_scaling

    is_moe = "num_experts" in text_config and text_config["num_experts"] > 0
    if is_moe:
        text_config["architectures"] = ["Qwen3_5MoeForCausalLM"]
    else:
        text_config["architectures"] = ["Qwen3_5ForCausalLM"]
        text_config.setdefault("num_experts", 0)
        text_config.setdefault("num_experts_per_tok", 0)
        text_config.setdefault("moe_intermediate_size", 0)
        text_config.setdefault("shared_expert_intermediate_size", 0)

    if text_config.get("model_type") == "qwen3_5":
        text_config["model_type"] = "qwen3_5_text"

    return text_config


def copy_item(src: Path, dst: Path) -> None:
    if src.is_file():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return

    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return

    raise ValueError(f"Unsupported path type: {src}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to the raw Qwen3.5-0.8B VLM checkpoint")
    parser.add_argument("--dst", required=True, help="Destination path for the text-only mirror")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source checkpoint does not exist: {src}")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    raw_config = json.loads((src / "config.json").read_text())
    normalized = normalize_qwen35_config(raw_config)
    (dst / "config.json").write_text(json.dumps(normalized, indent=2) + "\n")

    for item in src.iterdir():
        if item.name == "config.json":
            continue
        if item.name.startswith("."):
            continue
        copy_item(item, dst / item.name)

    print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
