#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import textwrap


DEFAULT_TARGET = {
    "os": "Ubuntu 24.04.3 LTS",
    "cpu_abi": "x86_64 / glibc 2.39",
    "python": "3.12.3 (cp312)",
    "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "driver": "580.65.06",
    "kernel": "6.8.0-78-generic",
    "cuda_runtime": "13.0",
    "cuda_symlink": "/usr/local/cuda -> /usr/local/cuda-12.8",
    "libcuda": "/usr/lib/x86_64-linux-gnu/libcuda.so.580.65.06",
    "tensorrt": "10.14.1.48 for CUDA 13.0",
    "tensorrt_root": "/usr/local/tensorrt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an off-box TensorRT-LLM wheel deployment kit."
    )
    parser.add_argument("--wheel", required=True, help="Path to custom tensorrt_llm wheel.")
    parser.add_argument(
        "--tensorrt-tar",
        required=True,
        help="Path to official TensorRT tarball for the target runtime.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the deployment kit directory and tarball will be written.",
    )
    parser.add_argument(
        "--bundle-name",
        help="Optional bundle directory name. Defaults to a name derived from the wheel.",
    )
    parser.add_argument(
        "--cuda-architectures",
        default="120-real",
        help='CUDA architectures passed to build_wheel.py, for example "120-real".',
    )
    parser.add_argument(
        "--build-deep-ep",
        choices=("ON", "OFF"),
        default="OFF",
        help="Value used for BUILD_DEEP_EP in the wheel build.",
    )
    parser.add_argument(
        "--tensorrt-version",
        default="10.14.1.48",
        help="TensorRT product version included in the kit.",
    )
    parser.add_argument(
        "--cuda-toolkit-version",
        default="13.0",
        help="CUDA toolkit version used on the builder.",
    )
    parser.add_argument(
        "--torch-version",
        default=importlib_version("torch", "2.9.1+cu130"),
        help="Pinned torch version for the target venv.",
    )
    parser.add_argument(
        "--torchvision-version",
        default=importlib_version("torchvision", "0.24.1+cu130"),
        help="Pinned torchvision version for the target venv.",
    )
    parser.add_argument(
        "--setuptools-version",
        default=importlib_version("setuptools", None),
        help="Pinned setuptools version for constraints.txt.",
    )
    parser.add_argument(
        "--wheel-version",
        default=importlib_version("wheel", None),
        help="Pinned wheel package version for constraints.txt.",
    )
    parser.add_argument(
        "--trt-root",
        default=None,
        help="TensorRT root path used on the builder. Defaults to /opt/tensorrt/<version>.",
    )
    parser.add_argument(
        "--build-command",
        default=None,
        help="Exact build command used for the wheel. A sensible default is generated if omitted.",
    )
    parser.add_argument(
        "--source-commit",
        default=None,
        help="TensorRT-LLM source commit hash. Defaults to the current repo commit if available.",
    )
    parser.add_argument(
        "--source-branch",
        default=None,
        help="TensorRT-LLM source branch or ref label. Defaults to the current repo ref.",
    )
    parser.add_argument(
        "--target-os",
        default=DEFAULT_TARGET["os"],
        help="Target server OS string recorded in the manifest.",
    )
    parser.add_argument(
        "--target-cpu-abi",
        default=DEFAULT_TARGET["cpu_abi"],
        help="Target server CPU/ABI string recorded in the manifest.",
    )
    parser.add_argument(
        "--target-python",
        default=DEFAULT_TARGET["python"],
        help="Target server Python string recorded in the manifest.",
    )
    parser.add_argument(
        "--target-gpu",
        default=DEFAULT_TARGET["gpu"],
        help="Target server GPU string recorded in the manifest.",
    )
    parser.add_argument(
        "--target-driver",
        default=DEFAULT_TARGET["driver"],
        help="Target server driver version string recorded in the manifest.",
    )
    parser.add_argument(
        "--target-kernel",
        default=DEFAULT_TARGET["kernel"],
        help="Target server kernel string recorded in the manifest.",
    )
    parser.add_argument(
        "--target-cuda-runtime",
        default=DEFAULT_TARGET["cuda_runtime"],
        help="Target server CUDA runtime string recorded in the manifest.",
    )
    parser.add_argument(
        "--target-cuda-symlink",
        default=DEFAULT_TARGET["cuda_symlink"],
        help="Target server /usr/local/cuda resolution recorded in the manifest.",
    )
    parser.add_argument(
        "--target-libcuda",
        default=DEFAULT_TARGET["libcuda"],
        help="Target server libcuda path recorded in the manifest.",
    )
    parser.add_argument(
        "--target-tensorrt",
        default=DEFAULT_TARGET["tensorrt"],
        help="Target server TensorRT string recorded in the manifest.",
    )
    parser.add_argument(
        "--target-venv",
        default="/opt/trtllm-wheel-venv",
        help="Target venv path used in README-deploy.md.",
    )
    parser.add_argument(
        "--target-trt-root",
        default=DEFAULT_TARGET["tensorrt_root"],
        help="Target TensorRT install root used in README-deploy.md.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3.5-0.8B",
        help="Model name used in the smoke build section.",
    )
    parser.add_argument(
        "--model-path",
        default="/workspace/trtllm-latency/models/Qwen3.5-0.8B-text-mirror",
        help="Model path used in the smoke build section.",
    )
    parser.add_argument(
        "--measurement-repo",
        default="/workspace/trtllm-latency",
        help="Target trtllm-latency repo path used in README-deploy.md.",
    )
    parser.add_argument(
        "--engine-tag",
        default="Qwen3.5-0.8B-text-mirror_bs1_seq127_tok127",
        help="Engine tag used in the smoke build section.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory and tarball.",
    )
    return parser.parse_args()


def importlib_version(package: str, fallback: str | None) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def run_checked(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def git_info(repo_dir: Path) -> dict[str, str | bool | None]:
    if not (repo_dir / ".git").exists():
        return {"commit": None, "branch": None, "dirty": None}
    info: dict[str, str | bool | None] = {}
    try:
        info["commit"] = run_checked(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    except subprocess.CalledProcessError:
        info["commit"] = None
    try:
        info["branch"] = run_checked(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir
        )
    except subprocess.CalledProcessError:
        info["branch"] = None
    try:
        status = run_checked(["git", "status", "--short"], cwd=repo_dir)
        info["dirty"] = bool(status)
    except subprocess.CalledProcessError:
        info["dirty"] = None
    return info


def read_os_release() -> dict[str, str]:
    os_release: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.exists():
        return os_release
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os_release[key] = value.strip().strip('"')
    return os_release


def glibc_version() -> str:
    libc, version = platform.libc_ver()
    if libc and version:
        return f"{libc} {version}"
    try:
        first_line = run_checked(["ldd", "--version"]).splitlines()[0]
        return first_line
    except Exception:
        return "unknown"


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_into_bundle(source: Path, bundle_dir: Path) -> Path:
    destination = bundle_dir / source.name
    if source.resolve() == destination.resolve():
        return destination
    shutil.copy2(source, destination)
    return destination


def render_constraints(args: argparse.Namespace) -> str:
    lines = [
        "# Target runtime pins for the off-box TensorRT-LLM wheel kit.",
        f"torch=={args.torch_version}",
        f"torchvision=={args.torchvision_version}",
    ]
    if args.setuptools_version:
        lines.append(f"setuptools=={args.setuptools_version}")
    else:
        lines.append("setuptools")
    if args.wheel_version:
        lines.append(f"wheel=={args.wheel_version}")
    else:
        lines.append("wheel")
    return "\n".join(lines) + "\n"


def render_deploy_readme(
    args: argparse.Namespace,
    wheel_name: str,
    tensorrt_tar_name: str,
    manifest_name: str,
    constraints_name: str,
    bundle_tar_name: str,
) -> str:
    trt_dir_name = f"TensorRT-{args.tensorrt_version}"
    venv_bin = f"{args.target_venv}/bin"
    return textwrap.dedent(
        f"""\
        # Off-box TensorRT-LLM Wheel Deploy

        This bundle was prepared for a target server with the following runtime contract:

        - OS: {args.target_os}
        - CPU/ABI: {args.target_cpu_abi}
        - Python: {args.target_python}
        - GPU: {args.target_gpu}
        - Driver: {args.target_driver}
        - Kernel: {args.target_kernel}
        - CUDA runtime: {args.target_cuda_runtime}
        - `/usr/local/cuda`: {args.target_cuda_symlink}
        - `libcuda.so.1`: {args.target_libcuda}
        - TensorRT: {args.target_tensorrt}
        - TensorRT root: {args.target_trt_root}

        Bundle contents:

        - `{wheel_name}`
        - `{tensorrt_tar_name}`
        - `{constraints_name}`
        - `{manifest_name}`
        - `{bundle_tar_name}`

        ## 1. Prepare the target host

        ```bash
        sudo apt-get update
        sudo apt-get install -y python3.12-venv libopenmpi-dev
        sudo mkdir -p /usr/local
        sudo tar -xzf {tensorrt_tar_name} -C /usr/local
        sudo ln -sfn /usr/local/{trt_dir_name} {args.target_trt_root}
        ```

        ## 2. Create the runtime venv

        ```bash
        python3.12 -m venv {args.target_venv}
        source {args.target_venv}/bin/activate
        export CUDA_HOME=/usr/local/cuda
        export LD_LIBRARY_PATH={args.target_trt_root}/lib:${{LD_LIBRARY_PATH:-}}

        pip install --upgrade pip setuptools wheel
        pip install {args.target_trt_root}/python/tensorrt-{args.tensorrt_version}-cp312-none-linux_x86_64.whl
        pip install --constraint {constraints_name} torch torchvision --index-url https://download.pytorch.org/whl/cu130
        pip install ./{wheel_name}
        ```

        ## 3. Sanity check the runtime

        ```bash
        export LD_LIBRARY_PATH={args.target_trt_root}/lib:${{LD_LIBRARY_PATH:-}}
        python -c "import tensorrt; print(tensorrt.__version__)"
        python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
        {venv_bin}/trtllm-bench --version
        ```

        If `import tensorrt` fails, check these first:

        1. `LD_LIBRARY_PATH={args.target_trt_root}/lib`
        2. the TensorRT Python wheel is installed from the extracted tarball
        3. TensorRT version matches the bundle manifest
        4. PyTorch matches the pinned CUDA 13.0 wheel

        ## 4. Use the wheel venv with trtllm-latency

        ```bash
        cd {args.measurement_repo}
        export TRTLLM_PYTHON={venv_bin}/python
        export LD_LIBRARY_PATH={args.target_trt_root}/lib:${{LD_LIBRARY_PATH:-}}
        export MODEL_NAME={args.model_name}
        export MODEL_PATH={args.model_path}
        export ENGINE_TAG={args.engine_tag}
        export TRUST_REMOTE_CODE=1
        ```

        ## 5. Run the tiny smoke build first

        ```bash
        cd {args.measurement_repo}
        MAX_BATCH_SIZE=1 MAX_NUM_TOKENS=127 MAX_SEQ_LEN=127 ./build_engine.sh
        ```

        The smoke build must create:

        - `${{ARTIFACT_ROOT:-{args.measurement_repo}/artifacts}}/workspaces/{args.engine_tag}/{args.model_name}/tp_1_pp_1`

        Only after that should you move on to the full `MAX_SEQ_LEN=9233` build and the long-context sweep.
        """
    )


def safe_bundle_name(raw_name: str) -> str:
    allowed = []
    for char in raw_name:
        if char.isalnum() or char in ("-", "_", "."):
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("._") or "offbox_trtllm_kit"


def main() -> int:
    args = parse_args()

    packaging_repo_dir = Path(__file__).resolve().parent
    packaging_git_meta = git_info(packaging_repo_dir)
    trtllm_source_dir = packaging_repo_dir.parent / "TensorRT-LLM"
    trtllm_source_git = git_info(trtllm_source_dir)

    wheel_path = Path(args.wheel).expanduser().resolve()
    tensorrt_tar_path = Path(args.tensorrt_tar).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not wheel_path.is_file():
        raise SystemExit(f"--wheel points to a missing file: {wheel_path}")
    if not tensorrt_tar_path.is_file():
        raise SystemExit(f"--tensorrt-tar points to a missing file: {tensorrt_tar_path}")

    bundle_name = args.bundle_name or f"offbox_{wheel_path.stem}"
    bundle_name = safe_bundle_name(bundle_name)
    bundle_dir = output_dir / bundle_name
    bundle_tar = output_dir / f"{bundle_name}.tar.gz"

    if bundle_dir.exists():
        if not args.force:
            raise SystemExit(f"bundle directory already exists: {bundle_dir} (use --force to overwrite)")
        shutil.rmtree(bundle_dir)
    if bundle_tar.exists():
        if not args.force:
            raise SystemExit(f"bundle tarball already exists: {bundle_tar} (use --force to overwrite)")
        bundle_tar.unlink()

    bundle_dir.mkdir(parents=True, exist_ok=True)

    copied_wheel = copy_into_bundle(wheel_path, bundle_dir)
    copied_trt_tar = copy_into_bundle(tensorrt_tar_path, bundle_dir)

    trt_root = args.trt_root or f"/opt/tensorrt/{args.tensorrt_version}"
    build_command = args.build_command or (
        f'python3 scripts/build_wheel.py --clean --benchmarks '
        f'--trt_root {trt_root} '
        f'--cuda_architectures "{args.cuda_architectures}" '
        f'-D "BUILD_DEEP_EP={args.build_deep_ep}"'
    )

    source_commit = args.source_commit or trtllm_source_git["commit"]
    source_branch = args.source_branch or trtllm_source_git["branch"]
    os_release = read_os_release()

    constraints_path = bundle_dir / "constraints.txt"
    constraints_path.write_text(render_constraints(args))

    manifest = {
        "generated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "packaging_repo": {
            "path": str(packaging_repo_dir),
            "commit": packaging_git_meta["commit"],
            "branch": packaging_git_meta["branch"],
            "dirty": packaging_git_meta["dirty"],
        },
        "tensorrt_llm_source": {
            "path": str(trtllm_source_dir) if trtllm_source_dir.exists() else None,
            "commit": source_commit,
            "branch": source_branch,
            "dirty": trtllm_source_git["dirty"],
        },
        "builder": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "glibc": glibc_version(),
            "os_release": os_release,
            "tensorrt_version": args.tensorrt_version,
            "cuda_toolkit_version": args.cuda_toolkit_version,
            "torch_version": args.torch_version,
            "torchvision_version": args.torchvision_version,
            "cuda_architectures": args.cuda_architectures,
            "build_deep_ep": args.build_deep_ep,
            "build_command": build_command,
        },
        "target_runtime": {
            "os": args.target_os,
            "cpu_abi": args.target_cpu_abi,
            "python": args.target_python,
            "gpu": args.target_gpu,
            "driver": args.target_driver,
            "kernel": args.target_kernel,
            "cuda_runtime": args.target_cuda_runtime,
            "cuda_symlink": args.target_cuda_symlink,
            "libcuda": args.target_libcuda,
            "tensorrt": args.target_tensorrt,
            "venv": args.target_venv,
            "tensorrt_root": args.target_trt_root,
        },
        "artifacts": {
            "wheel": {
                "filename": copied_wheel.name,
                "size_bytes": copied_wheel.stat().st_size,
                "sha256": sha256sum(copied_wheel),
            },
            "tensorrt_tarball": {
                "filename": copied_trt_tar.name,
                "size_bytes": copied_trt_tar.stat().st_size,
                "sha256": sha256sum(copied_trt_tar),
            },
            "constraints": constraints_path.name,
        },
        "measurement_defaults": {
            "measurement_repo": args.measurement_repo,
            "model_name": args.model_name,
            "model_path": args.model_path,
            "engine_tag": args.engine_tag,
        },
    }

    manifest_path = bundle_dir / "build-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    readme_path = bundle_dir / "README-deploy.md"
    readme_path.write_text(
        render_deploy_readme(
            args,
            wheel_name=copied_wheel.name,
            tensorrt_tar_name=copied_trt_tar.name,
            manifest_name=manifest_path.name,
            constraints_name=constraints_path.name,
            bundle_tar_name=bundle_tar.name,
        )
    )

    with tarfile.open(bundle_tar, "w:gz") as archive:
        archive.add(bundle_dir, arcname=bundle_dir.name)

    print(json.dumps(
        {
            "bundle_dir": str(bundle_dir),
            "bundle_tarball": str(bundle_tar),
            "wheel": copied_wheel.name,
            "tensorrt_tarball": copied_trt_tar.name,
            "manifest": manifest_path.name,
            "constraints": constraints_path.name,
            "readme": readme_path.name,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
