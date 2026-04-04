# Qwen3.5-0.8B Official TensorRT-LLM Long-Context: Status And Repro

Last updated: 2026-04-04 UTC

## Goal

Run `Qwen/Qwen3.5-0.8B` through the official TensorRT-LLM CLI path only:

- `trtllm-bench build`
- `trtllm-bench latency --backend tensorrt`

Target output location:

- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx`

Hard constraints used during this attempt:

- no Docker
- no PyTorch backend fallback
- no AutoDeploy fallback
- no mixed unofficial TensorRT root

## Repositories And Reference Docs

Main repos:

- `/workspace/TensorRT-LLM`
- `/workspace/trtllm-latency`

Reference docs used:

- `/workspace/TensorRT-LLM/docs/source/installation/build-from-source-linux.md`
- `/workspace/TensorRT-LLM/docker/develop.md`
- `/workspace/TensorRT-LLM/docker/common/install_ucx.sh`
- `/workspace/TensorRT-LLM/docker/common/install_nixl.sh`
- `/workspace/TensorRT-LLM/examples/disaggregated/README.md`
- `/workspace/TensorRT-LLM/examples/llm-api/qwen35_local_runtime/QWEN35_RTX6000_SETUP.md`

## Local Artifacts Already Prepared

Raw model checkpoint:

- `/workspace/trtllm-latency/models/Qwen3.5-0.8B-vlm`

Normalized text-only mirror:

- `/workspace/trtllm-latency/models/Qwen3.5-0.8B-text-mirror`

Confirmed on the mirror:

- `model_type = qwen3_5_text`
- `architectures = ["Qwen3_5ForCausalLM"]`
- `max_position_embeddings = 262144`
- `rope_scaling` is present

Measurement working directory:

- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx`

Subdirectories:

- `datasets/`
- `reports/`
- `logs/`

## Local Repo Changes In trtllm-latency

The measurement repo has helper changes to support the official host-build path:

- `/workspace/trtllm-latency/env.sh`
  - supports `TRTLLM_PYTHON`
  - prepends the interpreter's `bin` directory to `PATH`
  - prepends `/usr/local/tensorrt/lib` to `LD_LIBRARY_PATH` if present
  - exports `TLLM_WORKER_USE_SINGLE_PROCESS=1`
- `/workspace/trtllm-latency/build_engine.sh`
  - supports `TRUST_REMOTE_CODE=1`
- `/workspace/trtllm-latency/run_latency_sweep.sh`
  - supports `TRUST_REMOTE_CODE=1` for dataset preparation
- `/workspace/trtllm-latency/normalize_qwen35_vlm_checkpoint.py`
  - converts the raw VLM checkpoint into the text-only mirror
- `/workspace/trtllm-latency/generate_official_longctx_readme.py`
  - generates the final README from report JSON files

No source edits were made to `/workspace/TensorRT-LLM` itself. That repo was used as upstream source plus build tree only.

## Logs Collected So Far

- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/install_tensorrt.log`
- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/build_wheel.log`
- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/pip_editable.log`
- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/install_mpi.log`
- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/install_ucx.log`
- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/install_nixl.log`
- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/host_env_exports.sh`
- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/BLOCKERS.md`

## Host State Reached Before Shutdown

### 1. Official TensorRT was downloaded and installed

Downloaded archive:

- `/workspace/TensorRT-LLM/.downloads/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-13.0.tar.gz`

Installed root:

- `/usr/local/tensorrt`

Confirmed files:

- `/usr/local/tensorrt/include/NvInfer.h`
- `/usr/local/tensorrt/include/NvOnnxParser.h`
- `/usr/local/tensorrt/lib/libnvinfer.so`
- `/usr/local/tensorrt/lib/libnvonnxparser.so`

### 2. TensorRT Python package was aligned to the official install

The TensorRT-LLM build venv is:

- `/workspace/TensorRT-LLM/.venv-3.12`

That venv was switched to TensorRT Python package version:

- `10.14.1.48`

### 3. MPI dev packages were installed

Installed to resolve the first configure blocker:

- `openmpi-bin`
- `libopenmpi-dev`

See:

- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/install_mpi.log`

### 4. UCX was installed via the repo's official helper script

Installed root:

- `/usr/local/ucx`

Confirmed file:

- `/usr/local/ucx/lib/cmake/ucx/ucx-config.cmake`

The helper script appended this to:

- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/host_env_exports.sh`

Current content:

```bash
export LD_LIBRARY_PATH=/usr/local/ucx//lib:$LD_LIBRARY_PATH
```

### 5. NIXL install was started but failed

The repo's official helper script was used:

- `/workspace/TensorRT-LLM/docker/common/install_nixl.sh`

It failed because the script searches only under `/usr/local` for `libcuda.so.1`, but on this machine the driver library is at:

- `/usr/lib/x86_64-linux-gnu/libcuda.so.1`

Observed failure:

```text
+ find /usr/local -name libcuda.so.1
+ head -n1
+ CUDA_SO_PATH=
+ [[ -z '' ]]
+ echo 'libcuda.so.1 not found'
libcuda.so.1 not found
+ exit 1
```

See:

- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/install_nixl.log`

## TensorRT-LLM Build Status

The official source build was attempted with:

```bash
cd /workspace/TensorRT-LLM
export TRTLLM_SKIP_REQUIREMENTS_INSTALL=1
python3 scripts/build_wheel.py --clean --cuda_architectures 120-real --benchmarks
```

Then later with forced reconfigure:

```bash
cd /workspace/TensorRT-LLM
export TRTLLM_SKIP_REQUIREMENTS_INSTALL=1
python3 scripts/build_wheel.py --configure_cmake --cuda_architectures 120-real --benchmarks
```

Progress achieved:

- official TensorRT root was picked up correctly after installation
- CMake got past TensorRT detection
- CMake then exposed host-side blockers in this order:
  - missing MPI
  - stale build dir with missing `Makefile` after failed configure
  - `xgrammar` patch-step reentry
  - missing UCX CMake package
  - missing NIXL stack

Detailed write-up is in:

- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/BLOCKERS.md`

At shutdown time, `build_wheel.py` had not yet been rerun after the UCX success and NIXL failure.

## Exact Commands Already Used

### Official TensorRT install

The archive was downloaded with resume support and unpacked under `/usr/local/tensorrt`.

Relevant log:

- `/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/install_tensorrt.log`

### UCX install

```bash
mkdir -p /workspace/TensorRT-LLM/.deps-build
cd /workspace/TensorRT-LLM/.deps-build
export ENV=/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/host_env_exports.sh
export PIP_BREAK_SYSTEM_PACKAGES=1
bash /workspace/TensorRT-LLM/docker/common/install_ucx.sh \
  2>&1 | tee /workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/install_ucx.log
```

### NIXL install attempt

```bash
mkdir -p /workspace/TensorRT-LLM/.deps-build
cd /workspace/TensorRT-LLM/.deps-build
export ENV=/workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/host_env_exports.sh
export PIP_BREAK_SYSTEM_PACKAGES=1
export LD_LIBRARY_PATH=/usr/local/ucx/lib:${LD_LIBRARY_PATH:-}
bash /workspace/TensorRT-LLM/docker/common/install_nixl.sh \
  2>&1 | tee /workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/install_nixl.log
```

### TensorRT-LLM build attempts

```bash
cd /workspace/TensorRT-LLM
export TRTLLM_SKIP_REQUIREMENTS_INSTALL=1
python3 scripts/build_wheel.py --clean --cuda_architectures 120-real --benchmarks \
  2>&1 | tee /workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/build_wheel.log
```

```bash
cd /workspace/TensorRT-LLM
export TRTLLM_SKIP_REQUIREMENTS_INSTALL=1
python3 scripts/build_wheel.py --configure_cmake --cuda_architectures 120-real --benchmarks \
  2>&1 | tee -a /workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/build_wheel.log
```

## What Still Has Not Happened

None of these steps have been completed yet:

- `pip install -e /workspace/TensorRT-LLM` after a successful source build
- smoke engine build for the text mirror
- full engine build with `MAX_SEQ_LEN=9233`
- latency sweep across `8 ... 9192`
- final README generation from report JSON files

## Resume Checklist

On the next machine session, verify these first:

```bash
ls /usr/local/tensorrt/include/NvInfer.h
ls /usr/local/ucx/lib/cmake/ucx/ucx-config.cmake
ls /workspace/trtllm-latency/models/Qwen3.5-0.8B-text-mirror/config.json
```

If `/usr/local/tensorrt` or `/usr/local/ucx` did not survive the shutdown, reinstall them from the logs above before doing anything else.

## Recommended Next Step

The next blocking task is to make the official NIXL installer see `libcuda.so.1` on this host, then rerun the TensorRT-LLM configure.

The important fact discovered during this attempt is:

- `libcuda.so.1` exists on this machine
- it is at `/usr/lib/x86_64-linux-gnu/libcuda.so.1`
- the current `install_nixl.sh` only searches `/usr/local`

After that is solved, continue with:

```bash
cd /workspace/TensorRT-LLM
export TRTLLM_SKIP_REQUIREMENTS_INSTALL=1
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:/usr/local/ucx/lib:${LD_LIBRARY_PATH:-}
python3 scripts/build_wheel.py --configure_cmake --cuda_architectures 120-real --benchmarks \
  2>&1 | tee -a /workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/build_wheel.log
```

If that succeeds, then do:

```bash
/workspace/TensorRT-LLM/.venv-3.12/bin/pip install -e /workspace/TensorRT-LLM \
  2>&1 | tee -a /workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/pip_editable.log
```

Then continue with the planned smoke build and full measurement flow from `/workspace/trtllm-latency`.

## Short Status Summary

Completed:

- raw model download
- text-only mirror generation
- official TensorRT install
- MPI install
- official UCX install

Blocked:

- official NIXL install cannot find `libcuda.so.1`

Not started:

- successful TRT-LLM source build
- engine build
- latency sweep
- final report README
