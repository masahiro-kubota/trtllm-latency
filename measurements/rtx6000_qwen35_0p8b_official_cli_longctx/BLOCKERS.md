# Qwen3.5-0.8B Bring-up Blockers

This file tracks concrete blockers and slow points encountered while bringing up
the official TensorRT-LLM path for `Qwen/Qwen3.5-0.8B` on the RTX 6000 Runpod
host.

## Current Status

- Current stage: TensorRT-LLM source build
- Current action: rerunning `scripts/build_wheel.py` with `--configure_cmake`
  after fixing MPI and a stale failed build directory state
- Measurement stage reached: not yet

## Timeline

### 1. Official TensorRT root was missing

- Symptom:
  - `FindTensorRT.cmake` could not find TensorRT headers and libraries.
- Error shape:
  - `Could NOT find TensorRT (missing: TensorRT_LIBRARY TensorRT_LIBRARIES TensorRT_INCLUDE_DIR OnnxParser)`
- Root cause:
  - `/usr/local/tensorrt` did not exist yet.
  - The pip-installed Python package alone was not enough for CMake because the
    source build needs the official TensorRT headers and libs under the expected
    root.
- Resolution:
  - Downloaded the official NVIDIA tarball
    `TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-13.0.tar.gz`
  - Extracted it to `/usr/local/tensorrt`
  - Switched the Python package in `.venv-3.12` to the official
    `tensorrt-10.14.1.48` wheel from the tarball
- Relevant logs:
  - `logs/install_tensorrt.log`
  - `logs/pip_editable.log`

### 2. Fresh source build spent a long time in silent FetchContent clones

- Symptom:
  - `build_wheel.log` looked stuck for long stretches with no new log lines.
- Root cause:
  - CMake `FetchContent` was cloning and checking out external repos such as:
    - `cutlass`
    - `flashmla`
    - `nlohmann/json`
    - `deepgemm`
  - These steps are mostly silent in `build_wheel.log`, so `tail -f` appears
    frozen even when the build is still progressing.
- Notes:
  - This is not a failure by itself, but it is a real bring-up friction point.
  - On this host the slow phase was dominated by git clone / checkout /
    submodule update rather than compiler work.

### 3. CMake configure failed because MPI was not installed

- Symptom:
  - The source build failed during CMake configure.
- Error:

```text
Could NOT find MPI_C (missing: MPI_C_LIB_NAMES MPI_C_HEADER_DIR MPI_C_WORKS)
Could NOT find MPI_CXX (missing: MPI_CXX_LIB_NAMES MPI_CXX_HEADER_DIR MPI_CXX_WORKS)
Could NOT find MPI (missing: MPI_C_FOUND MPI_CXX_FOUND)
```

- Root cause:
  - `mpicc` / `mpicxx` and OpenMPI development headers were not installed on
    the host.
  - TensorRT-LLM's CMake enables `find_package(MPI REQUIRED)` when
    `ENABLE_MULTI_DEVICE` is active.
- Resolution:
  - Installed:
    - `openmpi-bin`
    - `libopenmpi-dev`
- Relevant logs:
  - `logs/install_mpi.log`
  - `logs/build_wheel.log`

### 4. `build_wheel.py` reused a failed build directory and skipped reconfigure

- Symptom:
  - After MPI was installed, a plain rerun of `build_wheel.py` failed with:

```text
CMake Build command:
cmake --build . --config Release --parallel 128 --target build_wheel_targets
gmake: Makefile: No such file or directory
gmake: *** No rule to make target 'Makefile'.  Stop.
```

- Root cause:
  - `scripts/build_wheel.py` decides whether this is a "first build" using the
    presence of `cpp/build/CMakeFiles`.
  - The earlier failed configure had already created `CMakeFiles` and
    `CMakeCache.txt`, so the script skipped the configure step on rerun.
  - Because the earlier configure had not completed cleanly, the build
    directory still had no top-level `Makefile`.
- Resolution:
  - Rerunning with `--configure_cmake` so CMake is forced to reconfigure
    against the now-correct host environment.
- Relevant logs:
  - `logs/build_wheel.log`

## Commands Used For Live Diagnosis

These were useful when `tail -f` looked frozen:

```bash
tail -f /workspace/trtllm-latency/measurements/rtx6000_qwen35_0p8b_official_cli_longctx/logs/build_wheel.log
ps -eo pid,etime,stat,cmd | rg 'build_wheel.py|cmake --build|git clone|git checkout|json-populate|deepgemm'
cat /proc/<pid>/io | sed -n '1,4p'
```
