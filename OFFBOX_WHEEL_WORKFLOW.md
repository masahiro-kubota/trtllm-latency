# Off-box TensorRT-LLM Wheel Workflow

This note documents the recommended workflow for building a custom
`tensorrt_llm` wheel on a separate machine and then carrying it into the target
GPU server for official CLI measurements.

The intended target in this repo is:

- Ubuntu `24.04.3`
- `x86_64`, `glibc 2.39`
- Python `3.12.3` / `cp312`
- `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Driver `580.65.06`
- CUDA runtime `13.0`
- TensorRT `10.14.1.48` for CUDA `13.0`

Current target-host fingerprint on this GPU server:

- kernel: `6.8.0-78-generic`
- `/usr/local/cuda -> /usr/local/cuda-12.8`
- `libcuda.so.1 -> /usr/lib/x86_64-linux-gnu/libcuda.so.580.65.06`
- TensorRT root: `/usr/local/tensorrt`

This matters because the import destination is this machine, not the builder.
The carried wheel kit should therefore record the target server's runtime
details, not only the builder's environment.

## Why the wheel alone is not enough

The custom `tensorrt_llm` wheel produced by `scripts/build_wheel.py` does not
fully replace host-side runtime requirements.

In practice you still need:

- TensorRT C++ libraries on the target host
- the matching TensorRT Python wheel
- a compatible CUDA runtime / driver stack
- `libopenmpi-dev` for the official runtime path

This is why the recommended artifact is a small deployment bundle, not just the
wheel by itself.

## Bundle contents

Each deployment bundle should contain:

- the custom `tensorrt_llm-*.whl`
- the official `TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-13.0.tar.gz`
- `constraints.txt`
- `build-manifest.json`
- `README-deploy.md`

This repo provides a helper to create that layout:

- [`package_offbox_wheel_kit.py`](./package_offbox_wheel_kit.py)

The generated `build-manifest.json` should also record the current target host's:

- kernel
- `/usr/local/cuda` resolution
- `libcuda.so.1` path
- TensorRT root

## Recommended builder contract

Use a build host that is as close as possible to the target runtime:

- Ubuntu `24.04`
- Python `3.12`
- CUDA `13.0` user-space
- official TensorRT `10.14.1.48`
- PyTorch `2.9.1` CUDA `13.0`

For the current single-GPU official CLI measurement flow, use:

- `--cuda_architectures "120-real"`
- `-D "BUILD_DEEP_EP=OFF"`

The `BUILD_DEEP_EP=OFF` default intentionally removes the `UCX/NIXL` dependency
from the carried wheel kit because this repo's target workflow is:

- single GPU
- `trtllm-bench build`
- `trtllm-bench latency --backend tensorrt`

If you later need DeepEP or disaggregated serving, rebuild the wheel with those
features enabled and carry the matching `UCX/NIXL` runtime as part of the host
contract too.

## Build the wheel on the separate machine

Clone the exact TensorRT-LLM source commit you want to reproduce, then run:

```bash
cd /path/to/TensorRT-LLM
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130

python3 scripts/build_wheel.py \
  --clean \
  --benchmarks \
  --trt_root /opt/tensorrt/10.14.1.48 \
  --cuda_architectures "120-real" \
  -D "BUILD_DEEP_EP=OFF"
```

Do not silently fall back to a different architecture if `120-real` is rejected.
That builder is not a correct Blackwell-targeted build machine for this flow.

## Package the deployment bundle

Once the wheel exists, create the bundle with:

```bash
cd /path/to/trtllm-latency
python3 ./package_offbox_wheel_kit.py \
  --wheel /path/to/TensorRT-LLM/build/tensorrt_llm-<version>-cp312-cp312-linux_x86_64.whl \
  --tensorrt-tar /path/to/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-13.0.tar.gz \
  --output-dir /path/to/output \
  --source-commit <TensorRT-LLM commit hash> \
  --force
```

The helper will:

- copy the wheel and TensorRT tarball into a named bundle directory
- generate `constraints.txt`
- generate `build-manifest.json`
- generate `README-deploy.md`
- create a `.tar.gz` bundle for transfer

## Install on the target GPU server

On the target host, unpack the bundle and follow the generated
`README-deploy.md`.

The short version is:

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv libopenmpi-dev
sudo tar -xzf TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-13.0.tar.gz -C /usr/local
sudo ln -sfn /usr/local/TensorRT-10.14.1.48 /usr/local/tensorrt

python3.12 -m venv /opt/trtllm-wheel-venv
source /opt/trtllm-wheel-venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:${LD_LIBRARY_PATH:-}

pip install --upgrade pip setuptools wheel
pip install /usr/local/tensorrt/python/tensorrt-10.14.1.48-cp312-none-linux_x86_64.whl
pip install --constraint constraints.txt torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install ./tensorrt_llm-<version>-cp312-cp312-linux_x86_64.whl
```

Then verify:

```bash
python -c "import tensorrt; print(tensorrt.__version__)"
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
trtllm-bench --version
```

If you want to re-capture the current target host fingerprint before importing a
new wheel bundle, run:

```bash
hostname
. /etc/os-release && echo "$PRETTY_NAME"
uname -r
python3 --version
ldd --version | sed -n '1p'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
readlink -f /usr/local/cuda
readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1
readlink -f /usr/local/tensorrt
```

## Use the carried wheel with trtllm-latency

This repo's [`env.sh`](./env.sh) already supports an external Python runtime
through `TRTLLM_PYTHON`, so you can point the latency scripts at the carried
wheel venv directly:

```bash
cd /workspace/trtllm-latency
export TRTLLM_PYTHON=/opt/trtllm-wheel-venv/bin/python
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:${LD_LIBRARY_PATH:-}
export MODEL_NAME=Qwen/Qwen3.5-0.8B
export MODEL_PATH=/workspace/trtllm-latency/models/Qwen3.5-0.8B-text-mirror
export TRUST_REMOTE_CODE=1
```

Always run a tiny smoke build first:

```bash
MAX_BATCH_SIZE=1 MAX_NUM_TOKENS=127 MAX_SEQ_LEN=127 ./build_engine.sh
```

Only continue to the long-context engine and full sweep after the smoke build
creates a non-empty `tp_1_pp_1` engine directory.

## Troubleshooting order

If the carried wheel does not import or the smoke build fails immediately, check
these in order:

1. `LD_LIBRARY_PATH` contains `/usr/local/tensorrt/lib`
2. the TensorRT Python wheel from the tarball is installed
3. TensorRT version matches the bundle manifest
4. PyTorch version matches the pinned CUDA 13.0 wheels
5. the wheel tag is `cp312` and `linux_x86_64`

## Related references

- [TensorRT-LLM build-from-source doc](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html)
- [TensorRT-LLM Linux pip install doc](https://nvidia.github.io/TensorRT-LLM/installation/linux.html)
- [TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/10.14.1/installing-tensorrt/installing.html)
- [`GPU_PLATFORM_NOTES.md`](./GPU_PLATFORM_NOTES.md)
- [`STATUS_AND_REPRO.md`](./measurements/rtx6000_qwen35_0p8b_official_cli_longctx/STATUS_AND_REPRO.md)
