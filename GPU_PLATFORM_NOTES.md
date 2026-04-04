# GPU Platform Notes

Last updated: 2026-04-04 UTC

This note records where the current `TensorRT-LLM` workflow fits well, where it
does not, and which GPU services are a better match for future development.

## Short answer

For this repo's current `TensorRT-LLM` workflow:

- Runpod is acceptable as a GPU execution and measurement host.
- Runpod is not a great primary development host when Docker is unavailable and
  you need repeated native `TensorRT-LLM` source builds.

That conclusion is based on this actual bring-up:

- official TensorRT had to be installed manually
- `TensorRT-LLM` native build exposed host dependencies one by one
- blockers included `MPI`, `UCX`, `NIXL`, `LD_LIBRARY_PATH`, and driver library
  discovery
- official docs are still strongly oriented around Docker / devel containers for
  source builds

The practical recommendation is:

- build the custom `tensorrt_llm` wheel on another machine
- carry the deployment bundle into this GPU server
- use this machine only for `import`, smoke builds, and measurements

## Current import target

The import destination is this server, so the target host details matter as much
as the builder details.

Current target-host fingerprint:

- hostname: `6c5f1b75e6d7`
- OS: `Ubuntu 24.04.3 LTS`
- kernel: `6.8.0-78-generic`
- Python: `3.12.3`
- glibc: `2.39`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- GPU memory: `97887 MiB`
- driver: `580.65.06`
- `/usr/local/cuda -> /usr/local/cuda-12.8`
- `libcuda.so.1 -> /usr/lib/x86_64-linux-gnu/libcuda.so.580.65.06`
- TensorRT root: `/usr/local/tensorrt`

This is why the off-box wheel bundle should record both:

- builder information
- target-host information

## Why Runpod is awkward here

This is not a general statement that Runpod is bad. It is a statement about this
specific workflow:

- `TensorRT-LLM` source builds are complex native builds with C++/CUDA/TensorRT
  pieces, not just Python packaging
- the official source-build docs lean heavily toward Docker or devel container
  environments
- when Docker is unavailable, the host itself becomes part of the build surface
- once the host becomes part of the build surface, reproducibility gets much
  worse

What that means in practice:

- development speed drops
- environment debugging starts to dominate code debugging
- wiping and re-trying is harder than in a disposable container
- native build caches can become stale in confusing ways after failed
  configure/build steps

For this reason, this machine should be treated as:

- a runtime/import/benchmark machine

not as:

- the primary place to iterate on `TensorRT-LLM` source builds

## Better options by use case

### 1. Best default for straightforward VM-based development: Lambda Cloud

Why it fits:

- Lambda documents its on-demand product as Linux-based GPU virtual machines
- instances are accessed over standard SSH
- Lambda says you can install Ubuntu-compatible libraries with `apt-get`
- this is much closer to a normal native dev box than the current Runpod setup

Tradeoffs:

- the current public docs show Ubuntu `22.04` + Lambda Stack, not the exact
  Ubuntu `24.04` target used on this server
- the public on-demand instance list currently shows `GH200`, `H100`, `A100`,
  `A10`, `A6000`, and older `RTX 6000`, but not the exact `RTX PRO 6000
  Blackwell Server Edition`

Use Lambda when:

- you want the least painful SSH-based development VM
- you want a more stable place to run native builds or your own Docker workflow
- exact GPU parity with this import server is not mandatory

Official references:

- `On-Demand Cloud (ODC) provides on-demand access to Linux-based, GPU-backed virtual machine instances.`
  https://docs.lambda.ai/public-cloud/on-demand/
- Lambda SSH/firewall details:
  https://docs.lambda.ai/public-cloud/on-demand/
- current on-demand instance catalog and pricing:
  https://lambda.ai/service/gpu-cloud

### 2. Best low-cost flexible builder: Vast.ai

Why it fits:

- Vast supports direct SSH access
- Vast supports custom images from Docker Hub and other registries
- Vast exposes launch modes for SSH, Jupyter, or original Docker entrypoint
- file transfer over `scp`/`sftp` and direct port forwarding are documented

Tradeoffs:

- host quality and exact environment vary by provider
- for SSH/Jupyter launch modes, Vast replaces the image entrypoint and expects
  you to use `onstart`, which can reduce reproducibility if you assume plain
  Docker behavior
- great for experimentation, less ideal when you want a tightly pinned and
  predictable builder

Use Vast when:

- you want the cheapest flexible builder
- you are comfortable managing some host variance
- you can tolerate extra setup discipline around launch mode and startup scripts

Official references:

- SSH access:
  https://docs.vast.ai/documentation/instances/connect/ssh
- template launch modes and custom image behavior:
  https://docs.vast.ai/documentation/templates/template-settings
- direct SSH/Jupyter runtype behavior:
  https://docs.vast.ai/api-reference/creating-instances-with-api

### 3. Best if exact Blackwell-family parity matters and you can use a heavier platform: CoreWeave

Why it fits:

- CoreWeave currently documents `RTX Pro 6000 Blackwell Server Edition`
  availability in `US-EAST-14A`
- CoreWeave documents GPU instance families and custom image workflows
- if exact GPU family is important, this is closer to the target machine than
  Lambda's currently listed self-serve on-demand GPUs

Tradeoffs:

- the platform is more cluster / Kubernetes / Slurm oriented
- it is likely overkill if you only want a simple single-node dev VM
- setup complexity can be higher than a straightforward SSH VM service

Use CoreWeave when:

- you need exact or closer Blackwell-family parity
- you are already comfortable with cluster-style environments
- your workflow can justify a heavier platform

Official references:

- GPU instance overview:
  https://docs.coreweave.com/docs/platform/instances/gpu-instances
- `RTX Pro 6000 Blackwell Server Edition` region listing:
  https://docs.coreweave.com/platform/regions/us-east/us-east-14
- custom image workflow:
  https://docs.coreweave.com/docs/products/sunk/development-on-slurm/custom-images

## Recommendation matrix

If the goal is:

- easiest native development VM:
  - use `Lambda Cloud`
- cheapest flexible build box with SSH and custom images:
  - use `Vast.ai`
- closest currently documented match to this Blackwell import server:
  - use `CoreWeave`
- lowest friction overall for this repo:
  - build elsewhere, import here, benchmark here

## Recommended future split for this repo

For this specific repo and workflow, the cleanest split is:

1. Build the wheel on a more development-friendly machine.
2. Package `wheel + TensorRT tarball + constraints + manifest`.
3. Carry that bundle into this GPU server.
4. Use this GPU server only for:
   - `import tensorrt`
   - `import tensorrt_llm`
   - tiny smoke build
   - full engine build
   - latency measurement

That split is already documented in:

- [`OFFBOX_WHEEL_WORKFLOW.md`](./OFFBOX_WHEEL_WORKFLOW.md)
- [`package_offbox_wheel_kit.py`](./package_offbox_wheel_kit.py)

## Bottom line

Runpod is not the best place to do repeated native `TensorRT-LLM` development
when Docker is unavailable.

Runpod is still useful as:

- a GPU runtime box
- an import target
- a benchmarking machine

For future development:

- prefer `Lambda Cloud` if you want the least painful VM experience
- prefer `Vast.ai` if you want flexibility and lower cost
- prefer `CoreWeave` if exact Blackwell-family matching matters more than setup
  simplicity
