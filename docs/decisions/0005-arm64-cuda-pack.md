# 0005: A CUDA pack for linux-arm64 (DGX Spark)

- Status: accepted, 2026-10-03 (issue #32)
- Applies to: the linux-arm64 CUDA pack (`ollaya-linux-arm64-cuda`, `lib/ollaya/cuda_v13`) and the
  linux-arm64 `ollaya-cuda-runner`. The x86-64 and Windows packs, the CPU builds and the Docker
  images are unchanged. There is no arm64 `:cuda` image.

## Context

Until now `linux-arm64` ran on the CPU only. The NVIDIA DGX Spark is an aarch64 machine with a GB10
Grace-Blackwell chip: compute capability 12.1 (sm_121), CUDA 13, driver R580 or newer.

ADR 0004's x86-64 design does not carry over as it is:

- **Microsoft publishes no aarch64 GPU archive.** ONNX Runtime's GitHub releases have an aarch64
  archive for the CPU only. The aarch64 GPU build I found is the `onnxruntime-gpu` wheel on
  PyPI, which Microsoft also publishes. Its first `manylinux_2_34_aarch64` build is 1.29.0. There
  is no `onnxruntime-gpu` 1.28.2 on PyPI at all.
- **pyke publishes no aarch64 GPU build.** `ort-sys` 2.0.0-rc.13 (`build/download/dist.tsv`) has
  one `aarch64-unknown-linux-gnu` entry, with no execution providers. A `bin/ollaya` built with
  `ollaya-runner/cuda` on aarch64 fails to link, on purpose. So the base binary cannot carry ORT's
  CUDA provider bridge (the internal interface that loads the CUDA provider library), as it does on
  x86-64.
- **`cuda-dynamic` still works.** That feature turns on `ort/load-dynamic`, which turns on
  `ort-sys/disable-linking`. The `ort-sys` build script then returns before it downloads or links
  anything, so `ollaya-cuda-runner` builds on aarch64 as it does on x86-64.

### Evidence (measured on a DGX Spark)

The wheel is `onnxruntime_gpu-1.29.0-cp312-cp312-manylinux_2_34_aarch64.whl`, sha256
`545d2966dd11208bbc98e67be4d92e54ecd793acc45c0532cc7bfd36f50692fa`.

- **Contents used.** `onnxruntime/capi/libonnxruntime.so.1.29.0` (soname `libonnxruntime.so.1`),
  `onnxruntime/capi/libonnxruntime_providers_shared.so` and
  `onnxruntime/capi/libonnxruntime_providers_cuda.so`, plus `onnxruntime/LICENSE` and
  `onnxruntime/ThirdPartyNotices.txt`. The Python bindings are not used.
- **glibc.** The libraries need glibc 2.34. Ollaya already needs 2.38 on Linux.
- **Kernels.** `cuobjdump` on `libonnxruntime_providers_cuda.so` lists SASS (machine code for one
  GPU generation) for sm_89, sm_90a, sm_120a and sm_121a, and **no PTX** (portable code that the
  driver can compile for a GPU the build did not list).
- **Links.** The provider's `DT_NEEDED` entries are `libcublasLt.so.13`, `libcublas.so.13`,
  `libcurand.so.10`, `libcudart.so.13` and `libcuda.so.1`, as on x86-64. cuDNN and cuFFT are loaded
  at run time, also as on x86-64.
- **NVIDIA wheels.** All seven wheels pinned in `packaging/cuda-requirements.txt` have aarch64
  builds at the pinned versions. The lock file already holds their hashes.
- **llama.cpp.** ggml-org's `llama-b11146-bin-ubuntu-cuda-13.4-arm64.tar.gz`, already pinned in
  `scripts/llama-cpp.sh`, has a `libggml-cuda.so` with SASS for sm_86, sm_89, sm_120a and sm_121a
  and PTX for sm_75, sm_80 and sm_90, from CUDA 13.4. That is the same set as the x86-64 CUDA 13
  build, and it matches the `CUDA13` table in `crates/ollaya-runner/src/llama/cuda_kernels.rs`.

## Decision

A linux-arm64 CUDA 13 pack, with the layout and runner of ADR 0004:

- **`bin/ollaya` stays CPU-only** on linux-arm64. It is built without `ollaya-runner/cuda` in the
  release, in `scripts/package.sh` and in the `Dockerfile`.
- **The GPU runner alone carries CUDA.** `ollaya-cuda-runner`, built with
  `--features ollaya-runner/cuda-dynamic`, ships in the base archive at
  `lib/ollaya/ollaya-cuda-runner` when the CUDA pack is built too. The daemon ignores a pack
  without that runner (`crates/ollaya-server/src/launch.rs`). Without it the pack installs and
  Ollaya still runs on the CPU. The arm64 Docker image stages no pack and no runner.
- **ONNX Runtime comes from the PyPI wheel.** `package.sh` downloads the wheel, checks its pinned
  sha256 and copies the three libraries byte for byte, as it does with NVIDIA's wheels. The core
  library is stored under its soname, `libonnxruntime.so.1`, the name the loader asks for. That is
  the same rename on copy that `scripts/llama-cpp.sh` uses.
- **ONNX Runtime 1.29.0 on arm64, 1.28.2 elsewhere.** ADR 0004 says the GPU pack's version moves
  with `ort`'s. This is a scoped exception to that rule, for linux-arm64 only, because no 1.28.x
  aarch64 GPU build exists. It is not a general policy change: the other packs stay on 1.28.2.
  `ort` rc.13 asks for C API version 24 (`api-24`), and ONNX Runtime's C API is backward
  compatible across minor versions, so 1.29.0 serves it.
- **The NVIDIA libraries and the llama.cpp backend are the x86-64 ones, built for aarch64.** The
  same pins, the same file list (`CUDA_LIBS_REQUIRED`), and llama.cpp's CUDA 13.4 arm64
  `libggml-cuda.so`. `cuda_kernels.rs` needs no change.
- **No CUDA 12 pack.** There is one aarch64 `onnxruntime-gpu` wheel per version, and it links
  `libcudart.so.13`. `package.sh` rejects `--cuda12` for linux-arm64.
- **The installer checks the release, not the architecture.** `install.sh` installs a GPU pack
  when the release's `sha256sum.txt` lists one for the platform. An older release has no arm64
  pack, so it gets the CPU and a warning.

## Consequences

- **This pack targets GB10 (sm_121a), the DGX Spark**, the only aarch64 GPU it was tested on. The
  wheel's provider has no PTX, so there is no JIT fallback for ONNX models: the driver cannot
  compile kernels for a GPU the build did not list. On other aarch64 NVIDIA systems, for ONNX
  models:
  - Grace-Hopper GH200 (sm_90a) has SASS in the provider. Untested.
  - Jetson Orin (sm_87), GB200 (sm_100a) and GB300 (sm_103a) have none, so the GPU runner is
    expected to fail to start. With `OLLAYA_DEVICE=auto` the daemon then starts a CPU runner. With
    an explicit `cuda` device the request fails with an error. Untested.

  GGUF models are a separate case: llama.cpp's `libggml-cuda.so` carries PTX for sm_75, sm_80 and
  sm_90, so `cuda_kernels.rs` can admit some of those GPUs through JIT even when ONNX cannot.

  ADR 0004's "Turing through Blackwell" range is for the x86-64 packs. It does not apply to arm64.
- **No pre-flight check for ONNX models.** For GGUF models,
  `crates/ollaya-runner/src/llama/cuda_kernels.rs` compares the GPU with the kernels in
  `libggml-cuda.so` before it loads a model. The ONNX side has no such check. The protection is
  the scheduler's CPU retry under `auto` (`crates/ollaya-server/src/scheduler.rs`, `spawn`), which
  costs one extra runner start.
- **GPU and CPU runners use different ONNX Runtime builds.** On arm64, GPU runners run Microsoft's
  1.29.0 and CPU runners pyke's 1.28.0. GPU parity on the DGX Spark, with the tolerances unchanged,
  is the check for this pack. That check ran on a GB10 and passed, against the staged pack and
  with the pack's own ONNX Runtime:
  - `winnow:e4b` (GGUF, Q8_0), against stock `llama-server` b11146 on the same GPU, as ADR 0003
    requires: `PASS`. 123 cases, 505 questions, decisions 505 of 505, option log-probabilities
    within 1.233e-5 of the reference. `parity_llama` applies log-softmax to both sides first, so
    this is not the raw-logit difference that `parity_decider` reports.
  - `laya:en` fp32 (ONNX): 0 encoding mismatches, 483 questions, decisions all agree, 0 answers
    outside rounding, probabilities within 1.3e-5. Only encoding mismatches make that example
    exit non-zero, so the decision and answer results come from its printed summary.
  - `decider:2b` (ONNX): 0 rejection mismatches, 0 encoding mismatches, 479 questions over 902
    rows, raw label and option logits within 5.2e-5, decisions all agree.
  - `laya:en` fp16 (ONNX): 481 of 483 decisions, probabilities within 5.9e-2, and 374 of 483
    answers outside rounding. Decision and answer differences are reported here, not enforced;
    encoding agreement still is. The x86-64 reference machine scores 482 of 483 decisions on the
    same fixture and publishes no answer count to compare. Reduced precision changes decisions
    near ties, which is why these are reported rather than enforced.

  The two logit checks have room: the GGUF figure is about 81 times below its 1e-3 limit, and both
  `decider` figures about 19 times below theirs. The other criteria demand exact agreement and
  matched. The tolerances were not changed. The reference machines run ONNX Runtime 1.28.2 against
  1.29.0 here, so these numbers do not rank hardware.
- **The base archive grows by one executable** on linux-arm64, as on x86-64.
- **Bumping `ort`.** When `ort` moves to ONNX Runtime 1.x, the arm64 pack moves to the
  `onnxruntime-gpu` 1.x wheel (1.29.0 or newer), and its parity runs again. When the arm64 version
  matches the x86-64 one again, the exception ends.
- **Docker.** There is no arm64 `:cuda` image. The same staging supports one when it is wanted.
