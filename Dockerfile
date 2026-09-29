# syntax=docker/dockerfile:1

# Ollaya container images. Four targets:
#   cpu (default)  linux/amd64 and linux/arm64   docker build -t ollaya .
#   cuda           linux/amd64                   docker build --target cuda -t ollaya:cuda .
#   cuda12         linux/amd64                   docker build --target cuda12 -t ollaya:cuda12 .
#   rocm           linux/amd64                   docker build --target rocm -t ollaya:rocm .
# All run `ollaya serve` as the unprivileged user ollaya (uid 1000), listen on 0.0.0.0:11435 and
# keep models in the volume /home/ollaya/.ollaya. The cuda images need the NVIDIA Container
# Toolkit and a host driver with CUDA 13 support (R580 or newer) for cuda, or CUDA 12 support
# (R525 or newer) for cuda12:
#   docker run -d --gpus=all -v ollaya:/home/ollaya/.ollaya -p 11435:11435 ghcr.io/ollaya-dev/ollaya:cuda
# The rocm image needs the AMDGPU driver (/dev/kfd and /dev/dri):
#   docker run -d --device=/dev/kfd --device=/dev/dri -v ollaya:/home/ollaya/.ollaya -p 11435:11435 ghcr.io/ollaya-dev/ollaya:rocm
# See docs/distribution.md.

ARG RUST_VERSION=1.98
ARG DEBIAN_RELEASE=trixie
ARG UV_VERSION=0.12.15
ARG ROCM_VERSION=6.3

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# --- build: compile ollaya and stage the release file tree -------------------------------------

# The full rust image already has g++ (ORT links libstdc++), unzip, file and binutils, which the
# build and scripts/package.sh need, so the builder installs nothing.
FROM rust:${RUST_VERSION}-${DEBIAN_RELEASE} AS build
ARG TARGETARCH
ARG OLLAYA_BUILD_VERSION=0.0.0-dev
WORKDIR /src
COPY . .
# One binary serves both images. On amd64 it is built with ORT's CUDA provider bridge, as in the
# release tarball; it runs on the CPU unless the CUDA libraries are present. amd64 also builds
# the GPU runner, lib/ollaya/ollaya-cuda-runner, which loads Microsoft's ONNX Runtime from the
# CUDA pack (docs/decisions/0004-cuda-onnxruntime-builds.md).
RUN --mount=type=cache,id=ollaya-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=ollaya-ort-${TARGETARCH},target=/root/.cache/ort.pyke.io,sharing=locked \
    --mount=type=cache,id=ollaya-target-${TARGETARCH},target=/src/target,sharing=locked \
    set -eu; \
    case "${TARGETARCH}" in \
        amd64) features=ollaya-runner/cuda ;; \
        arm64) features= ;; \
        *) echo "unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    OLLAYA_BUILD_VERSION="${OLLAYA_BUILD_VERSION}" \
        cargo build --release --locked -p ollaya ${features:+--features "$features"}; \
    mkdir -p /out; \
    cp target/release/ollaya /out/; \
    if [ "${TARGETARCH}" = amd64 ]; then \
        OLLAYA_BUILD_VERSION="${OLLAYA_BUILD_VERSION}" \
            cargo build --release --locked -p ollaya --features ollaya-runner/cuda-dynamic \
            --target-dir target/cuda-dynamic; \
        cp target/cuda-dynamic/release/ollaya /out/ollaya-cuda-runner; \
        export OLLAYA_CUDA_RUNNER=/out/ollaya-cuda-runner; \
        OLLAYA_BUILD_VERSION="${OLLAYA_BUILD_VERSION}" \
            cargo build --release --locked -p ollaya --features ollaya-runner/rocm-dynamic \
            --target-dir target/rocm-dynamic; \
        cp target/rocm-dynamic/release/ollaya /out/ollaya-rocm-runner; \
        export OLLAYA_ROCM_RUNNER=/out/ollaya-rocm-runner; \
    fi; \
    OLLAYA_CARGO_FEATURES="$features" scripts/package.sh --platform "linux-${TARGETARCH}" \
        --stage /stage /out "${OLLAYA_BUILD_VERSION}"; \
    mv "/stage/ollaya-linux-${TARGETARCH}" /stage/base

# Microsoft's ONNX Runtime CUDA build and the NVIDIA CUDA and cuDNN libraries from NVIDIA's
# wheels, exactly as in the -cuda tarball.
FROM build AS build-cuda
ARG OLLAYA_BUILD_VERSION=0.0.0-dev
COPY --from=uv /uv /uvx /usr/local/bin/
RUN --mount=type=cache,id=ollaya-nvidia-wheels,target=/root/.cache/ollaya-package/wheels,sharing=locked \
    scripts/package.sh --platform linux-amd64 --cuda --no-base --stage /stage /out \
        "${OLLAYA_BUILD_VERSION}"

# The same for CUDA 12 (Microsoft's cuda12 build), exactly as in the -cuda12 tarball.
FROM build AS build-cuda12
ARG OLLAYA_BUILD_VERSION=0.0.0-dev
COPY --from=uv /uv /uvx /usr/local/bin/
RUN --mount=type=cache,id=ollaya-nvidia-wheels,target=/root/.cache/ollaya-package/wheels,sharing=locked \
    scripts/package.sh --platform linux-amd64 --cuda12 --no-base --stage /stage /out \
        "${OLLAYA_BUILD_VERSION}"

# Optional HIP build environment container targeting key AMD architectures.
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION} AS rocm-builder
ARG HIP_PLATFORM=amd
ARG AMDGPU_TARGETS="gfx1100,gfx1101,gfx1102,gfx1200,gfx1201,gfx90a,gfx942"
ENV HIP_PLATFORM=${HIP_PLATFORM} \
    AMDGPU_TARGETS=${AMDGPU_TARGETS}

# The ROCm package: llama.cpp's ROCm/HIP backend and ONNX Runtime ROCm libraries,
# staged as in the -rocm tarball.
FROM build AS build-rocm
ARG OLLAYA_BUILD_VERSION=0.0.0-dev
RUN scripts/package.sh --platform linux-amd64 --rocm --no-base --stage /stage /out \
        "${OLLAYA_BUILD_VERSION}"

# --- runtime ---------------------------------------------------------------------------------

# Debian 13: pyke's ONNX Runtime needs glibc 2.38 or newer. bash (for the health check),
# libstdc++ and zlib (for cuDNN) are part of the slim image already; llama.cpp's libraries (GGUF
# models) also need GCC's OpenMP runtime, libgomp1.
FROM debian:${DEBIAN_RELEASE}-slim AS runtime
ARG OLLAYA_UID=1000
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid "${OLLAYA_UID}" ollaya \
    && useradd --uid "${OLLAYA_UID}" --gid ollaya --home-dir /home/ollaya --create-home \
        --shell /usr/sbin/nologin ollaya \
    && mkdir -p /home/ollaya/.ollaya/models \
    && chown -R ollaya:ollaya /home/ollaya/.ollaya
# CA roots for pulling models over HTTPS, without apt in this stage.
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --chmod=0755 packaging/docker-healthcheck.sh /usr/local/bin/ollaya-healthcheck
ENV HOME=/home/ollaya \
    OLLAYA_HOST=0.0.0.0:11435 \
    OLLAYA_MODELS=/home/ollaya/.ollaya/models
EXPOSE 11435
VOLUME ["/home/ollaya/.ollaya"]
WORKDIR /home/ollaya
# Numeric, so runtimes such as Kubernetes can tell it is not root.
USER ${OLLAYA_UID}:${OLLAYA_UID}
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["/usr/local/bin/ollaya-healthcheck"]
ENTRYPOINT ["/usr/bin/ollaya"]
CMD ["serve"]

FROM runtime AS cuda
COPY --from=build-cuda /stage/base/ /usr/
COPY --from=build-cuda /stage/ollaya-linux-amd64-cuda/ /usr/
# Read by the NVIDIA Container Toolkit, which mounts the host driver (libcuda.so.1) at run time.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

FROM runtime AS cuda12
COPY --from=build-cuda12 /stage/base/ /usr/
COPY --from=build-cuda12 /stage/ollaya-linux-amd64-cuda12/ /usr/
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

FROM runtime AS rocm
COPY --from=build-rocm /stage/base/ /usr/
COPY --from=build-rocm /stage/ollaya-linux-amd64-rocm/ /usr/
ENV ROCM_PATH=/opt/rocm \
    HIP_VISIBLE_DEVICES=all

FROM runtime AS cpu
COPY --from=build /stage/base/ /usr/
