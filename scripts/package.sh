#!/bin/sh
# Build Ollaya release archives from a cargo target directory.
#
#   scripts/package.sh [options] <target-dir> <version>
#   scripts/package.sh --checksums <dir>
#
# Archives, written to --out (default: ./dist):
#   ollaya-<platform>.tar.zst        bin/ollaya + lib/ollaya/llama/ (llama.cpp's libraries: CPU, and Metal
#                                    on macOS) + share/doc/ollaya/ (LICENSE, THIRD_PARTY_NOTICES)
#                                    and on x86-64 Linux and Windows, and on linux-arm64 with --cuda,
#                                    lib/ollaya/ollaya-cuda-runner (the GPU runner, see
#                                    OLLAYA_CUDA_RUNNER)
#   ollaya-linux-amd64-cuda.tar.zst  lib/ollaya/cuda_v13/ (Microsoft's ONNX Runtime CUDA build +
#                                    NVIDIA CUDA/cuDNN libraries + libggml-cuda.so, llama.cpp's
#                                    CUDA backend) + share/doc/ollaya/cuda_v13/ (notices, licenses)
#   ollaya-linux-arm64-cuda.tar.zst  the same for aarch64 (DGX Spark), with ONNX Runtime from the
#                                    onnxruntime-gpu wheel (docs/decisions/0005-arm64-cuda-pack.md)
#   ollaya-darwin-arm64.tgz          same content as the darwin .tar.zst; stock macOS has no zstd
#   ollaya-darwin-arm64-mlx.tar.zst  lib/ollaya/mlx_metal/ (mlx.metallib, the MLX engine's Metal
#                                    kernels) + share/doc/ollaya/mlx_metal/ (notices); also .tgz
#   ollaya-windows-amd64.zip         bin/ollaya.exe (with the DLLs it links), lib/ollaya/llama/ (CPU),
#                                    share/
#   ollaya-windows-amd64-cuda.zip    the same GPU pack as the Linux one, with the Windows DLLs
#   ollaya-<platform>-cuda12.*       the CUDA 12 pack, lib/ollaya/cuda_v12/, for drivers older than
#                                    R580: the same files built for CUDA 12 (.tar.zst or .zip)
#   ollaya-<platform>-cuda.sha256    sha256 of every library in the CUDA archive (FILES.sha256),
#                                    which the installers use to skip an unchanged CUDA download
#   ollaya-<platform>-cuda12.sha256  the same for the CUDA 12 archive
#   ollaya-darwin-arm64-mlx.sha256   the same for the MLX archive
#   sha256sum.txt                    over every archive in --out, and the files above
#
# Options:
#   --platform P  linux-amd64 | linux-arm64 | darwin-arm64 | windows-amd64 (default: this host)
#   --cuda        also build the CUDA archive (linux-amd64, linux-arm64 and windows-amd64), with
#                 Microsoft's ONNX Runtime GPU build (docs/decisions/0004-cuda-onnxruntime-builds.md,
#                 0005-arm64-cuda-pack.md). Its runner is the base archive's
#                 lib/ollaya/ollaya-cuda-runner.
#   --cuda12      also build the CUDA 12 archive (linux-amd64 and windows-amd64), with Microsoft's
#                 cuda12 build. The same runner loads either pack.
#   --mlx         also build the MLX archive (darwin-arm64 only). The binary must have been built
#                 with `--features ollaya-runner/mlx`, whose build script puts mlx.metallib in
#                 <target-dir> (docs/decisions/0001-mlx-engine.md).
#   --no-base     skip the base archive (only useful with --cuda, --cuda12 or --mlx)
#   --stage DIR   stage the file trees into DIR/<archive name>/ and stop: no archives, no checksums
#                 (the Dockerfile uses this)
#   --out DIR     output directory (default: dist)
#   --checksums D only (re)write D/sha256sum.txt
#
# Environment:
#   OLLAYA_BIN             binary to package (default: <target-dir>/ollaya)
#   OLLAYA_CUDA_RUNNER     ollaya built with `--features ollaya-runner/cuda-dynamic`; the base
#                          archive ships it as lib/ollaya/ollaya-cuda-runner, which GPU runners
#                          start from (required for the base archive of linux-amd64 and
#                          windows-amd64, and of linux-arm64 with --cuda)
#   OLLAYA_CARGO_PACKAGE   package whose dependency tree is listed in the notices (default: ollaya)
#   OLLAYA_CARGO_FEATURES  cargo features of the build (default: ollaya-runner/cuda on linux-amd64
#                          and windows-amd64, ollaya-runner/coreml on darwin-arm64, plus
#                          ollaya-runner/mlx with --mlx, none on linux-arm64)
#   OLLAYA_CACHE           download cache (default: ${XDG_CACHE_HOME:-~/.cache}/ollaya-package)
#   PYTHON                 interpreter used for `pip download` (default: python3 or python, else
#                          `uvx pip`)
#   ZSTD_LEVEL             zstd level (default: 19)
#   SOURCE_DATE_EPOCH      timestamp for archive entries (default: last commit, else now)

set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)

# ONNX Runtime version inside ort-sys (=2.0.0-rc.13 pins ORT 1.28.0). Bump together with `ort`.
ORT_VERSION=1.28.0
ORT_LICENSE_SHA256=2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c
ORT_NOTICES_SHA256=0e07b95f3a8d6230037707c5c4a2b554d12c4cb67369669ac255635528ffcee2
# The CUDA pack runs Microsoft's ONNX Runtime GPU release instead, unmodified: pyke's CUDA build
# has no kernels for sm_120 (docs/decisions/0004-cuda-onnxruntime-builds.md). Same minor version
# as ORT_VERSION; bump together. The one exception is linux-arm64, pinned below.
ORT_GPU_VERSION=1.28.2
ORT_GPU_LINUX_SHA256=118ca8dbc4e4bb9b3b7fea137d796a89d957c9aa70e1dc3a5199a302cdd5bb32
ORT_GPU_WINDOWS_SHA256=4b7a2d01a3cc96b12d06c8266af2c8f42c96365c4a0100d45fd874c71b4a2e19
ORT_GPU12_LINUX_SHA256=e172d4d52bc4399ca36553bd9705389adae900b1f1e08ade50078db1c84b6c1f
ORT_GPU12_WINDOWS_SHA256=5b5ceb06e90405c7de9acdaf5aa06d288768e6a1e5dc54337281e09c03fc60f9
# linux-arm64 (DGX Spark): Microsoft ships no aarch64 GPU archive, so the pack takes the same three
# libraries from the onnxruntime-gpu wheel on PyPI, whose aarch64 builds start at 1.29.0
# (docs/decisions/0005-arm64-cuda-pack.md).
ORT_GPU_ARM64_VERSION=1.29.0
ORT_GPU_ARM64_URL=https://files.pythonhosted.org/packages/fa/96/1be0b9711a614861fbf5c0c85c4da3aa2321f9da0b4734a717eb606f70ab/onnxruntime_gpu-1.29.0-cp312-cp312-manylinux_2_34_aarch64.whl
ORT_GPU_ARM64_SHA256=545d2966dd11208bbc98e67be4d92e54ecd793acc45c0532cc7bfd36f50692fa

# MLX (docs/decisions/0001-mlx-engine.md): the pins in crates/ollaya-mlx-sys/build.rs, and the
# notices of what the `mlx` feature links into bin/ollaya. MLX's ACKNOWLEDGMENTS.md carries the
# licenses of the code it includes (PocketFFT, metal-cpp); its CMake build also compiles in {fmt}
# and nlohmann/json, header-only.
MLX_VERSION=0.32.2
MLX_C_COMMIT=ebc88f10caa1b625e6b581437a8dea6df8a70085
MLX_LICENSE_SHA256=ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01
MLX_ACKNOWLEDGMENTS_SHA256=fbf1078b24ca057b31fef9297390bbe96e001258ecf4c0b108394c764e607799
MLX_C_LICENSE_SHA256=44326a4ea062241ae6fc26ee2ec90bdc81af7eb7b9d3966181b733fa69d42057
FMT_VERSION=12.1.0
FMT_LICENSE_SHA256=07580f2a3b35709ce703d523f447b242f6dfec7582a8c0df102c7fa2849375f8
JSON_VERSION=3.11.3
JSON_LICENSE_SHA256=86b998c792894ccb911a1cb7994f7a9652894e7a094c0b5e45be2f553f45cf14
# llama.cpp, which runs GGUF models, comes from scripts/llama-cpp.sh (its build and
# pins live there).
LLAMA_CPP_BUILD=$(sed -n 's/^LLAMA_CPP_BUILD=//p' "$ROOT/scripts/llama-cpp.sh")


say() { printf '>>> %s\n' "$*" >&2; }
die() { printf 'package.sh: error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

usage() {
    sed -n '2,/^$/s/^# \{0,1\}//p' "$0" >&2
    exit 2
}

sha256_of() {
    if have sha256sum; then
        sha256sum "$1" | cut -d' ' -f1
    elif have shasum; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        openssl dgst -sha256 -r "$1" | cut -d' ' -f1
    fi
}

human_size() {
    # MiB with one decimal; `du -h` rounds differently on GNU and BSD.
    wc -c <"$1" | awk '{ printf "%.1f MiB", $1 / 1048576 }'
}

write_checksums() {
    dir=$1
    [ -d "$dir" ] || die "no such directory: $dir"
    (
        cd "$dir"
        LC_ALL=C
        export LC_ALL
        : >sha256sum.txt.tmp
        for f in *; do
            case $f in
                *.tar.zst | *.tgz | *.zip | *-cuda.sha256 | *-cuda12.sha256 | *-mlx.sha256) [ -f "$f" ] || continue ;;
                *) continue ;;
            esac
            printf '%s  %s\n' "$(sha256_of "$f")" "$f" >>sha256sum.txt.tmp
        done
        [ -s sha256sum.txt.tmp ] || { rm -f sha256sum.txt.tmp; die "no archives in $dir"; }
        mv sha256sum.txt.tmp sha256sum.txt
    )
    say "Wrote $dir/sha256sum.txt"
}

# fetch URL DEST SHA256: download into the cache once, verify every time.
fetch() {
    url=$1 dest=$2 want=$3
    if [ ! -f "$dest" ] || [ "$(sha256_of "$dest")" != "$want" ]; then
        mkdir -p "$(dirname "$dest")"
        curl -fsSL --retry 3 -o "$dest.part" "$url" || die "download failed: $url"
        mv "$dest.part" "$dest"
    fi
    got=$(sha256_of "$dest")
    [ "$got" = "$want" ] || die "sha256 mismatch for $url: got $got, want $want"
}

# --- arguments ---------------------------------------------------------------------------------

PLATFORM='' CUDA=0 CUDA12=0 MLX=0 BASE=1 STAGE='' OUT=dist
while [ $# -gt 0 ]; do
    case $1 in
        --platform) [ $# -ge 2 ] || usage; PLATFORM=$2; shift 2 ;;
        --platform=*) PLATFORM=${1#*=}; shift ;;
        --cuda) CUDA=1; shift ;;
        --cuda12) CUDA12=1; shift ;;
        --mlx) MLX=1; shift ;;
        --no-base) BASE=0; shift ;;
        --stage) [ $# -ge 2 ] || usage; STAGE=$2; shift 2 ;;
        --out) [ $# -ge 2 ] || usage; OUT=$2; shift 2 ;;
        --checksums) [ $# -eq 2 ] || usage; write_checksums "$2"; exit 0 ;;
        -h | --help) usage ;;
        --) shift; break ;;
        -*) die "unknown option: $1" ;;
        *) break ;;
    esac
done
[ $# -eq 2 ] || usage
TARGET_DIR=$1
VERSION=${2#v}
case $VERSION in '' | *[!0-9A-Za-z.+-]*) die "bad version: $2" ;; esac
[ -d "$TARGET_DIR" ] || die "no such target directory: $TARGET_DIR"
TARGET_DIR=$(CDPATH='' cd -- "$TARGET_DIR" && pwd)

if [ -z "$PLATFORM" ]; then
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) PLATFORM=linux-amd64 ;;
        Linux-aarch64 | Linux-arm64) PLATFORM=linux-arm64 ;;
        Darwin-arm64) PLATFORM=darwin-arm64 ;;
        MINGW*-x86_64 | MSYS*-x86_64) PLATFORM=windows-amd64 ;;
        *) die "unsupported host $(uname -s)-$(uname -m); pass --platform" ;;
    esac
fi
case $PLATFORM in
    linux-amd64) TRIPLE=x86_64-unknown-linux-gnu DEFAULT_FEATURES=ollaya-runner/cuda ;;
    linux-arm64) TRIPLE=aarch64-unknown-linux-gnu DEFAULT_FEATURES= ;;
    darwin-arm64) TRIPLE=aarch64-apple-darwin DEFAULT_FEATURES=ollaya-runner/coreml ;;
    windows-amd64) TRIPLE=x86_64-pc-windows-msvc DEFAULT_FEATURES=ollaya-runner/cuda ;;
    *) die "unknown platform: $PLATFORM" ;;
esac
[ "$MLX" = 0 ] || DEFAULT_FEATURES=$DEFAULT_FEATURES,ollaya-runner/mlx
FEATURES=${OLLAYA_CARGO_FEATURES-$DEFAULT_FEATURES}
case $PLATFORM in
    linux-amd64 | windows-amd64) ;;
    # The only aarch64 ONNX Runtime GPU build is one onnxruntime-gpu wheel per version, and it links
    # libcudart.so.13: there is no CUDA 12 pack for linux-arm64.
    linux-arm64) [ "$CUDA12" = 0 ] || die "--cuda12 is not supported for linux-arm64 (there is no aarch64 CUDA 12 build of ONNX Runtime); use --cuda" ;;
    *) [ "$CUDA$CUDA12" = 00 ] || die "--cuda is only supported for linux-amd64, linux-arm64 and windows-amd64, --cuda12 for linux-amd64 and windows-amd64" ;;
esac
[ "$MLX" = 0 ] || [ "$PLATFORM" = darwin-arm64 ] || die "--mlx is only supported for darwin-arm64"
[ "$BASE" = 1 ] || [ "$CUDA" = 1 ] || [ "$CUDA12" = 1 ] || [ "$MLX" = 1 ] ||
    die "--no-base without --cuda, --cuda12 or --mlx leaves nothing to do"
# The `mlx` feature links MLX into bin/ollaya, so its notices go into the base archive too.
case ",$FEATURES," in *mlx,*) LINKS_MLX=1 ;; *) LINKS_MLX=0 ;; esac

EXE=
[ "$PLATFORM" != windows-amd64 ] || EXE=.exe

# The CUDA packs, lib/ollaya/cuda_v13 and lib/ollaya/cuda_v12: Microsoft's ONNX Runtime CUDA 13 or
# CUDA 12 build (the library and its shared and CUDA providers), and the NVIDIA libraries kept from
# the wheels in packaging/cuda-requirements.txt or packaging/cuda12-requirements.txt (see
# is_cuda_lib). Everything else in the wheels (static and import libs, headers, cufftw, nvblas,
# nvrtc .alt builds) is dropped. The TensorRT and NV TensorRT RTX providers are left out: they need
# TensorRT 10 (libnvinfer, nvinfer_10.dll), which is not shipped, and the runner only registers the
# CUDA execution provider.
ORT_LIBRARY=libonnxruntime.so.1
ORT_PROVIDERS="libonnxruntime_providers_shared.so libonnxruntime_providers_cuda.so"
WHEEL_LIB_DIR=lib
case $PLATFORM in
    windows-amd64)
        ORT_LIBRARY=onnxruntime.dll
        ORT_PROVIDERS="onnxruntime_providers_shared.dll onnxruntime_providers_cuda.dll"
        WHEEL_TAG=win_amd64
        WHEEL_PLATFORMS=win_amd64
        # Windows wheels keep their DLLs in nvidia/*/bin/, Linux wheels in nvidia/*/lib/.
        WHEEL_LIB_DIR=bin
        ;;
    linux-arm64)
        WHEEL_TAG=aarch64
        WHEEL_PLATFORMS="manylinux_2_28_aarch64 manylinux_2_27_aarch64 manylinux_2_17_aarch64
manylinux2014_aarch64"
        ;;
    *)
        WHEEL_TAG=x86_64
        WHEEL_PLATFORMS="manylinux_2_28_x86_64 manylinux_2_27_x86_64 manylinux_2_17_x86_64
manylinux2014_x86_64 manylinux_2_12_x86_64 manylinux2010_x86_64"
        ;;
esac

# cuda_pack MAJOR: set up the variables of the CUDA MAJOR pack (13 or 12). cuFFT's soname runs one
# major behind CUDA's (cufft 12 in CUDA 13, cufft 11 in CUDA 12).
cuda_pack() {
    M=$1
    CUDA_PACK=cuda_v$M
    case $M in
        13) CUDA_NAME=ollaya-$PLATFORM-cuda CUFFT=12 LLAMA_CUDA_KIND=${PLATFORM}-cuda
            CUDA_REQUIREMENTS=$ROOT/packaging/cuda-requirements.txt ;;
        12) CUDA_NAME=ollaya-$PLATFORM-cuda12 CUFFT=11 LLAMA_CUDA_KIND=${PLATFORM}-cuda12
            CUDA_REQUIREMENTS=$ROOT/packaging/cuda12-requirements.txt ;;
        *) die "no CUDA $M pack" ;;
    esac
    # ORT_GPU_URL: where the ONNX Runtime GPU build comes from. ORT_GPU_VER: its version.
    # ORT_GPU_DIR: the directory it unpacks to, which holds LICENSE and ThirdPartyNotices.txt.
    # ORT_GPU_LIBS: the directory of its libraries. ORT_GPU_CORE: the core library's name there,
    # copied to ORT_LIBRARY. Microsoft's GitHub archive, unless the platform says otherwise below.
    ORT_GPU_VER=$ORT_GPU_VERSION
    ORT_GPU_CORE=$ORT_LIBRARY
    CUDA_LIBS_REQUIRED="libcudart.so.$M libcublas.so.$M libcublasLt.so.$M libcufft.so.$CUFFT libcurand.so.10
libnvrtc.so.$M libnvJitLink.so.$M libcudnn.so.9 libcudnn_graph.so.9"
    case $PLATFORM in
        windows-amd64)
            ORT_GPU_ARCHIVE=onnxruntime-win-x64-gpu_cuda$M-$ORT_GPU_VER.zip
            ORT_GPU_SHA256=$ORT_GPU_WINDOWS_SHA256
            [ "$M" = 13 ] || ORT_GPU_SHA256=$ORT_GPU12_WINDOWS_SHA256
            CUDA_LIBS_REQUIRED="cudart64_$M.dll cublas64_$M.dll cublasLt64_$M.dll cufft64_$CUFFT.dll curand64_10.dll
nvrtc64_${M}0_0.dll nvJitLink_${M}0_0.dll cudnn64_9.dll cudnn_graph64_9.dll"
            ;;
        linux-arm64)
            # Microsoft's aarch64 GPU build ships only as a wheel (a zip), with the libraries in
            # onnxruntime/capi/ and the core one under its full version (libonnxruntime.so.1.29.0);
            # the loader asks for its soname (docs/decisions/0005-arm64-cuda-pack.md).
            ORT_GPU_VER=$ORT_GPU_ARM64_VERSION
            ORT_GPU_URL=$ORT_GPU_ARM64_URL
            ORT_GPU_ARCHIVE=${ORT_GPU_URL##*/}
            ORT_GPU_SHA256=$ORT_GPU_ARM64_SHA256
            ORT_GPU_DIR=onnxruntime
            ORT_GPU_LIBS=onnxruntime/capi
            ORT_GPU_CORE=libonnxruntime.so.$ORT_GPU_VER
            return
            ;;
        *)
            ORT_GPU_ARCHIVE=onnxruntime-linux-x64-gpu_cuda$M-$ORT_GPU_VER.tgz
            ORT_GPU_SHA256=$ORT_GPU_LINUX_SHA256
            [ "$M" = 13 ] || ORT_GPU_SHA256=$ORT_GPU12_LINUX_SHA256
            ;;
    esac
    ORT_GPU_URL=https://github.com/microsoft/onnxruntime/releases/download/v$ORT_GPU_VER/$ORT_GPU_ARCHIVE
    ORT_GPU_DIR=${ORT_GPU_ARCHIVE%.*}
    ORT_GPU_LIBS=$ORT_GPU_DIR/lib
}

# is_cuda_lib NAME: whether the wheel file NAME goes into the pack cuda_pack set up.
is_cuda_lib() {
    case $PLATFORM:$1 in
        linux-*:libcudart.so."$M" | linux-*:libcublas.so."$M" | linux-*:libcublasLt.so."$M" | \
            linux-*:libcufft.so."$CUFFT" | linux-*:libcurand.so.10 | linux-*:libnvrtc.so."$M" | \
            linux-*:libnvJitLink.so."$M" | linux-*:libcudnn*.so.9 | linux-*:libnvrtc-builtins.so."$M".*) ;;
        windows-*:cudart64_"$M".dll | windows-*:cublas64_"$M".dll | windows-*:cublasLt64_"$M".dll | \
            windows-*:cufft64_"$CUFFT".dll | windows-*:curand64_10.dll | windows-*:nvrtc64_"$M"0_0.dll | \
            windows-*:nvJitLink_"$M"0_0.dll | windows-*:cudnn*64_9.dll | windows-*:nvrtc-builtins64_"$M"*.dll) ;;
        *) return 1 ;;
    esac
}
BIN=${OLLAYA_BIN:-$TARGET_DIR/ollaya$EXE}
CARGO_PACKAGE=${OLLAYA_CARGO_PACKAGE:-ollaya}
CACHE=${OLLAYA_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/ollaya-package}
ZSTD_LEVEL=${ZSTD_LEVEL:-19}
if [ -z "${SOURCE_DATE_EPOCH:-}" ]; then
    SOURCE_DATE_EPOCH=$(git -C "$ROOT" log -1 --format=%ct 2>/dev/null || date +%s)
fi

for tool in curl tar; do have "$tool" || die "missing tool: $tool"; done
if [ -z "$STAGE" ] && [ "${PLATFORM%%-*}" != windows ]; then have zstd || die "missing tool: zstd"; fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/ollaya-package.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
trap 'exit 130' INT TERM

if [ -n "$STAGE" ]; then
    mkdir -p "$STAGE"
    STAGE=$(CDPATH='' cd -- "$STAGE" && pwd)
    TREES=$STAGE
else
    mkdir -p "$OUT"
    OUT=$(CDPATH='' cd -- "$OUT" && pwd)
    TREES=$WORK/trees
fi

# --- ONNX Runtime notices ----------------------------------------------------------------------

ORT_RAW=https://raw.githubusercontent.com/microsoft/onnxruntime/v$ORT_VERSION
fetch "$ORT_RAW/LICENSE" "$CACHE/onnxruntime-$ORT_VERSION/LICENSE" "$ORT_LICENSE_SHA256"
fetch "$ORT_RAW/ThirdPartyNotices.txt" "$CACHE/onnxruntime-$ORT_VERSION/ThirdPartyNotices.txt" \
    "$ORT_NOTICES_SHA256"

ort_notice() {
    cat <<EOF
ONNX Runtime $ORT_VERSION (https://github.com/microsoft/onnxruntime), prebuilt by pyke
(https://ort.pyke.io). $1
License: MIT. The components ONNX Runtime bundles are listed in
onnxruntime-ThirdPartyNotices.txt next to this file.

EOF
    sed 's/^/    /' "$CACHE/onnxruntime-$ORT_VERSION/LICENSE"
}

# --- MLX notices -------------------------------------------------------------------------------

if [ "$MLX" = 1 ] || [ "$LINKS_MLX" = 1 ]; then
    MLX_RAW=https://raw.githubusercontent.com/ml-explore
    fetch "$MLX_RAW/mlx/v$MLX_VERSION/LICENSE" "$CACHE/mlx-$MLX_VERSION/LICENSE" "$MLX_LICENSE_SHA256"
    fetch "$MLX_RAW/mlx/v$MLX_VERSION/ACKNOWLEDGMENTS.md" "$CACHE/mlx-$MLX_VERSION/ACKNOWLEDGMENTS.md" \
        "$MLX_ACKNOWLEDGMENTS_SHA256"
    fetch "$MLX_RAW/mlx-c/$MLX_C_COMMIT/LICENSE" "$CACHE/mlx-c-$MLX_C_COMMIT/LICENSE" "$MLX_C_LICENSE_SHA256"
    fetch "https://raw.githubusercontent.com/fmtlib/fmt/$FMT_VERSION/LICENSE" \
        "$CACHE/fmt-$FMT_VERSION/LICENSE" "$FMT_LICENSE_SHA256"
    fetch "https://raw.githubusercontent.com/nlohmann/json/v$JSON_VERSION/LICENSE.MIT" \
        "$CACHE/json-$JSON_VERSION/LICENSE.MIT" "$JSON_LICENSE_SHA256"
fi

# mlx_notice WHERE: MLX, mlx-c and what their build compiles in, with every license text.
mlx_notice() {
    cat <<EOF
MLX $MLX_VERSION (https://github.com/ml-explore/mlx) and mlx-c $MLX_C_COMMIT
(https://github.com/ml-explore/mlx-c), built from source. $1
License: MIT (both). MLX's build also compiles in {fmt} $FMT_VERSION (MIT) and nlohmann/json
$JSON_VERSION (MIT), and MLX includes PocketFFT (BSD-3-Clause) and metal-cpp (Apache-2.0), whose
notices follow MLX's own below.

MLX:

EOF
    sed 's/^/    /' "$CACHE/mlx-$MLX_VERSION/LICENSE"
    printf '\nmlx-c:\n\n'
    sed 's/^/    /' "$CACHE/mlx-c-$MLX_C_COMMIT/LICENSE"
    printf '\n{fmt}:\n\n'
    sed 's/^/    /' "$CACHE/fmt-$FMT_VERSION/LICENSE"
    printf '\nnlohmann/json:\n\n'
    sed 's/^/    /' "$CACHE/json-$JSON_VERSION/LICENSE.MIT"
    printf "\nMLX's third-party software (from its ACKNOWLEDGMENTS.md):\n\n"
    sed -n '/^# Third-Party Software/,$p' "$CACHE/mlx-$MLX_VERSION/ACKNOWLEDGMENTS.md" | sed 's/^/    /'
}

# --- Rust dependency notices -------------------------------------------------------------------

# Lists every crate compiled into the binary (normal dependencies of $CARGO_PACKAGE for $TRIPLE)
# with its license, then each distinct license text once, with the crates that ship it. License
# files are read from cargo's registry sources, which the build has already downloaded.
rust_notices() {
    have cargo || die "missing tool: cargo (needed to list Rust dependencies)"
    d=$WORK/rust
    registry=${CARGO_HOME:-$HOME/.cargo}/registry/src
    mkdir -p "$d/texts"
    set -- --locked -p "$CARGO_PACKAGE" --target "$TRIPLE" -e normal
    [ -z "$FEATURES" ] || set -- "$@" --features "$FEATURES"
    (cd "$ROOT" && cargo tree "$@" --prefix none --format '{p}|{l}|{r}') >"$d/tree" ||
        die "cargo tree failed for package $CARGO_PACKAGE (set OLLAYA_CARGO_PACKAGE?)"
    # "name vX.Y.Z[ (proc-macro)| (/path)]|license|repo[ (*)]". Path crates are this workspace.
    sed 's/ (\*)$//' "$d/tree" | awk -F '|' '$1 !~ / \(\// {
        split($1, p, " "); printf "%s|%s|%s|%s\n", p[1], substr(p[2], 2), $2, $3 }' |
        LC_ALL=C sort -u >"$d/crates"
    printf '%-32s %-14s %-36s %s\n' Crate Version License Repository
    # POSIX sh has no `local`: the loop variables are prefixed so they can't clobber the caller's.
    while IFS='|' read -r c_name c_version c_license c_repo; do
        printf '%-32s %-14s %-36s %s\n' "$c_name" "$c_version" "${c_license:-see license file}" "$c_repo"
        found=0
        for c_dir in "$registry"/*/"$c_name-$c_version"; do
            for c_file in "$c_dir"/LICENSE* "$c_dir"/LICENCE* "$c_dir"/COPYING* "$c_dir"/NOTICE* \
                "$c_dir"/UNLICENSE*; do
                [ -f "$c_file" ] || continue
                found=1
                h=$(sha256_of "$c_file")
                [ -f "$d/texts/$h" ] || cp "$c_file" "$d/texts/$h"
                printf '%s %s (%s)\n' "$c_name" "$c_version" "${c_file##*/}" >>"$d/texts/$h.users"
            done
            [ "$found" = 0 ] || break
        done
        [ "$found" = 1 ] || printf '%s %s\n' "$c_name" "$c_version" >>"$d/no-text"
    done <"$d/crates"
    count=$(wc -l <"$d/crates" | tr -d ' ')
    printf '\n%s crates. Their license texts follow; identical texts are printed once.\n' "$count"
    if [ -f "$d/no-text" ]; then
        printf 'These crates ship no license file; the license named above applies as published:\n'
        sed 's/^/  /' "$d/no-text"
    fi
    for users in "$d"/texts/*.users; do
        [ -f "$users" ] || continue
        printf '\n%s\n' '--------------------------------------------------------------------------------'
        sed 's/^/Used by: /' "$users"
        printf '\n'
        cat "${users%.users}"
    done
}

# --- staging -----------------------------------------------------------------------------------

# stage_runner ROOT: put OLLAYA_CUDA_RUNNER in ROOT/lib/ollaya/ollaya-cuda-runner.
stage_runner() {
    runner=${OLLAYA_CUDA_RUNNER:-}
    if [ -z "$runner" ] || [ ! -f "$runner" ]; then
        die "set OLLAYA_CUDA_RUNNER to ollaya built with --features ollaya-runner/cuda-dynamic"
    fi
    mkdir -p "$1/lib/ollaya"
    cp "$runner" "$1/lib/ollaya/ollaya-cuda-runner$EXE"
    chmod 0755 "$1/lib/ollaya/ollaya-cuda-runner$EXE"
}

stage_base() {
    name=ollaya-$PLATFORM
    root=$TREES/$name
    rm -rf "$root"
    mkdir -p "$root/bin" "$root/share/doc/ollaya"
    [ -f "$BIN" ] || die "binary not found: $BIN (build with cargo build --release -p ollaya)"
    if have file; then
        kind=$(file -bL "$BIN")
        case "$PLATFORM:$kind" in
            linux-amd64:*ELF*x86-64* | linux-arm64:*ELF*aarch64* | darwin-arm64:*Mach-O*arm64*) ;;
            windows-amd64:*PE32+*x86-64*) ;;
            *) die "$BIN is not a $PLATFORM executable: $kind" ;;
        esac
    fi
    if [ "${PLATFORM%%-*}" = linux ] && have readelf; then
        if readelf -d "$BIN" | grep -q 'NEEDED.*libonnxruntime\.so'; then
            die "$BIN links libonnxruntime.so dynamically; release builds link ORT statically"
        fi
        if have objdump; then
            glibc=$(objdump -T "$BIN" | grep -o 'GLIBC_[0-9.]*' | sort -t. -k1,1 -k2,2n -u | tail -n 1)
            say "$BIN needs ${glibc:-an unknown glibc} (install.sh enforces the floor)"
        fi
    fi
    cp "$BIN" "$root/bin/ollaya$EXE"
    chmod 0755 "$root/bin/ollaya$EXE"
    # The GPU runner: the same program, loading ONNX Runtime from the CUDA pack at run time. It is
    # here rather than in the pack so the pack keeps its FILES.sha256 from release to release.
    # linux-arm64 needs it only alongside a CUDA pack: the Dockerfile stages a CPU-only arm64 base
    # without one (its bin/ollaya has no ONNX Runtime CUDA support at all, ADR 0005).
    case $PLATFORM in
        linux-amd64 | windows-amd64) stage_runner "$root" ;;
        linux-arm64) [ "$CUDA" = 0 ] || stage_runner "$root" ;;
    esac
    # Windows: the DLLs ollaya.exe links (DirectML) sit next to it (copy-dylibs). ORT's provider
    # DLLs are loaded only on demand: the CUDA ones ship in the GPU pack, the others not at all.
    if [ -n "$EXE" ]; then
        for dll in "$TARGET_DIR"/*.dll; do
            case ${dll##*/} in onnxruntime_providers_*) continue ;; esac
            [ ! -f "$dll" ] || cp "$dll" "$root/bin/"
        done
    fi
    cp "$ROOT/LICENSE" "$root/share/doc/ollaya/LICENSE"
    # The agent skill (skills/ollaya-decisions), for agents on machines without the repository.
    mkdir -p "$root/share/ollaya/skills"
    cp -R "$ROOT/skills/ollaya-decisions" "$root/share/ollaya/skills/ollaya-decisions"
    cp "$CACHE/onnxruntime-$ORT_VERSION/ThirdPartyNotices.txt" \
        "$root/share/doc/ollaya/onnxruntime-ThirdPartyNotices.txt"
    # llama.cpp's libraries for GGUF models: the CPU backends (and Metal on macOS).
    OLLAYA_CACHE=$CACHE "$ROOT/scripts/llama-cpp.sh" "$PLATFORM" "$root/lib/ollaya/llama" \
        "$root/share/doc/ollaya/llama.cpp-THIRD_PARTY_NOTICES"
    {
        printf 'Ollaya %s (%s): third-party notices\n\n' "$VERSION" "$PLATFORM"
        printf 'Ollaya is licensed under the Apache License 2.0 (see LICENSE). bin/ollaya also\n'
        printf 'contains the third-party software below.\n\n'
        if [ -f "$root/lib/ollaya/ollaya-cuda-runner$EXE" ]; then
            printf 'lib/ollaya/ollaya-cuda-runner%s is the same program built to load ONNX Runtime from\n' "$EXE"
            printf 'the CUDA pack instead of linking it; it contains the Rust crates listed below.\n\n'
        fi
        printf '1. '
        ort_notice "Linked into bin/ollaya$EXE."
        printf '\n2. llama.cpp %s (MIT), in lib/ollaya/llama: see llama.cpp-THIRD_PARTY_NOTICES.\n' \
            "$LLAMA_CPP_BUILD"
        n=3
        if [ "$LINKS_MLX" = 1 ]; then
            printf '\n3. '
            mlx_notice "Linked into bin/ollaya$EXE (the MLX engine)."
            n=4
        fi
        printf '\n%s. Rust crates compiled into bin/ollaya\n\n' "$n"
        rust_notices
    } >"$root/share/doc/ollaya/THIRD_PARTY_NOTICES"
    say "Staged $name"
}

pip_download() {
    dest=$1
    set -- download --no-deps --only-binary=:all: --python-version 3.12
    for p in $WHEEL_PLATFORMS; do set -- "$@" --platform "$p"; done
    set -- "$@" --require-hashes -r "$CUDA_REQUIREMENTS" -d "$dest"
    # Windows has `python`, and often a `python3` that only opens the Microsoft Store.
    for py in ${PYTHON:-python3 python}; do
        if have "$py" && "$py" -m pip --version >/dev/null 2>&1; then
            "$py" -m pip --disable-pip-version-check -q "$@"
            return
        fi
    done
    have uvx || die "need pip (python3 -m pip) or uv (uvx) to download the NVIDIA wheels"
    uvx pip --disable-pip-version-check -q "$@"
}

# stage_cuda MAJOR: the CUDA MAJOR pack (13 or 12).
stage_cuda() {
    cuda_pack "$1"
    name=$CUDA_NAME
    root=$TREES/$name
    lib=$root/lib/ollaya/$CUDA_PACK
    doc=$root/share/doc/ollaya/$CUDA_PACK
    rm -rf "$root"
    mkdir -p "$lib" "$doc/licenses"
    have unzip || die "missing tool: unzip"

    # Microsoft's ONNX Runtime, byte for byte. The TensorRT provider in the archive is left out.
    fetch "$ORT_GPU_URL" "$CACHE/$ORT_GPU_ARCHIVE" "$ORT_GPU_SHA256"
    case $ORT_GPU_ARCHIVE in
        *.zip | *.whl) unzip -q "$CACHE/$ORT_GPU_ARCHIVE" -d "$WORK" ;;
        *) tar -xzf "$CACHE/$ORT_GPU_ARCHIVE" -C "$WORK" ;;
    esac
    ort=$WORK/$ORT_GPU_DIR
    ort_libs=$WORK/$ORT_GPU_LIBS
    for p in $ORT_PROVIDERS; do
        [ -e "$ort_libs/$p" ] || die "$p not found in $ORT_GPU_ARCHIVE"
        cp -L "$ort_libs/$p" "$lib/$p"
    done
    [ -e "$ort_libs/$ORT_GPU_CORE" ] || die "$ORT_GPU_CORE not found in $ORT_GPU_ARCHIVE"
    cp -L "$ort_libs/$ORT_GPU_CORE" "$lib/$ORT_LIBRARY"

    wheels=$CACHE/wheels
    mkdir -p "$wheels"
    say "Fetching NVIDIA wheels (about 1.2 GB on first run, cached in $wheels)"
    pip_download "$wheels"

    : >"$WORK/cuda-libs"
    sed -n 's/^\(nvidia-[a-z0-9-]*\)==\([^ ]*\).*/\1 \2/p' "$CUDA_REQUIREMENTS" \
        >"$WORK/cuda-pins"
    while read -r pname pver; do
        pkg=$pname==$pver
        wprefix=$(printf '%s' "$pname" | tr - _)-$pver-
        whl=
        for w in "$wheels/$wprefix"*"$WHEEL_TAG"*.whl; do [ -f "$w" ] && whl=$w; done
        [ -n "$whl" ] || die "wheel for $pkg missing from $wheels"
        for m in $(unzip -Z1 "$whl"); do
            base=${m##*/}
            case $m in
                */"$WHEEL_LIB_DIR"/*) ;;
                *.dist-info/licenses/* | *.dist-info/License.txt | *.dist-info/LICENSE*)
                    unzip -p "$whl" "$m" >"$doc/licenses/$pname-$base"
                    continue
                    ;;
                *) continue ;;
            esac
            is_cuda_lib "$base" || continue
            # Copied byte for byte: the NVIDIA EULA only allows redistribution of unmodified files.
            unzip -p "$whl" "$m" >"$lib/$base"
            chmod 0644 "$lib/$base"
            printf '%s\t%s\n' "$base" "$pkg" >>"$WORK/cuda-libs"
        done
    done <"$WORK/cuda-pins"
    for f in $CUDA_LIBS_REQUIRED; do
        [ -f "$lib/$f" ] || die "$f not found in the NVIDIA wheels"
    done
    builtins=
    for f in "$lib"/*nvrtc-builtins*; do [ ! -f "$f" ] || builtins=$f; done
    [ -n "$builtins" ] || die "the NVRTC builtins library was not found in the NVIDIA wheels"
    # llama.cpp's CUDA backend (GGUF models), which finds the CUDA libraries next to it. Every
    # pack has one except the Windows CUDA 12 pack: ggml-org's CUDA 12.4 build would take the zip
    # past GitHub's 2 GiB limit per release asset, so GGUF models use Vulkan or the CPU there.
    # linux-arm64 takes ggml-org's CUDA 13.4 arm64 build, with the same kernels as the x86-64 one.
    LLAMA_CUDA=
    case $LLAMA_CUDA_KIND in
        linux-amd64-cuda | linux-amd64-cuda12 | linux-arm64-cuda) LLAMA_CUDA=libggml-cuda.so ;;
        windows-amd64-cuda) LLAMA_CUDA=ggml-cuda.dll ;;
    esac
    if [ -n "$LLAMA_CUDA" ]; then
        OLLAYA_CACHE=$CACHE "$ROOT/scripts/llama-cpp.sh" "$LLAMA_CUDA_KIND" "$WORK/llama-$CUDA_PACK" \
            "$doc/llama.cpp-THIRD_PARTY_NOTICES"
        cp "$WORK/llama-$CUDA_PACK/$LLAMA_CUDA" "$lib/$LLAMA_CUDA"
    fi
    # FILES.sha256 fingerprints the libraries themselves (the archive's own checksum changes with
    # every release's timestamps). The installers compare it with the installed copy and skip the
    # ~1 GB download when nothing changed.
    (
        cd "$lib"
        LC_ALL=C
        export LC_ALL
        for f in *; do
            [ "$f" = FILES.sha256 ] || printf '%s  %s\n' "$(sha256_of "$f")" "$f"
        done
    ) >"$WORK/FILES.sha256"
    mv "$WORK/FILES.sha256" "$lib/FILES.sha256"

    cp "$ort/ThirdPartyNotices.txt" "$doc/onnxruntime-ThirdPartyNotices.txt"
    {
        printf 'Ollaya %s CUDA %s accelerator package (%s): third-party notices\n\n' "$VERSION" "$M" "$PLATFORM"
        printf 'Everything in lib/ollaya/%s is third-party software. None of it is covered by\n' "$CUDA_PACK"
        printf "Ollaya's Apache-2.0 license.\n\n"
        printf '1. ONNX Runtime %s (https://github.com/microsoft/onnxruntime), Microsoft'"'"'s\n' \
            "$ORT_GPU_VER"
        printf '%s, unmodified: %s.\n' "$ORT_GPU_ARCHIVE" "$ORT_LIBRARY $ORT_PROVIDERS"
        printf 'License: MIT. The components ONNX Runtime bundles are listed in\n'
        printf 'onnxruntime-ThirdPartyNotices.txt next to this file.\n\n'
        sed 's/^/    /' "$ort/LICENSE"
        cat <<'EOF'

2. NVIDIA CUDA and cuDNN runtime libraries

These files are NVIDIA's own binaries, unmodified, taken from NVIDIA's wheels on PyPI. They are
distributed under the NVIDIA Software License Agreement and CUDA Supplement (CUDA Toolkit EULA,
https://docs.nvidia.com/cuda/eula/) and the NVIDIA cuDNN Software License Agreement
(https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html). The license texts
shipped in each wheel are in licenses/. The libraries are licensed for use only on systems with
NVIDIA GPUs, and only by Ollaya.

EOF
        printf '    %-44s %s\n' File 'PyPI package'
        LC_ALL=C sort "$WORK/cuda-libs" | awk -F '\t' '{ printf "    %-44s %s\n", $1, $2 }'
        if [ -n "$LLAMA_CUDA" ]; then
            printf '\n3. llama.cpp %s CUDA backend (MIT), %s: see llama.cpp-THIRD_PARTY_NOTICES.\n' \
                "$LLAMA_CPP_BUILD" "$LLAMA_CUDA"
        fi
    } >"$doc/THIRD_PARTY_NOTICES"
    say "Staged $name ($(du -sk "$lib" | awk '{ printf "%.0f MiB", $1 / 1024 }') of libraries)"
}

stage_mlx() {
    name=ollaya-$PLATFORM-mlx
    root=$TREES/$name
    lib=$root/lib/ollaya/mlx_metal
    doc=$root/share/doc/ollaya/mlx_metal
    rm -rf "$root"
    mkdir -p "$lib" "$doc"
    [ -f "$TARGET_DIR/mlx.metallib" ] ||
        die "$TARGET_DIR/mlx.metallib not found; build with --features ollaya-runner/mlx"
    cp "$TARGET_DIR/mlx.metallib" "$lib/mlx.metallib"
    chmod 0644 "$lib/mlx.metallib"
    # As for CUDA: a fingerprint of the files themselves, for keep-if-unchanged installs.
    (cd "$lib" && printf '%s  %s\n' "$(sha256_of mlx.metallib)" mlx.metallib) >"$WORK/FILES.sha256"
    mv "$WORK/FILES.sha256" "$lib/FILES.sha256"
    {
        printf 'Ollaya %s MLX package (%s): third-party notices\n\n' "$VERSION" "$PLATFORM"
        printf "lib/ollaya/mlx_metal/mlx.metallib holds MLX's Metal kernels, compiled ahead of time.\n"
        printf "It is third-party software, not covered by Ollaya's Apache-2.0 license.\n\n"
        mlx_notice "Its Metal kernels, as compiled by this build."
    } >"$doc/THIRD_PARTY_NOTICES"
    say "Staged $name ($(human_size "$lib/mlx.metallib") metallib)"
}

# --- archives ----------------------------------------------------------------------------------

# tar_create DIR ENTRIES...: deterministic tar stream of DIR/ENTRIES on stdout. Only the named
# top-level entries go in (never "."), so extracting over a prefix leaves its own mode alone.
tar_create() {
    dir=$1
    shift
    if tar --version 2>/dev/null | grep -q 'GNU tar'; then
        gnutar=tar
    elif have gtar; then
        gnutar=gtar
    else
        gnutar=
    fi
    if [ -n "$gnutar" ]; then
        "$gnutar" -C "$dir" --sort=name --owner=0 --group=0 --numeric-owner \
            --mtime="@$SOURCE_DATE_EPOCH" --mode='u+rwX,go+rX,go-w' --format=gnu -cf - "$@"
    else
        # bsdtar (macOS without gtar): not byte-reproducible, but owner-neutral.
        tar -C "$dir" --uid 0 --gid 0 --uname root --gname wheel -cf - "$@"
    fi
}

archive() {
    name=$1
    shift
    src=$TREES/$name
    if [ "${PLATFORM%%-*}" = windows ]; then
        # A .zip: what Windows opens without extra tools. 7-Zip is on GitHub's Windows runners.
        have 7z || die "missing tool: 7z"
        (cd "$src" && 7z a -tzip -mx=9 -bd -bso0 "$OUT/$name.zip.part" "$@") || die "7z failed"
        mv "$OUT/$name.zip.part" "$OUT/$name.zip"
        say "Built $name.zip ($(human_size "$OUT/$name.zip"))"
        return
    fi
    tar_create "$src" "$@" | zstd -q -T0 -"$ZSTD_LEVEL" -o "$OUT/$name.tar.zst.part" -f
    # POSIX sh has no pipefail: read the archive back so a failed tar can't ship a truncated one.
    zstd -dc "$OUT/$name.tar.zst.part" | tar -tf - >"$WORK/listing"
    grep -q . "$WORK/listing" || die "$name.tar.zst is empty or corrupt"
    mv "$OUT/$name.tar.zst.part" "$OUT/$name.tar.zst"
    say "Built $name.tar.zst ($(human_size "$OUT/$name.tar.zst"))"
    if [ "${PLATFORM%%-*}" = darwin ]; then
        tar_create "$src" "$@" | gzip -n -9 >"$OUT/$name.tgz.part"
        tar -tzf "$OUT/$name.tgz.part" >/dev/null || die "$name.tgz is corrupt"
        mv "$OUT/$name.tgz.part" "$OUT/$name.tgz"
        say "Built $name.tgz ($(human_size "$OUT/$name.tgz"))"
    fi
}

[ "$BASE" = 0 ] || stage_base
[ "$CUDA" = 0 ] || stage_cuda 13
[ "$CUDA12" = 0 ] || stage_cuda 12
[ "$MLX" = 0 ] || stage_mlx

if [ -n "$STAGE" ]; then
    say "Staged trees are in $STAGE"
    exit 0
fi

[ "$BASE" = 0 ] || archive "ollaya-$PLATFORM" bin lib share
if [ "$CUDA" = 1 ]; then
    archive "ollaya-$PLATFORM-cuda" lib share
    cp "$TREES/ollaya-$PLATFORM-cuda/lib/ollaya/cuda_v13/FILES.sha256" "$OUT/ollaya-$PLATFORM-cuda.sha256"
fi
if [ "$CUDA12" = 1 ]; then
    archive "ollaya-$PLATFORM-cuda12" lib share
    cp "$TREES/ollaya-$PLATFORM-cuda12/lib/ollaya/cuda_v12/FILES.sha256" "$OUT/ollaya-$PLATFORM-cuda12.sha256"
fi
if [ "$MLX" = 1 ]; then
    archive "ollaya-$PLATFORM-mlx" lib share
    cp "$TREES/ollaya-$PLATFORM-mlx/lib/ollaya/mlx_metal/FILES.sha256" "$OUT/ollaya-$PLATFORM-mlx.sha256"
fi
write_checksums "$OUT"
