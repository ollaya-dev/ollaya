#!/bin/sh
# Build Ollaya release archives from a cargo target directory.
#
#   scripts/package.sh [options] <target-dir> <version>
#   scripts/package.sh --checksums <dir>
#
# Archives, written to --out (default: ./dist):
#   ollaya-<platform>.tar.zst        bin/ollaya + share/doc/ollaya/ (LICENSE, THIRD_PARTY_NOTICES)
#   ollaya-linux-amd64-cuda.tar.zst  lib/ollaya/cuda_v13/ (ORT CUDA provider + NVIDIA CUDA/cuDNN
#                                    libraries) + share/doc/ollaya/cuda_v13/ (notices, NVIDIA licenses)
#   ollaya-darwin-arm64.tgz          same content as the darwin .tar.zst; stock macOS has no zstd
#   ollaya-darwin-arm64-mlx.tar.zst  lib/ollaya/mlx_metal/ (mlx.metallib, the MLX engine's Metal
#                                    kernels) + share/doc/ollaya/mlx_metal/ (notices); also .tgz
#   ollaya-windows-amd64.zip         bin/ollaya.exe (with the DLLs it links), share/
#   ollaya-windows-amd64-cuda.zip    the same GPU pack as the Linux one, with the Windows DLLs
#   ollaya-<platform>-cuda.sha256    sha256 of every library in the CUDA archive (FILES.sha256),
#                                    which the installers use to skip an unchanged CUDA download
#   ollaya-darwin-arm64-mlx.sha256   the same for the MLX archive
#   sha256sum.txt                    over every archive in --out, and the files above
#
# Options:
#   --platform P  linux-amd64 | linux-arm64 | darwin-arm64 | windows-amd64 (default: this host)
#   --cuda        also build the CUDA archive (linux-amd64 and windows-amd64). The binary must
#                 have been built with `--features ollaya-runner/cuda`, which also puts the ORT
#                 provider libraries in <target-dir>.
#   --mlx         also build the MLX archive (darwin-arm64 only). The binary must have been built
#                 with `--features ollaya-runner/mlx`, whose build script puts mlx.metallib in
#                 <target-dir>. Not part of release builds yet (docs/decisions/0001-mlx-engine.md).
#   --no-base     skip the base archive (only useful with --cuda or --mlx)
#   --stage DIR   stage the file trees into DIR/<archive name>/ and stop: no archives, no checksums
#                 (the Dockerfile uses this)
#   --out DIR     output directory (default: dist)
#   --checksums D only (re)write D/sha256sum.txt
#
# Environment:
#   OLLAYA_BIN             binary to package (default: <target-dir>/ollaya)
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
                *.tar.zst | *.tgz | *.zip | *-cuda.sha256 | *-mlx.sha256) [ -f "$f" ] || continue ;;
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

PLATFORM='' CUDA=0 MLX=0 BASE=1 STAGE='' OUT=dist
while [ $# -gt 0 ]; do
    case $1 in
        --platform) [ $# -ge 2 ] || usage; PLATFORM=$2; shift 2 ;;
        --platform=*) PLATFORM=${1#*=}; shift ;;
        --cuda) CUDA=1; shift ;;
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
    *) [ "$CUDA" = 0 ] || die "--cuda is only supported for linux-amd64 and windows-amd64" ;;
esac
[ "$MLX" = 0 ] || [ "$PLATFORM" = darwin-arm64 ] || die "--mlx is only supported for darwin-arm64"
[ "$BASE" = 1 ] || [ "$CUDA" = 1 ] || [ "$MLX" = 1 ] ||
    die "--no-base without --cuda or --mlx leaves nothing to do"
# The `mlx` feature links MLX into bin/ollaya, so its notices go into the base archive too.
case ",$FEATURES," in *mlx,*) LINKS_MLX=1 ;; *) LINKS_MLX=0 ;; esac

EXE=
[ "$PLATFORM" != windows-amd64 ] || EXE=.exe

# The CUDA pack (lib/ollaya/cuda_v13): ORT's provider libraries, and the NVIDIA libraries kept from
# the wheels in packaging/cuda-requirements.txt (see is_cuda_lib). Everything else in the wheels
# (static and import libs, headers, cufftw, nvblas, nvrtc .alt builds) is dropped. The TensorRT
# and NV TensorRT RTX providers are left out: they need TensorRT 10 (libnvinfer, nvinfer_10.dll),
# which is not shipped, and the runner only registers the CUDA execution provider.
if [ "$PLATFORM" = windows-amd64 ]; then
    ORT_PROVIDERS="onnxruntime_providers_shared.dll onnxruntime_providers_cuda.dll"
    CUDA_LIBS_REQUIRED="cudart64_13.dll cublas64_13.dll cublasLt64_13.dll cufft64_12.dll curand64_10.dll
nvrtc64_130_0.dll nvJitLink_130_0.dll cudnn64_9.dll cudnn_graph64_9.dll"
    WHEEL_TAG=win_amd64
    WHEEL_PLATFORMS=win_amd64
    # Windows wheels keep their DLLs in nvidia/*/bin/, Linux wheels in nvidia/*/lib/.
    WHEEL_LIB_DIR=bin
else
    ORT_PROVIDERS="libonnxruntime_providers_shared.so libonnxruntime_providers_cuda.so"
    CUDA_LIBS_REQUIRED="libcudart.so.13 libcublas.so.13 libcublasLt.so.13 libcufft.so.12 libcurand.so.10
libnvrtc.so.13 libnvJitLink.so.13 libcudnn.so.9 libcudnn_graph.so.9"
    WHEEL_TAG=x86_64
    WHEEL_PLATFORMS="manylinux_2_28_x86_64 manylinux_2_27_x86_64 manylinux_2_17_x86_64
manylinux2014_x86_64 manylinux_2_12_x86_64 manylinux2010_x86_64"
    WHEEL_LIB_DIR=lib
fi

# is_cuda_lib NAME: whether the wheel file NAME goes into the CUDA pack.
is_cuda_lib() {
    case $PLATFORM:$1 in
        linux-*:libcudart.so.13 | linux-*:libcublas.so.13 | linux-*:libcublasLt.so.13 | \
            linux-*:libcufft.so.12 | linux-*:libcurand.so.10 | linux-*:libnvrtc.so.13 | \
            linux-*:libnvJitLink.so.13 | linux-*:libcudnn*.so.9 | linux-*:libnvrtc-builtins.so.13.*) ;;
        windows-*:cudart64_13.dll | windows-*:cublas64_13.dll | windows-*:cublasLt64_13.dll | \
            windows-*:cufft64_12.dll | windows-*:curand64_10.dll | windows-*:nvrtc64_130_0.dll | \
            windows-*:nvJitLink_130_0.dll | windows-*:cudnn*64_9.dll | windows-*:nvrtc-builtins64_13*.dll) ;;
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
    {
        printf 'Ollaya %s (%s): third-party notices\n\n' "$VERSION" "$PLATFORM"
        printf 'Ollaya is licensed under the Apache License 2.0 (see LICENSE). bin/ollaya also\n'
        printf 'contains the third-party software below.\n\n'
        printf '1. '
        ort_notice "Linked into bin/ollaya$EXE."
        n=2
        if [ "$LINKS_MLX" = 1 ]; then
            printf '\n2. '
            mlx_notice "Linked into bin/ollaya$EXE (the MLX engine)."
            n=3
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
    set -- "$@" --require-hashes -r "$ROOT/packaging/cuda-requirements.txt" -d "$dest"
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

stage_cuda() {
    name=ollaya-$PLATFORM-cuda
    root=$TREES/$name
    lib=$root/lib/ollaya/cuda_v13
    doc=$root/share/doc/ollaya/cuda_v13
    rm -rf "$root"
    mkdir -p "$lib" "$doc/licenses"
    have unzip || die "missing tool: unzip"

    for p in $ORT_PROVIDERS; do
        [ -e "$TARGET_DIR/$p" ] ||
            die "$TARGET_DIR/$p not found; build with --features ollaya-runner/cuda"
        cp -L "$TARGET_DIR/$p" "$lib/$p"
    done
    # Windows: GPU runners start from a copy of ollaya.exe in this folder, so the DLLs it imports
    # (DirectML.dll, the same file as in bin/) go here too. Otherwise the runner would pick up
    # whatever version System32 has, and ollaya.exe imports DirectML by ordinal.
    if [ -n "$EXE" ]; then
        for dll in "$TARGET_DIR"/*.dll; do
            case ${dll##*/} in onnxruntime_providers_*) continue ;; esac
            [ ! -f "$dll" ] || cp -L "$dll" "$lib/"
        done
    fi

    wheels=$CACHE/wheels
    mkdir -p "$wheels"
    say "Fetching NVIDIA wheels (about 1.2 GB on first run, cached in $wheels)"
    pip_download "$wheels"

    : >"$WORK/cuda-libs"
    sed -n 's/^\(nvidia-[a-z0-9-]*\)==\([^ ]*\).*/\1 \2/p' "$ROOT/packaging/cuda-requirements.txt" \
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

    cp "$CACHE/onnxruntime-$ORT_VERSION/ThirdPartyNotices.txt" "$doc/onnxruntime-ThirdPartyNotices.txt"
    {
        printf 'Ollaya %s CUDA accelerator package (%s): third-party notices\n\n' "$VERSION" "$PLATFORM"
        printf 'Everything in lib/ollaya/cuda_v13 is third-party software. None of it is covered by\n'
        printf "Ollaya's Apache-2.0 license.\n\n"
        printf '1. '
        ort_notice "Execution provider libraries: $ORT_PROVIDERS."
        if [ -f "$lib/DirectML.dll" ]; then
            printf '\nDirectML.dll is Microsoft DirectML as it ships with that ONNX Runtime build, the\n'
            printf 'same file as bin/DirectML.dll, for the runners that start from this folder.\n'
        fi
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
[ "$CUDA" = 0 ] || stage_cuda
[ "$MLX" = 0 ] || stage_mlx

if [ -n "$STAGE" ]; then
    say "Staged trees are in $STAGE"
    exit 0
fi

[ "$BASE" = 0 ] || archive "ollaya-$PLATFORM" bin share
if [ "$CUDA" = 1 ]; then
    archive "ollaya-$PLATFORM-cuda" lib share
    cp "$TREES/ollaya-$PLATFORM-cuda/lib/ollaya/cuda_v13/FILES.sha256" "$OUT/ollaya-$PLATFORM-cuda.sha256"
fi
if [ "$MLX" = 1 ]; then
    archive "ollaya-$PLATFORM-mlx" lib share
    cp "$TREES/ollaya-$PLATFORM-mlx/lib/ollaya/mlx_metal/FILES.sha256" "$OUT/ollaya-$PLATFORM-mlx.sha256"
fi
write_checksums "$OUT"
