#!/bin/sh
# Stage llama.cpp's libraries, which run Ollaya's GGUF models, for one platform.
#
#   scripts/llama-cpp.sh <kind> <dest-dir> [<notices-file>]
#
# <kind>:
#   linux-amd64       libllama, libggml, libggml-base and the 14 CPU backends
#   linux-amd64-cuda  libggml-cuda.so, the CUDA backend (for the CUDA pack, lib/ollaya/cuda_v13)
#   linux-arm64, windows-amd64   libllama, libggml, libggml-base and the CPU backends
#   darwin-arm64      libllama and libggml with its CPU, BLAS, Metal and RPC backends
#
# The files are ggml-org's own release build of llama.cpp v0.5.0 (build b11146), byte for byte,
# from the release archives pinned below by the sha256 GitHub publishes for each asset
# (docs/decisions/0001-llama-cpp-runtime.md). Both linux-amd64 kinds come from the CUDA 13.4
# archive, so the CPU and CUDA backends are one build. Nothing else from the archives is staged
# (no llama-server, no tools), and never as a symbolic link: each library is stored once, under
# the name the loader asks for (libllama.so.0, libllama.0.dylib). <notices-file> receives the
# third-party notices.
#
# Environment: OLLAYA_CACHE, the download cache (default ${XDG_CACHE_HOME:-~/.cache}/ollaya-package).
#
# Bumping llama.cpp: change LLAMA_CPP_BUILD, LLAMA_CPP_VERSION and the pins below, check
# crates/ollaya-runner/src/llama/ffi.rs against the new include/llama.h, regenerate
# packaging/llama.cpp/THIRD_PARTY_NOTICES from the new tag, then re-run the GGUF parity checks on
# every device (docs/families/winnow.md, "Parity").

set -eu

LLAMA_CPP_VERSION=0.5.0
LLAMA_CPP_BUILD=b11146
LLAMA_CPP_URL=https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_CPP_BUILD

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
CACHE=${OLLAYA_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/ollaya-package}

die() { printf 'llama-cpp.sh: error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

sha256_of() {
    if have sha256sum; then
        sha256sum "$1" | cut -d' ' -f1
    elif have shasum; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        openssl dgst -sha256 -r "$1" | cut -d' ' -f1
    fi
}

[ $# -ge 2 ] || { sed -n '2,/^$/s/^# \{0,1\}//p' "$0" >&2; exit 2; }
KIND=$1 DEST=$2 NOTICES=${3:-}

b=llama-$LLAMA_CPP_BUILD-bin
case $KIND in
    linux-amd64 | linux-amd64-cuda)
        ASSET=$b-ubuntu-cuda-13.4-x64.tar.gz SUM=1603d9c00a4b6eac8298c5c7868cdb080a3ac31948ab1e457441d71ce274dd7e ;;
    linux-arm64) ASSET=$b-ubuntu-arm64.tar.gz SUM=4aeda6fe68831547e49b7fa87607383ca5352b3d72ca5f70d52ed265f58c131f ;;
    darwin-arm64) ASSET=$b-macos-arm64.tar.gz SUM=1ad3f9eff80edb9dbef4259ad564d1720612ef7eea48fa4afed0e54f5f3d5711 ;;
    windows-amd64) ASSET=$b-win-cpu-x64.zip SUM=14cf1303ca9ac3abd94816850532f9f9a69ac66fbaca3776fc6f9061c2fac1d1 ;;
    *) die "unknown kind: $KIND" ;;
esac

# Download into the cache once, verify every time.
cached=$CACHE/llama.cpp-$LLAMA_CPP_BUILD/$ASSET
if [ ! -f "$cached" ] || [ "$(sha256_of "$cached")" != "$SUM" ]; then
    mkdir -p "$(dirname "$cached")"
    curl -fsSL --retry 3 -o "$cached.part" "$LLAMA_CPP_URL/$ASSET" || die "download failed: $LLAMA_CPP_URL/$ASSET"
    mv "$cached.part" "$cached"
fi
got=$(sha256_of "$cached")
[ "$got" = "$SUM" ] || die "sha256 mismatch for $ASSET: got $got, want $SUM"

WORK=$(mktemp -d "${TMPDIR:-/tmp}/ollaya-llama.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
case $ASSET in
    *.zip)
        if have unzip; then
            unzip -q "$cached" -d "$WORK"
        elif have 7z; then
            7z x -bd -bso0 -o"$WORK" "$cached" || die "7z failed on $ASSET"
        else
            die "missing tool: unzip (or 7z)"
        fi
        ;;
    *) tar -xzf "$cached" -C "$WORK" ;;
esac
# The tarballs hold one directory, llama-<build>/; the zips hold the files at the top level.
src=$WORK
[ ! -d "$WORK/llama-$LLAMA_CPP_BUILD" ] || src=$WORK/llama-$LLAMA_CPP_BUILD

# wanted NAME: the file belongs to this kind.
wanted() {
    case $KIND:$1 in
        linux-amd64-cuda:libggml-cuda.so) return 0 ;;
        linux-amd64-cuda:*) return 1 ;;
        *:libggml-cuda.so) return 1 ;;
        # libllama-common, libllama-server-impl and the other tools' libraries, and multimodal.
        *:libllama-* | *:libmtmd*) return 1 ;;
        # The RPC backend is a loadable plugin on Linux, not needed; macOS links it into libggml.
        linux-*:libggml-rpc.so*) return 1 ;;
        *:libllama.so* | *:libllama.*dylib | *:libggml*.so* | *:libggml*.dylib) return 0 ;;
        windows-amd64:ggml-rpc.dll) return 1 ;;
        # llama.dll, ggml.dll, ggml-base.dll, the CPU backends and LLVM's OpenMP runtime they load.
        windows-amd64:llama.dll | windows-amd64:ggml*.dll | windows-amd64:libomp.dll) return 0 ;;
        *) return 1 ;;
    esac
}

# linked NAME: some symbolic link in the build points at NAME.
linked() {
    for l in "$src"/*; do
        if [ -L "$l" ] && [ "$(readlink "$l")" = "$1" ]; then return 0; fi
    done
    return 1
}

rm -rf "$DEST"
mkdir -p "$DEST"
for f in "$src"/*; do
    name=${f##*/}
    wanted "$name" || continue
    if [ -L "$f" ]; then
        target=$(readlink "$f")
        # A development link (libllama.so -> libllama.so.0) is not used at run time.
        [ ! -L "$src/$target" ] || continue
        # The soname link: the library, under the name the loader asks for.
        cp "$src/$target" "$DEST/$name"
    elif ! linked "$name"; then
        cp "$f" "$DEST/$name"
    fi
done
case $KIND in
    linux-amd64-cuda) need=libggml-cuda.so ;;
    linux-*) need=libllama.so.0 ;;
    darwin-*) need=libllama.0.dylib ;;
    *) need=llama.dll ;;
esac
[ -f "$DEST/$need" ] || die "$ASSET has no $need"

if [ -n "$NOTICES" ]; then
    mkdir -p "$(dirname "$NOTICES")"
    {
        cat "$ROOT/packaging/llama.cpp/THIRD_PARTY_NOTICES"
        # LLVM's OpenMP runtime, which the Windows CPU backends load, ships its license in the zip.
        if [ -f "$src/LICENSE-LLVM-OpenMP" ]; then
            printf '\n%s\nLLVM OpenMP runtime (libomp.dll), shipped with the llama.cpp Windows build\n\n' \
                '--------------------------------------------------------------------------------'
            cat "$src/LICENSE-LLVM-OpenMP"
        fi
    } >"$NOTICES"
fi
printf '>>> Staged llama.cpp %s (%s, %s) in %s\n' "$LLAMA_CPP_VERSION" "$LLAMA_CPP_BUILD" "$KIND" "$DEST" >&2
