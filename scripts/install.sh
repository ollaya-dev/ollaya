#!/bin/sh
# Install Ollaya on Linux or macOS:
#
#   curl -fsSL https://ollaya.dev/install.sh | sh
#
# Uninstall (the models stay; the script says where and how to delete them):
#
#   curl -fsSL https://ollaya.dev/install.sh | sh -s -- --uninstall
#
# Settings, passed as environment variables to the shell that runs the script
# (curl -fsSL https://ollaya.dev/install.sh | OLLAYA_VERSION=0.1.0 sh):
#   OLLAYA_VERSION      version to install, such as 0.1.0 (default: the latest release)
#   OLLAYA_REPO         GitHub repository to download releases from (default: ollaya-dev/ollaya)
#   OLLAYA_INSTALL_DIR  install prefix, an absolute path; the binary goes to $OLLAYA_INSTALL_DIR/bin
#                       (default: /usr/local, or ~/.local without root or sudo)
#   OLLAYA_NO_SERVICE=1 don't create or start the systemd service
#   OLLAYA_NO_CUDA=1    don't download the CUDA libraries, even when there is an NVIDIA GPU
#   OLLAYA_NO_ROCM=1    don't download the ROCm libraries, even when there is an AMD GPU
#
# The script never installs GPU drivers. Downloads are checked against the release's sha256sum.txt.

# Everything runs inside main so that a truncated download can't execute half a script.
main() {
    set -eu

    if [ -t 2 ]; then
        bold=$(printf '\033[1m') red=$(printf '\033[31m') yellow=$(printf '\033[33m')
        plain=$(printf '\033[0m')
    else
        bold='' red='' yellow='' plain=''
    fi
    status() { printf '%s>>>%s %s\n' "$bold" "$plain" "$*" >&2; }
    warn() { printf '%sWARNING:%s %s\n' "$yellow" "$plain" "$*" >&2; }
    error() { printf '%sERROR:%s %s\n' "$red" "$plain" "$*" >&2; exit 1; }
    available() { command -v "$1" >/dev/null 2>&1; }
    enabled() { case ${1:-} in '' | 0 | false | no) return 1 ;; *) return 0 ;; esac; }

    UNINSTALL=false
    for arg in "$@"; do
        case $arg in
            --uninstall) UNINSTALL=true ;;
            *) error "unknown option: $arg (the only option is --uninstall)" ;;
        esac
    done

    REPO=${OLLAYA_REPO:-ollaya-dev/ollaya}
    PORT=11435

    # --- platform --------------------------------------------------------------------------

    OS=$(uname -s)
    ARCH=$(uname -m)
    case $ARCH in
        x86_64 | amd64) ARCH=amd64 ;;
        aarch64 | arm64) ARCH=arm64 ;;
        *) error "unsupported architecture: $ARCH" ;;
    esac
    WSL=
    case $OS in
        Linux)
            case $(uname -r) in
                *[Mm]icrosoft*WSL2* | *[Mm]icrosoft*wsl2*) WSL=2 ;;
                *Microsoft) WSL=1 ;;
                *) [ ! -e /proc/sys/fs/binfmt_misc/WSLInterop ] || WSL=2 ;;
            esac
            PLATFORM=linux-$ARCH
            ;;
        Darwin)
            # A shell running under Rosetta reports x86_64 on Apple silicon.
            if [ "$ARCH" = amd64 ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || :)" = 1 ]; then
                ARCH=arm64
            fi
            [ "$ARCH" = arm64 ] || error "Ollaya supports Apple silicon Macs only"
            # The MLX engine, linked into bin/ollaya from 0.7.0 on, needs macOS 14.
            case $(sw_vers -productVersion 2>/dev/null || :) in
                1[0-3].* | [1-9].*) error "Ollaya needs macOS 14 (Sonoma) or later; this Mac runs macOS $(sw_vers -productVersion)" ;;
            esac
            PLATFORM=darwin-arm64
            ;;
        *) error "this script supports Linux and macOS only (found $OS)" ;;
    esac
    [ "$WSL" != 1 ] || warn "WSL 1 is untested and has no GPU access; WSL 2 is recommended"

    if [ "$OS" = Linux ] && ! $UNINSTALL; then
        # pyke's ONNX Runtime build references glibc 2.38 symbols: Ubuntu 24.04, Debian 13,
        # Fedora 39, RHEL 10 or newer.
        glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{ print $2 }') || glibc=
        [ -n "$glibc" ] || error "Ollaya needs glibc 2.38 or newer (musl systems such as Alpine are not supported; use the Docker image)"
        if ! printf '%s\n' "$glibc" | awk -F. '{ exit !($1 > 2 || ($1 == 2 && $2 >= 38)) }'; then
            error "Ollaya needs glibc 2.38 or newer, this system has $glibc (Ubuntu 24.04, Debian 13, Fedora 39, RHEL 10 or newer; or use the Docker image)"
        fi
        # llama.cpp's CPU backends (GGUF models) use GCC's OpenMP runtime from the system.
        ldconfig=$(command -v ldconfig 2>/dev/null || echo /sbin/ldconfig)
        if [ -x "$ldconfig" ] && ! "$ldconfig" -p 2>/dev/null | grep -q 'libgomp\.so\.1 '; then
            warn "GCC's OpenMP runtime (libgomp.so.1) was not found: GGUF models such as winnow need it. Install your distribution's libgomp1 (or libgomp) package"
        fi
    fi

    missing=
    for tool in curl tar awk mktemp; do available "$tool" || missing="$missing $tool"; done
    available sha256sum || available shasum || available openssl || missing="$missing sha256sum"
    $UNINSTALL || [ -z "$missing" ] || error "required tools are missing:$missing"
    ZSTD=
    if available zstd; then
        ZSTD=zstd
    elif [ "$OS" = Linux ] && ! $UNINSTALL; then
        error "zstd is needed to unpack Ollaya. Install it and re-run:
  Debian/Ubuntu: sudo apt-get install zstd
  Fedora/RHEL:   sudo dnf install zstd
  Arch:          sudo pacman -S zstd"
    fi

    sha256() {
        if available sha256sum; then
            sha256sum "$1" | awk '{ print $1 }'
        elif available shasum; then
            shasum -a 256 "$1" | awk '{ print $1 }'
        else
            openssl dgst -sha256 -r "$1" | awk '{ print $1 }'
        fi
    }

    # --- install prefix and privileges -----------------------------------------------------

    # writable DIR: DIR, or the closest existing parent that would create it, is writable.
    writable() {
        d=$1
        while [ ! -e "$d" ]; do d=$(dirname "$d"); done
        [ -w "$d" ]
    }
    # The prefix itself too: the install stages in $PREFIX/.ollaya-install.*. On a Mac with
    # Homebrew, /usr/local/bin and /usr/local/share are often the user's while /usr/local is
    # root's (#31). lib/ollaya holds the CUDA pack on Linux and the MLX library on macOS.
    prefix_writable() {
        writable "$1" && writable "$1/bin" && writable "$1/share" && writable "$1/lib"
    }
    get_sudo() {
        available sudo || return 1
        sudo -n true 2>/dev/null && return 0
        # A password prompt needs a terminal; `curl | sh` still has one on /dev/tty.
        (: </dev/tty) 2>/dev/null || return 1
        status "Installing to $1 needs sudo (set OLLAYA_INSTALL_DIR=\$HOME/.local to avoid it)"
        sudo -v
    }

    SUDO=
    IS_ROOT=false
    [ "$(id -u)" -ne 0 ] || IS_ROOT=true
    # An install without root rights is in ~/.local; uninstall removes it without asking for sudo.
    if $UNINSTALL && [ -z "${OLLAYA_INSTALL_DIR:-}" ] && [ ! -e /usr/local/bin/ollaya ] &&
        [ -e "${HOME:-}/.local/bin/ollaya" ]; then
        OLLAYA_INSTALL_DIR=$HOME/.local
    fi
    if [ -n "${OLLAYA_INSTALL_DIR:-}" ]; then
        PREFIX=${OLLAYA_INSTALL_DIR%/}
        case $PREFIX in /*) ;; *) error "OLLAYA_INSTALL_DIR must be an absolute path" ;; esac
        if ! $IS_ROOT && ! prefix_writable "$PREFIX"; then
            get_sudo "$PREFIX" || error "cannot write to $PREFIX and sudo is not available"
            SUDO=sudo
        fi
    else
        PREFIX=/usr/local
        if ! $IS_ROOT && ! prefix_writable "$PREFIX"; then
            if get_sudo "$PREFIX"; then
                SUDO=sudo
            else
                ! $UNINSTALL || error "removing Ollaya from /usr/local needs root or sudo"
                [ -n "${HOME:-}" ] || error "cannot write to /usr/local, no sudo, and HOME is not set"
                PREFIX=$HOME/.local
                status "No root access: installing to $PREFIX instead"
            fi
        fi
    fi
    BINDIR=$PREFIX/bin
    # Root-level changes (system user, systemd unit) are only made with root rights we have anyway.
    PRIVILEGED=false
    if $IS_ROOT || [ -n "$SUDO" ]; then PRIVILEGED=true; fi

    # --- uninstall -------------------------------------------------------------------------

    if $UNINSTALL; then
        [ -e "$BINDIR/ollaya" ] || [ -d "$PREFIX/lib/ollaya" ] || error "no Ollaya install found in $PREFIX"
        # Only the service this install created, the one that runs this binary.
        unit=/etc/systemd/system/ollaya.service
        if $SUDO grep -qxF "ExecStart=$BINDIR/ollaya serve" "$unit" 2>/dev/null; then
            $PRIVILEGED || error "removing the ollaya systemd service needs root or sudo"
            status "Removing the ollaya systemd service"
            $SUDO systemctl disable --now ollaya >/dev/null 2>&1 || :
            $SUDO rm -f "$unit"
            $SUDO systemctl daemon-reload 2>/dev/null || :
        fi
        # A server this user's CLI started in the background.
        [ ! -x "$BINDIR/ollaya" ] || "$BINDIR/ollaya" stop >/dev/null 2>&1 || :
        $SUDO rm -rf "$BINDIR/ollaya" "$PREFIX/lib/ollaya" "$PREFIX/share/doc/ollaya" "$PREFIX/share/ollaya/skills"
        # With a /usr prefix, share/ollaya is also the service's home, which holds its models.
        $SUDO rmdir "$PREFIX/share/ollaya" 2>/dev/null || :
        status "Removed Ollaya from $PREFIX"
        if id ollaya >/dev/null 2>&1; then
            status "Kept the ollaya system user and the service's models in /usr/share/ollaya. To delete them: sudo userdel -r ollaya"
        fi
        if [ -d /etc/systemd/system/ollaya.service.d ]; then
            status "Kept the service settings in /etc/systemd/system/ollaya.service.d"
        fi
        if [ -d "${HOME:-}/.ollaya" ]; then
            status "Kept your models, logs and history in $HOME/.ollaya. To delete them: rm -rf ~/.ollaya"
        fi
        exit 0
    fi

    TMP=$(mktemp -d "${TMPDIR:-/tmp}/ollaya-install.XXXXXX")
    STAGE=
    cleanup() {
        rm -rf "$TMP"
        [ -z "$STAGE" ] || $SUDO rm -rf "$STAGE"
    }
    trap cleanup EXIT
    trap 'exit 130' INT TERM

    # --- release location ------------------------------------------------------------------

    if [ -n "${OLLAYA_DOWNLOAD_BASE:-}" ]; then
        # Undocumented: a directory with the release files, for testing this script.
        BASE_URL=${OLLAYA_DOWNLOAD_BASE%/}
        VERSION=${OLLAYA_VERSION:-from $BASE_URL}
    elif [ -n "${OLLAYA_VERSION:-}" ]; then
        VERSION=${OLLAYA_VERSION#v}
        BASE_URL=https://github.com/$REPO/releases/download/v$VERSION
    else
        # /releases/latest redirects to /releases/tag/<tag>; no API call, so no rate limit.
        latest=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest") ||
            error "could not find the latest release of $REPO (https://github.com/$REPO/releases)"
        case $latest in
            */releases/tag/*) TAG=${latest##*/releases/tag/} ;;
            *) error "$REPO has no published release yet (https://github.com/$REPO/releases)" ;;
        esac
        VERSION=${TAG#v}
        BASE_URL=https://github.com/$REPO/releases/download/$TAG
    fi

    download() {
        if [ -t 2 ]; then
            curl --fail --show-error --location --progress-bar -o "$2" "$1"
        else
            curl --fail --silent --show-error --location -o "$2" "$1"
        fi
    }

    status "Installing Ollaya $VERSION ($PLATFORM) to $PREFIX"
    download "$BASE_URL/sha256sum.txt" "$TMP/sha256sum.txt" ||
        error "could not download $BASE_URL/sha256sum.txt (is $VERSION a published release?)"

    # cuda_intact DIR: every library listed in DIR/FILES.sha256 is there with that checksum.
    cuda_intact() {
        (
            cd "$1" || exit 1
            while read -r sum file; do
                [ -f "$file" ] && [ "$(sha256 "$file")" = "$sum" ] || exit 1
            done <FILES.sha256
        )
    }

    # fetch_verified FILE: download FILE from the release and check it against sha256sum.txt.
    fetch_verified() {
        want=$(awk -v f="$1" '$2 == f || $2 == "*" f { print $1; exit }' "$TMP/sha256sum.txt")
        [ -n "$want" ] || error "$1 is not listed in $BASE_URL/sha256sum.txt"
        status "Downloading $1"
        download "$BASE_URL/$1" "$TMP/$1" || error "could not download $BASE_URL/$1"
        got=$(sha256 "$TMP/$1")
        [ "$got" = "$want" ] || error "checksum mismatch for $1: expected $want, got $got"
    }

    # --- GPU -------------------------------------------------------------------------------

    # NVIDIA_STATE: none | nodriver | oldriver | ready. Whether the release has a CUDA package for
    # this platform is checked below, against its sha256sum.txt.
    # CUDA_PACK: cuda_v13, or cuda_v12 for drivers without CUDA 13 support (R525 to R575) and for
    # pre-Turing cards (the CUDA 13 pack's kernels start at sm_75), which releases from 0.7.3 on
    # ship as ollaya-<platform>-cuda12.
    NVIDIA_STATE=none
    CUDA_DRIVER=
    CUDA_PACK=cuda_v13
    # The largest GPU's memory in MiB (nvidia-smi), for the model the summary suggests.
    GPU_MIB=
    nvidia_smi=
    if [ "$OS" = Linux ]; then
        if [ "$WSL" = 2 ]; then
            # WSL 2 uses the Windows driver, exposed in /usr/lib/wsl/lib; lspci shows no GPU.
            if [ -e /usr/lib/wsl/lib/libcuda.so.1 ]; then
                NVIDIA_STATE=ready
                if [ -x /usr/lib/wsl/lib/nvidia-smi ]; then nvidia_smi=/usr/lib/wsl/lib/nvidia-smi; fi
            fi
        elif available nvidia-smi || [ -r /proc/driver/nvidia/version ]; then
            NVIDIA_STATE=ready
        elif available lspci && lspci -d 10de: 2>/dev/null | grep -qiE 'vga|3d|display'; then
            NVIDIA_STATE=nodriver
        else
            for dev in /sys/bus/pci/devices/*; do
                if [ ! -r "$dev/vendor" ] || [ ! -r "$dev/class" ]; then
                    continue
                fi
                if [ "$(cat "$dev/vendor")" = 0x10de ]; then
                    case $(cat "$dev/class") in 0x03*) NVIDIA_STATE=nodriver ;; esac
                fi
            done
        fi
        if [ "$NVIDIA_STATE" = ready ]; then
            if [ -z "$nvidia_smi" ] && available nvidia-smi; then nvidia_smi=nvidia-smi; fi
            if [ -n "$nvidia_smi" ]; then
                # The highest CUDA version the driver supports. The header says "CUDA Version: 13.0"
                # on older drivers and "CUDA UMD Version: 13.4" on newer ones.
                CUDA_DRIVER=$("$nvidia_smi" 2>/dev/null | sed -n 's/.*CUDA[A-Z ]*Version: *\([0-9][0-9.]*\).*/\1/p' | head -n 1) ||
                    CUDA_DRIVER=
                if [ -n "$CUDA_DRIVER" ] && [ "${CUDA_DRIVER%%.*}" -lt 13 ]; then
                    if [ "${CUDA_DRIVER%%.*}" -ge 12 ] && grep -q " ollaya-$PLATFORM-cuda12\.tar\.zst\$" "$TMP/sha256sum.txt"; then
                        CUDA_PACK=cuda_v12
                    else
                        NVIDIA_STATE=oldriver
                    fi
                fi
                # The CUDA 13 pack's kernels start at sm_75, so Pascal (6.x) and Volta (7.0)
                # cards need the CUDA 12 pack even on a driver that reports CUDA 13: that
                # version is the newest runtime the driver supports, not what the cards can
                # run. The lowest compute capability of the host's cards decides, because a
                # machine with an old and a new card is only as fast as the oldest. compute_cap
                # is N/A only on drivers older than about R510, which never match here.
                GPU_MIB=$("$nvidia_smi" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null |
                    sed -n 's/^ *\([0-9][0-9]*\).*/\1/p' | sort -n | tail -n 1)
                min_cap=$("$nvidia_smi" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
                    sed -n 's/^ *\([0-9][0-9]*\)\.\([0-9]\).*/\1\2/p' | sort -n | head -n 1)
                if [ "$NVIDIA_STATE" = ready ] && [ -n "$min_cap" ] && [ "$min_cap" -lt 75 ] &&
                    grep -q " ollaya-$PLATFORM-cuda12\.tar\.zst\$" "$TMP/sha256sum.txt"; then
                    CUDA_PACK=cuda_v12
                fi
            fi
        fi
    fi

    WANT_CUDA=false
    CUDA_SUFFIX=''
    [ "$CUDA_PACK" = cuda_v13 ] || CUDA_SUFFIX=12
    case $NVIDIA_STATE in
        ready)
            if enabled "${OLLAYA_NO_CUDA:-}"; then
                status "NVIDIA GPU found; skipping the CUDA libraries (OLLAYA_NO_CUDA is set)"
            elif ! grep -q " ollaya-$PLATFORM-cuda$CUDA_SUFFIX\.tar\.zst\$" "$TMP/sha256sum.txt"; then
                # linux-arm64 has a CUDA package only in newer releases, and other platforms none.
                warn "NVIDIA GPU found, but Ollaya $VERSION has no GPU package for $PLATFORM; Ollaya will use the CPU"
            else
                WANT_CUDA=true
            fi
            ;;
        oldriver)
            if grep -q " ollaya-$PLATFORM-cuda12\.tar\.zst\$" "$TMP/sha256sum.txt"; then
                warn "the NVIDIA driver supports CUDA $CUDA_DRIVER, but Ollaya's GPU libraries need a driver with CUDA 12 support (R525 or newer)."
            else
                warn "the NVIDIA driver supports CUDA $CUDA_DRIVER, but Ollaya $VERSION's GPU libraries need a driver with CUDA 13 support (R580 or newer)."
            fi
            warn "Ollaya will use the CPU. Update the driver, then run this script again."
            ;;
        nodriver)
            warn "NVIDIA GPU found, but no NVIDIA driver is loaded. Ollaya will use the CPU."
            warn "Install the NVIDIA driver (R580 or newer; see https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), then run this script again."
            ;;
        *) ;;
    esac

    # ROCM_STATE: none | nodriver | unsupported | missing_libs | ready
    ROCM_STATE=none
    ROCM_ARCH=
    MISSING_ROCM_LIBS=
    # Check for ROCm runtime libraries required by the pack's libggml-hip.so
    has_rocm_lib() {
        if available ldconfig; then
            ldconfig -p 2>/dev/null | grep -q "/$1\$"
            return
        fi
        for d in /usr/lib64 /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib; do
            [ -f "$d/$1" ] && return 0
        done
        return 1
    }
    # Discrete AMD GPUs compiled by upstream llama.cpp. Integrated GPUs (APUs such as
    # gfx1036 or gfx1150) share system memory and are not faster than the CPU with AVX-512,
    # so Ollaya's auto-selection leaves them on the CPU.
    supported_rocm_arch() {
        case $1 in
            gfx1201 | gfx1200 | gfx1100 | gfx1101 | gfx1102 | gfx1030 | gfx1031 | gfx1032 | gfx1034 | gfx1010 | gfx1011 | gfx1012 | gfx908 | gfx90a | gfx942 | gfx950) return 0 ;;
            *) return 1 ;;
        esac
    }

    if [ "$OS" = Linux ] && ! $WANT_CUDA; then
        AMD_FOUND=false
        if available lspci && lspci -d 1002: 2>/dev/null | grep -qiE 'vga|3d|display'; then
            AMD_FOUND=true
        else
            for dev in /sys/bus/pci/devices/*; do
                if [ -r "$dev/vendor" ] && [ "$(cat "$dev/vendor")" = 0x1002 ]; then
                    if [ -r "$dev/class" ]; then
                        case $(cat "$dev/class") in 0x03*) AMD_FOUND=true; break ;; esac
                    fi
                fi
            done
        fi

        if $AMD_FOUND; then
            if [ -z "$GPU_MIB" ]; then
                bytes=$(cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | sort -n | tail -n 1) || bytes=
                if [ -n "$bytes" ] && [ "$bytes" -gt 0 ] 2>/dev/null; then
                    GPU_MIB=$((bytes / 1048576))
                fi
            fi
            # Verify /dev/kfd and /dev/dri for compute access
            if [ ! -e /dev/kfd ] || [ ! -d /dev/dri ]; then
                ROCM_STATE=nodriver
            else
                if [ -d /sys/class/kfd/kfd/topology/nodes ]; then
                    for node in /sys/class/kfd/kfd/topology/nodes/*; do
                        [ -r "$node/properties" ] || continue
                        ver=$(awk '$1 == "gfx_target_version" { print $2 }' "$node/properties" 2>/dev/null)
                        case $ver in '' | 0 | *[!0-9]*) continue ;; esac
                        case $ver in
                            120001) arch=gfx1201 ;;
                            120000) arch=gfx1200 ;;
                            110000) arch=gfx1100 ;;
                            110001) arch=gfx1101 ;;
                            110002) arch=gfx1102 ;;
                            100300) arch=gfx1030 ;;
                            100301) arch=gfx1031 ;;
                            100302) arch=gfx1032 ;;
                            100304) arch=gfx1034 ;;
                            100100) arch=gfx1010 ;;
                            100101) arch=gfx1011 ;;
                            100102) arch=gfx1012 ;;
                            90008)  arch=gfx908 ;;
                            90010)  arch=gfx90a ;;
                            90402)  arch=gfx942 ;;
                            90500)  arch=gfx950 ;;
                            *)
                                maj=$((ver / 10000))
                                min=$(((ver % 10000) / 100))
                                step=$((ver % 100))
                                if [ "$maj" -eq 9 ] && [ "$min" -eq 0 ] && [ "$step" -eq 10 ]; then
                                    arch=gfx90a
                                else
                                    arch="gfx${maj}${min}${step}"
                                fi
                                ;;
                        esac
                        if supported_rocm_arch "$arch"; then
                            ROCM_ARCH=$arch
                            break
                        fi
                        [ -n "$ROCM_ARCH" ] || ROCM_ARCH=$arch
                    done
                fi

                if [ -z "$ROCM_ARCH" ]; then
                    ROCM_STATE=nodriver
                elif supported_rocm_arch "$ROCM_ARCH"; then
                    # libggml-hip.so links libamdhip64.so.7, libhipblas.so.3 and librocblas.so.5 on the host.
                    missing=
                    for lib in libamdhip64.so.7 libhipblas.so.3 librocblas.so.5; do
                        has_rocm_lib "$lib" || missing="${missing:+$missing, }$lib"
                    done
                    if [ -n "$missing" ]; then
                        ROCM_STATE=missing_libs
                        MISSING_ROCM_LIBS=$missing
                    else
                        ROCM_STATE=ready
                    fi
                else
                    ROCM_STATE=unsupported
                fi
            fi
        fi
    fi

    WANT_ROCM=false
    case $ROCM_STATE in
        ready)
            if enabled "${OLLAYA_NO_ROCM:-}"; then
                status "AMD GPU ($ROCM_ARCH) found; skipping the ROCm libraries (OLLAYA_NO_ROCM is set)"
            elif [ "$ARCH" != amd64 ]; then
                warn "AMD GPU ($ROCM_ARCH) found, but ROCm GPU acceleration is only packaged for x86-64 so far; Ollaya will use the CPU"
            elif ! grep -q " ollaya-$PLATFORM-rocm\.tar\.zst\$" "$TMP/sha256sum.txt"; then
                warn "AMD GPU ($ROCM_ARCH) found, but Ollaya $VERSION has no ROCm GPU package for $PLATFORM; Ollaya will use the CPU"
            else
                WANT_ROCM=true
            fi
            ;;
        missing_libs)
            warn "AMD GPU ($ROCM_ARCH) found, but required ROCm 7 runtime libraries are missing: $MISSING_ROCM_LIBS."
            warn "Install ROCm 7 (see https://rocm.docs.amd.com/), add /opt/rocm/lib to /etc/ld.so.conf.d/ and run sudo ldconfig, then run this script again. Ollaya will use the CPU."
            ;;
        unsupported)
            status "AMD GPU ($ROCM_ARCH) found, but it is not currently supported by Ollaya's ROCm runtime (supported discrete cards: gfx1200/1201, gfx1100-1102, gfx1030-1032/1034, gfx1010-1012, gfx908/90a/942/950). Ollaya will use the CPU."
            ;;
        nodriver)
            warn "AMD GPU found, but /dev/kfd or /dev/dri was not found or accessible. Ollaya will use the CPU."
            warn "Ensure the amdgpu driver and ROCm compute support are loaded (see https://rocm.docs.amd.com/)."
            ;;
        *) ;;
    esac

    # --- download --------------------------------------------------------------------------

    if [ "$OS" = Darwin ] && [ -z "$ZSTD" ]; then
        BASE_ARCHIVE=ollaya-$PLATFORM.tgz
    else
        BASE_ARCHIVE=ollaya-$PLATFORM.tar.zst
    fi
    fetch_verified "$BASE_ARCHIVE"
    CUDA_ARCHIVE=
    CUDA_KEEP=false
    if $WANT_CUDA; then
        # The release lists the sha256 of every CUDA library (ollaya-<platform>-cuda.sha256, also
        # installed as FILES.sha256). When the installed libraries match it, keep them instead of
        # downloading the same ~1 GB again. Releases before 0.4.0 have no such file.
        CUDA_SIZE="about 1 GB"
        [ "$CUDA_PACK" = cuda_v13 ] || CUDA_SIZE="about 1.6 GB"
        CUDA_DIR=$PREFIX/lib/ollaya/$CUDA_PACK
        CUDA_FILES=ollaya-$PLATFORM-cuda$CUDA_SUFFIX.sha256
        if [ -f "$CUDA_DIR/FILES.sha256" ] && grep -q " $CUDA_FILES\$" "$TMP/sha256sum.txt" &&
            (fetch_verified "$CUDA_FILES") >/dev/null 2>&1 && cmp -s "$TMP/$CUDA_FILES" "$CUDA_DIR/FILES.sha256" &&
            cuda_intact "$CUDA_DIR"; then
            CUDA_KEEP=true
            status "The NVIDIA CUDA libraries are unchanged; keeping the installed copy"
        else
            CUDA_ARCHIVE=ollaya-$PLATFORM-cuda$CUDA_SUFFIX.tar.zst
            status "Downloading the NVIDIA CUDA ${CUDA_PACK#cuda_v} libraries ($CUDA_SIZE)"
            fetch_verified "$CUDA_ARCHIVE"
        fi
    fi
    # The MLX engine's Metal kernels (lib/ollaya/mlx_metal), kept like the CUDA libraries when the
    # installed copy matches the release's ollaya-darwin-arm64-mlx.sha256. Releases before 0.7.0
    # have no MLX archive.
    MLX_ARCHIVE=
    MLX_KEEP=false
    if [ "$PLATFORM" = darwin-arm64 ] && grep -q " ollaya-$PLATFORM-mlx\.tgz\$" "$TMP/sha256sum.txt"; then
        MLX_DIR=$PREFIX/lib/ollaya/mlx_metal
        MLX_FILES=ollaya-$PLATFORM-mlx.sha256
        if [ -f "$MLX_DIR/FILES.sha256" ] && grep -q " $MLX_FILES\$" "$TMP/sha256sum.txt" &&
            (fetch_verified "$MLX_FILES") >/dev/null 2>&1 && cmp -s "$TMP/$MLX_FILES" "$MLX_DIR/FILES.sha256" &&
            cuda_intact "$MLX_DIR"; then
            MLX_KEEP=true
            status "The MLX engine's Metal library is unchanged; keeping the installed copy"
        else
            if [ -z "$ZSTD" ]; then MLX_ARCHIVE=ollaya-$PLATFORM-mlx.tgz; else MLX_ARCHIVE=ollaya-$PLATFORM-mlx.tar.zst; fi
            status "Downloading the MLX engine's Metal library"
            fetch_verified "$MLX_ARCHIVE"
        fi
    fi
    ROCM_ARCHIVE=
    ROCM_KEEP=false
    if $WANT_ROCM && grep -q " ollaya-$PLATFORM-rocm\.tar\.zst\$" "$TMP/sha256sum.txt"; then
        ROCM_DIR=$PREFIX/lib/ollaya/rocm
        ROCM_FILES=ollaya-$PLATFORM-rocm.sha256
        if [ -f "$ROCM_DIR/FILES.sha256" ] && grep -q " $ROCM_FILES\$" "$TMP/sha256sum.txt" &&
            (fetch_verified "$ROCM_FILES") >/dev/null 2>&1 && cmp -s "$TMP/$ROCM_FILES" "$ROCM_DIR/FILES.sha256" &&
            cuda_intact "$ROCM_DIR"; then
            ROCM_KEEP=true
            status "The AMD ROCm libraries are unchanged; keeping the installed copy"
        else
            ROCM_ARCHIVE=ollaya-$PLATFORM-rocm.tar.zst
            status "Downloading the AMD ROCm libraries"
            fetch_verified "$ROCM_ARCHIVE"
        fi
    fi

    # --- install ---------------------------------------------------------------------------

    # Unpack next to the destination (same filesystem), then move into place: the binary is
    # replaced by a rename, which is safe while an old ollaya is running.
    $SUDO mkdir -p "$PREFIX"
    STAGE=$PREFIX/.ollaya-install.$$
    $SUDO rm -rf "$STAGE"
    $SUDO mkdir -p "$STAGE"
    unpack() {
        case $1 in
            *.tgz) $SUDO tar -xzf "$TMP/$1" -C "$STAGE" ;;
            *) zstd -dc "$TMP/$1" | $SUDO tar -xf - -C "$STAGE" ;;
        esac
        rm -f "$TMP/$1"
    }
    unpack "$BASE_ARCHIVE"
    [ -z "$CUDA_ARCHIVE" ] || unpack "$CUDA_ARCHIVE"
    [ -z "$MLX_ARCHIVE" ] || unpack "$MLX_ARCHIVE"
    [ -z "$ROCM_ARCHIVE" ] || unpack "$ROCM_ARCHIVE"
    [ -f "$STAGE/bin/ollaya" ] || error "$BASE_ARCHIVE does not contain bin/ollaya"
    if [ -n "$CUDA_ARCHIVE" ] && [ ! -f "$STAGE/lib/ollaya/$CUDA_PACK/libonnxruntime_providers_cuda.so" ]; then
        error "$CUDA_ARCHIVE does not contain lib/ollaya/$CUDA_PACK"
    fi

    $SUDO mkdir -p "$BINDIR" "$PREFIX/share/doc"
    $SUDO chmod 0755 "$STAGE/bin/ollaya"
    $SUDO mv -f "$STAGE/bin/ollaya" "$BINDIR/ollaya"
    # Kept CUDA libraries keep their notices too (the base archive replaces share/doc/ollaya).
    if $CUDA_KEEP && [ -d "$PREFIX/share/doc/ollaya/$CUDA_PACK" ] && [ ! -e "$STAGE/share/doc/ollaya/$CUDA_PACK" ]; then
        $SUDO mv "$PREFIX/share/doc/ollaya/$CUDA_PACK" "$STAGE/share/doc/ollaya/$CUDA_PACK"
    fi
    if $MLX_KEEP && [ -d "$PREFIX/share/doc/ollaya/mlx_metal" ] && [ ! -e "$STAGE/share/doc/ollaya/mlx_metal" ]; then
        $SUDO mv "$PREFIX/share/doc/ollaya/mlx_metal" "$STAGE/share/doc/ollaya/mlx_metal"
    fi
    if $ROCM_KEEP && [ -d "$PREFIX/share/doc/ollaya/rocm" ] && [ ! -e "$STAGE/share/doc/ollaya/rocm" ]; then
        $SUDO mv "$PREFIX/share/doc/ollaya/rocm" "$STAGE/share/doc/ollaya/rocm"
    fi
    $SUDO rm -rf "$PREFIX/share/doc/ollaya"
    $SUDO mv "$STAGE/share/doc/ollaya" "$PREFIX/share/doc/ollaya"
    # The agent skill (0.4.0 and later). Only share/ollaya/skills is replaced: with a /usr prefix,
    # share/ollaya is also the systemd service's home, which holds the models.
    if [ -d "$STAGE/share/ollaya/skills" ]; then
        $SUDO mkdir -p "$PREFIX/share/ollaya"
        $SUDO rm -rf "$PREFIX/share/ollaya/skills"
        $SUDO mv "$STAGE/share/ollaya/skills" "$PREFIX/share/ollaya/skills"
    fi
    # lib/ollaya is replaced as a whole, so libraries never outlive the binary they match. The
    # CUDA and ROCm libraries are the exception when they are byte for byte the ones this release ships:
    # they move into the new tree unchanged.
    if $CUDA_KEEP; then
        $SUDO mkdir -p "$STAGE/lib/ollaya"
        $SUDO rm -rf "$STAGE/lib/ollaya/$CUDA_PACK"
        $SUDO mv "$PREFIX/lib/ollaya/$CUDA_PACK" "$STAGE/lib/ollaya/$CUDA_PACK"
    fi
    if $MLX_KEEP; then
        $SUDO mkdir -p "$STAGE/lib/ollaya"
        $SUDO rm -rf "$STAGE/lib/ollaya/mlx_metal"
        $SUDO mv "$PREFIX/lib/ollaya/mlx_metal" "$STAGE/lib/ollaya/mlx_metal"
    fi
    if $ROCM_KEEP; then
        $SUDO mkdir -p "$STAGE/lib/ollaya"
        $SUDO rm -rf "$STAGE/lib/ollaya/rocm"
        $SUDO mv "$PREFIX/lib/ollaya/rocm" "$STAGE/lib/ollaya/rocm"
    fi
    $SUDO rm -rf "$PREFIX/lib/ollaya"
    if [ -d "$STAGE/lib/ollaya" ]; then
        $SUDO mkdir -p "$PREFIX/lib"
        $SUDO mv "$STAGE/lib/ollaya" "$PREFIX/lib/ollaya"
    fi
    $SUDO rm -rf "$STAGE"
    STAGE=
    # Moved files keep the SELinux label of the staging directory; give them their final labels.
    if available restorecon; then
        $SUDO restorecon -R "$BINDIR/ollaya" "$PREFIX/share/doc/ollaya" 2>/dev/null || :
        [ ! -d "$PREFIX/lib/ollaya" ] || $SUDO restorecon -R "$PREFIX/lib/ollaya" 2>/dev/null || :
    fi

    if ! out=$("$BINDIR/ollaya" --version 2>&1); then
        warn "$BINDIR/ollaya did not run: $out"
    fi
    case ":${PATH:-}:" in
        *":$BINDIR:"*) ;;
        *)
            # The file the user's login shell reads, so the hint works when pasted as is.
            case ${SHELL:-} in
                */zsh) rc="${HOME:-}/.zshrc" ;;
                */bash) if [ "$OS" = Darwin ]; then rc="${HOME:-}/.bash_profile"; else rc="${HOME:-}/.bashrc"; fi ;;
                *) rc="${HOME:-}/.profile" ;;
            esac
            case ${SHELL:-} in
                */fish) warn "$BINDIR is not on your PATH. Add it: fish_add_path $BINDIR" ;;
                *) warn "$BINDIR is not on your PATH. Add it: echo 'export PATH=\"$BINDIR:\$PATH\"' >>$rc" ;;
            esac
            ;;
    esac
    found=$(command -v ollaya 2>/dev/null || :)
    if [ -n "$found" ] && [ "$found" != "$BINDIR/ollaya" ]; then
        warn "another ollaya at $found comes first on your PATH"
    fi

    # A server that the CLI started in the background keeps running the old binary after an
    # upgrade. Stop this user's one; the next ollaya command starts the new version. (The systemd
    # service, which runs as another user, is restarted below.)
    if available pgrep; then
        old=$(pgrep -u "$(id -u)" -f "^$BINDIR/ollaya serve\$" 2>/dev/null || :)
        if [ -n "$old" ]; then
            for pid in $old; do kill "$pid" 2>/dev/null || :; done
            # It finishes open requests and stops its runners: up to 5 s.
            i=0
            for pid in $old; do
                while [ $i -lt 10 ] && kill -0 "$pid" 2>/dev/null; do
                    i=$((i + 1))
                    sleep 1
                done
            done
            status "Stopped the running Ollaya server; the next ollaya command starts version $VERSION"
        fi
    fi

    # --- systemd service -------------------------------------------------------------------

    systemd_unit() {
        # BEGIN packaging/ollaya.service (CI checks that this copy matches the file)
        cat <<EOF
# Ollaya daemon. scripts/install.sh writes this unit to /etc/systemd/system/ollaya.service with
# ExecStart pointing at the installed binary; CI checks that its copy matches this file.
# Change settings with a drop-in (sudo systemctl edit ollaya), for example:
#   [Service]
#   Environment="OLLAYA_HOST=0.0.0.0:11435"
#   Environment="OLLAYA_KEEP_ALIVE=30m"
#   Environment="OLLAYA_LOG=debug"
#   Environment="OLLAYA_LOG_DIR=/var/log/ollaya"
#   LogsDirectory=ollaya
[Unit]
Description=Ollaya Service
After=network-online.target

[Service]
ExecStart=$BINDIR/ollaya serve
User=ollaya
Group=ollaya
Restart=always
RestartSec=3
Environment="HOME=/usr/share/ollaya"
Environment="OLLAYA_HOST=127.0.0.1:11435"
Environment="OLLAYA_MODELS=/usr/share/ollaya/.ollaya/models"

[Install]
WantedBy=default.target
EOF
        # END packaging/ollaya.service
    }

    configure_systemd() {
        if ! id ollaya >/dev/null 2>&1; then
            status "Creating the ollaya system user"
            $SUDO useradd -r -s /bin/false -U -m -d /usr/share/ollaya ollaya
        fi
        for group in render video; do
            if getent group "$group" >/dev/null 2>&1; then
                $SUDO usermod -a -G "$group" ollaya
            fi
        done
        user=$(id -un)
        if [ "$user" != root ]; then
            status "Adding $user to the ollaya group"
            $SUDO usermod -a -G ollaya "$user"
        fi
        status "Creating the ollaya systemd service"
        unit=/etc/systemd/system/ollaya.service
        systemd_unit >"$TMP/ollaya.service"
        # Every install rewrites the unit, `ollaya update` too, so a setting added to it is lost;
        # drop-ins (ollaya.service.d) are never touched. When the unit has lines the new one does
        # not, comments aside, keep a copy next to it and name the lines, with the values of keys,
        # tokens and passwords masked.
        if [ -f "$unit" ]; then
            # Read through $SUDO: a unit that holds a key may be readable by root only.
            dropped=$($SUDO cat "$unit" | grep -v -e '^[[:space:]]*[#;]' -e '^[[:space:]]*$' |
                grep -vxF -f "$TMP/ollaya.service" || :)
            if [ -n "$dropped" ]; then
                saved=$unit.$(date +%Y%m%d-%H%M%S).bak
                $SUDO cp -p "$unit" "$saved"
                warn "$unit has settings this install replaces. The old file is saved as $saved; these lines are not in the new one:"
                printf '%s\n' "$dropped" |
                    sed -E 's/([A-Za-z0-9_]*(KEY|TOKEN|SECRET|PASSWORD)[A-Za-z0-9_]*=)[^" ]*/\1***/g; s/^/    /' >&2
                warn "every install and ollaya update rewrites $unit. Keep settings in a drop-in, which they never touch: sudo systemctl edit ollaya"
            fi
        fi
        $SUDO tee "$unit" <"$TMP/ollaya.service" >/dev/null
        $SUDO systemctl daemon-reload
        $SUDO systemctl enable ollaya >/dev/null
        $SUDO systemctl restart ollaya
        i=0
        while [ $i -lt 15 ]; do
            if curl -fsS "http://127.0.0.1:$PORT/" 2>/dev/null | grep -q 'Ollaya is running'; then
                return 0
            fi
            i=$((i + 1))
            sleep 1
        done
        warn "the ollaya service did not answer on 127.0.0.1:$PORT yet; check: journalctl -u ollaya"
    }

    SERVICE=false
    if [ "$OS" = Linux ]; then
        if enabled "${OLLAYA_NO_SERVICE:-}"; then
            status "Skipping the systemd service (OLLAYA_NO_SERVICE is set)"
        elif ! available systemctl || [ ! -d /run/systemd/system ]; then
            if [ "$WSL" = 2 ]; then
                status "systemd is not running in this WSL distro, so no service was created."
                status "To enable it: https://learn.microsoft.com/windows/wsl/systemd"
            else
                status "systemd is not running, so no service was created"
            fi
        elif ! $PRIVILEGED; then
            status "Installed without root rights, so no systemd service was created"
        else
            configure_systemd
            SERVICE=true
        fi
    fi

    # --- summary ---------------------------------------------------------------------------

    status "Installed Ollaya $VERSION: $BINDIR/ollaya"
    if [ -n "$CUDA_ARCHIVE" ] || $CUDA_KEEP; then
        status "NVIDIA GPU support: $PREFIX/lib/ollaya/$CUDA_PACK${CUDA_DRIVER:+ (driver supports CUDA $CUDA_DRIVER)}"
    elif [ -n "$ROCM_ARCHIVE" ] || $ROCM_KEEP; then
        status "AMD GPU support: $PREFIX/lib/ollaya/rocm${ROCM_ARCH:+ ($ROCM_ARCH)}"
    elif [ "$NVIDIA_STATE" = none ] && [ "$ROCM_STATE" = none ] && [ "$OS" = Linux ]; then
        status "No supported GPU found; Ollaya will run on the CPU"
    fi
    # winnow:e4b (the recommended model, 8 GB) with an accelerator GPU that holds it; laya, which is
    # fast on any CPU and on small GPUs, otherwise.
    START_MODEL=laya
    if { [ -n "$CUDA_ARCHIVE" ] || $CUDA_KEEP || [ -n "$ROCM_ARCHIVE" ] || $ROCM_KEEP; } && [ -n "$GPU_MIB" ] && [ "$GPU_MIB" -ge 10240 ]; then
        START_MODEL=winnow:e4b
    fi
    if $SERVICE; then
        status "The Ollaya API is available at http://127.0.0.1:$PORT (systemd service: ollaya)"
    fi
    # The CLI starts the server in the background when none is running.
    status "Get started:  ollaya run $START_MODEL"
}

main "$@"
