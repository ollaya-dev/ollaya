#!/bin/sh
# Install Ollaya on Linux or macOS:
#
#   curl -fsSL https://ollaya.cobanov.dev/install.sh | sh
#
# Settings, passed as environment variables to the shell that runs the script
# (curl -fsSL https://ollaya.cobanov.dev/install.sh | OLLAYA_VERSION=0.1.0 sh):
#   OLLAYA_VERSION      version to install, such as 0.1.0 (default: the latest release)
#   OLLAYA_REPO         GitHub repository to download releases from (default: ollaya-dev/ollaya)
#   OLLAYA_INSTALL_DIR  install prefix, an absolute path; the binary goes to $OLLAYA_INSTALL_DIR/bin
#                       (default: /usr/local, or ~/.local without root or sudo)
#   OLLAYA_NO_SERVICE=1 don't create or start the systemd service
#   OLLAYA_NO_CUDA=1    don't download the CUDA libraries, even when there is an NVIDIA GPU
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
            PLATFORM=darwin-arm64
            ;;
        *) error "this script supports Linux and macOS only (found $OS)" ;;
    esac
    [ "$WSL" != 1 ] || warn "WSL 1 is untested and has no GPU access; WSL 2 is recommended"

    if [ "$OS" = Linux ]; then
        # pyke's ONNX Runtime build references glibc 2.38 symbols: Ubuntu 24.04, Debian 13,
        # Fedora 39, RHEL 10 or newer.
        glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{ print $2 }') || glibc=
        [ -n "$glibc" ] || error "Ollaya needs glibc 2.38 or newer (musl systems such as Alpine are not supported; use the Docker image)"
        if ! printf '%s\n' "$glibc" | awk -F. '{ exit !($1 > 2 || ($1 == 2 && $2 >= 38)) }'; then
            error "Ollaya needs glibc 2.38 or newer, this system has $glibc (Ubuntu 24.04, Debian 13, Fedora 39, RHEL 10 or newer; or use the Docker image)"
        fi
    fi

    missing=
    for tool in curl tar awk mktemp; do available "$tool" || missing="$missing $tool"; done
    available sha256sum || available shasum || available openssl || missing="$missing sha256sum"
    [ -z "$missing" ] || error "required tools are missing:$missing"
    ZSTD=
    if available zstd; then
        ZSTD=zstd
    elif [ "$OS" = Linux ]; then
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
    prefix_writable() {
        writable "$1/bin" && writable "$1/share" && { [ "$OS" != Linux ] || writable "$1/lib"; }
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

    # NVIDIA_STATE: none | nodriver | oldriver | ready. Only linux-amd64 has a CUDA package.
    NVIDIA_STATE=none
    CUDA_DRIVER=
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
                    NVIDIA_STATE=oldriver
                fi
            fi
        fi
    fi

    WANT_CUDA=false
    case $NVIDIA_STATE in
        ready)
            if enabled "${OLLAYA_NO_CUDA:-}"; then
                status "NVIDIA GPU found; skipping the CUDA libraries (OLLAYA_NO_CUDA is set)"
            elif [ "$ARCH" != amd64 ]; then
                warn "NVIDIA GPU found, but GPU acceleration is only packaged for x86-64 so far; Ollaya will use the CPU"
            else
                WANT_CUDA=true
            fi
            ;;
        oldriver)
            warn "the NVIDIA driver supports CUDA $CUDA_DRIVER, but Ollaya's GPU libraries need a driver with CUDA 13 support (R580 or newer)."
            warn "Ollaya will use the CPU. Update the driver, then run this script again."
            ;;
        nodriver)
            warn "NVIDIA GPU found, but no NVIDIA driver is loaded. Ollaya will use the CPU."
            warn "Install the NVIDIA driver (R580 or newer; see https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), then run this script again."
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
    if $WANT_CUDA; then
        CUDA_ARCHIVE=ollaya-$PLATFORM-cuda.tar.zst
        status "Downloading the NVIDIA CUDA libraries (about 1 GB)"
        fetch_verified "$CUDA_ARCHIVE"
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
    [ -f "$STAGE/bin/ollaya" ] || error "$BASE_ARCHIVE does not contain bin/ollaya"
    if [ -n "$CUDA_ARCHIVE" ] && [ ! -f "$STAGE/lib/ollaya/cuda_v13/libonnxruntime_providers_cuda.so" ]; then
        error "$CUDA_ARCHIVE does not contain lib/ollaya/cuda_v13"
    fi

    $SUDO mkdir -p "$BINDIR" "$PREFIX/share/doc"
    $SUDO chmod 0755 "$STAGE/bin/ollaya"
    $SUDO mv -f "$STAGE/bin/ollaya" "$BINDIR/ollaya"
    $SUDO rm -rf "$PREFIX/share/doc/ollaya"
    $SUDO mv "$STAGE/share/doc/ollaya" "$PREFIX/share/doc/ollaya"
    # Remove GPU libraries from an earlier install, so they never outlive the binary they match.
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
#   Environment="OLLAYA_DEBUG=1"
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
        systemd_unit | $SUDO tee /etc/systemd/system/ollaya.service >/dev/null
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
    if [ -n "$CUDA_ARCHIVE" ]; then
        status "NVIDIA GPU support: $PREFIX/lib/ollaya/cuda_v13${CUDA_DRIVER:+ (driver supports CUDA $CUDA_DRIVER)}"
    elif [ "$NVIDIA_STATE" = none ] && [ "$OS" = Linux ]; then
        status "No NVIDIA GPU found; Ollaya will run on the CPU"
    fi
    if $SERVICE; then
        status "The Ollaya API is available at http://127.0.0.1:$PORT (systemd service: ollaya)"
        status "Get started:  ollaya run laya"
    else
        # The CLI starts the server in the background when none is running.
        status "Get started:  ollaya run laya"
    fi
}

main "$@"
