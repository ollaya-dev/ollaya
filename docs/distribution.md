# Distribution

How Ollaya is packaged, installed and released, and the contract between the install layout and the
daemon that spawns GPU runners.

| File | Role |
|---|---|
| `scripts/package.sh` | Builds the release archives from a cargo target directory |
| `scripts/install.sh` | The `curl -fsSL https://ollaya.dev/install.sh \| sh` installer (the site serves this file) |
| `packaging/ollaya.service` | The systemd unit. `install.sh` embeds a copy, and CI checks that the two match |
| `packaging/cuda-requirements.{in,txt}` | Pinned, hash-locked NVIDIA wheels that the CUDA libraries come from |
| `packaging/docker-healthcheck.sh` | The image's `HEALTHCHECK` |
| `Dockerfile`, `.dockerignore` | The `cpu` and `cuda` images |
| `.github/workflows/ci.yml` | fmt, clippy and tests (CPU), the site build, and lint for the packaging scripts |
| `.github/workflows/release.yml` | On a `v*` tag: tarballs, a GitHub Release and the GHCR images |

## Release artifacts

Each release on GitHub (`ollaya-dev/ollaya`, set with `OLLAYA_REPO`) carries:

| Archive | Contents | Size |
|---|---|---|
| `ollaya-linux-amd64.tar.zst` | `bin/ollaya`, `share/doc/ollaya/` | 7.7 MiB* |
| `ollaya-linux-amd64-cuda.tar.zst` | `lib/ollaya/cuda_v13/` (20 libraries), `share/doc/ollaya/cuda_v13/` | 1092 MiB (2069 MiB unpacked) |
| `ollaya-linux-arm64.tar.zst` | `bin/ollaya`, `share/doc/ollaya/` (CPU only) | |
| `ollaya-darwin-arm64.tar.zst` | `bin/ollaya`, `share/doc/ollaya/` (CPU and CoreML) | |
| `ollaya-darwin-arm64.tgz` | The same as the darwin `.tar.zst`. Stock macOS has no `zstd`, so `install.sh` falls back to it | |
| `sha256sum.txt` | SHA-256 of every archive, in `sha256sum -c` format | |

\* Measured with the 34 MB `parity` example standing in for `ollaya`, which isn't built yet.

Notes on the archives:
- **One linux-amd64 binary.** It is built with `--features ollaya-runner/cuda`, which puts ORT's
  provider bridge into the binary. The same binary runs on the CPU and only uses the GPU when the
  `-cuda` archive is installed next to it, as Ollama's is.
- **Deterministic.** The archives are built with GNU tar (`--sort=name`, owner 0, `SOURCE_DATE_EPOCH`
  set to the last commit). Two runs of `package.sh` produce the same SHA-256.
- **Top-level entries only.** An archive holds only `bin/`, `lib/` and `share/`, never `.`, so
  unpacking it over a prefix never changes the prefix directory's own mode.

`share/doc/ollaya/` holds `LICENSE`, `THIRD_PARTY_NOTICES` and `onnxruntime-ThirdPartyNotices.txt`.
The CUDA archive adds `share/doc/ollaya/cuda_v13/`, with its own `THIRD_PARTY_NOTICES` and
`licenses/`, the license text from each NVIDIA wheel.

### What `lib/ollaya/cuda_v13` contains

| File | Source | Loaded for Laya |
|---|---|---|
| `libonnxruntime_providers_shared.so`, `libonnxruntime_providers_cuda.so` | pyke's ORT 1.28 `cuda13,tensorrt,nvrtx` build (copied from `target/release` by `copy-dylibs`) | yes |
| `libcudart.so.13` | `nvidia-cuda-runtime==13.4.92` | yes |
| `libcublas.so.13`, `libcublasLt.so.13` | `nvidia-cublas==13.8.0.4` | yes |
| `libcurand.so.10` | `nvidia-curand==10.4.4.72` | yes |
| `libcudnn.so.9`, `libcudnn_graph.so.9` | `nvidia-cudnn-cu13==9.26.0.51` | yes |
| the other 8 `libcudnn_*.so.9` | `nvidia-cudnn-cu13==9.26.0.51` | no |
| `libcufft.so.12` | `nvidia-cufft==12.4.0.43` | no |
| `libnvrtc.so.13`, `libnvrtc-builtins.so.13.4` | `nvidia-cuda-nvrtc==13.4.92` | no |
| `libnvJitLink.so.13` | `nvidia-nvjitlink==13.4.92` | no |

- **Loaded for Laya.** This column was measured with `LD_DEBUG=files` on `laya:en` fp32 and fp16 and
  on `laya:multilingual` fp16. The loaded files add up to 900 MiB of the 2069 MiB. The rest covers
  ops that other architectures may need (convolutions, FFT, cuDNN fused attention, runtime-compiled
  kernels). The full set ships so that a new model never fails on a missing library. See the open
  questions at the end.
- **Left out.** Two providers stay out: `libonnxruntime_providers_tensorrt.so` and
  `libonnxruntime_providers_nv_tensorrt_rtx.so`. They need TensorRT 10 (`libnvinfer.so.10`), which
  isn't shipped, and the runner registers only the CUDA execution provider.
- **Changing the set.** The lists live in `ORT_PROVIDERS` and the `case` in `stage_cuda` in
  `package.sh`.
- **Updating the pins.** Edit the pins in `packaging/cuda-requirements.in`. Then regenerate the hash
  lock with the `uv pip compile` command in that file's header.
- **Download.** `package.sh` fetches the wheels with `pip download --require-hashes`. It falls back to
  `uvx pip` when there's no `pip`; `uv pip download` doesn't exist. The wheels are cached in
  `~/.cache/ollaya-package/wheels`, about 1.2 GB.

## Supported systems

| | Requirement | Why |
|---|---|---|
| Linux | glibc 2.38 or newer: Ubuntu 24.04, Debian 13, Fedora 39, RHEL 10 or newer | pyke's `libonnxruntime.a` references `__isoc23_strtol` and related symbols (`GLIBC_2.38`) and `GLIBCXX_3.4.31`. The provider `.so` needs glibc 2.38 too. `install.sh` checks this. Older hosts can use the Docker image. |
| Linux, GPU | NVIDIA driver R580 or newer (CUDA 13) on x86-64 | Uses pyke's CUDA 13 build. The driver's CUDA major version must be at least 13; minor-version compatibility covers the 13.4 runtime. |
| WSL 2 | Windows NVIDIA driver with CUDA 13 | The driver's `libcuda.so.1` comes from `/usr/lib/wsl/lib`. Never install a Linux driver inside WSL. |
| macOS | Apple silicon | The minimum macOS version comes from pyke's CoreML build. The release workflow prints it with `vtool -show-build`. |
| musl (Alpine) | not supported | glibc build. Use the Docker image. |

- **Runtime libraries.** The binary itself needs only `libc`, `libm`, `libstdc++` and `libgcc_s`.
- **Shared tested build.** The CUDA path was tested with the pyke build that the dev build uses
  (`b89451ba…`), under WSL 2 with driver 616.92 (CUDA UMD 13.4) on an RTX 4090.

## The runtime library contract (for the daemon)

### How ONNX Runtime finds its provider libraries

ORT's core is linked statically into `ollaya`. The CUDA execution provider isn't part of it: it lives
in `libonnxruntime_providers_cuda.so`, loaded at run time through `libonnxruntime_providers_shared.so`.
In ORT 1.28 (`onnxruntime/core/session/provider_bridge_ort.cc`), both are loaded by absolute path:

```
ProviderSharedLibrary::Initialize: dlopen(Env::Default().GetRuntimePath() + "libonnxruntime_providers_shared.so")
ProviderLibrary::Load:             dlopen(Env::Default().GetRuntimePath() + "libonnxruntime_providers_cuda.so")
```

`PosixEnv::GetRuntimePath()` (`onnxruntime/core/platform/posix/env.cc`) is the directory of whatever
`dladdr(&Env::Default)` reports, made absolute against the current directory:

- **In a shared `libonnxruntime.so`,** that is the library's directory, the usual "next to
  libonnxruntime" rule.
- **Statically linked, as in `ollaya`,** the symbol is in the main program. For the main program,
  glibc's `dladdr` returns **`argv[0]`**, not the executable's real path (`elf/dl-addr.c`, checked
  with a test program). So ORT looks for its providers in `dirname(absolute(argv[0]))`.

That rules out the other options:
- **Preloading doesn't work.** A `dlopen` of an absolute path never matches a library that was
  preloaded from elsewhere, so `ort::util::preload_dylib` can't redirect the providers. pyke's docs
  say there is no API for this path. `ort::ep::cuda::preload_dylibs` also hardcodes CUDA 12 file
  names.
- **`Environment::register_ep_library(path)` doesn't either.** It loads a provider from an absolute
  path, but `ProviderLibrary::Load` still loads `libonnxruntime_providers_shared.so` from
  `GetRuntimePath()` first. It would also mean moving to the EP-device session API.
- **An rpath can't reach the providers**, because they're loaded by absolute path. Patching the NVIDIA
  libraries' rpath (`patchelf`) is not allowed: the EULA only permits redistributing unmodified
  files.

So the daemon controls ORT's lookup through the runner's `argv[0]`, and the providers' own
dependencies through `LD_LIBRARY_PATH`. The provider has no RUNPATH:
- **Direct dependencies.** Its `DT_NEEDED` entries are `libcudart.so.13`, `libcublas.so.13`,
  `libcublasLt.so.13`, `libcurand.so.10` and `libcuda.so.1`.
- **Opened by name.** It `dlopen`s `libcudnn.so.9`, `libcufft.so.12` and `libnvrtc.so.13` by name.
- **Found without help.** cuDNN finds its sub-libraries through its own `$ORIGIN` rpath.
  `libcuda.so.1` belongs to the driver, which `ld.so.cache` finds. On WSL, that's through
  `/etc/ld.so.conf.d/ld.wsl.conf`.

### What the daemon must do

1. **Find the library directory**, first match wins:
   1. `$OLLAYA_LIBRARY_PATH`, if set. This override is only a proposal, for development.
   2. `dirname(current_exe())/../lib/ollaya`. That gives `/usr/local/lib/ollaya` for the tarball,
      `~/.local/lib/ollaya` for a user install and `/usr/lib/ollaya` in the Docker image.
      `std::env::current_exe()` already resolves symlinks.
   3. Development builds: `dirname(current_exe())` itself, when it contains
      `libonnxruntime_providers_cuda.so`. `copy-dylibs` puts the providers in `target/<profile>/`.
      The CUDA libraries then have to be on the caller's `LD_LIBRARY_PATH`, for example from the
      `convert/.venv` wheels.

   `CUDA_DIR` is `<libdir>/cuda_v13`, or the executable's directory in case 3.
2. **Decide on a GPU runner.** Use one only if both `CUDA_DIR/libonnxruntime_providers_shared.so` and
   `CUDA_DIR/libonnxruntime_providers_cuda.so` exist, and the daemon's own discovery (NVML, or
   `libcuda.so.1` with `cuInit` and `cuDeviceGetCount`) finds a usable NVIDIA GPU. Otherwise spawn a
   CPU runner with no special environment.
3. **Spawn the GPU runner** like this:

   ```rust
   use std::os::unix::process::CommandExt as _;

   let exe = std::env::current_exe()?;                  // absolute, symlinks resolved
   let mut cmd = std::process::Command::new(&exe);
   cmd.arg0(cuda_dir.join("ollaya"));                    // ORT's provider directory; the file need not exist
   let mut ld = std::ffi::OsString::from(&cuda_dir);    // prepend, keep what the caller had
   if let Some(old) = std::env::var_os("LD_LIBRARY_PATH").filter(|v| !v.is_empty()) {
       ld.push(":");
       ld.push(old);
   }
   cmd.env("LD_LIBRARY_PATH", ld);
   cmd.args(["runner", /* engine, model, device, port… */]);
   ```

   - **`argv[0]` must be absolute.** A relative one, such as a bare `ollaya` from a `PATH` lookup,
     makes ORT load its providers from the current working directory. That is a failure at best and
     a library-injection hole at worst. The same holds for any process that registers the CUDA EP
     in-process, so the CLI must never do that.
   - **`LD_LIBRARY_PATH` must be set at exec time.** glibc reads it once at startup, so setting it
     inside the runner has no effect.
   - **Optional:** `CUDA_VISIBLE_DEVICES` to pin a runner to a GPU, as Ollama does.
4. **Runner self-check (recommended).** For the CUDA device, check at startup that
   `dirname(argv[0])/libonnxruntime_providers_shared.so` exists. If it doesn't, fail with a clear
   message rather than ORT's `Failed to load library`. Keep `error_on_failure()` on the CUDA EP so a
   GPU runner never silently falls back to the CPU. Don't call `ort::ep::cuda::preload_dylibs`, which
   uses CUDA 12 names.

- **macOS.** Nothing to do: the CoreML EP is part of the static library.
- **Docker.** The same rules apply. The binary is `/usr/bin/ollaya` and the libraries are in
  `/usr/lib/ollaya/cuda_v13`.

### Evidence

The real archives were installed into a temporary prefix with `install.sh`, with the `parity` example
standing in for `bin/ollaya`. The runner was then run under `env -i`, on an RTX 4090 under WSL 2:

| `argv[0]` | `LD_LIBRARY_PATH` | Result |
|---|---|---|
| `<prefix>/lib/ollaya/cuda_v13/ollaya` | `<prefix>/lib/ollaya/cuda_v13` | Works. `laya:en` fp32: 97 cases, 483 questions, 100% same decisions, max probability difference 2.2e-4. `laya:multilingual` fp32: 477 cases, 2,383 questions, 100% same decisions, max difference 6.5e-6. |
| `<prefix>/bin/ollaya` (the real path) | set | `Failed to load library <prefix>/bin/libonnxruntime_providers_shared.so` |
| `<prefix>/lib/ollaya/cuda_v13/ollaya` | unset | `…/libonnxruntime_providers_cuda.so with error: libcublasLt.so.13: cannot open shared object file` |
| `ollaya` (relative), cwd `/` | set | `Failed to load library /libonnxruntime_providers_shared.so` |
| any | none, CPU device | Works |

## install.sh

```sh
curl -fsSL https://ollaya.dev/install.sh | sh
curl -fsSL https://ollaya.dev/install.sh | OLLAYA_VERSION=0.1.0 sh
```

| Variable | Effect |
|---|---|
| `OLLAYA_VERSION` | Version to install (`0.1.0` or `v0.1.0`). Default: the latest release, resolved from the `/releases/latest` redirect (no API call, no rate limit). |
| `OLLAYA_REPO` | Repository to download from (default `ollaya-dev/ollaya`) |
| `OLLAYA_INSTALL_DIR` | Absolute install prefix. Default `/usr/local`, or `~/.local` without root or sudo. |
| `OLLAYA_NO_SERVICE=1` | Don't create or start the systemd service |
| `OLLAYA_NO_CUDA=1` | Don't download the CUDA archive, even with an NVIDIA GPU |
| `OLLAYA_DOWNLOAD_BASE` | Undocumented in the script header. It downloads the release files from this URL instead of GitHub. For testing. |

What it does:
1. **Detects the platform.**
   - OS and architecture. Under Rosetta, a Mac is treated as arm64.
   - WSL 2, from the kernel string or `WSLInterop`. WSL 1 only gets a warning.
   - Linux: glibc 2.38 or newer, or it exits.
2. **Chooses the prefix.**
   - `/usr/local` when it's writable (or you're root), or with sudo. It uses `sudo -n` first, then a
     password prompt when there's a terminal.
   - Otherwise it falls back to `~/.local` with a message.
   - An explicit `OLLAYA_INSTALL_DIR` never falls back.
3. **Detects NVIDIA hardware** (Linux x86-64).
   - WSL 2: `/usr/lib/wsl/lib/libcuda.so.1` and the `nvidia-smi` next to it.
   - Elsewhere, in order: `nvidia-smi`, `/proc/driver/nvidia/version`, `lspci -d 10de:`, or a scan of
     `/sys/bus/pci/devices` for a display device (class `0x03`) with vendor `0x10de`.
   - The driver's CUDA version is read from the `nvidia-smi` header, in both the old
     `CUDA Version: 13.0` form and the newer `CUDA UMD Version: 13.4` form. Below 13, it warns and
     installs CPU only.
   - A GPU without a driver gets instructions. The script never installs drivers.
4. **Downloads `sha256sum.txt`, then each archive, and verifies it.** A mismatch, or an archive
   missing from the list, aborts before anything is installed.
5. **Unpacks into `<prefix>/.ollaya-install.<pid>`**, on the same filesystem, then moves things into
   place:
   - The binary is replaced by a rename, which is safe while an old daemon runs.
   - `lib/ollaya` is replaced as a whole, so GPU libraries never outlive the binary they match.
6. Runs `ollaya --version` and warns if it fails. It also warns when `<prefix>/bin` isn't on `PATH`,
   or when another `ollaya` comes first.
7. **Sets up the service** on Linux when systemd is running (`/run/systemd/system`), including WSL with
   systemd enabled, and root rights are already in use. It creates the `ollaya` system user, with home
   `/usr/share/ollaya` and membership in `render` and `video`. It adds you to the `ollaya` group,
   writes `/etc/systemd/system/ollaya.service`, enables it and restarts it. Then it waits up to 15 s
   for `GET /` to answer `Ollaya is running`. If any of those conditions is missing, it says why it
   skipped the service.
8. **Prints next steps:** `ollaya run laya`, or `ollaya serve` first when there's no service.

Models for the service live in `/usr/share/ollaya/.ollaya/models` (`OLLAYA_MODELS` in the unit). To
change settings, run `sudo systemctl edit ollaya` and add `Environment=` lines.

To uninstall:

```sh
sudo systemctl disable --now ollaya && sudo rm /etc/systemd/system/ollaya.service
sudo rm -rf /usr/local/bin/ollaya /usr/local/lib/ollaya /usr/local/share/doc/ollaya
sudo userdel -r ollaya    # also deletes /usr/share/ollaya, including the models
```

### Testing install.sh locally

Serve a directory that holds the archives and `sha256sum.txt`, then point the script at it. Use a
temporary prefix, and a fake `sudo` that always fails. This machine has passwordless sudo, and the
fake keeps the script from escalating:

```sh
(cd dist && python3 -m http.server 8000 --bind 127.0.0.1) &
mkdir -p /tmp/fakebin && printf '#!/bin/sh\nexit 1\n' >/tmp/fakebin/sudo && chmod +x /tmp/fakebin/sudo
PATH=/tmp/fakebin:$PATH OLLAYA_DOWNLOAD_BASE=http://127.0.0.1:8000 \
  OLLAYA_INSTALL_DIR=/tmp/ollaya-test sh scripts/install.sh
```

## Docker

```sh
docker run -d --name ollaya -v ollaya:/home/ollaya/.ollaya -p 11435:11435 ghcr.io/ollaya-dev/ollaya
docker run -d --name ollaya --gpus=all -v ollaya:/home/ollaya/.ollaya -p 11435:11435 ghcr.io/ollaya-dev/ollaya:cuda
```

| Tag | Platforms | Contents |
|---|---|---|
| `:<version>`, `:latest` | linux/amd64, linux/arm64 | `debian:trixie-slim` + `ollaya-linux-<arch>` |
| `:<version>-cuda`, `:cuda` | linux/amd64 | the above + `ollaya-linux-amd64-cuda` |

A prerelease tag (one with `-`) only gets the versioned tags.

**Build.**
- The Rust stage (`rust:1.98-trixie`) compiles the binary with `cargo build --release --locked`. It
  uses cache mounts for the registry, pyke's ORT download and `target/`.
- It then runs `scripts/package.sh --stage`, and the runtime stages copy the staged trees into
  `/usr`.
- The `cuda` target adds a stage that runs `package.sh --cuda --stage` (with `uvx pip`).
- So the images hold exactly the tarballs' files, and the library contract above applies unchanged
  (`/usr/bin/ollaya`, `/usr/lib/ollaya/cuda_v13`).

**Why debian-slim and our libraries, not `nvidia/cuda:13.x-cudnn-runtime`.**
- **Same bytes as the tarball.** They're tested once, through one code path.
- **Smaller.** The image carries only the 20 files we need. NVIDIA's runtime images also install
  CUDA libraries Ollaya never loads, such as cuSPARSE, cuSOLVER, NPP and nvJPEG.
- **New enough.** Debian 13 has glibc 2.41, above ORT's 2.38 floor.
- **The driver comes from the host.** The NVIDIA Container Toolkit injects the host driver
  (`libcuda.so.1`, `nvidia-smi`), driven by `NVIDIA_VISIBLE_DEVICES=all` and
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, as Ollama's image does.

The host needs the NVIDIA Container Toolkit and a CUDA 13 driver.

**User and volume.**
- **User.** The image runs as `ollaya`, uid and gid 1000 (build argument `OLLAYA_UID`), numeric so
  Kubernetes can check `runAsNonRoot`.
- **Environment.** `HOME=/home/ollaya`, `OLLAYA_MODELS=/home/ollaya/.ollaya/models` and
  `OLLAYA_HOST=0.0.0.0:11435`.
- **Volume.** `VOLUME /home/ollaya/.ollaya`. The directory already belongs to `ollaya`, so a new named
  volume inherits that ownership.
- **Bind mounts.** A bind mount must be writable by uid 1000. Either `chown 1000:1000` the host
  directory, or run with `--user $(id -u):$(id -g)`; `OLLAYA_MODELS` is absolute, so the path still
  holds.
- **Coming from Ollama-style images.** There is no `/root/.ollaya`. Mount the volume at
  `/home/ollaya/.ollaya`.

**Health check.**
- `packaging/docker-healthcheck.sh` sends `GET /` over bash's `/dev/tcp` and expects
  `Ollaya is running`, so the image needs no curl.
- It takes the port from `OLLAYA_HOST`, including a scheme, a path or `[::]`.
- It checks 127.0.0.1, so the daemon must listen on loopback too, which `0.0.0.0` does.

Compose, to replace laya-app on choso-wsl:

```yaml
services:
  ollaya:
    image: ghcr.io/ollaya-dev/ollaya:cuda
    ports: ["11435:11435"]
    volumes: ["ollaya:/home/ollaya/.ollaya"]
    environment:
      OLLAYA_KEEP_ALIVE: 30m
    deploy:
      resources:
        reservations:
          devices: [{ driver: nvidia, count: all, capabilities: [gpu] }]
    restart: unless-stopped
volumes:
  ollaya:
```

## Release process

1. **One-time setup.**
   - Create the `ollaya-dev/ollaya` repository and allow Actions.
   - After the first release, make the GHCR package `ollaya` public and link it to the repository.
   - The workflows need no secrets: they use `GITHUB_TOKEN`, with `contents: write` for the release
     and `packages: write` for GHCR.
2. **Version.**
   - The workflows pass the tag without its `v` as `OLLAYA_BUILD_VERSION`, in the environment of
     `cargo build` and as a Docker build argument.
   - The CLI should report
     `option_env!("OLLAYA_BUILD_VERSION").unwrap_or(env!("CARGO_PKG_VERSION"))` from
     `ollaya --version`. The release workflow runs that as a smoke test. Alternatively, bump
     `workspace.package.version` before tagging.
3. **Dry run.** Actions → Release → *Run workflow* (`workflow_dispatch`) builds all three platforms,
   packages them and builds the three images, but publishes nothing.
4. **Tag.** `git tag v0.1.0 && git push origin v0.1.0` starts the workflow:
   - **`build`** runs on ubuntu-latest, ubuntu-24.04-arm and macos-latest. It builds, smoke-tests and
     packages each platform, including `--cuda` on linux-amd64, and uploads the archives.
   - **`release`** writes `sha256sum.txt` over all archives and creates the GitHub Release. It uses
     `softprops/action-gh-release` with generated notes; a tag with `-` becomes a prerelease and
     doesn't become latest.
   - **`image`** builds the images natively per architecture (cpu amd64, cpu arm64, cuda amd64) and
     pushes them by digest only.
   - **`manifest`** runs after `release` and `image` succeed. It tags the multi-arch index with
     `docker buildx imagetools create`.
5. **Check.**
   - `curl -fsSL https://ollaya.dev/install.sh | OLLAYA_VERSION=0.1.0 sh`
   - `docker run --rm ghcr.io/ollaya-dev/ollaya:0.1.0 --version`
   - The `/download` page.
6. **Roll back.** Mark the previous release as latest, or delete the bad one; `install.sh` follows
   GitHub's latest flag. Point the image tags back with
   `docker buildx imagetools create -t …:latest …:<previous>`, and the same for `:cuda`.

All actions are pinned to full commit SHAs, with the version in a comment.

**Bumping `ort` or ONNX Runtime.**
- Update `ORT_VERSION` and the two ONNX Runtime notice hashes in `package.sh`.
- Check which CUDA and cuDNN versions pyke's new CUDA build needs, then update
  `packaging/cuda-requirements.in` and regenerate the `.txt`.
- Re-run the GPU parity check against an installed layout (see Evidence above).

## Licensing

- **Ollaya** is Apache-2.0. `LICENSE` ships in `share/doc/ollaya/`.
- **ONNX Runtime 1.28.0** is MIT. It's linked statically into `bin/ollaya`, and the CUDA providers
  ship as `.so` files. Both archives include the MIT text in `THIRD_PARTY_NOTICES` and ORT's
  `ThirdPartyNotices.txt`, which covers what ORT bundles: protobuf, ONNX, abseil, CUTLASS and more.
  `package.sh` fetches both files from the `v1.28.0` tag and checks their SHA-256.
- **Rust crates.**
  - `THIRD_PARTY_NOTICES` lists every crate compiled into the binary, from
    `cargo tree -e normal --target <triple>` with the release features, with version, SPDX license
    and repository.
  - After the list comes each distinct license file (`LICENSE*`, `COPYING*`, `NOTICE*`) once, with
    the crates that ship it, read from cargo's registry sources.
  - This is a lighter-weight equivalent of `cargo about`. It errs toward listing too much:
    proc-macro crates are included.
- **NVIDIA CUDA libraries.** They're covered by the NVIDIA Software License Agreement and CUDA
  Supplement (CUDA Toolkit EULA, v13.4, last updated 26 January 2026):
  - **Attachment A** lists as distributable, among others, `libcudart.so`, `libcublas.so`,
    `libcublasLt.so`, `libcufft.so`, `libcurand.so`, `libnvrtc.so`, `libnvrtc-builtins.so` and
    `libnvJitLink.so`, including versioned file names.
  - **§1.1.2** sets the conditions:
    - the application must add material functionality;
    - the distributed parts may only be accessed by the application;
    - our distribution terms must be consistent with the EULA;
    - NVIDIA must be told of any non-compliance we know of.
  - **§2.3** allows redistributing the Linux libraries "provided that the object code files are not
    modified in any way (except for unzipping of compressed files)".
- **NVIDIA cuDNN.** The NVIDIA cuDNN Software License Agreement has the same distribution
  requirements. Its supplement makes "the runtime files .so and .dll" distributable and licenses the
  SDK for use only on systems with NVIDIA GPUs.
- **How we comply.**
  - We ship only those runtime files, byte for byte from NVIDIA's hash-pinned PyPI wheels, with no
    patchelf and no strip.
  - They come in a separate, optional archive with each wheel's license text and a notice that they
    are not covered by Apache-2.0.
  - They stay in `lib/ollaya/cuda_v13`: never on the system library path, and loaded only by
    Ollaya's runner, through a per-process `LD_LIBRARY_PATH`.
  - Ollama redistributes the same kinds of files in the same way (`lib/ollama/cuda_v12|v13`).
  - This reading is ours, not legal advice. NVIDIA answers licensing questions at
    nvidia-compute-license-questions@nvidia.com.
- **Docker images** redistribute the same files, plus Debian packages whose copyright files stay in
  the image.
- **Model weights** are never in any archive or image. `.dockerignore` excludes `*.onnx`,
  `*.onnx.data` and `*.safetensors`.

## Open questions

- **A slimmer CUDA package.** Laya loads 900 MiB of the 2069 MiB. Leaving out cuFFT, NVRTC, nvJitLink
  and the unused cuDNN libraries would cut the archive roughly in half, at the risk of a missing
  library for a future model. Alternatively, ship them as a second, optional archive.
- **The glibc 2.38 floor.** It comes from pyke's build and rules out Ubuntu 22.04, Debian 12 and
  RHEL 9, which can only use Docker. Removing it would mean building ONNX Runtime ourselves.
- **macOS.**
  - The `ollaya-runner/coreml` feature doesn't exist yet. `release.yml` and `package.sh` assume it.
  - pyke only ships a CoreML build for `aarch64-apple-darwin`, so every macOS build uses that
    library whether or not the feature is enabled.
  - The minimum macOS version is still unknown.
  - Binaries aren't signed or notarized. `curl` downloads carry no quarantine attribute, so they run.
- **Which repository is canonical.** `Cargo.toml` and the plan say `github.com/cobanov/ollaya` and
  `ghcr.io/cobanov/ollaya`; this work uses `ollaya-dev/ollaya`. The workflows derive the image name
  from the repository they run in.
- **`OLLAYA_LIBRARY_PATH`.** The development override proposed above is not a documented setting yet.
