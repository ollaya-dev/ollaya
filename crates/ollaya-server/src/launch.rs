//! How runner processes are launched: which executable, `argv[0]` and library path.
//!
//! ONNX Runtime is linked statically. It loads its CUDA provider libraries at run time from its
//! "runtime path" (`Env::GetRuntimePath`), and the provider then loads the NVIDIA libraries:
//!
//! * **Linux:** the runtime path is the directory of the process's `argv[0]`, and the provider's
//!   own CUDA dependencies come from `LD_LIBRARY_PATH`, which glibc reads only at exec. So a GPU
//!   runner is spawned with `argv[0] = <cuda dir>/ollaya` and `LD_LIBRARY_PATH = <cuda dir>:...`.
//! * **Windows:** the runtime path is the directory of the executable file itself
//!   (`GetModuleFileNameW`); `argv[0]` plays no part. So a GPU runner is started from a copy of
//!   the executable inside the CUDA directory (see [`runner_copy`]). That also makes the CUDA
//!   directory the runner's application directory, which Windows searches first for every DLL
//!   the provider loads, so no `PATH` change is needed.
//!
//! A pack built from Microsoft's ONNX Runtime GPU release (see
//! `docs/decisions/0004-cuda-onnxruntime-builds.md`) also holds ONNX Runtime itself
//! ([`ORT_LIBRARY`]). Its runners start from a second build of this executable, [`CUDA_RUNNER`],
//! which loads that library at run time (`ORT_DYLIB_PATH`), so its runtime path is the pack. The
//! statically linked executable never loads such a pack's providers: they belong to another
//! ONNX Runtime build. CPU runners keep starting from the statically linked executable
//! ([`RunnerLaunch::cpu_exe`]), so CPU numbers do not depend on whether the pack is installed.
//!
//! See `docs/distribution.md`, "The runtime library contract".

use std::io::Read as _;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The ONNX Runtime provider libraries that make a directory a CUDA runtime pack.
#[cfg(windows)]
pub const CUDA_PROVIDERS: [&str; 2] = [
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_cuda.dll",
];
#[cfg(not(windows))]
pub const CUDA_PROVIDERS: [&str; 2] = [
    "libonnxruntime_providers_shared.so",
    "libonnxruntime_providers_cuda.so",
];

/// The ONNX Runtime provider libraries that make a directory a ROCm runtime pack.
#[cfg(windows)]
pub const ROCM_PROVIDERS: [&str; 2] = [
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_rocm.dll",
];
#[cfg(not(windows))]
pub const ROCM_PROVIDERS: [&str; 2] = [
    "libonnxruntime_providers_shared.so",
    "libonnxruntime_providers_rocm.so",
];

/// The llama.cpp ROCm/HIP backend library name.
pub const ROCM_BACKEND: &str = if cfg!(windows) {
    "ggml-hip.dll"
} else {
    "libggml-hip.so"
};

/// ONNX Runtime itself, in a pack built from Microsoft's GPU release.
#[cfg(windows)]
pub const ORT_LIBRARY: &str = "onnxruntime.dll";
#[cfg(not(windows))]
pub const ORT_LIBRARY: &str = "libonnxruntime.so.1";

/// The executable GPU runners start from when the pack holds [`ORT_LIBRARY`]: ollaya built with
/// `ollaya-runner/cuda-dynamic`, next to the pack directory (`lib/ollaya/`).
#[cfg(windows)]
pub const CUDA_RUNNER: &str = "ollaya-cuda-runner.exe";
#[cfg(not(windows))]
pub const CUDA_RUNNER: &str = "ollaya-cuda-runner";

/// The executable ROCm runners start from when the pack holds [`ORT_LIBRARY`]: ollaya built with
/// `ollaya-runner/rocm-dynamic`, next to the pack directory (`lib/ollaya/`).
#[cfg(windows)]
pub const ROCM_RUNNER: &str = "ollaya-rocm-runner.exe";
#[cfg(not(windows))]
pub const ROCM_RUNNER: &str = "ollaya-rocm-runner";

/// The pack directories, in order of preference: CUDA 13, then CUDA 12 (for drivers older than
/// R580). An install has at most one of them.
pub const CUDA_PACKS: [&str; 2] = ["cuda_v13", "cuda_v12"];

/// The ROCm pack directories (`lib/ollaya/rocm`).
pub const ROCM_PACKS: [&str; 1] = ["rocm"];

/// How runner processes are started.
#[derive(Debug, Clone)]
pub struct RunnerLaunch {
    /// The executable to spawn as `<exe> runner ...`.
    pub exe: PathBuf,
    /// `argv[0]` for runners, where it differs from `exe` (Linux GPU runners).
    pub arg0: Option<PathBuf>,
    /// Extra environment for runner processes (the GPU library path on Linux).
    pub env: Vec<(String, String)>,
    /// llama.cpp's libraries, for GGUF models (see [`llama_dir`]).
    pub llama_dir: Option<PathBuf>,
    /// Where ONNX models start when they run on the CPU, if not from `exe`: the statically linked
    /// executable, with no `argv[0]` or environment changes, when `exe` is [`CUDA_RUNNER`].
    pub cpu_exe: Option<PathBuf>,
}

impl RunnerLaunch {
    /// Runners are this executable's hidden `runner` subcommand, set up for the CUDA runtime pack
    /// when this install has one.
    pub fn current() -> std::io::Result<Self> {
        let exe = std::env::current_exe()?;
        Ok(RunnerLaunch {
            llama_dir: llama_dir(&exe),
            ..runner_launch(&exe)
        })
    }

    fn plain(exe: &Path) -> Self {
        RunnerLaunch {
            exe: exe.to_path_buf(),
            arg0: None,
            env: Vec::new(),
            llama_dir: None,
            cpu_exe: None,
        }
    }
}

/// Directory holding the CUDA runtime pack, if this install has one.
///
/// First match wins: `$OLLAYA_LIBRARY_PATH/<pack>`, `<exe dir>/../lib/ollaya/<pack>` (the
/// archive and Docker layouts), each pack of [`CUDA_PACKS`] in turn, then the executable's own
/// directory for development builds, where `copy-dylibs` places the providers next to the binary.
pub fn cuda_dir(exe: &Path) -> Option<PathBuf> {
    let has_providers = |d: &Path| CUDA_PROVIDERS.iter().all(|p| d.join(p).is_file());
    let exe_dir = exe.parent()?;
    let library_path = std::env::var_os("OLLAYA_LIBRARY_PATH").map(PathBuf::from);
    let roots = [library_path, Some(exe_dir.join("../lib/ollaya"))];
    roots
        .iter()
        .flatten()
        .flat_map(|root| CUDA_PACKS.map(|pack| root.join(pack)))
        .chain(std::iter::once(exe_dir.to_path_buf()))
        .find(|d| has_providers(d))
        .map(|d| resolve(&d))
}

/// Directory holding the ROCm runtime pack, if this install has one.
///
/// First match wins: `$OLLAYA_LIBRARY_PATH/<pack>`, `<exe dir>/../lib/ollaya/<pack>`,
/// each pack of [`ROCM_PACKS`] in turn, then the executable's own directory.
pub fn rocm_dir(exe: &Path) -> Option<PathBuf> {
    let has_providers = |d: &Path| {
        ROCM_PROVIDERS.iter().all(|p| d.join(p).is_file()) || d.join(ROCM_BACKEND).is_file()
    };
    let exe_dir = exe.parent()?;
    let library_path = std::env::var_os("OLLAYA_LIBRARY_PATH").map(PathBuf::from);
    let roots = [library_path, Some(exe_dir.join("../lib/ollaya"))];
    roots
        .iter()
        .flatten()
        .flat_map(|root| ROCM_PACKS.map(|pack| root.join(pack)))
        .chain(std::iter::once(exe_dir.to_path_buf()))
        .find(|d| has_providers(d))
        .map(|d| resolve(&d))
}

/// For a pack that holds its own ONNX Runtime: the runner executable and the library it loads.
/// `None` for a pack of provider libraries only, which the statically linked executable loads.
fn dynamic_runner(dir: &Path) -> std::io::Result<Option<(PathBuf, String)>> {
    let library = dir.join(ORT_LIBRARY);
    if !library.is_file() {
        return Ok(None);
    }
    let runner_name = if dir.ends_with("rocm") || dir.join(ROCM_BACKEND).is_file() {
        ROCM_RUNNER
    } else {
        CUDA_RUNNER
    };
    let runner = dir
        .parent()
        .map(|p| p.join(runner_name))
        .filter(|r| r.is_file())
        .or_else(|| {
            dir.parent()
                .map(|p| p.join(CUDA_RUNNER))
                .filter(|r| r.is_file())
        })
        .or_else(|| {
            dir.parent()
                .map(|p| p.join(ROCM_RUNNER))
                .filter(|r| r.is_file())
        })
        .ok_or_else(|| {
            std::io::Error::other(format!(
                "the GPU pack needs {runner_name} next to it; reinstall ollaya"
            ))
        })?;
    Ok(Some((resolve(&runner), library.display().to_string())))
}

/// `d` as an absolute path without `..`. Windows keeps the plain drive form: a canonical path
/// there starts with `\\?\`, which would then reach ONNX Runtime's `LoadLibraryExW` calls.
fn resolve(d: &Path) -> PathBuf {
    if cfg!(windows) {
        std::path::absolute(d).unwrap_or_else(|_| d.to_path_buf())
    } else {
        d.canonicalize().unwrap_or_else(|_| d.to_path_buf())
    }
}

/// The llama.cpp library that GGUF runners load, by platform (`scripts/llama-cpp.sh`).
pub const LIBLLAMA: &str = if cfg!(windows) {
    "llama.dll"
} else if cfg!(target_os = "macos") {
    "libllama.0.dylib"
} else {
    "libllama.so.0"
};

/// Directory holding llama.cpp's libraries (`lib/ollaya/llama`), if this install has them.
///
/// First match wins: `$OLLAYA_LIBRARY_PATH/llama` (development, and the desktop app, which
/// bundles them as resources), then `<exe dir>/../lib/ollaya/llama` (the tarball, zip and
/// Docker layouts). The CUDA backend is not here: it is in the CUDA pack, next to the CUDA
/// libraries it needs, where a GPU runner's `argv[0]` points.
pub fn llama_dir(exe: &Path) -> Option<PathBuf> {
    let exe_dir = exe.parent()?;
    let candidates = [
        std::env::var_os("OLLAYA_LIBRARY_PATH").map(|p| PathBuf::from(p).join("llama")),
        Some(exe_dir.join("../lib/ollaya/llama")),
    ];
    candidates
        .into_iter()
        .flatten()
        .find(|d| d.join(LIBLLAMA).is_file())
        .map(|d| resolve(&d))
}

/// How to start runners for the executable `exe`.
pub fn runner_launch(exe: &Path) -> RunnerLaunch {
    let want_rocm = std::env::var("OLLAYA_DEVICE").is_ok_and(|d| d.starts_with("rocm"));
    let pack = if want_rocm {
        rocm_dir(exe).or_else(|| cuda_dir(exe))
    } else {
        cuda_dir(exe).or_else(|| rocm_dir(exe))
    };
    match pack {
        Some(dir) => gpu_launch(exe, &dir).unwrap_or_else(|e| {
            tracing::warn!(
                "cannot use the GPU runtime in {}, runners will use the CPU: {e}",
                dir.display()
            );
            RunnerLaunch::plain(exe)
        }),
        None => RunnerLaunch::plain(exe),
    }
}

#[cfg(not(windows))]
fn gpu_launch(exe: &Path, dir: &Path) -> std::io::Result<RunnerLaunch> {
    let mut ld = dir.display().to_string();
    if let Some(old) = std::env::var("LD_LIBRARY_PATH")
        .ok()
        .filter(|v| !v.is_empty())
    {
        ld.push(':');
        ld.push_str(&old);
    }
    let mut env = vec![("LD_LIBRARY_PATH".into(), ld)];
    let (runner, cpu_exe) = match dynamic_runner(dir)? {
        Some((runner, library)) => {
            env.push(("ORT_DYLIB_PATH".into(), library));
            (runner, Some(exe.to_path_buf()))
        }
        None => (exe.to_path_buf(), None),
    };
    Ok(RunnerLaunch {
        exe: runner,
        arg0: Some(dir.join("ollaya")),
        env,
        llama_dir: None,
        cpu_exe,
    })
}

#[cfg(windows)]
fn gpu_launch(exe: &Path, dir: &Path) -> std::io::Result<RunnerLaunch> {
    // A development build has the providers next to it already (copy-dylibs), and the NVIDIA
    // DLLs come from the caller's PATH.
    if exe.parent().is_some_and(|p| resolve(p) == dir) {
        return Ok(RunnerLaunch::plain(exe));
    }
    let (source, env, cpu_exe) = match dynamic_runner(dir)? {
        Some((runner, library)) => (
            runner,
            vec![("ORT_DYLIB_PATH".into(), library)],
            Some(exe.to_path_buf()),
        ),
        None => (exe.to_path_buf(), Vec::new(), None),
    };
    let runner = runner_copy(&source, dir)?;
    tracing::debug!("GPU runners start from {}", runner.display());
    Ok(RunnerLaunch {
        env,
        cpu_exe,
        ..RunnerLaunch::plain(&runner)
    })
}

/// A copy of the executable `exe` in `dir`, for runners whose runtime path must be `dir`: on
/// Windows, ONNX Runtime loads its providers from the directory of the executable file.
///
/// The copy is named after a hash of `exe`'s contents (`ollaya-runner-<hash>.exe`), so the
/// runners of two different builds sharing one GPU pack (the command line and the desktop app,
/// or two versions) never replace each other's file. It is a hard link where the file system
/// allows one, and a copy otherwise; either appears under its final name only when complete.
/// The installer removes copies left by earlier versions.
pub fn runner_copy(exe: &Path, dir: &Path) -> std::io::Result<PathBuf> {
    let hash = file_sha256(exe)?;
    let name = format!(
        "ollaya-runner-{}{}",
        &hash[..16],
        std::env::consts::EXE_SUFFIX
    );
    let target = dir.join(name);
    let size = std::fs::metadata(exe)?.len();
    match std::fs::metadata(&target) {
        Ok(m) if m.len() == size => return Ok(target),
        Ok(_) => std::fs::remove_file(&target)?,
        Err(_) => {}
    }
    if std::fs::hard_link(exe, &target).is_ok() {
        return Ok(target);
    }
    let tmp = dir.join(format!(
        "ollaya-runner-{}.{}.tmp",
        &hash[..16],
        std::process::id()
    ));
    let copied = std::fs::copy(exe, &tmp).and_then(|_| std::fs::rename(&tmp, &target));
    if let Err(e) = copied {
        let _ = std::fs::remove_file(&tmp);
        // Another daemon may have made the same copy in the meantime.
        if !std::fs::metadata(&target).is_ok_and(|m| m.len() == size) {
            return Err(e);
        }
    }
    Ok(target)
}

fn file_sha256(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        for p in CUDA_PROVIDERS {
            std::fs::write(dir.join(p), b"provider").unwrap();
        }
    }

    /// `<root>/bin/ollaya`, as the archives lay it out.
    fn install(root: &Path) -> PathBuf {
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let exe = bin.join("ollaya");
        std::fs::write(&exe, b"exe").unwrap();
        exe
    }

    #[test]
    fn finds_the_pack_next_to_the_prefix() {
        let root = tempfile::tempdir().unwrap();
        let exe = install(root.path());
        assert_eq!(cuda_dir(&exe), None);

        let lib = root.path().join("lib/ollaya/cuda_v13");
        pack(&lib);
        let found = cuda_dir(&exe).unwrap();
        assert_eq!(found, resolve(&lib));
        assert!(found.is_absolute());
        assert!(!found.to_string_lossy().contains(".."));
    }

    #[test]
    fn a_pack_needs_both_providers() {
        let root = tempfile::tempdir().unwrap();
        let exe = install(root.path());
        let lib = root.path().join("lib/ollaya/cuda_v13");
        pack(&lib);
        assert!(cuda_dir(&exe).is_some());
        std::fs::remove_file(lib.join(CUDA_PROVIDERS[1])).unwrap();
        assert_eq!(cuda_dir(&exe), None);
    }

    #[test]
    fn runner_copy_is_named_by_content_and_reused() {
        let root = tempfile::tempdir().unwrap();
        let exe = root.path().join("ollaya.exe");
        std::fs::write(&exe, b"build one").unwrap();
        let dir = root.path().join("cuda_v13");
        std::fs::create_dir_all(&dir).unwrap();

        let first = runner_copy(&exe, &dir).unwrap();
        assert_eq!(first.parent(), Some(dir.as_path()));
        let name = first.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("ollaya-runner-"), "{name}");
        assert_eq!(std::fs::read(&first).unwrap(), b"build one");
        assert_eq!(runner_copy(&exe, &dir).unwrap(), first);

        // Another build gets its own file; the first one stays for the runners using it.
        let other = root.path().join("other.exe");
        std::fs::write(&other, b"build two").unwrap();
        let second = runner_copy(&other, &dir).unwrap();
        assert_ne!(second, first);
        assert_eq!(std::fs::read(&second).unwrap(), b"build two");
        assert_eq!(std::fs::read(&first).unwrap(), b"build one");

        // A leftover of the wrong size under the right name is replaced.
        std::fs::remove_file(&first).unwrap();
        std::fs::write(&first, b"cut").unwrap();
        assert_eq!(runner_copy(&exe, &dir).unwrap(), first);
        assert_eq!(std::fs::read(&first).unwrap(), b"build one");
        let leftovers = std::fs::read_dir(&dir)
            .unwrap()
            .filter(|e| e.as_ref().unwrap().path().extension() == Some("tmp".as_ref()))
            .count();
        assert_eq!(leftovers, 0);
    }

    #[cfg(not(windows))]
    #[test]
    fn linux_gpu_runners_get_argv0_and_library_path() {
        let root = tempfile::tempdir().unwrap();
        let exe = install(root.path());
        let lib = root.path().join("lib/ollaya/cuda_v13");
        pack(&lib);
        let launch = runner_launch(&exe);
        let dir = resolve(&lib);
        assert_eq!(launch.exe, exe);
        assert_eq!(launch.arg0, Some(dir.join("ollaya")));
        let (key, value) = &launch.env[0];
        assert_eq!(key, "LD_LIBRARY_PATH");
        assert!(value.starts_with(&dir.display().to_string()));
        assert_eq!(launch.env.len(), 1);
        assert_eq!(launch.cpu_exe, None);
    }

    #[test]
    fn finds_a_cuda_12_pack() {
        let root = tempfile::tempdir().unwrap();
        let exe = install(root.path());
        let lib = root.path().join("lib/ollaya/cuda_v12");
        pack(&lib);
        assert_eq!(cuda_dir(&exe), Some(resolve(&lib)));
    }

    #[cfg(not(windows))]
    #[test]
    fn a_pack_with_its_own_onnx_runtime_starts_the_cuda_runner() {
        let root = tempfile::tempdir().unwrap();
        let exe = install(root.path());
        let lib = root.path().join("lib/ollaya/cuda_v12");
        pack(&lib);
        std::fs::write(lib.join(ORT_LIBRARY), b"ort").unwrap();
        let dir = resolve(&lib);

        // Without its runner, the pack is not used: the static build must not load its providers.
        let launch = runner_launch(&exe);
        assert_eq!(launch.exe, exe);
        assert!(launch.env.is_empty());
        assert_eq!(launch.cpu_exe, None);

        let runner = root.path().join("lib/ollaya").join(CUDA_RUNNER);
        std::fs::write(&runner, b"cuda runner").unwrap();
        let launch = runner_launch(&exe);
        assert_eq!(launch.exe, resolve(&runner));
        assert_eq!(launch.arg0, Some(dir.join("ollaya")));
        // CPU runners stay on the statically linked build.
        assert_eq!(launch.cpu_exe, Some(exe.clone()));
        let env: std::collections::HashMap<_, _> = launch.env.into_iter().collect();
        assert!(env["LD_LIBRARY_PATH"].starts_with(&dir.display().to_string()));
        assert_eq!(
            env["ORT_DYLIB_PATH"],
            dir.join(ORT_LIBRARY).display().to_string()
        );
    }

    #[test]
    fn finds_a_rocm_pack() {
        let root = tempfile::tempdir().unwrap();
        let exe = install(root.path());
        let lib = root.path().join("lib/ollaya/rocm");
        std::fs::create_dir_all(&lib).unwrap();
        for p in ROCM_PROVIDERS {
            std::fs::write(lib.join(p), b"provider").unwrap();
        }
        assert_eq!(rocm_dir(&exe), Some(resolve(&lib)));
    }

    #[cfg(not(windows))]
    #[test]
    fn rocm_pack_with_its_own_onnx_runtime_starts_the_rocm_runner() {
        let root = tempfile::tempdir().unwrap();
        let exe = install(root.path());
        let lib = root.path().join("lib/ollaya/rocm");
        std::fs::create_dir_all(&lib).unwrap();
        for p in ROCM_PROVIDERS {
            std::fs::write(lib.join(p), b"provider").unwrap();
        }
        std::fs::write(lib.join(ORT_LIBRARY), b"ort").unwrap();
        let dir = resolve(&lib);

        let runner = root.path().join("lib/ollaya").join(ROCM_RUNNER);
        std::fs::write(&runner, b"rocm runner").unwrap();
        let launch = runner_launch(&exe);
        assert_eq!(launch.exe, resolve(&runner));
        assert_eq!(launch.arg0, Some(dir.join("ollaya")));
        assert_eq!(launch.cpu_exe, Some(exe.clone()));
    }
}
