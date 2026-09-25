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

/// How runner processes are started.
#[derive(Debug, Clone)]
pub struct RunnerLaunch {
    /// The executable to spawn as `<exe> runner ...`.
    pub exe: PathBuf,
    /// `argv[0]` for runners, where it differs from `exe` (Linux GPU runners).
    pub arg0: Option<PathBuf>,
    /// Extra environment for runner processes (the GPU library path on Linux).
    pub env: Vec<(String, String)>,
}

impl RunnerLaunch {
    /// Runners are this executable's hidden `runner` subcommand, set up for the CUDA runtime pack
    /// when this install has one.
    pub fn current() -> std::io::Result<Self> {
        Ok(runner_launch(&std::env::current_exe()?))
    }

    fn plain(exe: &Path) -> Self {
        RunnerLaunch {
            exe: exe.to_path_buf(),
            arg0: None,
            env: Vec::new(),
        }
    }
}

/// Directory holding the CUDA runtime pack, if this install has one.
///
/// First match wins: `$OLLAYA_LIBRARY_PATH/cuda_v13`, `<exe dir>/../lib/ollaya/cuda_v13` (the
/// archive and Docker layouts), then the executable's own directory for development builds,
/// where `copy-dylibs` places the providers next to the binary.
pub fn cuda_dir(exe: &Path) -> Option<PathBuf> {
    let has_providers = |d: &Path| CUDA_PROVIDERS.iter().all(|p| d.join(p).is_file());
    let exe_dir = exe.parent()?;
    let candidates = [
        std::env::var_os("OLLAYA_LIBRARY_PATH").map(|p| PathBuf::from(p).join("cuda_v13")),
        Some(exe_dir.join("../lib/ollaya/cuda_v13")),
        Some(exe_dir.to_path_buf()),
    ];
    candidates
        .into_iter()
        .flatten()
        .find(|d| has_providers(d))
        .map(|d| resolve(&d))
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

/// How to start runners for the executable `exe`.
pub fn runner_launch(exe: &Path) -> RunnerLaunch {
    match cuda_dir(exe) {
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
    Ok(RunnerLaunch {
        exe: exe.to_path_buf(),
        arg0: Some(dir.join("ollaya")),
        env: vec![("LD_LIBRARY_PATH".into(), ld)],
    })
}

#[cfg(windows)]
fn gpu_launch(exe: &Path, dir: &Path) -> std::io::Result<RunnerLaunch> {
    // A development build has the providers next to it already (copy-dylibs), and the NVIDIA
    // DLLs come from the caller's PATH.
    if exe.parent().is_some_and(|p| resolve(p) == dir) {
        return Ok(RunnerLaunch::plain(exe));
    }
    let runner = runner_copy(exe, dir)?;
    tracing::debug!("GPU runners start from {}", runner.display());
    Ok(RunnerLaunch::plain(&runner))
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
    }
}
