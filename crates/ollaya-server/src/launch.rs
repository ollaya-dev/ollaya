//! How runner processes are launched: which executable, `argv[0]` and library path.
//!
//! ONNX Runtime is linked statically, and it `dlopen`s its CUDA provider libraries from the
//! directory of the process's `argv[0]`. The provider's own CUDA dependencies come from
//! `LD_LIBRARY_PATH`, which glibc reads only at exec. So a GPU runner is spawned with
//! `argv[0] = <cuda dir>/ollaya` and `LD_LIBRARY_PATH = <cuda dir>:...`. See
//! `docs/distribution.md`, "The runtime library contract".

use std::path::{Path, PathBuf};

/// Directory holding the CUDA runtime pack, if this install has one.
///
/// First match wins: `$OLLAYA_LIBRARY_PATH/cuda_v13`, `<exe dir>/../lib/ollaya/cuda_v13` (the
/// tarball and Docker layouts), then the executable's own directory for development builds,
/// where `copy-dylibs` places the providers next to the binary.
pub fn cuda_dir(exe: &Path) -> Option<PathBuf> {
    let has_providers = |d: &Path| {
        d.join("libonnxruntime_providers_shared.so").is_file()
            && d.join("libonnxruntime_providers_cuda.so").is_file()
    };
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
        .map(|d| d.canonicalize().unwrap_or(d))
}

/// `argv[0]` and extra environment for runner processes.
pub fn runner_launch(exe: &Path) -> (Option<PathBuf>, Vec<(String, String)>) {
    match cuda_dir(exe) {
        Some(dir) => {
            let mut ld = dir.display().to_string();
            if let Some(old) = std::env::var("LD_LIBRARY_PATH")
                .ok()
                .filter(|v| !v.is_empty())
            {
                ld.push(':');
                ld.push_str(&old);
            }
            (
                Some(dir.join("ollaya")),
                vec![("LD_LIBRARY_PATH".into(), ld)],
            )
        }
        None => (None, Vec::new()),
    }
}
