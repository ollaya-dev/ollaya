//! The `ollaya` engine the app runs: which binary serves (the bundled one, or a command-line
//! install with a GPU pack), where its libraries are, and starting and stopping `ollaya serve`.
//!
//! The app ships `ollaya` next to its own executable and runs `ollaya serve` in the background,
//! as Ollama's app runs `ollama serve`. Everything else goes through the server's HTTP API with
//! `ollaya-api`'s client, so the app and the CLI always see the same models.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ollaya_api::Client;

/// This app's version, which is its bundled `ollaya`'s.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// This app started the server, so it stops it when it quits.
static STARTED: AtomicBool = AtomicBool::new(false);
/// Where the server this app started runs models, on Windows and Linux: "GPU" (the command-line
/// install with its GPU pack) or "CPU only" (the bundled server, #44).
static DEVICE: Mutex<Option<&'static str>> = Mutex::new(None);

pub fn client() -> Result<Client, String> {
    Client::from_env().map_err(|e| e.to_string())
}

/// The device of the server this app started, while it runs.
pub fn device() -> Option<&'static str> {
    if STARTED.load(Ordering::SeqCst) {
        *DEVICE.lock().unwrap_or_else(|e| e.into_inner())
    } else {
        None
    }
}

fn exe_name() -> &'static str {
    if cfg!(windows) {
        "ollaya.exe"
    } else {
        "ollaya"
    }
}

/// The `ollaya` binary: `OLLAYA_BIN`, else the one bundled next to this executable
/// (`Ollaya.app/Contents/MacOS/ollaya`), else one on `PATH`.
pub fn ollaya_exe() -> PathBuf {
    if let Some(bin) = std::env::var_os("OLLAYA_BIN") {
        return PathBuf::from(bin);
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(exe_name())))
        .filter(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from(exe_name()))
}

/// The directory the app's resources are installed in: the one holding `lib/ollaya/llama`.
/// macOS: `Contents/Resources`; Windows: beside the executable; Linux: where the package puts
/// them (desktop/Packager.toml).
pub fn resource_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    [
        dir.join("..").join("Resources"),
        dir.to_path_buf(),
        dir.join("..").join("lib").join("Ollaya"),
        dir.join("..").join("lib").join("ollaya-desktop"),
        dir.join("..").join("share").join("ollaya-desktop"),
    ]
    .into_iter()
    .find(|d| d.join("lib").join("ollaya").join("llama").is_dir())
}

/// A command for `exe` that never opens a console window on Windows.
pub fn command_for(exe: &Path) -> std::process::Command {
    let mut cmd = std::process::Command::new(exe);
    cmd.stdin(std::process::Stdio::null());
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut cmd, 0x0800_0000); // CREATE_NO_WINDOW
    cmd
}

/// Whether a server answers.
pub async fn running() -> bool {
    match client() {
        Ok(c) => c.version().await.is_ok(),
        Err(_) => false,
    }
}

/// Start `ollaya serve` in the background (logging where the CLI's server logs), then wait for it.
pub async fn start() -> Result<(), String> {
    if running().await {
        return Ok(());
    }
    let logs = dirs::home_dir()
        .unwrap_or_default()
        .join(".ollaya")
        .join("logs");
    std::fs::create_dir_all(&logs).map_err(|e| format!("creating {}: {e}", logs.display()))?;
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(logs.join("server.log"))
        .map_err(|e| e.to_string())?;
    // GGUF models run on the llama.cpp build bundled in the app's resources (desktop.yml stages
    // it there), and on Windows GPU runners start from the app's own runner; the bundled
    // `ollaya serve` finds them through OLLAYA_LIBRARY_PATH. A command-line install uses its own,
    // next to its GPU pack.
    let app_lib = resource_dir()
        .map(|dir| dir.join("lib").join("ollaya"))
        .filter(|_| std::env::var_os("OLLAYA_LIBRARY_PATH").is_none());
    let (exe, note, gpu, shared) = server_exe(app_lib.as_deref());
    if let Some(note) = note {
        let _ = writeln!(&log, "{note}");
    }
    let mut cmd = command_for(&exe);
    cmd.arg("serve")
        .stdout(log.try_clone().map_err(|e| e.to_string())?)
        .stderr(log);
    if exe == ollaya_exe()
        && let Some(lib) = app_lib
    {
        // The app's directory first: the server takes each library from the first that has it.
        let dirs =
            std::env::join_paths(std::iter::once(lib).chain(shared)).map_err(|e| e.to_string())?;
        cmd.env("OLLAYA_LIBRARY_PATH", dirs);
    }
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut cmd, 0x0800_0000 | 0x0000_0200); // no window, new group
    cmd.spawn()
        .map_err(|e| format!("could not start {}: {e}", exe.display()))?;
    STARTED.store(true, Ordering::SeqCst);
    *DEVICE.lock().unwrap_or_else(|e| e.into_inner()) =
        (!cfg!(target_os = "macos")).then_some(if gpu { "GPU" } else { "CPU only" });

    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if running().await {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    Err("the server did not answer within 30 s; see ~/.ollaya/logs/server.log".into())
}

/// `ollaya stop`: stops the server this user started (never another user's service).
pub async fn stop() -> Result<(), String> {
    let out = tokio::task::spawn_blocking(|| command_for(&ollaya_exe()).arg("stop").output())
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| format!("could not run {}: {e}", ollaya_exe().display()))?;
    if !out.status.success() {
        let msg = String::from_utf8_lossy(&out.stderr)
            .trim()
            .trim_start_matches("Error: ")
            .to_owned();
        return Err(if msg.is_empty() {
            "could not stop the server".into()
        } else {
            msg
        });
    }
    STARTED.store(false, Ordering::SeqCst);
    *DEVICE.lock().unwrap_or_else(|e| e.into_inner()) = None;
    Ok(())
}

/// On quit: a server this app started goes away with it, as with Ollama.
pub fn stop_if_started() {
    if STARTED.swap(false, Ordering::SeqCst) {
        let _ = command_for(&ollaya_exe()).arg("stop").status();
    }
}

/// The `ollaya` that runs the server, with a line for `server.log` that says which and why,
/// whether it uses the GPU, and the command-line install's `lib/ollaya` whose GPU pack the
/// bundled one uses.
///
/// On Windows and Linux the app bundles no GPU pack (1.4 GB and more); `install.ps1` and
/// `install.sh` add one next to the command-line `ollaya` when they find an NVIDIA GPU. When it is
/// the pack this app's release ships, the bundled server uses it (see [`shared_pack`]). Otherwise
/// a server started from that install uses the GPU as `ollaya serve` in a terminal does (#44),
/// but only when it is as new as this app, so the app never serves models with an older server.
fn server_exe(app_lib: Option<&Path>) -> (PathBuf, Option<String>, bool, Option<PathBuf>) {
    let bundled = ollaya_exe();
    if cfg!(target_os = "macos") {
        return (bundled, None, false, None);
    }
    let mut note = None;
    for cli in cli_candidates(&bundled) {
        if !has_gpu_pack(&cli) {
            continue;
        }
        if let Some(lib) = app_lib.and_then(|l| shared_pack(l, &cli)) {
            let msg = format!(
                "desktop app: the server uses the GPU pack in {}, the same as this app's release",
                lib.display()
            );
            return (bundled, Some(msg), true, Some(lib));
        }
        match cli_version(&cli) {
            Some(v) if version_at_least(&v, VERSION) => {
                let msg = format!(
                    "desktop app: starting the server from {} ({v}), which has a GPU pack",
                    cli.display()
                );
                return (cli, Some(msg), true, None);
            }
            v => {
                note = Some(format!(
                    "desktop app: {} has a GPU pack but is {}, older than this app ({VERSION}); \
                     the server runs on the CPU. Run `ollaya update` to use the GPU.",
                    cli.display(),
                    v.as_deref().unwrap_or("of an unknown version"),
                ));
            }
        }
    }
    (bundled, note, false, None)
}

/// The `lib/ollaya` of the command-line install around `cli`, when the bundled server can use its
/// GPU pack: the app bundles its own GPU runner in `app_lib` (desktop.yml), and the pack has the
/// libraries of this app's release, by its `FILES.sha256`, as `install.ps1` decides to keep a pack.
/// A pack of another release may hold another ONNX Runtime and llama.cpp than the app's runner
/// and llama.cpp were built for.
fn shared_pack(app_lib: &Path, cli: &Path) -> Option<PathBuf> {
    let runner = if cfg!(windows) {
        "ollaya-cuda-runner.exe"
    } else {
        "ollaya-cuda-runner"
    };
    if !app_lib.join(runner).is_file() {
        return None;
    }
    let lib = cli.parent()?.parent()?.join("lib").join("ollaya");
    // The pack the server picks: cuda_v13, else cuda_v12 (an install has one).
    let pack = ["cuda_v13", "cuda_v12"]
        .into_iter()
        .find(|p| lib.join(p).join("FILES.sha256").is_file())?;
    let want = std::fs::read(app_lib.join(format!("{pack}.FILES.sha256"))).ok()?;
    (std::fs::read(lib.join(pack).join("FILES.sha256")).ok()? == want).then_some(lib)
}

/// Command-line installs of `ollaya` other than the bundled one: on `PATH`, then where the
/// install scripts put it by default.
fn cli_candidates(bundled: &Path) -> Vec<PathBuf> {
    let name = bundled.file_name().unwrap_or_default();
    let mut places: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect())
        .unwrap_or_default();
    if cfg!(windows) {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            places.push(
                PathBuf::from(local)
                    .join("Programs")
                    .join("Ollaya")
                    .join("bin"),
            );
        }
    } else {
        places.push(PathBuf::from("/usr/local/bin"));
        if let Some(home) = dirs::home_dir() {
            places.push(home.join(".local").join("bin"));
        }
    }
    let real = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let mut seen = vec![real(bundled)];
    let mut found = Vec::new();
    for exe in places
        .into_iter()
        .map(|d| d.join(name))
        .filter(|p| p.is_file())
    {
        let r = real(&exe);
        if !seen.contains(&r) {
            seen.push(r);
            found.push(exe);
        }
    }
    found
}

/// Whether the install around `exe` (`<prefix>/bin/ollaya`) has a CUDA pack in
/// `<prefix>/lib/ollaya`, as the server looks for one (`ollaya_server::launch::cuda_dir`).
fn has_gpu_pack(exe: &Path) -> bool {
    let provider = if cfg!(windows) {
        "onnxruntime_providers_cuda.dll"
    } else {
        "libonnxruntime_providers_cuda.so"
    };
    let Some(lib) = exe
        .parent()
        .and_then(Path::parent)
        .map(|p| p.join("lib").join("ollaya"))
    else {
        return false;
    };
    ["cuda_v13", "cuda_v12"]
        .iter()
        .any(|pack| lib.join(pack).join(provider).is_file())
}

/// `exe`'s own version. With no server running, `ollaya --version` prints it as
/// "Warning: client version is 0.8.0".
fn cli_version(exe: &Path) -> Option<String> {
    let out = command_for(exe).arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&out.stderr);
    text.lines()
        .find_map(|l| l.split_once("client version is "))
        .map(|(_, v)| v.trim().to_owned())
        .filter(|v| !v.is_empty())
}

/// Whether version `v` is `min` or newer, comparing the numbers of `major.minor.patch`.
pub fn version_at_least(v: &str, min: &str) -> bool {
    match (parse_version(v), parse_version(min)) {
        (Some(a), Some(b)) => a >= b,
        _ => false,
    }
}

/// Whether version `a` is newer than `b`.
pub fn newer(a: &str, b: &str) -> bool {
    matches!((parse_version(a), parse_version(b)), (Some(x), Some(y)) if x > y)
}

fn parse_version(s: &str) -> Option<Vec<u64>> {
    let core = s.trim_start_matches('v').split(['-', '+']).next()?;
    core.split('.').map(|n| n.parse().ok()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_compares_versions_as_numbers() {
        assert!(newer("0.8.1", "0.8.0"));
        assert!(newer("0.10.0", "0.9.9"));
        assert!(!newer("0.8.0", "0.8.0"));
        assert!(!newer("0.7.5", "0.8.0"));
        assert!(!newer("garbage", "0.8.0"));
    }

    #[test]
    fn versions_compare_by_number() {
        assert!(version_at_least("0.8.0", "0.8.0"));
        assert!(version_at_least("0.10.0", "0.9.2"));
        assert!(version_at_least("v1.0.0-rc1", "0.8.0"));
        assert!(!version_at_least("0.7.5", "0.8.0"));
        assert!(!version_at_least("unknown", "0.8.0"));
    }

    #[test]
    fn a_gpu_pack_sits_in_the_install_beside_bin() {
        let root = std::env::temp_dir().join(format!("ollaya-desktop-test-{}", std::process::id()));
        let exe = root.join("bin").join(exe_name());
        std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
        std::fs::write(&exe, b"").unwrap();
        assert!(!has_gpu_pack(&exe));
        let pack = root.join("lib").join("ollaya").join("cuda_v12");
        std::fs::create_dir_all(&pack).unwrap();
        let provider = if cfg!(windows) {
            "onnxruntime_providers_cuda.dll"
        } else {
            "libonnxruntime_providers_cuda.so"
        };
        std::fs::write(pack.join(provider), b"").unwrap();
        assert!(has_gpu_pack(&exe));
        // The bundled binary itself is never a candidate.
        assert!(cli_candidates(&exe).iter().all(|p| p != &exe));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_bundled_server_shares_only_the_pack_of_its_release() {
        let root =
            std::env::temp_dir().join(format!("ollaya-desktop-share-{}", std::process::id()));
        let (app, cli) = (root.join("app"), root.join("cli"));
        let exe = cli.join("bin").join("ollaya");
        let pack = cli.join("lib").join("ollaya").join("cuda_v13");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::create_dir_all(&pack).unwrap();
        std::fs::write(pack.join("FILES.sha256"), b"aaaa  onnxruntime.dll\n").unwrap();
        std::fs::write(
            app.join("cuda_v13.FILES.sha256"),
            b"aaaa  onnxruntime.dll\n",
        )
        .unwrap();
        // Without a runner of its own, the app never uses another install's pack.
        assert_eq!(shared_pack(&app, &exe), None);
        let runner = if cfg!(windows) {
            "ollaya-cuda-runner.exe"
        } else {
            "ollaya-cuda-runner"
        };
        std::fs::write(app.join(runner), b"").unwrap();
        assert_eq!(
            shared_pack(&app, &exe),
            Some(cli.join("lib").join("ollaya"))
        );
        // Another release's libraries.
        std::fs::write(pack.join("FILES.sha256"), b"bbbb  onnxruntime.dll\n").unwrap();
        assert_eq!(shared_pack(&app, &exe), None);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
