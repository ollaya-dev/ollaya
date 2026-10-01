//! Ollaya's desktop app: a window over the same local server the CLI uses.
//!
//! The app ships the `ollaya` binary next to its own executable (Tauri's `externalBin`) and runs
//! `ollaya serve` in the background, as Ollama's app runs `ollama serve`. Everything else goes
//! through the server's HTTP API with `ollaya-api`'s client, so the app and the CLI always see
//! the same models. The web page in `ui/` calls the commands below.
//!
//! On macOS the app lives in the menu bar (`tray.rs`), with no Dock icon until its window is open.

#[cfg(target_os = "macos")]
mod tray;

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use ollaya_api::{
    Client, DecideRequest, DecideResponse, PullRequest, Questions, TagsResponse, presets,
};
use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager, RunEvent};

/// Where the model library is listed: the website's search index.
const LIBRARY_URL: &str = "https://ollaya.dev/search.json";

/// Where a newer release is downloaded (the page picks the right installer for this system).
const DOWNLOAD_URL: &str = "https://ollaya.dev/download";

#[derive(Default)]
struct AppState {
    /// This app started the server, so it stops it when it quits.
    started_server: AtomicBool,
    /// The last release check: when, and the newer version it found (`None`: up to date).
    release: Mutex<Option<(Instant, Option<String>)>>,
    /// Downloads in progress: model → (bytes done, bytes in total).
    pulling: Mutex<HashMap<String, (u64, u64)>>,
    /// Wakes the menu bar to redraw now rather than at its next poll.
    refresh: tokio::sync::Notify,
    /// `Status::device` of the server this app started.
    server_device: Mutex<Option<&'static str>>,
}

impl AppState {
    fn changed(&self) {
        self.refresh.notify_one();
    }
}

#[derive(Serialize)]
struct Status {
    running: bool,
    version: Option<String>,
    url: String,
    /// Where the server this app started runs models, on Windows and Linux: "GPU" (the
    /// command-line install with its GPU pack) or "CPU only" (the bundled server, #44).
    device: Option<&'static str>,
}

#[derive(Clone, Serialize)]
struct PullProgress {
    model: String,
    status: String,
    completed: u64,
    total: u64,
    done: bool,
    error: Option<String>,
}

#[derive(Serialize)]
struct Preset {
    name: String,
    questions: Value,
    /// Built into Ollaya, as against one saved with `ollaya preset create` (0.8.0 and newer).
    builtin: bool,
}

fn client() -> Result<Client, String> {
    Client::from_env().map_err(|e| e.to_string())
}

/// The `ollaya` binary bundled next to this executable (`externalBin`), else one on `PATH`.
fn ollaya_exe() -> PathBuf {
    let name = if cfg!(windows) {
        "ollaya.exe"
    } else {
        "ollaya"
    };
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(name)))
        .filter(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from(name))
}

/// A command for the bundled binary that never opens a console window on Windows.
fn ollaya_command() -> std::process::Command {
    command_for(&ollaya_exe())
}

/// A command for `exe` that never opens a console window on Windows.
fn command_for(exe: &Path) -> std::process::Command {
    let mut cmd = std::process::Command::new(exe);
    cmd.stdin(std::process::Stdio::null());
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut cmd, 0x0800_0000); // CREATE_NO_WINDOW
    cmd
}

/// A newer Ollaya release than this app, if there is one. GitHub redirects `/releases/latest` to
/// `/releases/tag/v<version>`, which `ollaya update` and the install scripts read too: no API
/// call and no rate limit. Checked at most every six hours; a failed check says nothing.
async fn newer_release(app: &AppHandle) -> Option<String> {
    let state = app.state::<AppState>();
    if let Some((at, found)) = state.release.lock().expect("release lock").clone()
        && at.elapsed() < Duration::from_secs(6 * 3600)
    {
        return found;
    }
    let found = latest_release()
        .await
        .filter(|v| newer(v, env!("CARGO_PKG_VERSION")));
    *state.release.lock().expect("release lock") = Some((Instant::now(), found.clone()));
    found
}

async fn latest_release() -> Option<String> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .user_agent(concat!("ollaya-desktop/", env!("CARGO_PKG_VERSION")))
        .timeout(Duration::from_secs(10))
        .build()
        .ok()?;
    let resp = client
        .head("https://github.com/ollaya-dev/ollaya/releases/latest")
        .send()
        .await
        .ok()?;
    let location = resp
        .headers()
        .get(reqwest::header::LOCATION)?
        .to_str()
        .ok()?;
    let (_, tag) = location.rsplit_once("/releases/tag/")?;
    Some(tag.trim_start_matches('v').to_owned())
}

/// Whether version `a` is newer than `b`, comparing `major.minor.patch` as numbers.
fn newer(a: &str, b: &str) -> bool {
    let parse = |s: &str| -> Option<Vec<u64>> {
        s.split(['-', '+'])
            .next()?
            .split('.')
            .map(|n| n.parse().ok())
            .collect()
    };
    matches!((parse(a), parse(b)), (Some(x), Some(y)) if x > y)
}

/// Open `url` in the default browser.
fn open_url(url: &str) {
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(windows)]
    let _ = windows_start(url);
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
}

#[cfg(windows)]
fn windows_start(url: &str) -> std::io::Result<std::process::Child> {
    let mut cmd = std::process::Command::new("cmd");
    cmd.args(["/C", "start", "", url]);
    std::os::windows::process::CommandExt::creation_flags(&mut cmd, 0x0800_0000); // CREATE_NO_WINDOW
    cmd.spawn()
}

/// The newer release the window offers, if any.
#[tauri::command]
async fn update_available(app: AppHandle) -> Option<String> {
    newer_release(&app).await
}

#[tauri::command]
fn open_download() {
    open_url(DOWNLOAD_URL);
}

/// The `ollaya` that runs the server: the command-line install's when it can use the GPU, else
/// the bundled one, with a line for `server.log` that says which and why.
///
/// On Windows and Linux the app bundles no GPU pack (1.4 GB and more); `install.ps1` and
/// `install.sh` add one next to the command-line `ollaya` when they find an NVIDIA GPU. A server
/// started from that install uses the GPU as `ollaya serve` in a terminal does (#44). It is used
/// only when it is as new as this app, so the app never serves models with an older server.
fn server_exe() -> (PathBuf, Option<String>, bool) {
    let bundled = ollaya_exe();
    if cfg!(target_os = "macos") {
        return (bundled, None, false);
    }
    let app = VERSION;
    let mut note = None;
    for cli in cli_candidates(&bundled) {
        if !has_gpu_pack(&cli) {
            continue;
        }
        match cli_version(&cli) {
            Some(v) if version_at_least(&v, app) => {
                let msg = format!(
                    "desktop app: starting the server from {} ({v}), which has a GPU pack",
                    cli.display()
                );
                return (cli, Some(msg), true);
            }
            v => {
                note = Some(format!(
                    "desktop app: {} has a GPU pack but is {}, older than this app ({app}); \
                     the server runs on the CPU. Run `ollaya update` to use the GPU.",
                    cli.display(),
                    v.as_deref().unwrap_or("of an unknown version"),
                ));
            }
        }
    }
    (bundled, note, false)
}

/// This app's version, which is its bundled `ollaya`'s.
const VERSION: &str = env!("CARGO_PKG_VERSION");

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
fn version_at_least(v: &str, min: &str) -> bool {
    let parse = |s: &str| -> Option<Vec<u64>> {
        let core = s.trim_start_matches('v').split(['-', '+']).next()?;
        core.split('.').map(|n| n.parse().ok()).collect()
    };
    match (parse(v), parse(min)) {
        (Some(a), Some(b)) => a >= b,
        _ => false,
    }
}

async fn status_now() -> Status {
    match client() {
        Ok(c) => {
            let version = c.version().await.ok().map(|v| v.version);
            Status {
                running: version.is_some(),
                version,
                url: c.base_url().to_owned(),
                device: None,
            }
        }
        Err(_) => Status {
            running: false,
            version: None,
            url: String::new(),
            device: None,
        },
    }
}

/// `s` with the device of the server, when this app started it (Windows and Linux).
fn with_device(app: &AppHandle, mut s: Status) -> Status {
    let state = app.state::<AppState>();
    if s.running && state.started_server.load(Ordering::SeqCst) {
        s.device = *state.server_device.lock().expect("device lock");
    }
    s
}

#[tauri::command]
async fn status(app: AppHandle) -> Status {
    with_device(&app, status_now().await)
}

#[tauri::command]
async fn start_server(app: AppHandle) -> Result<Status, String> {
    start_server_now(&app).await
}

#[tauri::command]
async fn stop_server(app: AppHandle) -> Result<Status, String> {
    stop_server_now(&app).await
}

/// Start `ollaya serve` in the background (logging where the CLI's server logs), then wait for it.
async fn start_server_now(app: &AppHandle) -> Result<Status, String> {
    let state = app.state::<AppState>();
    let current = status_now().await;
    if current.running {
        return Ok(current);
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
    let (exe, note, gpu) = server_exe();
    if let Some(note) = note {
        let _ = writeln!(&log, "{note}");
    }
    let mut cmd = command_for(&exe);
    cmd.arg("serve")
        .stdout(log.try_clone().map_err(|e| e.to_string())?)
        .stderr(log);
    // GGUF models run on the llama.cpp build bundled in the app's resources (desktop.yml stages
    // it there); the bundled `ollaya serve` finds it through OLLAYA_LIBRARY_PATH. A command-line
    // install uses its own, next to its GPU pack.
    if exe == ollaya_exe()
        && std::env::var_os("OLLAYA_LIBRARY_PATH").is_none()
        && let Ok(dir) = app.path().resource_dir()
    {
        let lib = dir.join("lib").join("ollaya");
        if lib.join("llama").is_dir() {
            cmd.env("OLLAYA_LIBRARY_PATH", lib);
        }
    }
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut cmd, 0x0800_0000 | 0x0000_0200); // no window, new group
    cmd.spawn()
        .map_err(|e| format!("could not start {}: {e}", exe.display()))?;
    state.started_server.store(true, Ordering::SeqCst);
    *state.server_device.lock().expect("device lock") =
        (!cfg!(target_os = "macos")).then_some(if gpu { "GPU" } else { "CPU only" });

    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        let s = status_now().await;
        if s.running {
            state.changed();
            return Ok(with_device(app, s));
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    Err("the server did not answer within 30 s; see ~/.ollaya/logs/server.log".into())
}

/// `ollaya stop`: stops the server this user started (never another user's service).
async fn stop_server_now(app: &AppHandle) -> Result<Status, String> {
    let state = app.state::<AppState>();
    let out = tokio::task::spawn_blocking(|| ollaya_command().arg("stop").output())
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
    state.started_server.store(false, Ordering::SeqCst);
    *state.server_device.lock().expect("device lock") = None;
    state.changed();
    Ok(status_now().await)
}

/// The public model library, as the website lists it.
#[tauri::command]
async fn library() -> Result<Value, String> {
    library_now().await
}

async fn library_now() -> Result<Value, String> {
    let body = reqwest::get(LIBRARY_URL)
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("could not load the model library: {e}"))?;
    let index: Value = body.json().await.map_err(|e| e.to_string())?;
    Ok(index["models"].clone())
}

#[tauri::command]
async fn installed() -> Result<TagsResponse, String> {
    client()?.tags().await.map_err(|e| e.to_string())
}

/// Pull `model`, emitting `pull-progress` events with the bytes done over every layer.
#[tauri::command]
async fn pull(app: AppHandle, model: String) -> Result<(), String> {
    pull_now(&app, model).await
}

/// Pull `model`, telling the window (`pull-progress` events) and the menu bar (`AppState`) how far
/// it got, in bytes done over every layer.
async fn pull_now(app: &AppHandle, model: String) -> Result<(), String> {
    let state = app.state::<AppState>();
    let emit = |p: PullProgress| {
        {
            let mut pulling = state.pulling.lock().expect("pulling lock");
            if p.done {
                pulling.remove(&p.model);
            } else {
                pulling.insert(p.model.clone(), (p.completed, p.total));
            }
        }
        state.changed();
        let _ = app.emit("pull-progress", p);
    };
    let request = PullRequest {
        model: model.clone(),
        insecure: false,
        stream: Some(true),
    };
    emit(PullProgress {
        model: model.clone(),
        status: "starting".into(),
        completed: 0,
        total: 0,
        done: false,
        error: None,
    });
    let mut stream = match client()?.pull_stream(&request).await {
        Ok(s) => s,
        Err(e) => {
            emit(PullProgress {
                model: model.clone(),
                status: "failed".into(),
                completed: 0,
                total: 0,
                done: true,
                error: Some(e.to_string()),
            });
            return Err(e.to_string());
        }
    };
    let mut layers: std::collections::HashMap<String, (u64, u64)> = Default::default();
    let mut last = Instant::now() - Duration::from_secs(1);
    while let Some(line) = stream.next().await {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                emit(PullProgress {
                    model: model.clone(),
                    status: "failed".into(),
                    completed: 0,
                    total: 0,
                    done: true,
                    error: Some(e.to_string()),
                });
                return Err(e.to_string());
            }
        };
        if let (Some(digest), Some(total)) = (&line.digest, line.total) {
            layers.insert(digest.clone(), (line.completed.unwrap_or(0), total));
        }
        // At most ten updates a second: the page redraws a progress bar, not a log.
        if last.elapsed() >= Duration::from_millis(100) {
            last = Instant::now();
            let (completed, total) = layers
                .values()
                .fold((0, 0), |(d, t), (c, n)| (d + c, t + n));
            emit(PullProgress {
                model: model.clone(),
                status: line.status.clone(),
                completed,
                total,
                done: false,
                error: None,
            });
        }
    }
    let (completed, total) = layers
        .values()
        .fold((0, 0), |(d, t), (c, n)| (d + c, t + n));
    emit(PullProgress {
        model,
        status: "success".into(),
        completed,
        total,
        done: true,
        error: None,
    });
    Ok(())
}

#[tauri::command]
async fn remove(model: String) -> Result<(), String> {
    client()?.delete(&model).await.map_err(|e| e.to_string())
}

/// The built-in presets, then the custom ones the server has saved (`/api/presets`). A server
/// older than 0.8.0 has no custom presets, and the built-in ones are listed alone.
#[tauri::command]
async fn preset_list() -> Vec<Preset> {
    let mut list: Vec<Preset> = presets::NAMES
        .iter()
        .map(|&name| Preset {
            name: name.to_owned(),
            questions: presets::get(name).unwrap_or(Value::Null),
            builtin: true,
        })
        .collect();
    let Ok(c) = client() else { return list };
    let Ok(saved) = c.presets().await else {
        return list;
    };
    for p in saved.presets.into_iter().filter(|p| !p.builtin) {
        if let Ok(full) = c.show_preset(&p.name).await {
            list.push(Preset {
                name: full.name,
                questions: serde_json::to_value(full.questions).unwrap_or(Value::Null),
                builtin: false,
            });
        }
    }
    list
}

/// The questions `model` asks by itself (`/api/show`'s `questions`, from the manifest's questions
/// layer), or `None` when every request must bring its own. A model such as qwen3guard answers
/// only these, so the Run panel sends it the state alone.
#[tauri::command]
async fn builtin_questions(model: String) -> Result<Option<Questions>, String> {
    let show = client()?.show(&model).await.map_err(|e| e.to_string())?;
    Ok(show.questions)
}

/// The questions to send: a preset, the JSON text of the Custom box, or neither, which leaves
/// them out so the model answers its built-in questions (as `ollaya run` without flags).
fn request_questions(
    preset: Option<&str>,
    questions: Option<&str>,
) -> Result<Option<Questions>, String> {
    match (preset, questions) {
        (Some(name), _) => presets::get(name)
            .map(serde_json::from_value)
            .ok_or_else(|| format!("unknown preset {name:?}"))?
            .map(Some)
            .map_err(|e| e.to_string()),
        (None, Some(text)) => serde_json::from_str(text)
            .map(Some)
            .map_err(|e| format!("the questions are not valid: {e}")),
        (None, None) => Ok(None),
    }
}

/// Answer `questions` (JSON text), a preset, or with neither the model's built-in questions,
/// about `state`, which is sent as JSON when it is a JSON object or array and as text otherwise
/// (as `ollaya run` does).
#[tauri::command]
async fn decide(
    model: String,
    state: String,
    preset: Option<String>,
    questions: Option<String>,
) -> Result<DecideResponse, String> {
    // A built-in preset's questions are sent as questions, so any server answers them; a custom
    // preset lives on the server, which resolves it by name (`preset` on /api/decide).
    let custom = preset
        .as_deref()
        .filter(|p| presets::get(p).is_none())
        .map(str::to_owned);
    let questions = match custom {
        Some(_) => None,
        None => request_questions(preset.as_deref(), questions.as_deref())?,
    };
    let trimmed = state.trim();
    let state = match serde_json::from_str::<Value>(trimmed) {
        Ok(v @ (Value::Object(_) | Value::Array(_))) if trimmed.starts_with(['{', '[']) => v,
        _ => Value::String(state),
    };
    let mut request = DecideRequest::new(model, state, questions);
    request.preset = custom;
    client()?.decide(&request).await.map_err(|e| e.to_string())
}

pub fn run() {
    let app = tauri::Builder::default()
        .manage(AppState::default())
        .setup(|app| {
            #[cfg(target_os = "macos")]
            tray::init(app.handle())?;
            // Elsewhere the window is the app: show it (it starts hidden for the macOS menu bar).
            #[cfg(not(target_os = "macos"))]
            if let Some(window) = app.get_webview_window("main") {
                window.show()?;
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            // On macOS, closing the window leaves Ollaya in the menu bar.
            #[cfg(target_os = "macos")]
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                tray::hide_window(window.app_handle());
            }
            #[cfg(not(target_os = "macos"))]
            let _ = (window, event);
        })
        .invoke_handler(tauri::generate_handler![
            status,
            start_server,
            stop_server,
            library,
            installed,
            pull,
            remove,
            preset_list,
            builtin_questions,
            update_available,
            open_download,
            decide
        ])
        .build(tauri::generate_context!())
        .expect("building the Ollaya app");
    app.run(|handle, event| {
        // Quit what we started: a server this app launched goes away with it, as with Ollama.
        if let RunEvent::Exit = event
            && handle
                .state::<AppState>()
                .started_server
                .load(Ordering::SeqCst)
        {
            let _ = ollaya_command().arg("stop").status();
        }
    });
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
        let exe = root.join("bin").join(if cfg!(windows) {
            "ollaya.exe"
        } else {
            "ollaya"
        });
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
    fn questions_come_from_a_preset_the_custom_box_or_the_model() {
        let triage = request_questions(Some("triage"), None).unwrap().unwrap();
        assert_eq!(triage.keys().next().map(String::as_str), Some("intent"));
        let custom = r#"{"ok": {"type": "noul", "instructions": "Is it fine?"}}"#;
        let custom = request_questions(None, Some(custom)).unwrap().unwrap();
        assert_eq!(custom.keys().collect::<Vec<_>>(), ["ok"]);
        assert!(request_questions(Some("nope"), None).is_err());
        assert!(request_questions(None, Some("{")).is_err());

        // Neither: the request leaves `questions` out, and the model asks its built-in ones.
        assert!(request_questions(None, None).unwrap().is_none());
        let body = DecideRequest::new("qwen3guard", Value::String("Hi".into()), None);
        let body = serde_json::to_value(body).unwrap();
        assert!(body.get("questions").is_none());
    }
}
