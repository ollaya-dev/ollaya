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
use std::path::PathBuf;
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

#[derive(Default)]
struct AppState {
    /// This app started the server, so it stops it when it quits.
    started_server: AtomicBool,
    /// Downloads in progress: model → (bytes done, bytes in total).
    pulling: Mutex<HashMap<String, (u64, u64)>>,
    /// Wakes the menu bar to redraw now rather than at its next poll.
    refresh: tokio::sync::Notify,
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
    name: &'static str,
    questions: Value,
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
    let mut cmd = std::process::Command::new(ollaya_exe());
    cmd.stdin(std::process::Stdio::null());
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut cmd, 0x0800_0000); // CREATE_NO_WINDOW
    cmd
}

async fn status_now() -> Status {
    match client() {
        Ok(c) => {
            let version = c.version().await.ok().map(|v| v.version);
            Status {
                running: version.is_some(),
                version,
                url: c.base_url().to_owned(),
            }
        }
        Err(_) => Status {
            running: false,
            version: None,
            url: String::new(),
        },
    }
}

#[tauri::command]
async fn status() -> Status {
    status_now().await
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
    let mut cmd = ollaya_command();
    cmd.arg("serve")
        .stdout(log.try_clone().map_err(|e| e.to_string())?)
        .stderr(log);
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut cmd, 0x0800_0000 | 0x0000_0200); // no window, new group
    cmd.spawn()
        .map_err(|e| format!("could not start {}: {e}", ollaya_exe().display()))?;
    state.started_server.store(true, Ordering::SeqCst);

    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        let s = status_now().await;
        if s.running {
            state.changed();
            return Ok(s);
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

#[tauri::command]
fn preset_list() -> Vec<Preset> {
    presets::NAMES
        .iter()
        .map(|&name| Preset {
            name,
            questions: presets::get(name).unwrap_or(Value::Null),
        })
        .collect()
}

/// Answer `questions` (JSON text) or a preset about `state`, which is sent as JSON when it is a
/// JSON object or array and as text otherwise (as `ollaya run` does).
#[tauri::command]
async fn decide(
    model: String,
    state: String,
    preset: Option<String>,
    questions: Option<String>,
) -> Result<DecideResponse, String> {
    let questions: Questions = match (preset.as_deref(), questions.as_deref()) {
        (Some(name), _) => presets::get(name)
            .map(serde_json::from_value)
            .ok_or_else(|| format!("unknown preset {name:?}"))?
            .map_err(|e| e.to_string())?,
        (None, Some(text)) => {
            serde_json::from_str(text).map_err(|e| format!("the questions are not valid: {e}"))?
        }
        (None, None) => return Err("choose a preset or write questions".into()),
    };
    let trimmed = state.trim();
    let state = match serde_json::from_str::<Value>(trimmed) {
        Ok(v @ (Value::Object(_) | Value::Array(_))) if trimmed.starts_with(['{', '[']) => v,
        _ => Value::String(state),
    };
    client()?
        .decide(&DecideRequest::new(model, state, Some(questions)))
        .await
        .map_err(|e| e.to_string())
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
