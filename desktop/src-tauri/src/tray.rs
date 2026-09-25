//! The macOS menu bar: Ollaya lives there, like Ollama's app, with no Dock icon until its window
//! is open. The menu shows whether the server runs and where, starts and stops it, lists the
//! models (a check mark means loaded: click to load or unload) and downloads new ones.
//!
//! The menu is rebuilt from a [`Snapshot`] of the server whenever that snapshot changes: every
//! two seconds, or at once after an action (`AppState::refresh`).

use std::time::Duration;

use ollaya_api::KeepAlive;
use tauri::image::Image;
use tauri::menu::{
    CheckMenuItemBuilder, Menu, MenuBuilder, MenuEvent, MenuItemBuilder, PredefinedMenuItem,
    SubmenuBuilder,
};
use tauri::tray::TrayIconBuilder;
use tauri::{ActivationPolicy, AppHandle, Manager, Wry};

use crate::{
    AppState, client, library_now, pull_now, start_server_now, status_now, stop_server_now,
};

const TRAY: &str = "ollaya";
/// Template images (alpha only): an owl awake while the server runs, asleep while it doesn't.
const ICON_AWAKE: &[u8] = include_bytes!("../icons/tray-awake.png");
const ICON_ASLEEP: &[u8] = include_bytes!("../icons/tray-asleep.png");

/// What the menu shows. Equal snapshots draw equal menus, so an unchanged one isn't redrawn
/// (redrawing would close a menu the user has open).
#[derive(Clone, Default, PartialEq)]
struct Snapshot {
    running: bool,
    version: String,
    url: String,
    installed: Vec<String>,
    loaded: Vec<String>,
    /// Library models not on this machine.
    available: Vec<String>,
    /// Downloads in progress: (model, percent).
    pulling: Vec<(String, u32)>,
    open_at_login: bool,
}

pub fn init(app: &AppHandle) -> tauri::Result<()> {
    app.set_activation_policy(ActivationPolicy::Accessory)?;
    TrayIconBuilder::with_id(TRAY)
        .icon(Image::from_bytes(ICON_ASLEEP)?)
        .icon_as_template(true)
        .tooltip("Ollaya")
        .menu(&build_menu(app, &Snapshot::default())?)
        .show_menu_on_left_click(true)
        .on_menu_event(on_menu_event)
        .build(app)?;

    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        // Like Ollama's app: the server runs while Ollaya is in the menu bar.
        if !status_now().await.running {
            let _ = start_server_now(&app).await;
        }
        let mut library: Vec<String> = Vec::new();
        let mut shown: Option<Snapshot> = None;
        loop {
            if library.is_empty() {
                library = library_names().await;
            }
            let now = snapshot(&app, &library).await;
            if shown.as_ref() != Some(&now) {
                redraw(&app, &now);
                shown = Some(now);
            }
            let state = app.state::<AppState>();
            tokio::select! {
                () = tokio::time::sleep(Duration::from_secs(2)) => {}
                () = state.refresh.notified() => {}
            }
        }
    });
    Ok(())
}

async fn library_names() -> Vec<String> {
    library_now()
        .await
        .ok()
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|m| m["name"].as_str().map(str::to_owned))
        .collect()
}

async fn snapshot(app: &AppHandle, library: &[String]) -> Snapshot {
    let status = status_now().await;
    let mut s = Snapshot {
        running: status.running,
        version: status.version.unwrap_or_default(),
        url: status.url,
        open_at_login: login_item().exists(),
        ..Default::default()
    };
    if s.running
        && let Ok(c) = client()
    {
        if let Ok(tags) = c.tags().await {
            s.installed = tags.models.into_iter().map(|m| m.name).collect();
            s.installed.sort();
        }
        if let Ok(ps) = c.ps().await {
            s.loaded = ps.models.into_iter().map(|m| m.name).collect();
            s.loaded.sort();
        }
    }
    let family = |name: &str| name.split(':').next().unwrap_or(name).to_owned();
    s.available = library
        .iter()
        .filter(|m| !s.installed.iter().any(|i| family(i) == **m))
        .cloned()
        .collect();
    let state = app.state::<AppState>();
    let pulling = state.pulling.lock().expect("pulling lock");
    s.pulling = pulling
        .iter()
        .map(|(m, (done, total))| {
            (
                m.clone(),
                if *total > 0 {
                    (done * 100 / total) as u32
                } else {
                    0
                },
            )
        })
        .collect();
    s.pulling.sort();
    s
}

fn redraw(app: &AppHandle, s: &Snapshot) {
    let Some(tray) = app.tray_by_id(TRAY) else {
        return;
    };
    if let Ok(menu) = build_menu(app, s) {
        let _ = tray.set_menu(Some(menu));
    }
    let icon = if s.running { ICON_AWAKE } else { ICON_ASLEEP };
    if let Ok(image) = Image::from_bytes(icon) {
        let _ = tray.set_icon_with_as_template(Some(image), true);
    }
    let tip = if s.running {
        format!("Ollaya {} is running at {}", s.version, s.url)
    } else {
        "Ollaya is stopped".to_owned()
    };
    let _ = tray.set_tooltip(Some(tip));
}

fn build_menu(app: &AppHandle, s: &Snapshot) -> tauri::Result<Menu<Wry>> {
    let host = s.url.trim_start_matches("http://").to_owned();
    let status = if s.running {
        format!("Ollaya {} is running", s.version)
    } else if s.url.is_empty() {
        "Starting…".to_owned()
    } else {
        "Ollaya is stopped".to_owned()
    };
    let mut menu = MenuBuilder::new(app).item(
        &MenuItemBuilder::with_id("status", status)
            .enabled(false)
            .build(app)?,
    );
    if s.running {
        menu = menu.item(&MenuItemBuilder::with_id("copy-url", format!("Copy {host}")).build(app)?);
    }
    menu = menu
        .item(
            &MenuItemBuilder::with_id(
                "toggle",
                if s.running {
                    "Stop Server"
                } else {
                    "Start Server"
                },
            )
            .build(app)?,
        )
        .separator();

    for (model, pct) in &s.pulling {
        menu = menu.item(
            &MenuItemBuilder::with_id(
                format!("pulling:{model}"),
                format!("Downloading {model}… {pct}%"),
            )
            .enabled(false)
            .build(app)?,
        );
    }

    let mut models = SubmenuBuilder::new(app, "Models").enabled(s.running);
    if s.installed.is_empty() {
        models = models.item(
            &MenuItemBuilder::with_id("no-models", "No models yet")
                .enabled(false)
                .build(app)?,
        );
    }
    for name in &s.installed {
        let loaded = s.loaded.contains(name);
        models = models.item(
            &CheckMenuItemBuilder::with_id(format!("model:{name}"), name)
                .checked(loaded)
                .build(app)?,
        );
    }
    if !s.installed.is_empty() {
        models = models.separator().item(
            &MenuItemBuilder::with_id(
                "models-hint",
                "A check mark means loaded: click to load or unload",
            )
            .enabled(false)
            .build(app)?,
        );
    }
    menu = menu.item(&models.build()?);

    let mut download = SubmenuBuilder::new(app, "Download").enabled(s.running);
    if s.available.is_empty() {
        download = download.item(
            &MenuItemBuilder::with_id("all-installed", "Every model is installed")
                .enabled(false)
                .build(app)?,
        );
    }
    for name in &s.available {
        let busy = s.pulling.iter().any(|(m, _)| m == name);
        download = download.item(
            &MenuItemBuilder::with_id(format!("pull:{name}"), name)
                .enabled(!busy)
                .build(app)?,
        );
    }
    menu = menu
        .item(&download.build()?)
        .separator()
        .item(&MenuItemBuilder::with_id("open", "Open Ollaya…").build(app)?)
        .item(
            &CheckMenuItemBuilder::with_id("login", "Open at Login")
                .checked(s.open_at_login)
                .build(app)?,
        )
        .separator()
        .item(&PredefinedMenuItem::quit(app, Some("Quit Ollaya"))?);
    menu.build()
}

fn on_menu_event(app: &AppHandle, event: MenuEvent) {
    let id = event.id().as_ref().to_owned();
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        match id.as_str() {
            "toggle" => {
                if status_now().await.running {
                    let _ = stop_server_now(&app).await;
                } else {
                    let _ = start_server_now(&app).await;
                }
            }
            "copy-url" => copy(&status_now().await.url),
            "open" => show_window(&app),
            "login" => set_open_at_login(!login_item().exists()),
            other => {
                if let Some(model) = other.strip_prefix("model:") {
                    toggle_loaded(model).await;
                } else if let Some(model) = other.strip_prefix("pull:") {
                    let _ = pull_now(&app, model.to_owned()).await;
                }
            }
        }
        app.state::<AppState>().changed();
    });
}

/// Load a model and keep it loaded, or unload it if it is loaded.
async fn toggle_loaded(model: &str) {
    let Ok(c) = client() else { return };
    let loaded = c
        .ps()
        .await
        .map(|ps| ps.models.iter().any(|m| m.name == model))
        .unwrap_or(false);
    if loaded {
        let _ = c.unload(model).await;
    } else {
        let _ = c.load(model, Some(KeepAlive::Forever)).await;
    }
}

fn copy(text: &str) {
    use std::io::Write;
    if let Ok(mut child) = std::process::Command::new("pbcopy")
        .stdin(std::process::Stdio::piped())
        .spawn()
    {
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(text.as_bytes());
        }
        let _ = child.wait();
    }
}

/// Show the window, with a Dock icon while it is open.
fn show_window(app: &AppHandle) {
    let _ = app.set_activation_policy(ActivationPolicy::Regular);
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

/// Hide the window and go back to living in the menu bar only.
pub fn hide_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.hide();
    }
    let _ = app.set_activation_policy(ActivationPolicy::Accessory);
}

/// A LaunchAgent that opens the app at login (what "Open at Login" toggles).
fn login_item() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join("Library/LaunchAgents/dev.ollaya.app.plist")
}

fn set_open_at_login(on: bool) {
    let path = login_item();
    if !on {
        let _ = std::fs::remove_file(&path);
        return;
    }
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>dev.ollaya.app</string>
  <key>ProgramArguments</key><array><string>{}</string></array>
  <key>RunAtLoad</key><true/>
  <key>ProcessType</key><string>Interactive</string>
</dict>
</plist>
"#,
        exe.display()
            .to_string()
            .replace('&', "&amp;")
            .replace('<', "&lt;")
    );
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&path, plist);
}
