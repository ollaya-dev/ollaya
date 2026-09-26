//! The macOS menu bar: Ollaya lives there, like Ollama's app, with no Dock icon until its window
//! is open. The menu follows native menu bar apps such as Tailscale: a header with the server's
//! state and a switch that starts and stops it, the server address (click to copy), the models
//! (a check mark means loaded: click to load or unload; library models download on a click),
//! then the window, Open at Login and Quit.
//!
//! Tauri builds the menu; [`appkit`] then dresses the native `NSMenu` with what Tauri's menu API
//! has no words for: the header view, subtitles and section headers.
//!
//! The menu follows a [`Snapshot`] of the server, taken every two seconds or at once after an
//! action (`AppState::refresh`). When only check marks, download progress or the header change,
//! the items are updated in place, so an open menu stays open and shows the progress live. When
//! the items themselves change (a model installed, the server started), the menu is rebuilt, but
//! never while it is open: replacing it would close it. The rebuild waits for the menu to close.

use std::collections::HashMap;
use std::time::Duration;

use ollaya_api::KeepAlive;
use tauri::image::Image;
use tauri::menu::{
    CheckMenuItem, CheckMenuItemBuilder, Menu, MenuBuilder, MenuEvent, MenuItem, MenuItemBuilder,
    PredefinedMenuItem, SubmenuBuilder,
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

/// What the menu shows. Equal snapshots draw equal menus, so an unchanged one isn't redrawn.
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

impl Snapshot {
    /// Whether `self` and `other` draw menus with the same items, which differ at most in their
    /// check marks, titles, subtitles and whether they are enabled.
    fn same_items(&self, other: &Self) -> bool {
        self.running == other.running
            && self.version == other.version
            && self.url == other.url
            && self.installed == other.installed
            && self.available == other.available
    }
}

/// The menu on screen: the snapshot it shows, and the items a redraw can update in place.
struct Drawn {
    shown: Snapshot,
    items: Items,
}

/// Handles to the menu items that change without changing the menu's shape.
struct Items {
    /// Installed models, checked while loaded.
    models: HashMap<String, CheckMenuItem<Wry>>,
    /// Library models, disabled while downloading.
    pulls: HashMap<String, MenuItem<Wry>>,
    login: CheckMenuItem<Wry>,
}

impl Items {
    /// Updates the items for `s`, which must have the same items as the menu they belong to.
    fn update(&self, s: &Snapshot) {
        let rich = appkit::has_subtitles();
        for (name, item) in &self.models {
            let _ = item.set_checked(s.loaded.contains(name));
        }
        for (name, item) in &self.pulls {
            let pulling = s.pulling.iter().find(|(m, _)| m == name);
            let _ = item.set_enabled(pulling.is_none());
            if !rich {
                let _ = item.set_text(pull_title(name, pulling));
            }
        }
        let _ = self.login.set_checked(s.open_at_login);
    }
}

pub fn init(app: &AppHandle) -> tauri::Result<()> {
    app.set_activation_policy(ActivationPolicy::Accessory)?;
    appkit::set_app(app.clone());
    appkit::watch_menus();
    TrayIconBuilder::with_id(TRAY)
        .icon(Image::from_bytes(ICON_ASLEEP)?)
        .icon_as_template(true)
        .tooltip("Ollaya")
        .menu(&build_menu(app, &Snapshot::default())?.0)
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
        let mut drawn: Option<Drawn> = None;
        loop {
            if library.is_empty() {
                library = library_names().await;
            }
            let now = snapshot(&app, &library).await;
            if drawn.as_ref().map(|d| &d.shown) != Some(&now) {
                redraw(&app, now, &mut drawn);
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

/// Draws `s`, and records in `drawn` what the menu now shows. A menu with the same items is
/// updated in place, open or not. Otherwise the menu is rebuilt if it is closed; if it is open,
/// only the header is updated and the rebuild waits for a pass after the menu closes.
fn redraw(app: &AppHandle, s: Snapshot, drawn: &mut Option<Drawn>) {
    let Some(tray) = app.tray_by_id(TRAY) else {
        return;
    };
    if let Some(d) = drawn.as_mut().filter(|d| d.shown.same_items(&s)) {
        d.items.update(&s);
        let header = s.clone();
        let _ = tray.with_inner_tray_icon(move |t| {
            if let Some(item) = t.ns_status_item() {
                appkit::update(&item, &header);
            }
        });
        d.shown = s;
        return;
    }
    let open = tray
        .with_inner_tray_icon(|t| t.ns_status_item().is_some_and(|i| appkit::menu_is_open(&i)))
        .unwrap_or(false);
    let header = s.clone();
    if open {
        let _ = tray.with_inner_tray_icon(move |_| appkit::update_header(&header));
        return;
    }
    let Ok((menu, items)) = build_menu(app, &s) else {
        return;
    };
    let _ = tray.set_menu(Some(menu));
    let _ = tray.with_inner_tray_icon(move |t| {
        if let Some(item) = t.ns_status_item() {
            appkit::dress(&item, &header);
        }
    });
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
    *drawn = Some(Drawn { shown: s, items });
}

/// The header's second line, and whether its switch is on and can be used.
fn header_state(s: &Snapshot) -> (String, bool, bool) {
    if s.url.is_empty() {
        ("Starting…".to_owned(), true, false)
    } else if !s.running {
        ("Stopped".to_owned(), false, true)
    } else if let Some((model, pct)) = s.pulling.first() {
        (format!("Downloading {model}, {pct}%"), true, true)
    } else {
        (format!("Running, version {}", s.version), true, true)
    }
}

/// A library model's title where subtitles aren't available.
fn pull_title(name: &str, pulling: Option<&(String, u32)>) -> String {
    match pulling {
        Some((_, pct)) => format!("Downloading {name}… {pct}%"),
        None => format!("Download {name}"),
    }
}

fn build_menu(app: &AppHandle, s: &Snapshot) -> tauri::Result<(Menu<Wry>, Items)> {
    // Subtitles and section headers need macOS 14; older systems get plainer titles.
    let rich = appkit::has_subtitles();
    let host = s.url.trim_start_matches("http://").to_owned();

    // The header's title is replaced by a view (`appkit::dress`).
    let mut menu = MenuBuilder::new(app)
        .item(&MenuItemBuilder::with_id("header", "Ollaya").build(app)?)
        .separator();
    menu = if s.running {
        menu.item(
            &MenuItemBuilder::with_id(
                "copy-url",
                if rich {
                    format!("Server: {host}")
                } else {
                    format!("Copy {host}")
                },
            )
            .accelerator("CmdOrCtrl+C")
            .build(app)?,
        )
    } else {
        menu.item(
            &MenuItemBuilder::with_id("server", "Server: not running")
                .enabled(false)
                .build(app)?,
        )
    };

    let mut models = SubmenuBuilder::new(app, "Models").enabled(s.running);
    if s.installed.is_empty() {
        models = models.item(
            &MenuItemBuilder::with_id("no-models", "No models yet")
                .enabled(false)
                .build(app)?,
        );
    }
    let mut installed = HashMap::new();
    for name in &s.installed {
        let item = CheckMenuItemBuilder::with_id(format!("model:{name}"), name)
            .checked(s.loaded.contains(name))
            .build(app)?;
        models = models.item(&item);
        installed.insert(name.clone(), item);
    }
    if !rich && !s.installed.is_empty() {
        models = models.separator().item(
            &MenuItemBuilder::with_id(
                "models-hint",
                "A check mark means loaded: click to load or unload",
            )
            .enabled(false)
            .build(app)?,
        );
    }
    if !s.available.is_empty() {
        models = models.separator();
    }
    let mut pulls = HashMap::new();
    for name in &s.available {
        let pulling = s.pulling.iter().find(|(m, _)| m == name);
        let title = if rich {
            name.clone()
        } else {
            pull_title(name, pulling)
        };
        let item = MenuItemBuilder::with_id(format!("pull:{name}"), title)
            .enabled(pulling.is_none())
            .build(app)?;
        models = models.item(&item);
        pulls.insert(name.clone(), item);
    }

    let login = CheckMenuItemBuilder::with_id("login", "Open at Login")
        .checked(s.open_at_login)
        .build(app)?;
    menu = menu
        .item(&models.build()?)
        .separator()
        .item(
            &MenuItemBuilder::with_id("open", "Open Ollaya…")
                .accelerator("CmdOrCtrl+O")
                .build(app)?,
        )
        .item(&login)
        .separator()
        .item(&PredefinedMenuItem::quit(app, Some("Quit Ollaya"))?);
    let items = Items {
        models: installed,
        pulls,
        login,
    };
    Ok((menu.build()?, items))
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

/// The native side of the menu: what `NSMenu` can show and Tauri's menu API can't ask for.
/// Everything here runs on the main thread (inside `with_inner_tray_icon`, or as an AppKit action).
mod appkit {
    use std::cell::RefCell;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicBool, Ordering};

    use objc2::rc::Retained;
    use objc2::runtime::NSObject;
    use objc2::{ClassType, MainThreadMarker, MainThreadOnly, define_class, msg_send, sel};
    use objc2_app_kit::{
        NSAutoresizingMaskOptions, NSColor, NSControlStateValueOff, NSControlStateValueOn, NSFont,
        NSFontWeightSemibold, NSMenu, NSMenuDidBeginTrackingNotification,
        NSMenuDidEndTrackingNotification, NSMenuItem, NSStatusItem, NSSwitch, NSTextField, NSView,
    };
    use objc2_foundation::{
        NSNotification, NSNotificationCenter, NSPoint, NSRect, NSSize, NSString,
    };
    use tauri::{AppHandle, Manager};

    use super::{AppState, Snapshot, header_state, start_server_now, stop_server_now};

    /// Left and right inset of the header, in line with the titles of the items below it.
    const INSET: f64 = 14.0;
    const HEADER_WIDTH: f64 = 280.0;
    const HEADER_HEIGHT: f64 = 44.0;

    static APP: OnceLock<AppHandle> = OnceLock::new();

    /// Whether one of the app's menus is open, from AppKit's tracking notifications.
    static TRACKING: AtomicBool = AtomicBool::new(false);

    pub fn set_app(app: AppHandle) {
        let _ = APP.set(app);
    }

    define_class!(
        /// Receives the notifications that a menu opened or closed. Notifications, not a menu
        /// delegate: tray-icon makes the status item the menu's delegate.
        // SAFETY: NSObject has no subclassing requirements, and MenuWatcher doesn't implement Drop.
        #[unsafe(super(NSObject))]
        #[thread_kind = MainThreadOnly]
        #[name = "OllayaMenuWatcher"]
        struct MenuWatcher;

        impl MenuWatcher {
            #[unsafe(method(menuOpened:))]
            fn opened(&self, _note: &NSNotification) {
                TRACKING.store(true, Ordering::Relaxed);
            }

            #[unsafe(method(menuClosed:))]
            fn closed(&self, _note: &NSNotification) {
                TRACKING.store(false, Ordering::Relaxed);
                // A rebuild may be waiting for the menu to close.
                if let Some(app) = APP.get() {
                    app.state::<AppState>().changed();
                }
            }
        }
    );

    impl MenuWatcher {
        fn new(mtm: MainThreadMarker) -> Retained<Self> {
            let this = Self::alloc(mtm).set_ivars(());
            // SAFETY: NSObject's designated initializer.
            unsafe { msg_send![super(this), init] }
        }
    }

    /// Starts following whether a menu is open. Call once, on the main thread.
    pub fn watch_menus() {
        let Some(mtm) = MainThreadMarker::new() else {
            return;
        };
        let watcher = MenuWatcher::new(mtm);
        let center = NSNotificationCenter::defaultCenter();
        // SAFETY: both selectors take the notification; the names are constants AppKit exports.
        unsafe {
            center.addObserver_selector_name_object(
                &watcher,
                sel!(menuOpened:),
                Some(NSMenuDidBeginTrackingNotification),
                None,
            );
            center.addObserver_selector_name_object(
                &watcher,
                sel!(menuClosed:),
                Some(NSMenuDidEndTrackingNotification),
                None,
            );
        }
        // The center doesn't retain its observers, and this one lives as long as the app.
        std::mem::forget(watcher);
    }

    /// The header on screen, so a redraw while the menu is open can update it in place.
    struct Header {
        subtitle: Retained<NSTextField>,
        switch: Retained<NSSwitch>,
        /// A control holds its target weakly.
        _target: Retained<SwitchTarget>,
    }

    thread_local! {
        static HEADER: RefCell<Option<Header>> = const { RefCell::new(None) };
    }

    define_class!(
        /// Receives the header switch's action: start or stop the server.
        // SAFETY: NSObject has no subclassing requirements, and SwitchTarget doesn't implement Drop.
        #[unsafe(super(NSObject))]
        #[thread_kind = MainThreadOnly]
        #[name = "OllayaServerSwitchTarget"]
        struct SwitchTarget;

        impl SwitchTarget {
            #[unsafe(method(toggle:))]
            fn toggle(&self, sender: &NSSwitch) {
                let on = sender.state() == NSControlStateValueOn;
                sender.setEnabled(false);
                HEADER.with(|h| {
                    if let Some(h) = &*h.borrow() {
                        h.subtitle.setStringValue(&NSString::from_str(if on {
                            "Starting…"
                        } else {
                            "Stopping…"
                        }));
                    }
                });
                let Some(app) = APP.get().cloned() else {
                    return;
                };
                tauri::async_runtime::spawn(async move {
                    if on {
                        let _ = start_server_now(&app).await;
                    } else {
                        let _ = stop_server_now(&app).await;
                    }
                    app.state::<AppState>().changed();
                });
            }
        }
    );

    impl SwitchTarget {
        fn new(mtm: MainThreadMarker) -> Retained<Self> {
            let this = Self::alloc(mtm).set_ivars(());
            // SAFETY: NSObject's designated initializer.
            unsafe { msg_send![super(this), init] }
        }
    }

    /// `NSMenuItem` subtitles and section headers arrived in macOS 14.
    pub fn has_subtitles() -> bool {
        NSMenuItem::class().responds_to(sel!(setSubtitle:))
    }

    /// Whether a menu is open: AppKit is tracking one, or the status item's button is highlighted.
    /// The highlight alone isn't enough: on macOS 27 it is off while the status item's menu is
    /// open.
    pub fn menu_is_open(item: &NSStatusItem) -> bool {
        let highlighted = MainThreadMarker::new()
            .and_then(|mtm| item.button(mtm))
            .is_some_and(|b| b.isHighlighted());
        let tracking = TRACKING.load(Ordering::Relaxed);
        tracking || highlighted
    }

    /// Dresses the menu Tauri just attached to the status item.
    pub fn dress(item: &NSStatusItem, s: &Snapshot) {
        let Some(mtm) = MainThreadMarker::new() else {
            return;
        };
        let Some(menu) = item.menu(mtm) else {
            return;
        };
        if let Some(first) = menu.itemAtIndex(0) {
            let (view, header) = header_view(mtm, s);
            first.setView(Some(&view));
            HEADER.with(|h| *h.borrow_mut() = Some(header));
        }
        if !has_subtitles() {
            return;
        }
        for i in 0..menu.numberOfItems() {
            let Some(entry) = menu.itemAtIndex(i) else {
                continue;
            };
            if entry.title().to_string().starts_with("Server: ") && entry.isEnabled() {
                entry.setSubtitle(Some(&NSString::from_str("Click to copy")));
            }
            if let Some(models) = entry.submenu() {
                dress_models(mtm, &models, s);
            }
        }
    }

    /// Updates a menu with the same items as `s` (the header, and the models' subtitles).
    pub fn update(item: &NSStatusItem, s: &Snapshot) {
        update_header(s);
        let Some(mtm) = MainThreadMarker::new() else {
            return;
        };
        let Some(menu) = item.menu(mtm) else {
            return;
        };
        if !has_subtitles() {
            return;
        }
        for i in 0..menu.numberOfItems() {
            if let Some(models) = menu.itemAtIndex(i).and_then(|entry| entry.submenu()) {
                subtitle_models(&models, s);
            }
        }
    }

    /// Subtitles on the models: loaded, downloading or click to download. Returns the index of
    /// the first library model.
    fn subtitle_models(menu: &NSMenu, s: &Snapshot) -> Option<isize> {
        let mut first_library = None;
        for i in 0..menu.numberOfItems() {
            let Some(entry) = menu.itemAtIndex(i) else {
                continue;
            };
            let title = entry.title().to_string();
            let subtitle = if s.loaded.contains(&title) {
                Some("Loaded".to_owned())
            } else if s.available.contains(&title) {
                first_library.get_or_insert(i);
                Some(match s.pulling.iter().find(|(m, _)| *m == title) {
                    Some((_, pct)) => format!("Downloading, {pct}%"),
                    None => "Click to download".to_owned(),
                })
            } else if s.installed.contains(&title) {
                None
            } else {
                continue;
            };
            entry.setSubtitle(subtitle.map(|t| NSString::from_str(&t)).as_deref());
        }
        first_library
    }

    /// Subtitles on the models, and a section header over the installed and the library ones.
    fn dress_models(mtm: MainThreadMarker, menu: &NSMenu, s: &Snapshot) {
        let first_library = subtitle_models(menu, s);
        // Insert from the bottom up so the first index stays valid.
        if let Some(i) = first_library {
            let header = NSMenuItem::sectionHeaderWithTitle(&NSString::from_str("Library"), mtm);
            menu.insertItem_atIndex(&header, i);
        }
        if !s.installed.is_empty() {
            let header =
                NSMenuItem::sectionHeaderWithTitle(&NSString::from_str("On This Mac"), mtm);
            menu.insertItem_atIndex(&header, 0);
        }
    }

    /// Updates the header of an open menu.
    pub fn update_header(s: &Snapshot) {
        let (subtitle, on, enabled) = header_state(s);
        HEADER.with(|h| {
            if let Some(h) = &*h.borrow() {
                h.subtitle.setStringValue(&NSString::from_str(&subtitle));
                h.switch.setState(if on {
                    NSControlStateValueOn
                } else {
                    NSControlStateValueOff
                });
                h.switch.setEnabled(enabled);
            }
        });
    }

    /// "Ollaya" over the server's state, with a switch on the right, like Tailscale's header.
    fn header_view(mtm: MainThreadMarker, s: &Snapshot) -> (Retained<NSView>, Header) {
        let (subtitle, on, enabled) = header_state(s);
        let view = NSView::initWithFrame(
            NSView::alloc(mtm),
            NSRect::new(
                NSPoint::new(0.0, 0.0),
                NSSize::new(HEADER_WIDTH, HEADER_HEIGHT),
            ),
        );
        view.setAutoresizingMask(NSAutoresizingMaskOptions::ViewWidthSizable);

        let switch = NSSwitch::new(mtm);
        switch.sizeToFit();
        let size = switch.frame().size;
        switch.setFrameOrigin(NSPoint::new(
            HEADER_WIDTH - INSET - size.width,
            ((HEADER_HEIGHT - size.height) / 2.0).round(),
        ));
        switch.setAutoresizingMask(NSAutoresizingMaskOptions::ViewMinXMargin);
        switch.setState(if on {
            NSControlStateValueOn
        } else {
            NSControlStateValueOff
        });
        switch.setEnabled(enabled);
        let target = SwitchTarget::new(mtm);
        // SAFETY: `toggle:` takes the sender, and the header keeps the target alive.
        unsafe {
            switch.setTarget(Some(&target));
            switch.setAction(Some(sel!(toggle:)));
        }
        view.addSubview(&switch);

        let text_width = HEADER_WIDTH - 2.0 * INSET - size.width - 8.0;
        let title = NSTextField::labelWithString(&NSString::from_str("Ollaya"), mtm);
        // SAFETY: a constant AppKit exports.
        title.setFont(Some(&NSFont::systemFontOfSize_weight(13.0, unsafe {
            NSFontWeightSemibold
        })));
        title.setTextColor(Some(&NSColor::labelColor()));
        title.sizeToFit();
        title.setFrameOrigin(NSPoint::new(INSET, 22.0));
        view.addSubview(&title);

        let second = NSTextField::labelWithString(&NSString::from_str(&subtitle), mtm);
        second.setFont(Some(&NSFont::systemFontOfSize(11.0)));
        second.setTextColor(Some(&NSColor::secondaryLabelColor()));
        second.sizeToFit();
        let height = second.frame().size.height;
        second.setFrame(NSRect::new(
            NSPoint::new(INSET, 7.0),
            NSSize::new(text_width, height),
        ));
        second.setAutoresizingMask(NSAutoresizingMaskOptions::ViewWidthSizable);
        view.addSubview(&second);

        (
            view,
            Header {
                subtitle: second,
                switch,
                _target: target,
            },
        )
    }
}
