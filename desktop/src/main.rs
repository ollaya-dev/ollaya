//! Ollaya's desktop app, on Rune (https://github.com/anomalyco/rune), a Rust UI framework that
//! draws on the GPU itself, with no webview. It runs `ollaya serve` in the background, as
//! Ollama's app runs `ollama serve`, and shows four pages: Discover (the model library), My
//! Models, Playground and Server. On macOS it lives in the menu bar.
//!
//! One process, three parts: Rune's UI thread owns [`state::State`] and draws it ([`ui`]);
//! [`backend`] talks to the server and the website on a Tokio thread and answers through
//! `AppHandle::update`; on macOS [`menubar`] is an AppKit status item, since Rune has none yet
//! (anomalyco/rune#141).

// Windows: a GUI app, with no console window beside it.
#![cfg_attr(windows, windows_subsystem = "windows")]

mod backend;
mod engine;
mod library;
#[cfg(target_os = "macos")]
mod login;
#[cfg(target_os = "macos")]
mod menubar;
mod scenes;
mod selftest;
mod state;
mod theme;
mod ui;

use std::time::{Duration, Instant};

use rune::{App, AppSpec, Chrome, Tx};

use state::State;

impl State {
    pub fn quit(&mut self, tx: &mut Tx) {
        engine::stop_if_started();
        tx.quit();
    }

    #[cfg(target_os = "macos")]
    pub fn menu(&mut self, action: menubar::Action, tx: &mut Tx) {
        use menubar::Action;
        match action {
            Action::Server(on) => {
                if on != self.snap.running {
                    self.toggle_server();
                }
                // AppKit flipped the switch itself: when nothing changed (already there, or busy
                // starting or stopping), put it back.
                menubar::redraw(self);
            }
            Action::CopyAddress => {
                if self.snap.running {
                    tx.copy(self.snap.url.clone());
                }
            }
            Action::ToggleModel(name) => self.toggle_model(name),
            Action::Pull(tag) => self.pull(tag),
            Action::Update => {
                let _ = tx.open_url(library::DOWNLOAD_URL);
            }
            Action::Open => self.show_window(),
            Action::OpenAtLogin => {
                if let Err(e) = login::set(!login::is_on()) {
                    self.error = Some(format!("Open at Login: {e}"));
                }
            }
            Action::Quit => self.quit(tx),
        }
    }
}

fn on_close(s: &mut State, _: &mut Tx) {
    // Closing the window leaves Ollaya in the menu bar, with no Dock icon.
    #[cfg(target_os = "macos")]
    s.handle.set_dock_icon(false);
    #[cfg(not(target_os = "macos"))]
    let _ = s;
}

/// Quitting stops a server this app started, as Ollama's app does.
fn on_quit(_: &mut State, _: &mut Tx) {
    engine::stop_if_started();
}

/// Clears a copy button's check mark once it has shown long enough.
fn poll(s: &mut State, now: f64, tx: &mut Tx) -> bool {
    match &s.copied {
        Some((_, at)) if now - at >= ui::COPIED_FOR => {
            s.copied = None;
            true
        }
        Some((_, at)) => {
            tx.wake_after(Duration::from_secs_f64(ui::COPIED_FOR - (now - at)));
            false
        }
        None => false,
    }
}

fn main() {
    let started = Instant::now();
    // Linux: the Wayland app_id and X11 WM_CLASS, matching the installed ollaya-desktop.desktop
    // (Packager.toml), so the desktop shows its icon and name.
    rune::set_app_id("ollaya-desktop");
    // Windows: the icon of the window, the taskbar and Alt+Tab (the executable's own is build.rs's;
    // macOS takes the bundle's).
    #[cfg(windows)]
    if let Err(e) = rune::set_app_icon(include_bytes!("../icons/128x128@2x.png").as_slice()) {
        eprintln!("{e}");
    }
    let spec = AppSpec {
        min_size: Some((820.0, 580.0)),
        scenes: scenes::SCENES,
        poll: Some(poll),
        ..AppSpec::new("Ollaya", (1080.0, 740.0), || unreachable!(), ui::view)
    };
    // The content runs under a transparent titlebar; the traffic lights sit in the rail, level
    // with the page's title.
    let app = App::new(spec).chrome(Chrome {
        traffic_lights: Some((12.0, 13.0)),
        ..Chrome::DEFAULT
    });
    let handle = app.handle();
    // A headless preview (`--snap`, `--tree`, `--script`) shows its scene, not the real server.
    let headless = std::env::args()
        .any(|a| matches!(a.as_str(), "--snap" | "--tree" | "--script" | "--video"));
    let cmd = if headless {
        tokio::sync::mpsc::unbounded_channel().0
    } else {
        backend::spawn(handle.clone())
    };
    let app = app
        .init({
            let handle = handle.clone();
            move || State::new(handle, cmd)
        })
        .on_close(on_close)
        .on_quit(on_quit)
        .on_ready(move |_| {
            if std::env::var_os("OLLAYA_TIMING").is_some() {
                eprintln!(
                    "first frame {} ms after start",
                    started.elapsed().as_millis()
                );
            }
        });

    // macOS: Ollaya lives in the menu bar, with no Dock icon until its window opens.
    // `OLLAYA_OPEN_WINDOW=1` opens the window at launch (for measuring).
    #[cfg(target_os = "macos")]
    let app = {
        let menu_handle = handle.clone();
        handle.update_if(move |s, _| {
            let h = menu_handle.clone();
            if let Err(e) = menubar::install(Box::new(move |action| {
                h.update(move |s, tx| s.menu(action, tx));
            })) {
                eprintln!("menu bar: {e}; opening the window");
                s.show_window();
            }
            s.sync_menu();
            false
        });
        if std::env::var_os("OLLAYA_OPEN_WINDOW").is_some() {
            // An open window has a Dock icon, as after "Open Ollaya".
            handle.set_dock_icon(true);
            app.accessory().keep_running()
        } else {
            app.accessory().windowless()
        }
    };

    // Elsewhere the window is the app: closing it quits, stopping a server it started.
    app.run();
}
