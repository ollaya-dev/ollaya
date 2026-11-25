//! The window: the rail of pages, the page, and the status bar, drawn from [`State`] each frame.
//! The look is the website's (`theme.rs`): figures in tabular numbers, labels in sentence case,
//! one off-white primary button per view, green only for what is running or done.

mod discover;
mod models;
mod playground;
mod server;

use std::time::Duration;

use rune::{
    Align, Color, Cx, El, Family, Icon, ScrollSpec, TextStyle, Tx, col, el, icon, row, spacer, text,
};
use rune_kit::{Kit, PaletteFocus, PaletteSeed};

use crate::backend::Progress;
use crate::state::{Page, State};
use crate::theme::*;

// --- look ----------------------------------------------------------------------------------------

/// Figures line up in columns.
const TNUM: &str = "'tnum' 1";
/// Running text: descriptions, empty states, the playground's text.
const S_BODY: TextStyle = TextStyle::new(13.0, 400.0);
/// The page's title in the window's top band.
const S_TITLE: TextStyle = TextStyle::new(13.0, 500.0);
/// Tables, the status bar, rows of the API panel.
const S_SMALL: TextStyle = TextStyle::new(12.0, 400.0);
const S_SMALL_NUM: TextStyle = S_SMALL.features(TNUM);
/// The platform's own mono (SF Mono on macOS), as the Tauri app's `ui-monospace`.
const S_MONO: TextStyle = TextStyle::new(12.0, 400.0).family(Family::SystemMono);
/// Buttons and a section's label.
const S_STRONG: TextStyle = TextStyle::new(12.0, 500.0);
/// Notes under figures, table headers, the rail's labels.
const S_META: TextStyle = TextStyle::new(11.0, 400.0);
const S_META_NUM: TextStyle = S_META.features(TNUM);
const S_HEAD: TextStyle = TextStyle::new(11.0, 500.0);
const S_FIGURE: TextStyle = TextStyle::new(20.0, 500.0).features(TNUM);
/// A family's name over its detail, and in the library's list.
const S_NAME: TextStyle = TextStyle::new(20.0, 600.0).family(Family::SystemMono);
const S_LIST_NAME: TextStyle = TextStyle::new(13.0, 400.0).family(Family::SystemMono);
const S_MONO_META: TextStyle = TextStyle::new(11.0, 400.0).family(Family::SystemMono);
/// An answer in the playground.
const S_ANSWER: TextStyle = TextStyle::new(13.0, 500.0);
/// A measured figure in running text: "0.812 accuracy".
const S_STAT: TextStyle = TextStyle::new(12.0, 500.0).features(TNUM);
/// The count on the rail's My Models.
const BADGE: TextStyle = TextStyle::new(9.0, 500.0).features(TNUM);

/// The window's rail: wide enough for the traffic lights over it, with 12 pt either side.
pub const RAIL: f32 = 84.0;
/// The window's top band, level with the traffic lights.
const BAND: f32 = 40.0;
/// The pages' column.
const COLUMN: f32 = 768.0;
/// How long a copy button shows its check mark.
pub const COPIED_FOR: f64 = 1.5;

/// Rune's controls in the app's colours: the pickers' menus.
pub fn kit() -> Kit {
    Kit::from_palette_focus(
        PaletteSeed {
            bg: CANVAS,
            fg: FG,
            accent: FG,
        },
        PaletteFocus::Ink,
    )
}

/// Ollaya's mark, the gaze: two rings and one pupil (site/src/components/Logo.tsx).
pub static GAZE: Icon = Icon {
    name: "gaze",
    view: 32.0,
    body: r#"<circle cx="9.5" cy="16" r="6.5" fill="none" stroke="currentColor" stroke-width="3.2"/><circle cx="22.5" cy="16" r="6.5" fill="none" stroke="currentColor" stroke-width="3.2"/><circle cx="22.5" cy="16" r="2.6" fill="currentColor"/>"#,
};

/// Line icons, as the Tauri app drew them: 20 pt, a 1.6 stroke, round ends.
macro_rules! line_icon {
    ($name:ident, $body:literal) => {
        line_icon!($name, $body, "1.6");
    };
    ($name:ident, $body:literal, $width:literal) => {
        pub static $name: Icon = Icon {
            name: stringify!($name),
            view: 20.0,
            body: concat!(
                r#"<g fill="none" stroke="currentColor" stroke-width=""#,
                $width,
                r#"" stroke-linecap="round" stroke-linejoin="round">"#,
                $body,
                "</g>"
            ),
        };
    };
}

line_icon!(
    DISCOVER,
    r#"<circle cx="10" cy="10" r="7"/><path d="m12.8 7.2-1.6 4-4 1.6 1.6-4z"/>"#
);
line_icon!(
    MODELS,
    r#"<path d="M3.5 6.5 10 3l6.5 3.5v7L10 17l-6.5-3.5zM3.5 6.5 10 10l6.5-3.5M10 10v7"/>"#
);
line_icon!(PLAY, r#"<path d="M6.5 4.5v11l9-5.5z"/>"#);
line_icon!(SERVER, r#"<path d="M3 10h3l2-5 4 10 2-5h3"/>"#);
line_icon!(
    SEARCH,
    r#"<circle cx="9" cy="9" r="5"/><path d="m16 16-3.5-3.5"/>"#
);
line_icon!(DOWNLOAD, r#"<path d="M10 3v10m0 0 4-4m-4 4-4-4M4 16h12"/>"#);
line_icon!(X, r#"<path d="m6 6 8 8M14 6l-8 8"/>"#);
line_icon!(
    COPY,
    r#"<rect x="7" y="7" width="9" height="9" rx="1.5"/><path d="M13 7V5.5A1.5 1.5 0 0 0 11.5 4h-6A1.5 1.5 0 0 0 4 5.5v6A1.5 1.5 0 0 0 5.5 13H7"/>"#
);
line_icon!(CHECK, r#"<path d="m5 10.5 3.2 3.2L15 7"/>"#, "1.8");
line_icon!(
    EXTERNAL,
    r#"<path d="M11 4h5v5m0-5-7 7M8 5H5a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-3"/>"#
);

// --- pieces --------------------------------------------------------------------------------------

fn hairline() -> El {
    el().w_full().h(1.0).fill(LINE)
}

/// A panel: the site's bordered surface, its rows clipped to its corners.
fn panel() -> El {
    col()
        .w_full()
        .radius(8.0)
        .clip_rounded(8.0)
        .fill(PANEL)
        .border(1.0, LINE)
}

/// A row inside a panel, with a hairline above all but the first.
fn panel_row(first: bool) -> El {
    let r = row().w_full().px(12.0).gap(12.0).items_center();
    if first {
        r
    } else {
        r.border_sides([1.0, 0.0, 0.0, 0.0], LINE)
    }
}

/// A section's label over its panel, with an optional note at its right.
fn section(label: &'static str, note: Option<El>) -> El {
    let mut r = row()
        .w_full()
        .pt(24.0)
        .pb(8.0)
        .items_center()
        .child(text(label).font(S_STRONG).color(MUTED))
        .child(spacer());
    if let Some(note) = note {
        r = r.child(note);
    }
    r
}

/// The off-white primary button, or the outlined secondary one. A disabled one is dimmed and
/// takes no press: attach the press with [`on_press`].
fn button(key: impl Into<String>, label: impl Into<String>, primary: bool, enabled: bool) -> El {
    button_with(key, None, label, primary, enabled)
}

/// A button with an icon before its label.
fn icon_button(
    key: impl Into<String>,
    glyph: &'static Icon,
    label: impl Into<String>,
    primary: bool,
    enabled: bool,
) -> El {
    button_with(key, Some(glyph), label, primary, enabled)
}

fn button_with(
    key: impl Into<String>,
    glyph: Option<&'static Icon>,
    label: impl Into<String>,
    primary: bool,
    enabled: bool,
) -> El {
    let label = label.into();
    let b = row()
        .key(key)
        .role("button")
        .label(label.clone())
        .focusable(enabled)
        .disabled(!enabled)
        .h(28.0)
        .px(12.0)
        .gap(6.0)
        .items_center()
        .radius(6.0)
        .opacity(if enabled { 1.0 } else { 0.5 });
    let b = match (primary, enabled) {
        (true, true) => b.fill(BTN).hover(BTN_HOVER),
        (true, false) => b.fill(BTN),
        (false, true) => b.border(1.0, LINE).hover(FILL),
        (false, false) => b.border(1.0, LINE),
    };
    let ink = if primary { BTN_FG } else { FG };
    let b = match glyph {
        Some(g) => b.child(icon(g, 14.0).color(ink)),
        None => b,
    };
    b.child(text(label).font(S_STRONG).color(ink))
}

/// A quiet square button with only an icon, named for assistive technology and its tooltip.
fn quiet(key: impl Into<String>, glyph: &'static Icon, label: String, enabled: bool) -> El {
    let b = row()
        .key(key)
        .role("button")
        .label(label.clone())
        .help(label)
        .focusable(enabled)
        .disabled(!enabled)
        .size(28.0, 28.0)
        .center()
        .radius(6.0)
        .opacity(if enabled { 1.0 } else { 0.5 });
    let b = if enabled { b.hover(FILL) } else { b };
    b.child(icon(glyph, 16.0).color(MUTED))
}

/// `on_press`, only when the control can be used.
fn on_press(b: El, enabled: bool, f: impl Fn(&mut State) + 'static) -> El {
    if enabled { b.on_press(f) } else { b }
}

/// A text link: muted, brighter under the pointer.
fn link(key: impl Into<String>, label: impl Into<String>, external: bool) -> El {
    let label = label.into();
    let mut r = row()
        .key(key)
        .role("link")
        .label(label.clone())
        .focusable(true)
        .gap(4.0)
        .items_center()
        .child(text(label).font(S_META).color(MUTED));
    if external {
        r = r.child(icon(&EXTERNAL, 12.0).color(MUTED));
    }
    r
}

/// A quiet icon button that copies `what` and shows a check mark for a moment.
fn copy_button(s: &State, key: String, what: String, label: &str) -> El {
    let done = s.copied.as_ref().is_some_and(|(k, _)| *k == key);
    let k = key.clone();
    row()
        .key(key)
        .role("button")
        .label(label.to_owned())
        .help(label.to_owned())
        .focusable(true)
        .size(28.0, 28.0)
        .center()
        .radius(6.0)
        .hover(FILL)
        .on_press_tx(move |s: &mut State, tx: &mut Tx| {
            tx.copy(what.clone());
            s.copied = Some((k.clone(), tx.clock()));
            tx.wake_after(Duration::from_secs_f64(COPIED_FOR));
        })
        .child(icon(if done { &CHECK } else { &COPY }, 16.0).color(if done { OK } else { MUTED }))
}

/// A cell `w` wide whose content sits at its right edge, for figures.
fn right(w: f32, child: El) -> El {
    row().w(w).justify_end().child(child)
}

/// A thin bar of how far a download is, and its percentage.
fn progress(p: Progress) -> El {
    const W: f32 = 104.0;
    let f = p.fraction().unwrap_or(0.0) as f32;
    row()
        .gap(8.0)
        .items_center()
        .child(
            row()
                .w(W)
                .h(4.0)
                .radius(2.0)
                .clip_rounded(2.0)
                .fill(FILL_STRONG)
                .child(el().w((W * f).max(2.0)).h_full().fill(OK)),
        )
        .child(right(
            32.0,
            text(
                p.fraction()
                    .map_or("…".into(), |f| format!("{}%", (f * 100.0).floor())),
            )
            .font(S_META_NUM)
            .color(MUTED),
        ))
}

/// The Download button, or how far a download is, or that the version is on this computer.
fn pull_state(s: &State, tag: &str) -> El {
    if let Some(p) = s.pulling.get(tag) {
        return progress(*p);
    }
    if s.is_installed(tag) {
        return row()
            .gap(4.0)
            .items_center()
            .child(icon(&CHECK, 14.0).color(OK))
            .child(text("Installed").font(S_SMALL).color(MUTED));
    }
    let running = s.snap.running;
    let t = tag.to_owned();
    let b = icon_button(format!("pull-{tag}"), &DOWNLOAD, "Download", false, running);
    let b = if running {
        b
    } else {
        b.help("Start the server first")
    };
    on_press(b, running, move |s: &mut State| s.pull(t.clone()))
}

/// What fills a page when it has nothing to show: the gaze, a line, and the way on.
fn empty(title: &str, line: &str, action: Option<El>, error: Option<&str>) -> El {
    let mut c = col()
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .center()
        .px(24.0)
        .gap(8.0)
        .child(icon(&GAZE, 32.0).color(FAINT))
        .child(el().h(4.0))
        .child(text(title.to_owned()).font(S_TITLE).color(FG))
        .child(
            text(line.to_owned())
                .font(S_BODY)
                .color(MUTED)
                .align(Align::Center)
                .wrap(),
        );
    if let Some(action) = action {
        c = c.child(el().h(8.0)).child(action);
    }
    if let Some(error) = error {
        c = c.child(
            text(error.to_owned())
                .font(S_SMALL)
                .color(BAD)
                .align(Align::Center)
                .max_w(480.0)
                .wrap(),
        );
    }
    c
}

/// The server switch's label: what pressing it does, or what it is doing.
fn switch_label(s: &State, long: bool) -> String {
    let label = match (s.busy || !s.checked, s.snap.running) {
        (true, true) => "Stopping…",
        (true, false) => "Starting…",
        (false, true) => "Stop",
        (false, false) => "Start",
    };
    if long && !s.busy && s.checked {
        format!("{label} server")
    } else {
        label.to_owned()
    }
}

fn start_button(s: &State) -> El {
    let ok = !s.busy && s.checked;
    on_press(
        button("empty-start", switch_label(s, true), true, ok),
        ok,
        |s: &mut State| s.toggle_server(),
    )
}

/// The empty page of a stopped server.
fn stopped(s: &State, line: &str) -> El {
    if !s.checked {
        return empty("Checking the server…", "", None, None);
    }
    empty(
        "The server is stopped",
        line,
        Some(start_button(s)),
        s.error.as_deref(),
    )
}

fn discover_button() -> El {
    button("go-discover", "Discover models", false, true)
        .on_press(|s: &mut State| s.go(Page::Discover))
}

/// A page's scrolling column, 768 pt at most, centred.
fn page(key: &'static str) -> (El, El) {
    let outer = col()
        .key(key)
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .items_center()
        .scroll(ScrollSpec::default());
    let inner = col().w_full().max_w(COLUMN).px(24.0).pt(0.0).pb(32.0);
    (outer, inner)
}

// --- the window ----------------------------------------------------------------------------------

pub fn view(s: &State, _: &mut Cx<State>) -> El {
    let mut layer = Vec::new();
    let (title, tools, body) = match s.page {
        Page::Discover => ("Discover", None, discover::page(s)),
        Page::Models => ("My Models", None, models::page(s)),
        Page::Playground => {
            let (tools, body) = playground::page(s, &mut layer);
            ("Playground", tools, body)
        }
        Page::Server => ("Server", None, server::page(s)),
    };
    let mut band = row()
        .w_full()
        .h(BAND)
        .px(24.0)
        .gap(8.0)
        .items_center()
        .window_drag()
        .child(text(title).font(S_TITLE).color(FG))
        .child(spacer());
    if let Some(tools) = tools {
        band = band.pr(12.0).child(tools);
    }
    let base = col()
        .w_full()
        .h_full()
        .fill(CANVAS)
        .child(
            row()
                .w_full()
                .grow(1.0)
                .min_h(0.0)
                .child(rail(s))
                .child(el().w(1.0).h_full().fill(LINE))
                .child(
                    col()
                        .grow(1.0)
                        .min_w(0.0)
                        .h_full()
                        .child(band)
                        .child(hairline())
                        .child(body),
                ),
        )
        .child(hairline())
        .child(status_bar(s));
    rune_kit::root(base, layer)
}

/// The rail: the gaze under the traffic lights, then the pages.
fn rail(s: &State) -> El {
    let have = s.snap.models.len();
    col()
        .w(RAIL)
        .h_full()
        .pt(BAND + 8.0)
        .px(8.0)
        .gap(4.0)
        .items_center()
        .window_drag()
        .child(icon(&GAZE, 28.0).color(FG))
        .child(el().h(8.0))
        .child(nav(s, Page::Discover, "Discover", &DISCOVER, 0))
        .child(nav(s, Page::Models, "My Models", &MODELS, have))
        .child(nav(s, Page::Playground, "Playground", &PLAY, 0))
        .child(nav(s, Page::Server, "Server", &SERVER, 0))
}

fn nav(s: &State, page: Page, label: &'static str, glyph: &'static Icon, count: usize) -> El {
    let on = s.page == page;
    let ink = if on { FG } else { MUTED };
    let mut glyph_row = row().child(icon(glyph, 20.0).color(ink));
    if count > 0 {
        // How many versions are on this computer, at the icon's corner.
        glyph_row = glyph_row.child(
            row()
                .absolute(14.0, -4.0)
                .h(14.0)
                .min_w(14.0)
                .px(4.0)
                .radius(7.0)
                .fill(FILL_STRONG)
                .center()
                .child(text(count.to_string()).font(BADGE).color(BODY)),
        );
    }
    col()
        .key(label)
        .role("button")
        .label(label)
        .focusable(true)
        .w_full()
        .py(8.0)
        .gap(4.0)
        .items_center()
        .radius(8.0)
        .fill(if on { FILL_STRONG } else { Color::TRANSPARENT })
        .hover(if on { FILL_STRONG } else { FILL })
        .on_press(move |s: &mut State| s.go(page))
        .child(glyph_row)
        .child(text(label).font(S_META).color(ink))
}

fn status_bar(s: &State) -> El {
    let snap = &s.snap;
    let what = if !s.checked {
        text("Checking the server…").font(S_SMALL).color(MUTED)
    } else if snap.running {
        text(format!(
            "Running on {}",
            snap.url.trim_start_matches("http://")
        ))
        .font(S_SMALL)
        .color(BODY)
    } else {
        text("Server stopped").font(S_SMALL).color(MUTED)
    };
    let mut lead = row()
        .gap(8.0)
        .items_center()
        .child(
            el().size(8.0, 8.0)
                .radius(4.0)
                .fill(if snap.running { OK } else { FAINT }),
        )
        .child(what);
    if let Some(device) = snap.device.filter(|_| snap.running) {
        lead = lead
            .child(text(format!("· {device}")).font(S_SMALL).color(MUTED))
            .help(device_hint(device));
    }
    let mut bar = row()
        .w_full()
        .h(32.0)
        .px(12.0)
        .gap(12.0)
        .items_center()
        .child(lead);
    if snap.running {
        let loaded: Vec<&str> = snap
            .models
            .iter()
            .filter(|m| m.loaded)
            .map(|m| m.name.as_str())
            .collect();
        let line = if loaded.is_empty() {
            "· nothing loaded".to_owned()
        } else {
            format!("· loaded: {}", loaded.join(", "))
        };
        bar = bar.child(text(line).font(S_SMALL).color(MUTED).shrink().truncate());
    }
    if let Some(err) = s.error.as_ref().filter(|_| !page_shows_error(s)) {
        bar = bar.child(
            text(err.clone())
                .font(S_SMALL)
                .color(BAD)
                .shrink()
                .truncate()
                .help(err.clone()),
        );
    }
    bar = bar.child(spacer());
    if let Some(v) = &s.update {
        bar = bar.child(
            row()
                .key("update")
                .role("link")
                .label(format!("Update to {v}"))
                .help("Opens the download page")
                .focusable(true)
                .on_press_tx(|_: &mut State, tx: &mut Tx| {
                    let _ = tx.open_url(crate::library::DOWNLOAD_URL);
                })
                .child(text(format!("Update to {v}")).font(S_SMALL).color(FG)),
        );
    }
    let ok = !s.busy && s.checked;
    bar = bar.child(on_press(
        button("status-switch", switch_label(s, false), false, ok)
            .h(24.0)
            .px(10.0),
        ok,
        |s: &mut State| s.toggle_server(),
    ));
    if let Some(v) = &snap.version {
        bar = bar.child(text(format!("v{v}")).font(S_SMALL_NUM).color(MUTED));
    }
    bar
}

/// The status bar's tooltip for where the server runs models.
fn device_hint(device: &str) -> String {
    if device == "GPU" {
        "The server runs models on the NVIDIA GPU, with the GPU pack of the command-line install"
            .into()
    } else {
        "The server runs models on the CPU. Install Ollaya's command line with its GPU pack to use an NVIDIA GPU"
            .into()
    }
}

/// Whether the page shows the last server error itself: a stopped server's page does.
fn page_shows_error(s: &State) -> bool {
    !s.snap.running && s.page != Page::Discover
}

// --- figures -------------------------------------------------------------------------------------

pub fn bytes(n: u64) -> String {
    let n = n as f64;
    if n >= 1e9 {
        format!("{:.1} GB", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.0} MB", n / 1e6)
    } else {
        format!("{:.0} kB", n / 1e3)
    }
}

pub fn ms(n: f64) -> String {
    if n < 10.0 {
        format!("{n:.1}")
    } else {
        format!("{}", n.round() as u64)
    }
}

/// `1284` as `1,284`.
pub fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn figures_read_as_the_tauri_app_writes_them() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1284), "1,284");
        assert_eq!(thousands(1_234_567), "1,234,567");
        assert_eq!(ms(3.06), "3.1");
        assert_eq!(ms(12.4), "12");
        assert_eq!(bytes(1_240_000_000), "1.2 GB");
        assert_eq!(bytes(92_000_000), "92 MB");
    }
}
