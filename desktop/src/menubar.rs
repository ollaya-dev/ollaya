//! The macOS menu bar, in AppKit: Ollaya lives there, like Ollama's app, with no Dock icon until
//! its window is open. Rune has windows, a tray on Windows and Linux, and an accessory activation
//! policy, but no macOS status item yet (anomalyco/rune#141); Anomaly's own Hex does the same.
//!
//! The menu follows native menu bar apps such as Tailscale, as the Tauri app's did: a header with
//! the server's state and a switch that starts and stops it, the server address (click to copy),
//! the models (a check mark means loaded: click to load or unload; library models download on a
//! click), a newer release, then the window, Open at Login and Quit.
//!
//! The menu is built when it opens (`menuNeedsUpdate:`) from the latest [`View`] of the app's
//! state. While it is open, [`sync`] updates its header, check marks and download progress in
//! place; anything that changes its items waits for the next time it opens, since replacing an
//! open menu would close it. Everything here runs on the main thread, which is Rune's UI thread.

use std::cell::RefCell;
use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObjectProtocol, ProtocolObject, Sel};
use objc2::{AnyThread, DefinedClass, MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::{
    NSAutoresizingMaskOptions, NSColor, NSControlStateValueOff, NSControlStateValueOn,
    NSEventModifierFlags, NSFont, NSFontWeightSemibold, NSImage, NSMenu, NSMenuDelegate,
    NSMenuItem, NSStatusBar, NSStatusItem, NSSwitch, NSTextField, NSVariableStatusItemLength,
    NSView,
};
use objc2_foundation::{MainThreadMarker, NSData, NSObject, NSPoint, NSRect, NSSize, NSString};

use crate::state::State;

/// Template images (alpha only) of the gaze mark, eyes open while the server runs
/// (desktop/scripts/tray-icons.swift).
const ICON_AWAKE: &[u8] = include_bytes!("../icons/tray-awake.png");
const ICON_ASLEEP: &[u8] = include_bytes!("../icons/tray-asleep.png");

/// Left and right inset of the header, in line with the titles of the items below it.
const INSET: f64 = 14.0;
const HEADER_WIDTH: f64 = 280.0;
const HEADER_HEIGHT: f64 = 44.0;

pub enum Action {
    /// The header's switch: start the server (`true`) or stop it.
    Server(bool),
    CopyAddress,
    /// Load an installed model, or unload it.
    ToggleModel(String),
    /// Download a library model's default version.
    Pull(String),
    Update,
    Open,
    OpenAtLogin,
    Quit,
}

pub type OnAction = Box<dyn Fn(Action)>;

/// What the menu shows, from the app's state. Equal views draw equal menus.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct View {
    checked: bool,
    busy: bool,
    running: bool,
    version: String,
    host: String,
    /// Installed versions, and whether each is loaded.
    installed: Vec<(String, bool)>,
    /// Library families with no version on this computer.
    available: Vec<String>,
    /// Downloads in progress: (version, percent).
    pulling: Vec<(String, u32)>,
    update: Option<String>,
}

impl View {
    pub fn of(s: &State) -> View {
        let installed: Vec<(String, bool)> = s
            .snap
            .models
            .iter()
            .map(|m| (m.name.clone(), m.loaded))
            .collect();
        let family = |name: &str| name.split(':').next().unwrap_or(name).to_owned();
        View {
            checked: s.checked,
            busy: s.busy,
            running: s.snap.running,
            version: s.snap.version.clone().unwrap_or_default(),
            host: s.snap.url.trim_start_matches("http://").to_owned(),
            available: s
                .library
                .iter()
                .map(|m| m.name.clone())
                .filter(|m| !installed.iter().any(|(i, _)| family(i) == *m))
                .collect(),
            installed,
            pulling: s
                .pulling
                .iter()
                .map(|(m, p)| {
                    let pct = p.fraction().map_or(0, |f| (f * 100.0) as u32);
                    (m.clone(), pct)
                })
                .collect(),
            update: s.update.clone(),
        }
    }

    /// The header's second line, and whether its switch is on and can be used.
    fn header(&self) -> (String, bool, bool) {
        if !self.checked {
            ("Starting…".into(), true, false)
        } else if self.busy {
            if self.running {
                ("Stopping…".into(), false, false)
            } else {
                ("Starting…".into(), true, false)
            }
        } else if !self.running {
            ("Stopped".into(), false, true)
        } else if let Some((model, pct)) = self.pulling.first() {
            (format!("Downloading {model}, {pct}%"), true, true)
        } else {
            (format!("Running, version {}", self.version), true, true)
        }
    }

    /// A library family's version that the menu downloads, and how far it is.
    fn pulling(&self, family: &str) -> Option<u32> {
        let tag = format!("{family}:latest");
        self.pulling
            .iter()
            .find(|(m, _)| *m == tag)
            .map(|(_, pct)| *pct)
    }
}

pub struct Ivars {
    on_action: OnAction,
}

define_class!(
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    #[ivars = Ivars]
    struct Target;

    unsafe impl NSObjectProtocol for Target {}

    unsafe impl NSMenuDelegate for Target {
        #[unsafe(method(menuNeedsUpdate:))]
        fn menu_needs_update(&self, menu: &NSMenu) {
            if is_main(menu) {
                fill(menu, self);
            }
        }

        #[unsafe(method(menuWillOpen:))]
        fn menu_will_open(&self, menu: &NSMenu) {
            if is_main(menu) {
                with_bar(|bar| bar.open = true);
            }
        }

        #[unsafe(method(menuDidClose:))]
        fn menu_did_close(&self, menu: &NSMenu) {
            if is_main(menu) {
                with_bar(|bar| bar.open = false);
            }
        }
    }

    impl Target {
        #[unsafe(method(switchServer:))]
        fn switch_server(&self, sender: &NSSwitch) {
            let on = sender.state() == NSControlStateValueOn;
            sender.setEnabled(false);
            with_bar(|bar| {
                if let Some(h) = &bar.header {
                    h.subtitle.setStringValue(&NSString::from_str(if on {
                        "Starting…"
                    } else {
                        "Stopping…"
                    }));
                }
            });
            (self.ivars().on_action)(Action::Server(on));
        }

        #[unsafe(method(copyAddress:))]
        fn copy_address(&self, _sender: &AnyObject) {
            (self.ivars().on_action)(Action::CopyAddress);
        }

        #[unsafe(method(toggleModel:))]
        fn toggle_model(&self, sender: &NSMenuItem) {
            if let Some(name) = tagged(sender, |bar| &bar.models_shown) {
                (self.ivars().on_action)(Action::ToggleModel(name));
            }
        }

        #[unsafe(method(pull:))]
        fn pull(&self, sender: &NSMenuItem) {
            if let Some(name) = tagged(sender, |bar| &bar.library_shown) {
                (self.ivars().on_action)(Action::Pull(format!("{name}:latest")));
            }
        }

        #[unsafe(method(update:))]
        fn update(&self, _sender: &AnyObject) {
            (self.ivars().on_action)(Action::Update);
        }

        #[unsafe(method(open:))]
        fn open(&self, _sender: &AnyObject) {
            (self.ivars().on_action)(Action::Open);
        }

        #[unsafe(method(openAtLogin:))]
        fn open_at_login(&self, _sender: &AnyObject) {
            (self.ivars().on_action)(Action::OpenAtLogin);
        }

        #[unsafe(method(quit:))]
        fn quit(&self, _sender: &AnyObject) {
            (self.ivars().on_action)(Action::Quit);
        }
    }
);

impl Target {
    fn new(on_action: OnAction, mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(Ivars { on_action });
        unsafe { msg_send![super(this), init] }
    }
}

/// The header on screen, so a sync while the menu is open can update it in place.
struct Header {
    subtitle: Retained<NSTextField>,
    switch: Retained<NSSwitch>,
}

struct Bar {
    item: Retained<NSStatusItem>,
    awake: Retained<NSImage>,
    asleep: Retained<NSImage>,
    /// Kept alive: the menu's delegate and every item's target are weak references to it.
    target: Retained<Target>,
    menu: Retained<NSMenu>,
    open: bool,
    /// What the open menu was built from, and handles to what can change in place.
    built: View,
    header: Option<Header>,
    models: HashMap<String, Retained<NSMenuItem>>,
    library: HashMap<String, Retained<NSMenuItem>>,
    login: Option<Retained<NSMenuItem>>,
    /// The names behind the items' tags.
    models_shown: Vec<String>,
    library_shown: Vec<String>,
}

thread_local! {
    static BAR: RefCell<Option<Bar>> = const { RefCell::new(None) };
    /// The latest view, kept even before `install`.
    static LATEST: RefCell<View> = RefCell::new(View::default());
}

fn with_bar<R>(f: impl FnOnce(&mut Bar) -> R) -> Option<R> {
    BAR.with_borrow_mut(|bar| bar.as_mut().map(f))
}

fn is_main(menu: &NSMenu) -> bool {
    BAR.with_borrow(|bar| bar.as_ref().is_some_and(|b| std::ptr::eq(&*b.menu, menu)))
}

fn tagged(sender: &NSMenuItem, list: impl Fn(&Bar) -> &Vec<String>) -> Option<String> {
    let i = usize::try_from(sender.tag()).ok()?;
    BAR.with_borrow(|bar| bar.as_ref().and_then(|b| list(b).get(i).cloned()))
}

pub fn install(on_action: OnAction) -> Result<(), String> {
    let mtm = MainThreadMarker::new().ok_or("the menu bar needs the main thread")?;
    let target = Target::new(on_action, mtm);
    let menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), &NSString::from_str("Ollaya"));
    menu.setAutoenablesItems(false);
    menu.setDelegate(Some(ProtocolObject::from_ref(&*target)));
    let item = NSStatusBar::systemStatusBar().statusItemWithLength(NSVariableStatusItemLength);
    item.setMenu(Some(&menu));
    let bar = Bar {
        item,
        awake: icon(ICON_AWAKE, "Ollaya is running")?,
        asleep: icon(ICON_ASLEEP, "Ollaya is stopped")?,
        target,
        menu,
        open: false,
        built: View::default(),
        header: None,
        models: HashMap::new(),
        library: HashMap::new(),
        login: None,
        models_shown: Vec::new(),
        library_shown: Vec::new(),
    };
    BAR.with_borrow_mut(|b| *b = Some(bar));
    draw_icon();
    Ok(())
}

/// The app's state changed: redraw the icon, and an open menu's header, check marks and
/// progress. The menu reads the rest the next time it opens.
/// Draw the menu even if the app's state did not change: a control moved on its own.
pub fn redraw(s: &State) {
    LATEST.with_borrow_mut(|l| *l = View::default());
    sync(s);
}

pub fn sync(s: &State) {
    let view = View::of(s);
    let changed = LATEST.with_borrow(|l| *l != view);
    if !changed {
        return;
    }
    LATEST.with_borrow_mut(|l| *l = view.clone());
    draw_icon();
    with_bar(|bar| {
        if bar.open {
            update_in_place(bar, &view);
        }
    });
}

fn draw_icon() {
    let view = LATEST.with_borrow(|l| l.clone());
    with_bar(|bar| {
        let Some(mtm) = MainThreadMarker::new() else {
            return;
        };
        if let Some(button) = bar.item.button(mtm) {
            button.setImage(Some(if view.running {
                &bar.awake
            } else {
                &bar.asleep
            }));
            let tip = if view.running {
                format!("Ollaya {} is running at {}", view.version, view.host)
            } else {
                "Ollaya is stopped".to_owned()
            };
            button.setToolTip(Some(&NSString::from_str(&tip)));
        }
    });
}

fn icon(png: &[u8], description: &str) -> Result<Retained<NSImage>, String> {
    let data = NSData::with_bytes(png);
    let image = NSImage::initWithData(NSImage::alloc(), &data).ok_or("a menu bar icon")?;
    image.setSize(NSSize::new(18.0, 18.0));
    image.setTemplate(true);
    image.setAccessibilityDescription(Some(&NSString::from_str(description)));
    Ok(image)
}

/// The menu, from the latest view.
fn fill(menu: &NSMenu, target: &Target) {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let v = LATEST.with_borrow(|l| l.clone());
    menu.removeAllItems();

    let first = disabled("Ollaya", mtm);
    first.setEnabled(true);
    let (view, header) = header_view(mtm, &v, target);
    first.setView(Some(&view));
    menu.addItem(&first);
    menu.addItem(&NSMenuItem::separatorItem(mtm));

    if v.running {
        let copy = action(
            target,
            &format!("Server: {}", v.host),
            sel!(copyAddress:),
            "c",
            mtm,
        );
        copy.setSubtitle(Some(&NSString::from_str("Click to copy")));
        menu.addItem(&copy);
    } else {
        menu.addItem(&disabled("Server: not running", mtm));
    }

    let models = NSMenu::initWithTitle(NSMenu::alloc(mtm), &NSString::from_str("Models"));
    models.setAutoenablesItems(false);
    let mut model_items = HashMap::new();
    let mut models_shown = Vec::new();
    if v.installed.is_empty() {
        models.addItem(&disabled("No models yet", mtm));
    } else {
        models.addItem(&NSMenuItem::sectionHeaderWithTitle(
            &NSString::from_str("On This Mac"),
            mtm,
        ));
    }
    for (i, (name, loaded)) in v.installed.iter().enumerate() {
        let item = action(target, name, sel!(toggleModel:), "", mtm);
        item.setTag(i as isize);
        set_loaded(&item, *loaded);
        models.addItem(&item);
        model_items.insert(name.clone(), item);
        models_shown.push(name.clone());
    }
    let mut library_items = HashMap::new();
    let mut library_shown = Vec::new();
    if !v.available.is_empty() {
        models.addItem(&NSMenuItem::separatorItem(mtm));
        models.addItem(&NSMenuItem::sectionHeaderWithTitle(
            &NSString::from_str("Library"),
            mtm,
        ));
    }
    for (i, name) in v.available.iter().enumerate() {
        let item = action(target, name, sel!(pull:), "", mtm);
        item.setTag(i as isize);
        set_pulling(&item, v.pulling(name));
        models.addItem(&item);
        library_items.insert(name.clone(), item);
        library_shown.push(name.clone());
    }
    let models_item = disabled("Models", mtm);
    models_item.setEnabled(v.running);
    models_item.setSubmenu(Some(&models));
    menu.addItem(&models_item);
    menu.addItem(&NSMenuItem::separatorItem(mtm));

    if let Some(version) = &v.update {
        menu.addItem(&action(
            target,
            &format!("Download Ollaya {version}…"),
            sel!(update:),
            "",
            mtm,
        ));
        menu.addItem(&NSMenuItem::separatorItem(mtm));
    }
    menu.addItem(&action(target, "Open Ollaya…", sel!(open:), "o", mtm));
    let login = action(target, "Open at Login", sel!(openAtLogin:), "", mtm);
    login.setState(if crate::login::is_on() {
        NSControlStateValueOn
    } else {
        NSControlStateValueOff
    });
    menu.addItem(&login);
    menu.addItem(&NSMenuItem::separatorItem(mtm));
    menu.addItem(&action(target, "Quit Ollaya", sel!(quit:), "q", mtm));

    with_bar(|bar| {
        bar.built = v;
        bar.header = Some(header);
        bar.models = model_items;
        bar.library = library_items;
        bar.login = Some(login);
        bar.models_shown = models_shown;
        bar.library_shown = library_shown;
    });
}

/// Updates an open menu for `v`: the header always, the items as far as they still match.
fn update_in_place(bar: &mut Bar, v: &View) {
    if let Some(h) = &bar.header {
        let (subtitle, on, enabled) = v.header();
        h.subtitle.setStringValue(&NSString::from_str(&subtitle));
        h.switch.setState(if on {
            NSControlStateValueOn
        } else {
            NSControlStateValueOff
        });
        h.switch.setEnabled(enabled);
    }
    for (name, loaded) in &v.installed {
        if let Some(item) = bar.models.get(name) {
            set_loaded(item, *loaded);
        }
    }
    for (name, item) in &bar.library {
        set_pulling(item, v.pulling(name));
    }
}

fn set_loaded(item: &NSMenuItem, loaded: bool) {
    item.setState(if loaded {
        NSControlStateValueOn
    } else {
        NSControlStateValueOff
    });
    item.setSubtitle(loaded.then(|| NSString::from_str("Loaded")).as_deref());
}

fn set_pulling(item: &NSMenuItem, pct: Option<u32>) {
    item.setEnabled(pct.is_none());
    let subtitle = match pct {
        Some(pct) => format!("Downloading, {pct}%"),
        None => "Click to download".to_owned(),
    };
    item.setSubtitle(Some(&NSString::from_str(&subtitle)));
}

/// "Ollaya" over the server's state, with a switch on the right, like Tailscale's header.
fn header_view(mtm: MainThreadMarker, v: &View, target: &Target) -> (Retained<NSView>, Header) {
    let (subtitle, on, enabled) = v.header();
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
    // SAFETY: `switchServer:` takes the sender, and the bar keeps the target alive.
    unsafe {
        switch.setTarget(Some(target));
        switch.setAction(Some(sel!(switchServer:)));
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
        },
    )
}

/// The menu's items as it would open now, built by the same code as a click on the icon, the
/// Models submenu inline and indented; a check mark shows as "✓ ", a disabled item in brackets,
/// a subtitle after " · ", a separator as "—" (for the self-test).
pub fn titles() -> Vec<String> {
    let menu = BAR.with_borrow(|bar| bar.as_ref().map(|b| (b.menu.clone(), b.target.clone())));
    let Some((menu, target)) = menu else {
        return vec!["(no menu bar item)".into()];
    };
    fill(&menu, &target);
    let mut out = Vec::new();
    list(&menu, "", &mut out);
    out
}

fn list(menu: &NSMenu, indent: &str, out: &mut Vec<String>) {
    for item in menu.itemArray().iter() {
        if item.isSeparatorItem() {
            out.push(format!("{indent}—"));
            continue;
        }
        let mut t = item.title().to_string();
        if item.view().is_some() {
            let (subtitle, on, enabled) = LATEST.with_borrow(View::header);
            t = format!(
                "{t} · {subtitle} [switch {}{}]",
                if on { "on" } else { "off" },
                if enabled { "" } else { ", disabled" }
            );
        }
        if let Some(sub) = item.subtitle() {
            t = format!("{t} · {sub}");
        }
        if item.state() == NSControlStateValueOn {
            t = format!("✓ {t}");
        }
        if !item.isEnabled() {
            t = format!("[{t}]");
        }
        out.push(format!("{indent}{t}"));
        if let Some(sub) = item.submenu() {
            list(&sub, &format!("{indent}  "), out);
        }
    }
}

fn disabled(title: &str, mtm: MainThreadMarker) -> Retained<NSMenuItem> {
    let item = unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            &NSString::from_str(title),
            None,
            &NSString::new(),
        )
    };
    item.setEnabled(false);
    item
}

fn action(
    target: &Target,
    title: &str,
    sel: Sel,
    key: &str,
    mtm: MainThreadMarker,
) -> Retained<NSMenuItem> {
    let item = unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            &NSString::from_str(title),
            Some(sel),
            &NSString::from_str(key),
        )
    };
    if !key.is_empty() {
        item.setKeyEquivalentModifierMask(NSEventModifierFlags::Command);
    }
    unsafe { item.setTarget(Some(target)) };
    item
}
