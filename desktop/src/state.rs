//! The app's state, which Rune's UI thread owns: what the window and the menu bar show, the
//! backend's answers (`on_*`), and what the buttons do.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use ollaya_api::DecideResponse;
use rune::{AppHandle, TextEdit, Tx};

use crate::backend::{self, Cmd, Decide, Preset, Progress, Snapshot};
use crate::library::LibModel;
use crate::ui;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Page {
    Discover,
    Models,
    Playground,
    Server,
}

/// The playground's pickers, one open at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Picker {
    Model,
    Questions,
}

/// The preset a run without one asks, and the questions the Custom box starts with.
pub const DEFAULT_PRESET: &str = "triage";
/// The Custom box's entry in the questions picker.
pub const CUSTOM: &str = "custom";

/// What `/api/decide` with the triage preset is shown with when someone asks for an example.
pub const EXAMPLE: &str = "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor.";

pub struct State {
    pub handle: AppHandle<State>,
    pub cmd: backend::Sender,
    /// Rune's controls (the pickers), in the app's colours.
    pub kit: rune_kit::Kit,

    /// The server as last seen.
    pub snap: Snapshot,
    /// Whether the backend has looked at the server yet.
    pub checked: bool,
    pub page: Page,
    /// The person picked a page, so launch no longer picks one.
    pub navigated: bool,
    /// The server is starting or stopping: its switch waits.
    pub busy: bool,
    pub error: Option<String>,
    /// A newer Ollaya release than this app.
    pub update: Option<String>,

    pub library: Vec<LibModel>,
    pub library_error: Option<String>,
    pub library_loading: bool,
    pub query: TextEdit,
    pub only_installed: bool,
    /// The family open in Discover's detail pane.
    pub selected: Option<String>,
    /// Downloads in progress.
    pub pulling: BTreeMap<String, Progress>,
    pub removing: BTreeSet<String>,
    /// The last download or removal that failed.
    pub pull_error: Option<String>,

    pub presets: Vec<Preset>,
    /// Per installed version: the ids of its built-in questions, or `None` when it takes
    /// questions. Missing while unknown.
    pub builtin: HashMap<String, Option<Vec<String>>>,
    /// Versions whose questions the backend is looking up.
    pub asking: BTreeSet<String>,
    pub run_model: String,
    /// A preset's name, or [`CUSTOM`].
    pub preset: String,
    pub input: TextEdit,
    pub custom: TextEdit,
    /// The run on its way, by id.
    pub running: Option<u64>,
    pub runs: u64,
    pub result: Option<DecideResponse>,
    pub run_error: Option<String>,
    pub picker: Option<Picker>,

    /// The copy button that was just pressed and when (the window's clock), for its check mark.
    pub copied: Option<(String, f64)>,
    /// The self-test's next step (`OLLAYA_SELFTEST`).
    pub selftest: Option<u8>,
    /// Launch is done: the server checked, and started if it was stopped.
    pub booted: bool,
}

impl State {
    pub fn new(handle: AppHandle<State>, cmd: backend::Sender) -> State {
        let presets = backend::builtin_presets();
        let custom = presets
            .iter()
            .find(|p| p.name == DEFAULT_PRESET)
            .and_then(|p| serde_json::to_string_pretty(&p.questions).ok())
            .unwrap_or_default();
        State {
            handle,
            cmd,
            kit: ui::kit(),
            snap: Snapshot::default(),
            checked: false,
            page: Page::Discover,
            navigated: false,
            busy: false,
            error: None,
            update: None,
            library: Vec::new(),
            library_error: None,
            library_loading: true,
            query: TextEdit::new(""),
            only_installed: false,
            selected: None,
            pulling: BTreeMap::new(),
            removing: BTreeSet::new(),
            pull_error: None,
            presets,
            builtin: HashMap::new(),
            asking: BTreeSet::new(),
            run_model: String::new(),
            preset: DEFAULT_PRESET.into(),
            input: TextEdit::new(""),
            custom: TextEdit::new(custom),
            running: None,
            runs: 0,
            result: None,
            run_error: None,
            picker: None,
            copied: None,
            selftest: std::env::var_os("OLLAYA_SELFTEST").map(|_| 0),
            booted: false,
        }
    }

    // --- what is installed -------------------------------------------------------------------

    pub fn is_installed(&self, tag: &str) -> bool {
        self.snap.models.iter().any(|m| m.name == tag)
    }

    pub fn is_loaded(&self, tag: &str) -> bool {
        self.snap.models.iter().any(|m| m.name == tag && m.loaded)
    }

    pub fn size_of(&self, tag: &str) -> u64 {
        self.snap
            .models
            .iter()
            .find(|m| m.name == tag)
            .map_or(0, |m| m.size)
    }

    /// A family's versions on this computer, in the order Discover lists them.
    pub fn installed_tags(&self, m: &LibModel) -> Vec<String> {
        m.main_tags()
            .into_iter()
            .filter(|t| self.is_installed(t))
            .collect()
    }

    /// Every installed version: the library's order first, then what the library does not list.
    pub fn installed(&self) -> Vec<String> {
        let mut known: Vec<String> = self
            .library
            .iter()
            .flat_map(|m| self.installed_tags(m))
            .collect();
        let mut rest: Vec<String> = self
            .snap
            .models
            .iter()
            .map(|m| m.name.clone())
            .filter(|t| !known.contains(t))
            .collect();
        rest.sort();
        known.extend(rest);
        known
    }

    pub fn library_model(&self, name: &str) -> Option<&LibModel> {
        self.library.iter().find(|m| m.name == name)
    }

    /// Discover's list, as the search and the tabs leave it.
    pub fn shown_library(&self) -> Vec<&LibModel> {
        let q = self.query.text.trim().to_lowercase();
        self.library
            .iter()
            .filter(|m| !self.only_installed || !self.installed_tags(m).is_empty())
            .filter(|m| m.matches(&q))
            .collect()
    }

    /// The family Discover's detail pane shows: the one picked, else the first listed.
    pub fn selected_model(&self) -> Option<&LibModel> {
        self.selected
            .as_deref()
            .and_then(|n| self.library_model(n))
            .or_else(|| self.shown_library().first().copied())
    }

    // --- actions -----------------------------------------------------------------------------

    fn post(&self, cmd: Cmd) {
        let _ = self.cmd.send(cmd);
    }

    pub fn go(&mut self, page: Page) {
        self.page = page;
        self.navigated = true;
        self.picker = None;
        if page == Page::Playground {
            self.pick_model();
        }
    }

    pub fn toggle_server(&mut self) {
        if self.busy || !self.checked {
            return;
        }
        self.busy = true;
        self.error = None;
        self.post(Cmd::Server(!self.snap.running));
        self.sync_menu();
    }

    pub fn toggle_model(&mut self, name: String) {
        self.post(Cmd::Toggle(name));
    }

    pub fn pull(&mut self, tag: String) {
        if self.pulling.contains_key(&tag) || !self.snap.running {
            return;
        }
        self.pull_error = None;
        self.pulling.insert(tag.clone(), Progress::default());
        self.post(Cmd::Pull(tag));
        self.sync_menu();
    }

    pub fn remove(&mut self, tag: String) {
        if !self.removing.insert(tag.clone()) {
            return;
        }
        self.pull_error = None;
        self.post(Cmd::Remove(tag));
    }

    pub fn reload_library(&mut self) {
        self.library_error = None;
        self.library_loading = true;
        self.post(Cmd::Library);
    }

    /// Open the playground with `tag` picked.
    pub fn try_out(&mut self, tag: String) {
        self.run_model = tag;
        self.result = None;
        self.run_error = None;
        self.go(Page::Playground);
    }

    pub fn choose_model(&mut self, tag: &str) {
        if self.run_model != tag {
            self.run_model = tag.to_owned();
            self.result = None;
            self.run_error = None;
        }
        self.picker = None;
        self.pick_model();
    }

    pub fn choose_preset(&mut self, name: &str) {
        self.preset = name.to_owned();
        self.picker = None;
    }

    pub fn use_example(&mut self) {
        self.input.replace_text(EXAMPLE);
    }

    /// Keep the playground's model one that is installed, and learn what it asks.
    fn pick_model(&mut self) {
        let tags = self.installed();
        if !tags.contains(&self.run_model) {
            self.run_model = tags.first().cloned().unwrap_or_default();
        }
        let m = self.run_model.clone();
        if !m.is_empty()
            && self.snap.running
            && !self.builtin.contains_key(&m)
            && self.asking.insert(m.clone())
        {
            self.post(Cmd::Builtin(m));
        }
    }

    /// What the picked model asks: its own questions, or `None` when it takes questions; outer
    /// `None` while unknown.
    pub fn own_questions(&self) -> Option<Option<&[String]>> {
        self.builtin.get(&self.run_model).map(|q| q.as_deref())
    }

    pub fn can_run(&self) -> bool {
        self.running.is_none()
            && self.snap.running
            && !self.run_model.is_empty()
            && self.own_questions().is_some()
            && !self.input.text.trim().is_empty()
    }

    pub fn run(&mut self) {
        if !self.can_run() {
            return;
        }
        self.runs += 1;
        self.running = Some(self.runs);
        self.run_error = None;
        self.picker = None;
        let own = matches!(self.own_questions(), Some(Some(_)));
        // A model with built-in questions gets the state alone.
        let (preset, questions) = if own {
            (None, None)
        } else if self.preset == CUSTOM {
            (None, Some(self.custom.text.clone()))
        } else {
            (Some(self.preset.clone()), None)
        };
        self.post(Cmd::Decide(Decide {
            id: self.runs,
            model: self.run_model.clone(),
            state: self.input.text.clone(),
            preset,
            questions,
        }));
    }

    pub fn show_window(&mut self) {
        #[cfg(target_os = "macos")]
        self.handle.set_dock_icon(true);
        self.handle.open();
        self.handle.focus();
    }

    // --- the backend's answers ---------------------------------------------------------------

    pub fn on_snapshot(&mut self, snap: Snapshot, tx: &mut Tx) {
        self.set_snapshot(snap);
        self.after(false, tx);
    }

    /// Launch found the server stopped and starts it.
    pub fn on_starting(&mut self) {
        self.busy = true;
        self.error = None;
        self.sync_menu();
    }

    pub fn on_server(&mut self, result: Result<(), String>, snap: Snapshot, tx: &mut Tx) {
        self.busy = false;
        self.error = result.err();
        self.set_snapshot(snap);
        self.after(true, tx);
    }

    pub fn on_toggled(&mut self, result: Result<(), String>, snap: Snapshot, tx: &mut Tx) {
        if let Err(e) = result {
            self.error = Some(e);
        }
        self.set_snapshot(snap);
        self.after(false, tx);
    }

    /// Launch is done: with models on this computer, start where they run; a first run starts
    /// where they are found.
    pub fn on_booted(&mut self, tx: &mut Tx) {
        self.booted = true;
        if !self.navigated && !self.snap.models.is_empty() {
            self.page = Page::Playground;
            self.pick_model();
        }
        self.after(true, tx);
    }

    pub fn on_library(&mut self, result: Result<Vec<LibModel>, String>) {
        self.library_loading = false;
        match result {
            Ok(models) => {
                self.library = models;
                self.library_error = None;
            }
            Err(e) => self.library_error = Some(e),
        }
        self.sync_menu();
    }

    pub fn on_presets(&mut self, list: Vec<Preset>) {
        self.presets = list;
        if self.preset != CUSTOM && !self.presets.iter().any(|p| p.name == self.preset) {
            self.preset = DEFAULT_PRESET.into();
        }
    }

    pub fn on_pull(&mut self, model: String, p: Progress) {
        if let Some(slot) = self.pulling.get_mut(&model) {
            *slot = p;
            self.sync_menu();
        }
    }

    pub fn on_pulled(
        &mut self,
        model: String,
        result: Result<(), String>,
        snap: Snapshot,
        tx: &mut Tx,
    ) {
        self.pulling.remove(&model);
        if let Err(e) = result {
            self.pull_error = Some(format!("{model}: {e}"));
        }
        self.forget(&model);
        self.set_snapshot(snap);
        self.after(false, tx);
    }

    pub fn on_removed(
        &mut self,
        model: String,
        result: Result<(), String>,
        snap: Snapshot,
        tx: &mut Tx,
    ) {
        self.removing.remove(&model);
        if let Err(e) = result {
            self.pull_error = Some(e);
        }
        self.forget(&model);
        if self.run_model == model {
            self.run_model.clear();
            self.result = None;
        }
        self.set_snapshot(snap);
        self.after(false, tx);
    }

    pub fn on_builtin(&mut self, model: String, ids: Result<Option<Vec<String>>, String>) {
        self.asking.remove(&model);
        // Unknown for now: ask again the next time the model is picked.
        if let Ok(ids) = ids {
            self.builtin.insert(model, ids);
        }
    }

    pub fn on_decided(&mut self, id: u64, result: Result<DecideResponse, String>) {
        if self.running != Some(id) {
            return;
        }
        self.running = None;
        match result {
            Ok(r) => {
                self.result = Some(r);
                self.run_error = None;
            }
            Err(e) => {
                self.result = None;
                self.run_error = Some(e);
            }
        }
    }

    pub fn on_update(&mut self, version: Option<String>) {
        self.update = version;
        self.sync_menu();
    }

    /// Forget what a version asks, once it is downloaded again or removed.
    fn forget(&mut self, tag: &str) {
        self.builtin.remove(tag);
    }

    fn set_snapshot(&mut self, snap: Snapshot) {
        let started = snap.running && !self.snap.running;
        self.snap = snap;
        self.checked = true;
        if started {
            // What a version asks may have been unknown while the server was away.
            self.asking.clear();
        }
        if self.page == Page::Playground || !self.run_model.is_empty() {
            self.pick_model();
        }
    }

    /// After every snapshot: the menu bar, and the self-test's next step.
    fn after(&mut self, answered: bool, tx: &mut Tx) {
        self.sync_menu();
        if self.selftest.is_some() {
            crate::selftest::step(self, answered, tx);
        }
    }

    pub fn sync_menu(&self) {
        #[cfg(target_os = "macos")]
        crate::menubar::sync(self);
    }
}
