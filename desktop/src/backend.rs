//! Everything that talks to the server or the website, on a thread of its own with a Tokio
//! runtime: Rune's UI thread never waits on the network. The window and the menu bar send
//! [`Cmd`]s; each runs as a task of its own and answers through `AppHandle::update`, calling one
//! of `State`'s `on_*` methods. A [`Snapshot`] of the server follows every two seconds, and at
//! once after a command that changes it, only when something changed, so an idle app draws
//! nothing.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use ollaya_api::{DecideRequest, DecideResponse, KeepAlive, PullRequest, Questions, presets};
use serde_json::Value;
use tokio::sync::Notify;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

use crate::engine::{self, client};
use crate::library;
use crate::state::State;

/// What the window and the menu bar ask the backend to do.
#[derive(Debug)]
pub enum Cmd {
    /// Start the server (`true`) or stop it.
    Server(bool),
    /// Load a model, or unload it when it is loaded (the menu bar's check marks).
    Toggle(String),
    Pull(String),
    Remove(String),
    /// Fetch the library again, after it failed.
    Library,
    /// Which questions a version asks by itself.
    Builtin(String),
    Decide(Decide),
}

pub type Sender = UnboundedSender<Cmd>;

/// A playground run: `questions` is the Custom box's JSON text, `preset` a preset's name; with
/// neither, the model answers its built-in questions.
#[derive(Debug)]
pub struct Decide {
    pub id: u64,
    pub model: String,
    pub state: String,
    pub preset: Option<String>,
    pub questions: Option<String>,
}

/// What the app shows of the server. Equal snapshots draw equal screens.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
    pub running: bool,
    pub version: Option<String>,
    pub url: String,
    /// Where the server this app started runs models, on Windows and Linux: "GPU" or "CPU only".
    pub device: Option<&'static str>,
    pub models: Vec<Model>,
    /// The server keeps an activity log (0.17.0 and newer).
    pub activity: bool,
    /// Decision requests since the server started (`GET /api/activity`).
    pub requests: u64,
    /// The latest decision requests, newest first (the server keeps 100).
    pub recent: Vec<Request>,
    pub speed: Speed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub name: String,
    pub size: u64,
    pub loaded: bool,
}

/// One decision request from the server's activity log.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub at: DateTime<Utc>,
    pub endpoint: String,
    pub model: String,
    pub questions: u64,
    pub took_ms: f64,
    /// The error code in words (`model not found`) when the request failed.
    pub error: Option<String>,
}

/// How fast the server answers, from its recent requests (the Tauri app's `speed()`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Speed {
    /// Requests per second over the last minute.
    pub per_second: f64,
    /// The median response time of the requests that succeeded.
    pub median_ms: Option<f64>,
    /// How many requests the median is of.
    pub samples: usize,
}

/// How far a download is, in bytes over all its layers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Progress {
    pub completed: u64,
    pub total: u64,
}

impl Progress {
    /// Done, from 0 to 1; unknown until the first layer reports its size.
    pub fn fraction(&self) -> Option<f64> {
        (self.total > 0).then(|| (self.completed as f64 / self.total as f64).clamp(0.0, 1.0))
    }
}

/// A preset the playground offers.
#[derive(Clone, Debug, PartialEq)]
pub struct Preset {
    pub name: String,
    pub questions: Value,
    /// Built into Ollaya, as against one saved with `ollaya preset create` (0.8.0 and newer).
    pub builtin: bool,
}

pub fn speed(recent: &[Request], now: DateTime<Utc>) -> Speed {
    let mut times: Vec<f64> = recent
        .iter()
        .filter(|r| r.error.is_none())
        .map(|r| r.took_ms)
        .collect();
    times.sort_by(f64::total_cmp);
    // The rate between the first and the last request of the past minute: the log holds 100
    // requests, which a busy client sends in a few seconds.
    let at: Vec<DateTime<Utc>> = recent
        .iter()
        .map(|r| r.at)
        .filter(|t| (now - *t).num_milliseconds() < 60_000)
        .collect();
    let per_second = match (at.first(), at.last()) {
        (Some(first), Some(last)) if at.len() > 1 => {
            let seconds = ((*first - *last).num_milliseconds() as f64 / 1000.0).max(1.0);
            (at.len() - 1) as f64 / seconds
        }
        _ => at.len() as f64 / 60.0,
    };
    Speed {
        per_second,
        median_ms: times.get(times.len() / 2).copied(),
        samples: times.len(),
    }
}

/// What the tasks share: the way back to the UI thread, and the snapshot it last got.
#[derive(Clone)]
struct Ctx {
    handle: rune::AppHandle<State>,
    shown: Arc<Mutex<Option<Snapshot>>>,
    refresh: Arc<Notify>,
}

impl Ctx {
    fn update(&self, f: impl FnOnce(&mut State, &mut rune::Tx) + Send + 'static) {
        self.handle.update(f);
    }

    /// A snapshot taken now, which the UI is about to get with an answer.
    async fn snapshot(&self) -> Snapshot {
        let snap = snapshot().await;
        *self.shown.lock().unwrap_or_else(|e| e.into_inner()) = Some(snap.clone());
        snap
    }

    /// Send the UI a snapshot if it differs from the last one it got.
    async fn publish(&self) {
        let snap = snapshot().await;
        let mut shown = self.shown.lock().unwrap_or_else(|e| e.into_inner());
        if shown.as_ref() != Some(&snap) {
            *shown = Some(snap.clone());
            drop(shown);
            self.update(move |s, tx| s.on_snapshot(snap, tx));
        }
    }
}

pub fn spawn(handle: rune::AppHandle<State>) -> Sender {
    let (tx, rx) = unbounded_channel();
    std::thread::Builder::new()
        .name("ollaya-backend".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("a Tokio runtime");
            rt.block_on(run(rx, handle));
        })
        .expect("the backend thread");
    tx
}

async fn run(mut rx: UnboundedReceiver<Cmd>, handle: rune::AppHandle<State>) {
    let ctx = Ctx {
        handle,
        shown: Arc::default(),
        refresh: Arc::default(),
    };
    tokio::spawn(boot(ctx.clone()));
    let mut tick = tokio::time::interval(Duration::from_secs(2));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = tick.tick() => ctx.publish().await,
            _ = ctx.refresh.notified() => ctx.publish().await,
            cmd = rx.recv() => match cmd {
                None => return,
                Some(cmd) => {
                    tokio::spawn(command(cmd, ctx.clone()));
                }
            },
        }
    }
}

/// At launch, as the Tauri app: the library and the server at once; like Ollama's app, the server
/// runs while the app is open, so start it if it is stopped; then the presets saved on it, and
/// whether a newer release is out (again every six hours).
async fn boot(ctx: Ctx) {
    let lib = ctx.clone();
    tokio::spawn(async move {
        let models = library::fetch().await;
        lib.update(move |s, _| s.on_library(models));
    });
    let snap = ctx.snapshot().await;
    let running = snap.running;
    ctx.update(move |s, tx| s.on_snapshot(snap, tx));
    if !running && std::env::var_os("OLLAYA_NO_AUTOSTART").is_none() {
        ctx.update(|s, _| s.on_starting());
        let result = engine::start().await;
        let snap = ctx.snapshot().await;
        ctx.update(move |s, tx| s.on_server(result, snap, tx));
    }
    ctx.update(|s, tx| s.on_booted(tx));
    let list = preset_list().await;
    ctx.update(move |s, _| s.on_presets(list));
    loop {
        let found = library::latest_release()
            .await
            .filter(|v| engine::newer(v, engine::VERSION));
        ctx.update(move |s, _| s.on_update(found));
        tokio::time::sleep(Duration::from_secs(6 * 3600)).await;
    }
}

async fn command(cmd: Cmd, ctx: Ctx) {
    match cmd {
        Cmd::Server(on) => {
            let result = if on {
                engine::start().await
            } else {
                engine::stop().await
            };
            let snap = ctx.snapshot().await;
            ctx.update(move |s, tx| s.on_server(result, snap, tx));
            if on {
                // Custom presets live on the server, which may only have started just now.
                let list = preset_list().await;
                ctx.update(move |s, _| s.on_presets(list));
            }
        }
        Cmd::Toggle(model) => {
            let result = toggle(&model).await;
            let snap = ctx.snapshot().await;
            ctx.update(move |s, tx| s.on_toggled(result, snap, tx));
        }
        Cmd::Pull(model) => {
            let result = pull(&ctx, &model).await;
            let snap = ctx.snapshot().await;
            ctx.update(move |s, tx| s.on_pulled(model, result, snap, tx));
        }
        Cmd::Remove(model) => {
            let result = match client() {
                Ok(c) => c.delete(&model).await.map_err(|e| e.to_string()),
                Err(e) => Err(e),
            };
            let snap = ctx.snapshot().await;
            ctx.update(move |s, tx| s.on_removed(model, result, snap, tx));
        }
        Cmd::Library => {
            let models = library::fetch().await;
            ctx.update(move |s, _| s.on_library(models));
        }
        Cmd::Builtin(model) => {
            let ids = builtin_questions(&model).await;
            ctx.update(move |s, _| s.on_builtin(model, ids));
        }
        Cmd::Decide(d) => {
            let id = d.id;
            let result = decide(d).await;
            ctx.update(move |s, _| s.on_decided(id, result));
            // The request shows in the server's activity at once.
            ctx.refresh.notify_one();
        }
    }
}

async fn toggle(model: &str) -> Result<(), String> {
    let c = client()?;
    let loaded = c
        .ps()
        .await
        .map_err(|e| e.to_string())?
        .models
        .iter()
        .any(|m| m.name == model);
    if loaded {
        c.unload(model).await.map(drop)
    } else {
        // Loaded by hand, it stays until unloaded by hand, as the Tauri tray did.
        c.load(model, Some(KeepAlive::Forever)).await.map(drop)
    }
    .map_err(|e| e.to_string())
}

/// Pull `model`, telling the window and the menu bar how far it got, in bytes done over every
/// layer, at most ten times a second: they draw a progress bar, not a log.
async fn pull(ctx: &Ctx, model: &str) -> Result<(), String> {
    let mut stream = client()?
        .pull_stream(&PullRequest::new(model))
        .await
        .map_err(|e| e.to_string())?;
    let mut layers: std::collections::HashMap<String, (u64, u64)> = Default::default();
    let mut last = Instant::now() - Duration::from_secs(1);
    let sum = |layers: &std::collections::HashMap<String, (u64, u64)>| {
        let (completed, total) = layers
            .values()
            .fold((0, 0), |(d, t), (c, n)| (d + c, t + n));
        Progress { completed, total }
    };
    while let Some(line) = stream.next().await {
        let line = line.map_err(|e| e.to_string())?;
        if let (Some(digest), Some(total)) = (&line.digest, line.total) {
            layers.insert(digest.clone(), (line.completed.unwrap_or(0), total));
        }
        if last.elapsed() >= Duration::from_millis(100) {
            last = Instant::now();
            let (m, p) = (model.to_owned(), sum(&layers));
            ctx.update(move |s, _| s.on_pull(m, p));
        }
    }
    Ok(())
}

/// The built-in presets, as this app knows them, before the server answers.
pub fn builtin_presets() -> Vec<Preset> {
    presets::NAMES
        .iter()
        .map(|&name| Preset {
            name: name.to_owned(),
            questions: presets::get(name).unwrap_or(Value::Null),
            builtin: true,
        })
        .collect()
}

/// The built-in presets, then the custom ones the server has saved (`/api/presets`). A server
/// older than 0.8.0 has no custom presets, and the built-in ones are listed alone.
async fn preset_list() -> Vec<Preset> {
    let mut list = builtin_presets();
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

/// The ids of the questions `model` asks by itself (`/api/show`'s `questions`, from the
/// manifest's questions layer), or `None` when every request must bring its own. A model such as
/// qwen3guard answers only these, so the playground sends it the state alone.
async fn builtin_questions(model: &str) -> Result<Option<Vec<String>>, String> {
    let show = client()?.show(model).await.map_err(|e| e.to_string())?;
    Ok(show.questions.map(|q| q.keys().cloned().collect()))
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

/// Answer `d`: its state is sent as JSON when it is a JSON object or array and as text
/// otherwise (as `ollaya run` does).
async fn decide(d: Decide) -> Result<DecideResponse, String> {
    // A built-in preset's questions are sent as questions, so any server answers them; a custom
    // preset lives on the server, which resolves it by name (`preset` on /api/decide).
    let custom = d
        .preset
        .as_deref()
        .filter(|p| presets::get(p).is_none())
        .map(str::to_owned);
    let questions = match custom {
        Some(_) => None,
        None => request_questions(d.preset.as_deref(), d.questions.as_deref())?,
    };
    let state = parse_state(&d.state);
    // A built-in preset puts plain text into its state template (the server does it for custom ones).
    let state = match d.preset.as_deref().filter(|_| custom.is_none()) {
        Some(name) => presets::fill_state(presets::state_template(name).as_ref(), state),
        None => state,
    };
    let mut request = DecideRequest::new(d.model, state, questions);
    request.preset = custom;
    client()?.decide(&request).await.map_err(|e| e.to_string())
}

/// The playground's text as a decision's state: a JSON object or array as JSON, else the text.
fn parse_state(text: &str) -> Value {
    let trimmed = text.trim();
    match serde_json::from_str::<Value>(trimmed) {
        Ok(v @ (Value::Object(_) | Value::Array(_))) if trimmed.starts_with(['{', '[']) => v,
        _ => Value::String(text.to_owned()),
    }
}

async fn snapshot() -> Snapshot {
    let Ok(c) = client() else {
        return Snapshot::default();
    };
    let url = c.base_url().to_owned();
    let Ok(version) = c.version().await else {
        return Snapshot {
            url,
            ..Snapshot::default()
        };
    };
    let loaded: Vec<String> = c
        .ps()
        .await
        .map(|p| p.models.into_iter().map(|m| m.name).collect())
        .unwrap_or_default();
    let models = c
        .tags()
        .await
        .map(|t| t.models)
        .unwrap_or_default()
        .into_iter()
        .map(|m| Model {
            loaded: loaded.contains(&m.name),
            name: m.name,
            size: m.size,
        })
        .collect();
    // A server older than 0.17.0 (a command-line install) has no activity log.
    let (activity, requests, recent) = match c.activity().await {
        Ok(a) => (
            true,
            a.total,
            a.requests
                .into_iter()
                .map(|r| Request {
                    at: r.at,
                    endpoint: r.endpoint,
                    model: r.model,
                    questions: r.questions,
                    took_ms: r.total_duration as f64 / 1e6,
                    error: r.error.map(|e| e.as_str().to_lowercase().replace('_', " ")),
                })
                .collect(),
        ),
        Err(_) => (false, 0, Vec::new()),
    };
    Snapshot {
        running: true,
        version: Some(version.version),
        url,
        device: engine::device(),
        models,
        activity,
        requests,
        speed: speed(&recent, Utc::now()),
        recent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(secs_ago: i64, took_ms: f64, error: bool, now: DateTime<Utc>) -> Request {
        Request {
            at: now - chrono::Duration::seconds(secs_ago),
            endpoint: "/api/decide".into(),
            model: "kev:latest".into(),
            questions: 3,
            took_ms,
            error: error.then(|| "model not found".into()),
        }
    }

    #[test]
    fn speed_takes_the_median_of_successes_and_the_rate_of_the_last_minute() {
        let now = Utc::now();
        // Newest first: five requests over the last 8 seconds, one failed, one from long ago.
        let recent = [
            req(0, 12.0, false, now),
            req(2, 30.0, true, now),
            req(4, 10.0, false, now),
            req(8, 14.0, false, now),
            req(600, 11.0, false, now),
        ];
        let s = speed(&recent, now);
        assert_eq!(s.samples, 4);
        assert_eq!(s.median_ms, Some(12.0)); // 10, 11, 12, 14
        assert!((s.per_second - 3.0 / 8.0).abs() < 1e-9, "{}", s.per_second);
    }

    #[test]
    fn speed_of_nothing() {
        let s = speed(&[], Utc::now());
        assert_eq!(s, Speed::default());
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

    #[test]
    fn the_state_is_json_only_when_it_is_an_object_or_an_array() {
        assert_eq!(parse_state(r#" {"a": 1} "#), serde_json::json!({"a": 1}));
        assert_eq!(parse_state("[1, 2]"), serde_json::json!([1, 2]));
        assert_eq!(parse_state("42"), Value::String("42".into()));
        assert_eq!(parse_state("Refund me"), Value::String("Refund me".into()));
        assert_eq!(parse_state("{broken"), Value::String("{broken".into()));
    }

    #[test]
    fn progress_is_unknown_until_a_layer_reports_its_size() {
        assert_eq!(Progress::default().fraction(), None);
        let p = Progress {
            completed: 50,
            total: 200,
        };
        assert_eq!(p.fraction(), Some(0.25));
    }
}
