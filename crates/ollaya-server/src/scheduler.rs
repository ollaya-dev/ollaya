//! The runner scheduler: one child process per loaded model.
//!
//! Requests lease a runner. When the last lease ends, the runner's keep-alive clock starts; a
//! reaper unloads runners whose clock ran out. Loads happen one at a time. When `max_loaded` is
//! reached, the least recently used idle runner is unloaded first. Killing a runner process
//! returns all of its memory, including GPU memory.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use serde::Deserialize;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

use ollaya_decision::{cygnet, jebadiah, jevk5, llm_logits, snap, winnow};
use ollaya_registry::pull::RunCheck;

use crate::Error;
use crate::models::{AdapterFiles, EngineFiles, Loadable};

/// How long a model stays loaded after its last request (the API's `keep_alive`).
pub use ollaya_api::KeepAlive;

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub keep_alive: KeepAlive,
    pub max_loaded: usize,
    /// Adapter models one runner holds over its base (`OLLAYA_MAX_ADAPTERS`).
    pub max_adapters: usize,
    /// `auto`, `cpu`, `cuda`, `cuda:<n>` or `metal`, passed to runners.
    pub device: String,
    /// CPU threads per runner (`--threads`); `None` keeps each engine's default.
    pub threads: Option<usize>,
    pub load_timeout: Duration,
    /// The executable to spawn as `<exe> runner ...`: the running binary, or on Windows with a GPU
    /// pack, its copy inside the pack (see [`crate::launch`]).
    pub exe: PathBuf,
    /// `argv[0]` for runners. On Linux, ONNX Runtime loads its GPU provider libraries from the
    /// directory of `argv[0]`, so GPU runners get `<cuda dir>/ollaya` (absolute; the file need
    /// not exist).
    pub arg0: Option<PathBuf>,
    /// Extra environment for runner processes (e.g. the GPU library path).
    pub env: Vec<(String, String)>,
    /// llama.cpp's libraries, which GGUF models run on (`launch::llama_dir`); `None` when this
    /// install has none.
    pub llama_dir: Option<PathBuf>,
    /// The statically linked executable, when `exe` is the GPU pack's runner
    /// (`launch::CUDA_RUNNER`): ONNX models that run on the CPU start from it, plainly.
    pub cpu_exe: Option<PathBuf>,
}

impl SchedulerConfig {
    /// Whether runners start with a GPU pack (see [`crate::launch`]): a GPU `argv[0]`, the pack's
    /// runner, or the pack's library environment.
    pub fn has_gpu_pack(&self) -> bool {
        self.arg0.is_some() || self.cpu_exe.is_some() || !self.env.is_empty()
    }
}

/// Which runner a model starts from, given the configured device (see [`Scheduler::spawn`]).
#[derive(Debug, PartialEq)]
enum Plan {
    /// `exe`, with `arg0` and `env`, and the configured device.
    Configured,
    /// The CPU launch: `cpu_exe` for ONNX models, `exe` with `--device cpu` for GGUF models.
    Cpu,
    /// `Configured`; if it fails to load, `Cpu`.
    GpuThenCpu,
}

/// With a separate CPU executable, an ONNX model on `cpu` runs there, and on `auto` tries the GPU
/// runner first (which never falls back to the CPU itself, see `ollaya_runner::server`).
///
/// A GGUF model on `auto` falls back to the CPU inside its runner when llama.cpp reports an
/// error, but not when the GPU backend takes the process down before the runner answers (a
/// driver fault, an abort in ggml): it then starts again with `--device cpu`.
fn plan(device: &str, onnx: bool, has_cpu_exe: bool) -> Plan {
    match (device, onnx, has_cpu_exe) {
        ("cpu", true, true) => Plan::Cpu,
        ("auto", true, true) | ("auto", false, _) => Plan::GpuThenCpu,
        _ => Plan::Configured,
    }
}

/// One way to start a runner.
struct Launch<'a> {
    exe: &'a Path,
    // argv[0] can only be set on Unix; Windows runners never read it.
    #[cfg_attr(not(unix), allow(dead_code))]
    arg0: Option<&'a Path>,
    env: &'a [(String, String)],
    device: &'a str,
}

/// The GGUF layouts this build's runner can run (`ollaya_runner::llama::LAYOUTS`).
pub const LLAMA_LAYOUTS: &[&str] = &[
    llm_logits::LAYOUT,
    winnow::LAYOUT,
    jevk5::LAYOUT,
    jebadiah::LAYOUT,
    cygnet::LAYOUT,
    snap::LAYOUT,
    ollaya_decision::d1::LAYOUT,
];

#[derive(Debug, Deserialize)]
struct Hello {
    port: u16,
    device: String,
    precision: String,
    /// `onnx` or `mlx`; runners before 0.6 do not send it.
    #[serde(default)]
    engine: String,
    /// Bytes the runner's tensors occupy, when it reports them (the MLX engine does).
    #[serde(default)]
    bytes: u64,
}

/// An adapter model loaded in a runner over its base.
struct AdapterState {
    name: String,
    size: u64,
    loaded_at: SystemTime,
    leases: usize,
    last_used: Instant,
    /// When an idle adapter may be unloaded; `None` while leased or kept forever.
    expires: Option<Instant>,
    keep_forever: bool,
    pinned: bool,
    /// Being unloaded: no request may lease it any more.
    removing: bool,
}

enum Slot {
    Free,
    Evict(String),
    Unloading,
    Full,
}

/// What a runner shared by adapter models knows about its base.
struct Base {
    /// The name the adapters were created `FROM`.
    name: String,
    /// The base's manifest digest.
    manifest: String,
}

pub struct Runner {
    pub name: String,
    pub digest: String,
    pub device: String,
    pub precision: String,
    /// The runner's engine: `onnx` or `mlx`.
    pub engine: String,
    pub size: u64,
    pub loaded_at: SystemTime,
    port: u16,
    /// Set when the runner holds a base for adapter models.
    base: Option<Base>,
    adapters: Mutex<HashMap<String, AdapterState>>,
    child: tokio::sync::Mutex<Child>,
    http: reqwest::Client,
    /// How long registering or dropping an adapter may take (`OLLAYA_LOAD_TIMEOUT`): both run
    /// while the scheduler's load lock is held, behind the runner's queue of decisions.
    load_timeout: Duration,
    leases: AtomicUsize,
    /// When an idle runner may be unloaded; `None` while leased or when kept forever.
    expires: Mutex<Option<Instant>>,
    keep_forever: Mutex<bool>,
    /// Loaded by `OLLAYA_PRELOAD`: a request without its own `keep_alive` leaves it loaded.
    pinned: AtomicBool,
    last_used: Mutex<Instant>,
}

/// What `ollaya ps` shows.
#[derive(Debug, Clone)]
pub struct RunningInfo {
    pub name: String,
    pub digest: String,
    pub device: String,
    pub precision: String,
    pub size: u64,
    pub loaded_at: SystemTime,
    /// `None`: kept loaded until stopped.
    pub expires_at: Option<SystemTime>,
}

impl Runner {
    /// Forward a request to the runner process. `images` (vision models) and `audio` (models
    /// with an audio encoder) are base64. With `timeout`, the runner drops the request if it is
    /// still queued when that has passed.
    pub async fn decide(
        &self,
        state: &Value,
        questions: &Value,
        images: &[String],
        audio: &[String],
        timeout: Option<Duration>,
        adapter: Option<&str>,
    ) -> Result<Value, Error> {
        let mut body = serde_json::json!({"state": state, "questions": questions});
        if let Some(adapter) = adapter {
            body["adapter"] = serde_json::json!(adapter);
        }
        if !images.is_empty() {
            body["images"] = serde_json::json!(images);
        }
        if !audio.is_empty() {
            body["audio"] = serde_json::json!(audio);
        }
        if let Some(timeout) = timeout {
            body["timeout_ms"] = serde_json::json!(timeout.as_millis() as u64);
        }
        let resp = self
            .http
            .post(format!("http://127.0.0.1:{}/decide", self.port))
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Runner(format!("{}: {e}", self.name)))?;
        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .map_err(|e| Error::Runner(format!("{}: {e}", self.name)))?;
        if status.is_success() {
            return Ok(body);
        }
        let err = &body["error"];
        let message = err["message"].as_str().unwrap_or("runner error").to_owned();
        if err["code"] == "DEADLINE_EXCEEDED" {
            return Err(Error::DeadlineExceeded);
        }
        if err["code"] == "TOO_MANY_OPTIONS" {
            return Err(Error::TooManyOptions {
                question: err["question"].as_str().unwrap_or_default().to_owned(),
                options: err["options"].as_u64().unwrap_or(0) as usize,
                model: self.name.clone(),
            });
        }
        if status.is_client_error() {
            Err(Error::InvalidRequest(message))
        } else {
            Err(Error::Runner(message))
        }
    }

    async fn kill(&self) {
        let mut child = self.child.lock().await;
        let _ = child.kill().await;
    }

    /// Register an adapter model with the runner (`POST /adapters`): its tensors, and its own
    /// `decision.json` and tokenizer. Returns the bytes the adapter occupies there.
    async fn post_adapter(&self, model: &Loadable, adapter: &AdapterFiles) -> Result<u64, Error> {
        let EngineFiles::Onnx(files) = &model.files else {
            return Err(Error::Corrupt(format!(
                "{}: an adapter model runs on MLX",
                model.name
            )));
        };
        let body = serde_json::json!({
            "digest": model.digest,
            "spec": adapter.spec,
            "weights": adapter.weights,
            "decision": files.decision,
            "tokenizer": files.tokenizer,
        });
        let resp = self
            .http
            .post(format!("http://127.0.0.1:{}/adapters", self.port))
            .timeout(self.load_timeout)
            .json(&body)
            .send()
            .await;
        let resp = match resp {
            Ok(r) => r,
            Err(e) => return Err(self.load_error(model, &e).await),
        };
        let ok = resp.status().is_success();
        let body: Value = match resp.json().await {
            Ok(b) => b,
            Err(e) => return Err(self.load_error(model, &e).await),
        };
        if !ok {
            let message = body["error"]["message"]
                .as_str()
                .unwrap_or("the runner refused it");
            return Err(Error::LoadFailed(format!(
                "{} failed to load as an adapter: {message}",
                model.name
            )));
        }
        Ok(body["bytes"].as_u64().unwrap_or(0))
    }

    /// The error of a failed `POST /adapters`. After a timeout the runner may still finish loading
    /// the adapter later and serve it outside the count, so a best-effort `DELETE` follows (the
    /// runner takes it after the load, in its queue's order); its own answer is not waited for
    /// beyond a few seconds.
    async fn load_error(&self, model: &Loadable, e: &reqwest::Error) -> Error {
        if e.is_timeout() {
            let wait = self.load_timeout.min(Duration::from_secs(5));
            if let Err(d) = self.delete_adapter(&model.digest, wait).await {
                tracing::warn!(model = %self.name, "after the adapter load timed out: {d}");
            }
        }
        Error::LoadFailed(format!("{}: {}", model.name, self.timed_out(e)))
    }

    /// An error of a request to the runner, with the reason a timeout stands for.
    fn timed_out(&self, e: &reqwest::Error) -> String {
        if e.is_timeout() {
            format!(
                "the runner did not answer within the load timeout ({} s)",
                self.load_timeout.as_secs()
            )
        } else {
            e.to_string()
        }
    }

    /// Set the base's clock from the adapters that are left: kept forever only while one of them
    /// is, and until the latest of their clocks. Called after an adapter goes, so a `-1` adapter
    /// that is gone no longer keeps the base loaded.
    fn recompute_clock(&self) {
        let now = Instant::now();
        let clock = base_clock(
            self.adapters
                .lock()
                .unwrap()
                .values()
                .map(|a| (a.expires, a.keep_forever)),
        );
        *self.keep_forever.lock().unwrap() = clock.1;
        *self.expires.lock().unwrap() = clock.0.or(Some(now));
    }

    /// Drop an adapter from the runner (`DELETE /adapters/<digest>`).
    async fn delete_adapter(&self, digest: &str, timeout: Duration) -> Result<(), Error> {
        let resp = self
            .http
            .delete(format!("http://127.0.0.1:{}/adapters/{digest}", self.port))
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| Error::Runner(format!("{}: {}", self.name, self.timed_out(&e))))?;
        if resp.status().is_success() || resp.status() == reqwest::StatusCode::NOT_FOUND {
            Ok(())
        } else {
            Err(Error::Runner(format!(
                "{}: dropping an adapter answered {}",
                self.name,
                resp.status()
            )))
        }
    }

    /// Where a new adapter goes: a free slot, one made free by unloading the least recently used
    /// idle adapter (marked as being unloaded here), or nowhere.
    fn adapter_slot(&self, digest: &str, max: usize) -> Slot {
        let mut adapters = self.adapters.lock().unwrap();
        if adapters.contains_key(digest) {
            return Slot::Unloading;
        }
        if adapters.len() < max {
            return Slot::Free;
        }
        // Adapters go before bases.
        let candidates: Vec<(String, usize, Instant)> = adapters
            .iter()
            .filter(|(_, a)| !a.removing)
            .map(|(d, a)| (d.clone(), a.leases, a.last_used))
            .collect();
        match least_recently_used(&candidates) {
            Some(victim) => {
                if let Some(a) = adapters.get_mut(&victim) {
                    a.removing = true;
                }
                Slot::Evict(victim)
            }
            None => Slot::Full,
        }
    }

    /// Lease the adapter if it is loaded and not being unloaded.
    fn lease_adapter(&self, digest: &str) -> bool {
        let mut adapters = self.adapters.lock().unwrap();
        match adapters.get_mut(digest) {
            Some(a) if !a.removing => {
                a.leases += 1;
                a.expires = None;
                true
            }
            _ => false,
        }
    }

    /// Mark the adapters `pick` chooses as being unloaded, so no request leases them meanwhile.
    fn begin_remove(
        &self,
        pick: impl FnOnce(&HashMap<String, AdapterState>) -> Vec<String>,
    ) -> Vec<String> {
        let mut adapters = self.adapters.lock().unwrap();
        let picked = pick(&adapters);
        for d in &picked {
            if let Some(a) = adapters.get_mut(d) {
                a.removing = true;
            }
        }
        picked
    }

    /// Drop an adapter that [`Self::begin_remove`] marked, once its leases are over. Spawned so a
    /// cancelled `unload` future cannot leave the adapter stuck in `removing = true`.
    async fn finish_remove(self: &Arc<Self>, digest: &str) {
        let runner = self.clone();
        let digest = digest.to_owned();
        let _ = tokio::spawn(async move {
            loop {
                let leased = runner
                    .adapters
                    .lock()
                    .unwrap()
                    .get(&digest)
                    .is_some_and(|a| a.leases > 0);
                if !leased {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            if let Err(e) = runner.delete_adapter(&digest, runner.load_timeout).await {
                tracing::warn!(model = %runner.name, "{e}");
            }
            if let Some(a) = runner.adapters.lock().unwrap().remove(&digest) {
                tracing::info!(model = %a.name, "unloaded adapter");
            }
            runner.recompute_clock();
        })
        .await;
    }
}

/// One row per loaded model name and digest when a model is loaded both as itself and as the
/// base of adapters (decision 0011, section 3): its row counts both copies (`size`), is kept
/// until stopped if either is, and otherwise lasts as long as the later one. Two plain runners
/// with the same name (e.g. a re-pulled tag whose old digest is still loaded) stay two rows.
/// Rows must be sorted by `(name, digest)`.
fn merge_rows(rows: Vec<(RunningInfo, bool)>) -> Vec<RunningInfo> {
    let mut out: Vec<(RunningInfo, bool)> = Vec::with_capacity(rows.len());
    for (row, shared) in rows {
        match out.last_mut() {
            Some((last, last_shared))
                if last.name == row.name
                    && last.digest == row.digest
                    && (*last_shared || shared) =>
            {
                last.size += row.size;
                last.loaded_at = last.loaded_at.min(row.loaded_at);
                last.expires_at = match (last.expires_at, row.expires_at) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    _ => None,
                };
                *last_shared = true;
            }
            _ => out.push((row, shared)),
        }
    }
    out.into_iter().map(|(info, _)| info).collect()
}

/// The idle entry used longest ago: `(key, leases, last_used)` for each candidate. Eviction of
/// runners and of adapters both pick this way.
fn least_recently_used<K: Clone>(candidates: &[(K, usize, Instant)]) -> Option<K> {
    candidates
        .iter()
        .filter(|(_, leases, _)| *leases == 0)
        .min_by_key(|(_, _, used)| *used)
        .map(|(key, _, _)| key.clone())
}

/// The clock of a base shared by adapters: kept forever while any adapter is, else until the
/// latest of the adapters' clocks. `(expires, keep_forever)` for each adapter.
fn base_clock(
    adapters: impl IntoIterator<Item = (Option<Instant>, bool)>,
) -> (Option<Instant>, bool) {
    let mut clock = (None, false);
    for (expires, forever) in adapters {
        clock.1 |= forever;
        clock.0 = clock.0.max(expires);
    }
    clock
}

impl Runner {
    /// Keep this runner loaded until it is unloaded explicitly or evicted to make room.
    pub fn pin(&self) {
        self.pinned.store(true, Ordering::SeqCst);
        *self.keep_forever.lock().unwrap() = true;
    }
}

/// A runner in use; dropping it starts the runner's keep-alive clock.
pub struct Lease {
    pub runner: Arc<Runner>,
    keep_alive: KeepAlive,
    /// The manifest digest of the adapter model this lease is for, when it is one.
    adapter: Option<String>,
}

impl Lease {
    /// Ask the runner, as the adapter model this lease is for.
    pub async fn decide(
        &self,
        state: &Value,
        questions: &Value,
        images: &[String],
        audio: &[String],
        timeout: Option<Duration>,
    ) -> Result<Value, Error> {
        self.runner
            .decide(
                state,
                questions,
                images,
                audio,
                timeout,
                self.adapter.as_deref(),
            )
            .await
    }

    /// Keep the model this lease is for loaded until it is unloaded explicitly or evicted.
    pub fn pin(&self) {
        match &self.adapter {
            None => self.runner.pin(),
            Some(d) => {
                if let Some(a) = self.runner.adapters.lock().unwrap().get_mut(d) {
                    a.pinned = true;
                    a.keep_forever = true;
                }
                *self.runner.keep_forever.lock().unwrap() = true;
            }
        }
    }
}

impl std::ops::Deref for Lease {
    type Target = Runner;
    fn deref(&self) -> &Runner {
        &self.runner
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let r = &self.runner;
        *r.last_used.lock().unwrap() = Instant::now();
        if let Some(digest) = &self.adapter {
            let now = Instant::now();
            let clock = {
                let mut adapters = r.adapters.lock().unwrap();
                if let Some(a) = adapters.get_mut(digest) {
                    a.leases = a.leases.saturating_sub(1);
                    a.last_used = now;
                    match self.keep_alive {
                        KeepAlive::Forever => a.keep_forever = true,
                        KeepAlive::For(d) => {
                            a.keep_forever = false;
                            a.expires = Some(now + d);
                        }
                    }
                }
                base_clock(adapters.values().map(|a| (a.expires, a.keep_forever)))
            };
            *r.keep_forever.lock().unwrap() = clock.1;
            *r.expires.lock().unwrap() = clock.0.or(Some(now));
            r.leases.fetch_sub(1, Ordering::SeqCst);
            return;
        }
        match self.keep_alive {
            KeepAlive::Forever => *r.keep_forever.lock().unwrap() = true,
            KeepAlive::For(d) => {
                *r.keep_forever.lock().unwrap() = false;
                *r.expires.lock().unwrap() = Some(Instant::now() + d);
            }
        }
        r.leases.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct Scheduler {
    config: SchedulerConfig,
    runners: Mutex<HashMap<String, Arc<Runner>>>,
    load_lock: tokio::sync::Mutex<()>,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Arc<Self> {
        let s = Arc::new(Scheduler {
            config,
            runners: Mutex::new(HashMap::new()),
            load_lock: tokio::sync::Mutex::new(()),
        });
        let reaper = Arc::downgrade(&s);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_millis(500));
            loop {
                tick.tick().await;
                let Some(s) = reaper.upgrade() else { break };
                s.reap().await;
            }
        });
        s
    }

    pub fn default_keep_alive(&self) -> KeepAlive {
        self.config.keep_alive
    }

    /// Whether this install can run a model with `config`, judged before a pull downloads it.
    pub fn run_check(&self) -> RunCheck {
        let llama = self.config.llama_dir.is_some();
        Arc::new(move |name, c| match c.model_format.as_str() {
            "" | "onnx" | "router" | ollaya_registry::adapter::MODEL_FORMAT => Ok(()),
            "gguf" => match c.layout.as_deref() {
                Some(l) if !LLAMA_LAYOUTS.contains(&l) => Err(format!(
                    "{name} uses the {l} layout, which this version of ollaya cannot run; upgrade ollaya"
                )),
                _ if !llama => Err(format!(
                    "{name} runs on llama.cpp, and this installation of ollaya has no llama.cpp \
                     libraries (lib/ollaya/llama); nothing was downloaded"
                )),
                _ => Ok(()),
            },
            other => Err(format!(
                "{name} is a {other} model, which this version of ollaya cannot run; upgrade ollaya"
            )),
        })
    }

    /// Lease a runner for `model`, loading it if needed. Returns the lease and the load time.
    ///
    /// An adapter model leases the runner of its base (loading that if needed), then loads
    /// itself into it.
    pub async fn acquire(
        &self,
        model: &Loadable,
        keep_alive: Option<KeepAlive>,
    ) -> Result<(Lease, Duration), Error> {
        let started = Instant::now();
        let key = model.runner_key();
        let mut loaded = false;
        let fast = match &model.adapter {
            None => self.lease_existing(key),
            Some(_) => self.lease_existing_adapter(key, &model.digest),
        };
        let runner = match fast {
            Some(r) => r,
            None => {
                let _guard = self.load_lock.lock().await;
                let runner = match self.lease_existing(key) {
                    Some(r) => r,
                    None => {
                        self.make_room().await;
                        let runner = Arc::new(self.spawn(model).await?);
                        runner.leases.fetch_add(1, Ordering::SeqCst);
                        self.runners
                            .lock()
                            .unwrap()
                            .insert(key.to_owned(), runner.clone());
                        loaded = true;
                        tracing::info!(model = %model.name, device = %runner.device, precision = %runner.precision,
                            engine = %runner.engine, threads = self.config.threads,
                            load_ms = started.elapsed().as_millis() as u64, "loaded model");
                        runner
                    }
                };
                // The runner is leased from here: a failure gives the lease back.
                if let Some(files) = &model.adapter {
                    match self.lease_adapter_locked(&runner, model, files).await {
                        Ok(did_load) => loaded |= did_load,
                        Err(e) => {
                            self.give_back(&runner);
                            return Err(e);
                        }
                    }
                }
                runner
            }
        };
        let keep_alive = keep_alive.unwrap_or_else(|| {
            let pinned = match &model.adapter {
                Some(_) => runner
                    .adapters
                    .lock()
                    .unwrap()
                    .get(&model.digest)
                    .is_some_and(|a| a.pinned),
                None => runner.pinned.load(Ordering::SeqCst),
            };
            if pinned {
                KeepAlive::Forever
            } else {
                self.config.keep_alive
            }
        });
        let lease = Lease {
            keep_alive,
            adapter: model.adapter.as_ref().map(|_| model.digest.clone()),
            runner,
        };
        Ok((
            lease,
            if loaded {
                started.elapsed()
            } else {
                Duration::ZERO
            },
        ))
    }

    /// Lease `model` in `runner`, loading it into the runner when it is not there. Caller must
    /// hold `self.load_lock`. Returns whether it was loaded. Only that adapter fails when it
    /// cannot load: the base and the other adapters stay.
    async fn lease_adapter_locked(
        &self,
        runner: &Arc<Runner>,
        model: &Loadable,
        files: &AdapterFiles,
    ) -> Result<bool, Error> {
        if runner.lease_adapter(&model.digest) {
            return Ok(false);
        }
        let max = self.config.max_adapters.max(1);
        let deadline = Instant::now() + self.config.load_timeout;
        let victim = loop {
            match runner.adapter_slot(&model.digest, max) {
                // An unload of this very adapter is still running: wait it out, then load again.
                Slot::Unloading => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return Err(Error::LoadFailed(format!(
                            "{}: timed out waiting for a previous unload to finish ({} s)",
                            model.name,
                            self.config.load_timeout.as_secs()
                        )));
                    }
                    tokio::time::sleep(remaining.min(Duration::from_millis(20))).await;
                }
                Slot::Free => break None,
                Slot::Evict(victim) => break Some(victim),
                Slot::Full => {
                    return Err(Error::LoadFailed(format!(
                        "{} cannot load: the runner holds OLLAYA_MAX_ADAPTERS ({max}) adapters and all of them are in use",
                        model.name
                    )));
                }
            }
        };
        if let Some(victim) = victim {
            runner.finish_remove(&victim).await;
        }
        let started = Instant::now();
        let bytes = runner.post_adapter(model, files).await?;
        runner.adapters.lock().unwrap().insert(
            model.digest.clone(),
            AdapterState {
                name: model.name.to_string(),
                size: bytes,
                loaded_at: SystemTime::now(),
                leases: 1,
                last_used: Instant::now(),
                expires: None,
                keep_forever: false,
                pinned: false,
                removing: false,
            },
        );
        tracing::info!(model = %model.name, base = %runner.name, bytes,
            load_ms = started.elapsed().as_millis() as u64, "loaded adapter");
        Ok(true)
    }

    /// Give back a runner lease taken for a load that failed; the runner's clock resumes from its
    /// remaining adapters (as [`Lease::drop`] does), or starts from the default keep-alive when no
    /// adapters are loaded yet.
    fn give_back(&self, runner: &Runner) {
        let now = Instant::now();
        *runner.last_used.lock().unwrap() = now;
        let (empty, clock) = {
            let adapters = runner.adapters.lock().unwrap();
            (
                adapters.is_empty(),
                base_clock(adapters.values().map(|a| (a.expires, a.keep_forever))),
            )
        };
        if empty {
            match self.config.keep_alive {
                KeepAlive::Forever => *runner.keep_forever.lock().unwrap() = true,
                KeepAlive::For(d) => {
                    *runner.keep_forever.lock().unwrap() = false;
                    *runner.expires.lock().unwrap() = Some(now + d);
                }
            }
        } else {
            *runner.keep_forever.lock().unwrap() = clock.1;
            *runner.expires.lock().unwrap() = clock.0.or(Some(now));
        }
        runner.leases.fetch_sub(1, Ordering::SeqCst);
    }

    fn lease_existing(&self, digest: &str) -> Option<Arc<Runner>> {
        let runners = self.runners.lock().unwrap();
        let r = runners.get(digest)?.clone();
        r.leases.fetch_add(1, Ordering::SeqCst);
        *r.expires.lock().unwrap() = None;
        Some(r)
    }

    /// Lease an existing runner together with an already-loaded adapter, atomically under
    /// `runners`. Returns `None` without taking a runner lease when the adapter is not loaded, so
    /// no caller ever waits on `load_lock` while holding a runner lease.
    fn lease_existing_adapter(&self, key: &str, adapter: &str) -> Option<Arc<Runner>> {
        let runners = self.runners.lock().unwrap();
        let r = runners.get(key)?.clone();
        if !r.lease_adapter(adapter) {
            return None;
        }
        r.leases.fetch_add(1, Ordering::SeqCst);
        *r.expires.lock().unwrap() = None;
        Some(r)
    }

    /// Take down the runner stored under `key` (the reaper's and `make_room`'s eviction): that
    /// runner only. A plain `laya:en` that expires or is evicted leaves a shared base of adapters
    /// created on it, and their keep-alive, alone.
    async fn evict(&self, key: &str) {
        let runner = self.runners.lock().unwrap().remove(key);
        if let Some(r) = runner {
            while r.leases.load(Ordering::SeqCst) > 0 {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            r.kill().await;
            tracing::info!(model = %r.name, "unloaded model");
        }
    }

    /// Unload a model now, at the user's word (`stop`, `delete`; the runner finishes in-flight
    /// requests first). `digest` is the model's manifest digest. An adapter model leaves its
    /// runner; the runner goes with its last adapter. A base also takes down the shared runner of
    /// adapters created on it, with those adapters. Internal eviction does not: see `evict`.
    pub async fn unload(&self, digest: &str) -> bool {
        // The runner keyed by this digest, and the shared runner of any adapters created on this
        // manifest: `ollaya stop <base>` reaches the base whether it was loaded as itself, as the
        // base of adapters, or both. Its adapters go with it, as when it is evicted.
        let gone: Vec<Arc<Runner>> = {
            let mut runners = self.runners.lock().unwrap();
            let keys: Vec<String> = runners
                .iter()
                .filter(|(key, r)| {
                    key.as_str() == digest || r.base.as_ref().is_some_and(|b| b.manifest == digest)
                })
                .map(|(key, _)| key.clone())
                .collect();
            keys.iter().filter_map(|k| runners.remove(k)).collect()
        };
        if !gone.is_empty() {
            for r in gone {
                while r.leases.load(Ordering::SeqCst) > 0 {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                r.kill().await;
                tracing::info!(model = %r.name, "unloaded model");
            }
            return true;
        }
        let holder = self
            .runners
            .lock()
            .unwrap()
            .iter()
            .find(|(_, r)| r.adapters.lock().unwrap().contains_key(digest))
            .map(|(key, r)| (key.clone(), r.clone()));
        let Some((key, runner)) = holder else {
            return false;
        };
        let marked = runner.begin_remove(|a| {
            a.contains_key(digest)
                .then(|| digest.to_owned())
                .into_iter()
                .collect()
        });
        for d in marked {
            runner.finish_remove(&d).await;
        }
        if runner.adapters.lock().unwrap().is_empty() {
            let gone = {
                let mut runners = self.runners.lock().unwrap();
                match runners.get(&key) {
                    Some(r) if Arc::ptr_eq(r, &runner) && r.leases.load(Ordering::SeqCst) == 0 => {
                        runners.remove(&key)
                    }
                    _ => None,
                }
            };
            if let Some(r) = gone {
                r.kill().await;
                tracing::info!(model = %r.name, "unloaded model");
            }
        }
        true
    }

    pub fn running(&self) -> Vec<RunningInfo> {
        let now_i = Instant::now();
        let now_s = SystemTime::now();
        let at = |e: Instant| now_s + e.saturating_duration_since(now_i);
        let mut out: Vec<(RunningInfo, bool)> = Vec::new();
        for r in self.runners.lock().unwrap().values() {
            let busy = r.leases.load(Ordering::SeqCst) > 0;
            let expires_at = if *r.keep_forever.lock().unwrap() {
                None
            } else if busy {
                Some(now_s + Duration::from_secs(0))
            } else {
                r.expires.lock().unwrap().map(at)
            };
            // One row per loaded model name: the base, and one for each adapter over it. The
            // base's `size` is the base's, an adapter's only its own, so the rows add up to what
            // the runner holds.
            out.push((
                RunningInfo {
                    name: r.name.clone(),
                    digest: r
                        .base
                        .as_ref()
                        .map_or_else(|| r.digest.clone(), |b| b.manifest.clone()),
                    device: r.device.clone(),
                    precision: r.precision.clone(),
                    size: r.size,
                    loaded_at: r.loaded_at,
                    expires_at,
                },
                r.base.is_some(),
            ));
            for (digest, a) in r.adapters.lock().unwrap().iter() {
                let expires_at = if a.keep_forever {
                    None
                } else if a.leases > 0 {
                    Some(now_s)
                } else {
                    a.expires.map(at)
                };
                out.push((
                    RunningInfo {
                        name: a.name.clone(),
                        digest: digest.clone(),
                        device: r.device.clone(),
                        precision: r.precision.clone(),
                        size: a.size,
                        loaded_at: a.loaded_at,
                        expires_at,
                    },
                    false,
                ));
            }
        }
        out.sort_by(|(a, _), (b, _)| a.name.cmp(&b.name).then_with(|| a.digest.cmp(&b.digest)));
        merge_rows(out)
    }

    /// Unload every runner (daemon shutdown).
    pub async fn shutdown(&self) {
        let all: Vec<Arc<Runner>> = self
            .runners
            .lock()
            .unwrap()
            .drain()
            .map(|(_, r)| r)
            .collect();
        for r in all {
            r.kill().await;
        }
    }

    async fn reap(&self) {
        let now = Instant::now();
        // Adapters unload on their own clocks, before their base's.
        let holders: Vec<Arc<Runner>> = self
            .runners
            .lock()
            .unwrap()
            .values()
            .filter(|r| r.base.is_some())
            .cloned()
            .collect();
        for r in holders {
            let due = r.begin_remove(|adapters| {
                adapters
                    .iter()
                    .filter(|(_, a)| {
                        a.leases == 0
                            && !a.keep_forever
                            && !a.removing
                            && a.expires.is_some_and(|e| e <= now)
                    })
                    .map(|(d, _)| d.clone())
                    .collect()
            });
            for d in due {
                r.finish_remove(&d).await;
            }
        }
        let mut expired = Vec::new();
        for (digest, r) in self.runners.lock().unwrap().iter() {
            let idle = r.leases.load(Ordering::SeqCst) == 0;
            let due = r.expires.lock().unwrap().is_some_and(|e| e <= now);
            if idle && due && !*r.keep_forever.lock().unwrap() {
                expired.push(digest.clone());
            }
        }
        for digest in expired {
            self.evict(&digest).await;
        }
        // A runner that died on its own (crash, OOM kill) is forgotten so the next request reloads it.
        let mut dead = Vec::new();
        for (digest, r) in self.runners.lock().unwrap().iter() {
            if let Ok(mut child) = r.child.try_lock()
                && let Ok(Some(status)) = child.try_wait()
            {
                tracing::warn!(model = %r.name, %status, "runner exited");
                dead.push(digest.clone());
            }
        }
        for digest in dead {
            self.runners.lock().unwrap().remove(&digest);
        }
    }

    async fn make_room(&self) {
        let victim = {
            let runners = self.runners.lock().unwrap();
            if runners.len() < self.config.max_loaded.max(1) {
                return;
            }
            // Idle runners only; its adapters go with a runner that is evicted.
            let candidates: Vec<(String, usize, Instant)> = runners
                .iter()
                .map(|(d, r)| {
                    (
                        d.clone(),
                        r.leases.load(Ordering::SeqCst),
                        *r.last_used.lock().unwrap(),
                    )
                })
                .collect();
            least_recently_used(&candidates)
        };
        if let Some(digest) = victim {
            self.evict(&digest).await;
        }
    }

    async fn spawn(&self, model: &Loadable) -> Result<Runner, Error> {
        if model.adapter.is_some() {
            return self.spawn_adapter_base(model).await;
        }
        let configured = Launch {
            exe: &self.config.exe,
            arg0: self.config.arg0.as_deref(),
            env: &self.config.env,
            device: &self.config.device,
        };
        let onnx = matches!(model.files, EngineFiles::Onnx(_));
        let cpu_exe = self.config.cpu_exe.as_deref();
        let cpu = if onnx {
            cpu_exe.map(|exe| Launch {
                exe,
                arg0: None,
                env: &[],
                device: "cpu",
            })
        } else {
            Some(Launch {
                device: "cpu",
                ..configured
            })
        };
        match (plan(&self.config.device, onnx, cpu_exe.is_some()), cpu) {
            (Plan::Cpu, Some(cpu)) => self.spawn_with(model, &cpu).await,
            (Plan::GpuThenCpu, Some(cpu)) => match self.spawn_with(model, &configured).await {
                Ok(runner) => Ok(runner),
                Err(e) => {
                    tracing::warn!("{e}; loading it on the CPU");
                    self.spawn_with(model, &cpu).await
                }
            },
            _ => self.spawn_with(model, &configured).await,
        }
    }

    /// The runner for an adapter model's base: MLX only, so `auto` and `metal` start it on Metal
    /// and any other device is refused (no fallback to the CPU).
    async fn spawn_adapter_base(&self, model: &Loadable) -> Result<Runner, Error> {
        if !matches!(self.config.device.as_str(), "auto" | "metal") {
            return Err(Error::LoadFailed(format!(
                "{} is an adapter model, which runs on MLX (the Metal GPU) only, and the device is {}",
                model.name, self.config.device
            )));
        }
        let launch = Launch {
            exe: &self.config.exe,
            arg0: self.config.arg0.as_deref(),
            env: &self.config.env,
            device: "metal",
        };
        self.spawn_with(model, &launch).await.map_err(|e| match e {
            Error::LoadFailed(m) => Error::LoadFailed(format!(
                "{m} (an adapter model runs on MLX only, on the Metal GPU)"
            )),
            other => other,
        })
    }

    async fn spawn_with(&self, model: &Loadable, launch: &Launch<'_>) -> Result<Runner, Error> {
        let mut cmd = Command::new(launch.exe);
        cmd.arg("runner");
        match &model.files {
            EngineFiles::Onnx(f) => {
                cmd.arg("--tokenizer")
                    .arg(&f.tokenizer)
                    .arg("--decision")
                    .arg(&f.decision);
                if let Some(g) = &f.graph_fp32 {
                    cmd.arg("--graph-fp32").arg(g);
                }
                if let Some(g) = &f.graph_fp16 {
                    cmd.arg("--graph-fp16").arg(g);
                }
                if let Some(g) = &f.vision_graph {
                    cmd.arg("--vision-graph").arg(g);
                }
                if let Some(arch) = &f.arch {
                    cmd.arg("--arch").arg(arch);
                    for w in &f.weights {
                        cmd.arg("--weights").arg(w);
                    }
                }
            }
            EngineFiles::Llama(f) => {
                let dir = self.config.llama_dir.as_ref().ok_or_else(|| {
                    Error::Unsupported(format!(
                        "{} runs on llama.cpp, and this installation of ollaya has no llama.cpp \
                         libraries (lib/ollaya/llama); reinstall ollaya",
                        model.name
                    ))
                })?;
                if let Some(mmproj) = &f.mmproj {
                    cmd.arg("--mmproj").arg(mmproj);
                }
                if let Some(tokenizer) = &f.tokenizer {
                    cmd.arg("--tokenizer").arg(tokenizer);
                }
                cmd.arg("--gguf")
                    .arg(&f.gguf)
                    .arg("--decision")
                    .arg(&f.decision)
                    .arg("--llama-dir")
                    .arg(dir);
            }
        }
        cmd.arg("--device").arg(launch.device);
        if let Some(n) = self.config.threads {
            cmd.arg("--threads").arg(n.to_string());
        }
        #[cfg(unix)]
        if let Some(arg0) = launch.arg0 {
            cmd.arg0(arg0);
        }
        // A runner is a console program; from a server without a console (the desktop app's, or
        // one the CLI started), Windows would open a window for each one.
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        cmd.envs(launch.env.iter().map(|(k, v)| (k, v)))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd
            .spawn()
            .map_err(|e| Error::LoadFailed(format!("spawn {}: {e}", launch.exe.display())))?;
        let stdout = child.stdout.take().expect("stdout is piped");
        let stderr = child.stderr.take().expect("stderr is piped");

        // Drain the runner's stderr into our log from the start, so it never blocks on a full
        // pipe: a runner that logs more than the pipe holds before it announces itself (4 KiB on
        // Windows, for example with OLLAYA_LOG=info,ort=info) would otherwise hang until the load
        // times out. The last lines explain a failed load.
        let tail = Arc::new(Mutex::new(VecDeque::<String>::new()));
        let name = model.name.to_string();
        let drain = tokio::spawn({
            let tail = tail.clone();
            async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(l)) = lines.next_line().await {
                    // Runners before 0.7.4 colour their log even into a pipe.
                    let l = strip_ansi(&l);
                    // A runner's own warnings (a GPU that failed, a fallback to the CPU) are the
                    // only record of why a model runs where it does, so they stay visible at the
                    // default level; the rest of its log is debug output.
                    if runner_warns(&l) {
                        tracing::warn!(runner = %name, "{l}");
                    } else {
                        tracing::debug!(runner = %name, "{l}");
                    }
                    if !l.trim().is_empty() {
                        let mut tail = tail.lock().unwrap();
                        if tail.len() == STDERR_TAIL {
                            tail.pop_front();
                        }
                        tail.push_back(l);
                    }
                }
            }
        });

        let mut line = String::new();
        let read = tokio::time::timeout(
            self.config.load_timeout,
            BufReader::new(stdout).read_line(&mut line),
        )
        .await;
        let hello: Option<Hello> = match read {
            Ok(Ok(n)) if n > 0 => serde_json::from_str(line.trim()).ok(),
            _ => None,
        };
        let Some(hello) = hello else {
            let _ = child.kill().await;
            let _ = tokio::time::timeout(Duration::from_secs(1), drain).await;
            let reason = if read.is_err() {
                "timed out loading".to_owned()
            } else {
                let tail = tail.lock().unwrap();
                tail.iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(" | ")
            };
            return Err(Error::LoadFailed(format!(
                "{} failed to load: {reason}",
                model.name
            )));
        };
        if launch.device == "auto" && hello.device == "cpu" && self.config.has_gpu_pack() {
            tracing::warn!(
                "{} runs on the CPU although a GPU pack is installed: no usable GPU for it (see \
                 the runner's warnings above, or set OLLAYA_DEVICE=cuda to see the error)",
                model.name
            );
        }
        let base = model.adapter.as_ref().map(|a| Base {
            name: a.parent.clone(),
            manifest: a.base_manifest.clone(),
        });
        // What the runner reports it holds; else, for a base, its weights on disk.
        let size = match (&model.adapter, hello.bytes) {
            (Some(a), 0) => a.base_size,
            (Some(_), bytes) => bytes,
            (None, _) => model.size,
        };
        Ok(Runner {
            name: match &base {
                Some(b) if !b.name.is_empty() => b.name.clone(),
                _ => model.name.to_string(),
            },
            digest: model.digest.clone(),
            device: hello.device,
            precision: hello.precision,
            engine: hello.engine,
            size,
            loaded_at: SystemTime::now(),
            port: hello.port,
            base,
            adapters: Mutex::new(HashMap::new()),
            child: tokio::sync::Mutex::new(child),
            // Plain HTTP to 127.0.0.1: no TLS roots, so a host without CA certificates works.
            http: reqwest::Client::builder()
                .no_proxy()
                .tls_certs_only(std::iter::empty())
                .build()
                .map_err(|e| Error::LoadFailed(e.to_string()))?,
            load_timeout: self.config.load_timeout,
            leases: AtomicUsize::new(0),
            expires: Mutex::new(None),
            keep_forever: Mutex::new(false),
            pinned: AtomicBool::new(false),
            last_used: Mutex::new(Instant::now()),
        })
    }
}

/// Whether a line of a runner's log is a warning or an error (`tracing`'s format: timestamp, level,
/// target, message), whatever its colouring.
fn runner_warns(line: &str) -> bool {
    strip_ansi(line)
        .split_whitespace()
        .nth(1)
        .is_some_and(|level| level == "WARN" || level == "ERROR")
}

/// `line` without ANSI colour sequences (`ESC [ ... m`).
fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Non-empty stderr lines of a runner kept to explain a failed load.
const STDERR_TAIL: usize = 5;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::process::Stdio;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant, SystemTime};

    use axum::Json;
    use axum::routing::{delete, post};
    use ollaya_api::KeepAlive;
    use ollaya_decision::Calibration;
    use ollaya_registry::{ModelConfig, ModelName};
    use serde_json::json;
    use tokio::process::{Child, Command};

    use super::{
        AdapterState, Base, Plan, Runner, RunningInfo, Scheduler, SchedulerConfig, base_clock,
        least_recently_used, merge_rows, plan, runner_warns,
    };
    use crate::Error;
    use crate::models::{AdapterFiles, EngineFiles, Inputs, Loadable, RunnerFiles};

    fn dummy_child() -> Child {
        #[cfg(unix)]
        let mut cmd = {
            let mut c = Command::new("sleep");
            c.arg("60");
            c
        };
        #[cfg(windows)]
        let mut cmd = {
            let mut c = Command::new("ping");
            c.args(["-n", "60", "127.0.0.1"]);
            c
        };
        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .unwrap()
    }

    async fn mock_adapter_server() -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = axum::Router::new()
            .route("/adapters", post(|| async { Json(json!({"bytes": 1000})) }))
            .route(
                "/adapters/{digest}",
                delete(|| async { Json(json!({"status": "ok"})) }),
            );
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        port
    }

    fn test_scheduler(max_loaded: usize, load_timeout: Duration) -> Arc<Scheduler> {
        Scheduler::new(SchedulerConfig {
            keep_alive: KeepAlive::For(Duration::from_secs(300)),
            max_loaded,
            max_adapters: 4,
            device: "metal".into(),
            threads: None,
            load_timeout,
            exe: PathBuf::from("nonexistent-ollaya-test-exe"),
            arg0: None,
            env: vec![],
            llama_dir: None,
            cpu_exe: None,
        })
    }

    fn test_adapter_loadable(dir: &Path, name: &str, digest: &str, base_key: &str) -> Loadable {
        let placeholder = |file: &str| {
            let p = dir.join(file);
            if !p.exists() {
                std::fs::write(&p, b"{}").unwrap();
            }
            p
        };
        let config: ModelConfig =
            serde_json::from_value(json!({"model_format": "adapter", "family": "laya"})).unwrap();
        Loadable {
            name: ModelName::parse(name).unwrap(),
            digest: digest.into(),
            files: EngineFiles::Onnx(RunnerFiles {
                graph_fp32: None,
                graph_fp16: None,
                tokenizer: placeholder("tokenizer.json"),
                decision: placeholder("decision.json"),
                arch: Some(placeholder("arch.json")),
                weights: vec![placeholder("base.safetensors")],
                vision_graph: None,
            }),
            calibration: Calibration::default(),
            inputs: Inputs::default(),
            config,
            questions: None,
            size: 1000,
            adapter: Some(AdapterFiles {
                spec: placeholder("spec.json"),
                weights: placeholder("adapter.safetensors"),
                key: base_key.into(),
                parent: "laya:en".into(),
                base_manifest: "sha256:base".into(),
                base_size: 5000,
            }),
        }
    }

    fn test_shared_runner(
        port: u16,
        load_timeout: Duration,
        adapters: HashMap<String, AdapterState>,
        leases: usize,
    ) -> Arc<Runner> {
        Arc::new(Runner {
            name: "laya:en".into(),
            digest: "sha256:base".into(),
            device: "metal".into(),
            precision: "fp32".into(),
            engine: "mlx".into(),
            size: 5000,
            loaded_at: SystemTime::now(),
            port,
            base: Some(Base {
                name: "laya:en".into(),
                manifest: "sha256:base".into(),
            }),
            adapters: Mutex::new(adapters),
            child: tokio::sync::Mutex::new(dummy_child()),
            http: reqwest::Client::builder().no_proxy().build().unwrap(),
            load_timeout,
            leases: AtomicUsize::new(leases),
            expires: Mutex::new(None),
            keep_forever: Mutex::new(false),
            pinned: AtomicBool::new(false),
            last_used: Mutex::new(Instant::now()),
        })
    }

    #[test]
    fn runner_warnings_are_recognised_with_or_without_colour() {
        assert!(runner_warns(
            "2026-09-28T00:34:25.166279Z  WARN ollaya_runner::server: GPU 0 failed its first request, using CPU"
        ));
        assert!(runner_warns(
            "\x1b[2m2026-09-28T00:34:25Z\x1b[0m \x1b[33m WARN\x1b[0m \x1b[2mollaya_runner::server\x1b[0m: x"
        ));
        assert!(runner_warns(
            "2026-09-28T00:34:25Z ERROR ollaya_runner: boom"
        ));
        assert!(!runner_warns(
            "2026-09-28T00:34:25Z  INFO ollaya_runner::server: loaded"
        ));
        assert!(!runner_warns("a line that merely mentions WARN"));
        assert!(!runner_warns(""));
    }

    #[test]
    fn onnx_models_on_the_cpu_start_from_the_static_build() {
        // A GPU pack with its own ONNX Runtime: CPU runners keep today's build.
        assert_eq!(plan("cpu", true, true), Plan::Cpu);
        assert_eq!(plan("auto", true, true), Plan::GpuThenCpu);
        assert_eq!(plan("cuda", true, true), Plan::Configured);
        assert_eq!(plan("cuda:1", true, true), Plan::Configured);
        // GGUF models run on llama.cpp and keep the configured runner; on `auto`, a runner that
        // dies on the GPU before it answers starts again on the CPU.
        assert_eq!(plan("cpu", false, true), Plan::Configured);
        assert_eq!(plan("auto", false, true), Plan::GpuThenCpu);
        assert_eq!(plan("auto", false, false), Plan::GpuThenCpu);
        assert_eq!(plan("cuda", false, false), Plan::Configured);
        // Without a separate CPU executable, nothing changes for ONNX models.
        assert_eq!(plan("cpu", true, false), Plan::Configured);
        assert_eq!(plan("auto", true, false), Plan::Configured);
    }

    #[test]
    fn the_victim_is_the_idle_entry_used_longest_ago() {
        let t0 = Instant::now();
        let at = |s| t0 + Duration::from_secs(s);
        // The oldest is leased: it is never the victim; adapters and runners pick alike.
        let list = [("a", 1, at(1)), ("b", 0, at(5)), ("c", 0, at(3))];
        assert_eq!(least_recently_used(&list), Some("c"));
        assert_eq!(least_recently_used(&[("a", 2, at(1))]), None);
        assert_eq!(least_recently_used::<&str>(&[]), None);
    }

    #[test]
    fn a_base_lives_as_long_as_its_longest_lived_adapter() {
        let t0 = Instant::now();
        let at = |s| Some(t0 + Duration::from_secs(s));
        // The latest clock wins; an adapter still leased (no clock) does not shorten it.
        assert_eq!(
            base_clock([(at(10), false), (at(30), false), (None, false)]),
            (at(30), false)
        );
        // `-1` on one adapter keeps the base forever.
        assert_eq!(base_clock([(at(10), false), (None, true)]), (at(10), true));
        assert_eq!(base_clock([]), (None, false));
    }

    #[tokio::test]
    async fn dropped_unload_completes_and_allows_later_load() {
        let dir = tempfile::tempdir().unwrap();
        let port = mock_adapter_server().await;
        let scheduler = test_scheduler(2, Duration::from_secs(2));
        let base_key = "shared-base";
        let a_digest = "sha256:adapter-a";
        let b_digest = "sha256:adapter-b";
        let adapter_a = test_adapter_loadable(dir.path(), "ft-a:latest", a_digest, base_key);
        let mut adapters = HashMap::new();
        adapters.insert(
            a_digest.into(),
            AdapterState {
                name: "ft-a:latest".into(),
                size: 1000,
                loaded_at: SystemTime::now(),
                leases: 1,
                last_used: Instant::now(),
                expires: None,
                keep_forever: false,
                pinned: false,
                removing: false,
            },
        );
        adapters.insert(
            b_digest.into(),
            AdapterState {
                name: "ft-b:latest".into(),
                size: 1000,
                loaded_at: SystemTime::now(),
                leases: 0,
                last_used: Instant::now(),
                expires: None,
                keep_forever: true,
                pinned: false,
                removing: false,
            },
        );
        let runner = test_shared_runner(port, Duration::from_secs(2), adapters, 1);
        scheduler
            .runners
            .lock()
            .unwrap()
            .insert(base_key.into(), runner.clone());

        // Drop `unload` while `finish_remove` waits for adapter `a`'s active lease.
        let cancelled = tokio::time::timeout(Duration::from_millis(60), scheduler.unload(a_digest))
            .await
            .is_err();
        assert!(cancelled, "unload should block while lease is active");

        // End the lease; a spawned `finish_remove` still finishes removing `a`.
        if let Some(a) = runner.adapters.lock().unwrap().get_mut(a_digest) {
            a.leases = 0;
        }
        runner.leases.store(0, Ordering::SeqCst);

        let (lease, _) =
            tokio::time::timeout(Duration::from_secs(2), scheduler.acquire(&adapter_a, None))
                .await
                .expect("acquire hung after cancelled unload")
                .unwrap();
        drop(lease);
        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn unloading_wait_is_bounded_by_load_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let port = mock_adapter_server().await;
        let scheduler = test_scheduler(2, Duration::from_millis(120));
        let base_key = "shared-base";
        let a_digest = "sha256:adapter-a";
        let b_digest = "sha256:adapter-b";
        let adapter_a = test_adapter_loadable(dir.path(), "ft-a:latest", a_digest, base_key);
        let adapter_b = test_adapter_loadable(dir.path(), "ft-b:latest", b_digest, base_key);
        let mut adapters = HashMap::new();
        adapters.insert(
            a_digest.into(),
            AdapterState {
                name: "ft-a:latest".into(),
                size: 1000,
                loaded_at: SystemTime::now(),
                leases: 1,
                last_used: Instant::now(),
                expires: None,
                keep_forever: false,
                pinned: false,
                removing: true,
            },
        );
        let runner = test_shared_runner(port, Duration::from_millis(120), adapters, 1);
        scheduler
            .runners
            .lock()
            .unwrap()
            .insert(base_key.into(), runner);

        let err = tokio::time::timeout(Duration::from_secs(2), scheduler.acquire(&adapter_a, None))
            .await
            .expect("Slot::Unloading loop did not respect load_timeout")
            .err()
            .expect("expected LoadFailed");
        assert!(
            matches!(err, Error::LoadFailed(ref m) if m.contains("unload") || m.contains("timed out")),
            "unexpected error: {err:?}"
        );

        // `load_lock` was released, so another adapter loads cleanly.
        let (lease, _) =
            tokio::time::timeout(Duration::from_secs(2), scheduler.acquire(&adapter_b, None))
                .await
                .expect("load_lock stayed held after bounded Unloading timeout")
                .unwrap();
        drop(lease);
        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn make_room_evict_does_not_deadlock_with_cold_adapter_request() {
        let dir = tempfile::tempdir().unwrap();
        let port = mock_adapter_server().await;
        let scheduler = test_scheduler(1, Duration::from_secs(2));
        let base_key = "shared-base";
        let a_digest = "sha256:adapter-a";
        let b_digest = "sha256:adapter-b";
        let adapter_b = test_adapter_loadable(dir.path(), "ft-b:latest", b_digest, base_key);

        let idle_a = || {
            let mut adapters = HashMap::new();
            adapters.insert(
                a_digest.into(),
                AdapterState {
                    name: "ft-a:latest".into(),
                    size: 1000,
                    loaded_at: SystemTime::now(),
                    leases: 0,
                    last_used: Instant::now(),
                    expires: Some(Instant::now() + Duration::from_secs(300)),
                    keep_forever: false,
                    pinned: false,
                    removing: false,
                },
            );
            adapters
        };

        // Part 1: `make_room` holds `load_lock` and evicts idle runner R while a cold adapter
        // request for B waits on `load_lock`.
        let runner = test_shared_runner(port, Duration::from_secs(2), idle_a(), 0);
        scheduler
            .runners
            .lock()
            .unwrap()
            .insert(base_key.into(), runner);

        let guard = scheduler.load_lock.lock().await;
        let task2 = tokio::spawn({
            let s = scheduler.clone();
            let b = adapter_b.clone();
            async move { s.acquire(&b, None).await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        tokio::time::timeout(Duration::from_secs(2), scheduler.evict(base_key))
            .await
            .expect("evict deadlocked while adapter request waited on load_lock");
        drop(guard);

        let res = tokio::time::timeout(Duration::from_secs(2), task2)
            .await
            .expect("adapter request hung")
            .unwrap();
        assert!(
            matches!(res, Err(Error::LoadFailed(_))),
            "expected LoadFailed after victim was evicted"
        );

        // Part 2: when R is not evicted, a waiting cold-adapter request holds no runner lease
        // across `load_lock` and finishes loading B once `load_lock` is released.
        let runner2 = test_shared_runner(port, Duration::from_secs(2), idle_a(), 0);
        scheduler
            .runners
            .lock()
            .unwrap()
            .insert(base_key.into(), runner2.clone());

        let guard = scheduler.load_lock.lock().await;
        let task2 = tokio::spawn({
            let s = scheduler.clone();
            let b = adapter_b.clone();
            async move { s.acquire(&b, None).await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            runner2.leases.load(Ordering::SeqCst),
            0,
            "cold adapter request held runner lease while waiting on load_lock"
        );
        drop(guard);

        let (lease, _) = tokio::time::timeout(Duration::from_secs(2), task2)
            .await
            .expect("cold adapter request hung")
            .unwrap()
            .unwrap();
        drop(lease);
        scheduler.shutdown().await;
    }

    #[test]
    fn merge_rows_merges_only_a_shared_base_with_its_plain_copy() {
        let t0 = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let row = |digest: &str, size: u64, loaded_s: u64, expires_s: Option<u64>| RunningInfo {
            name: "laya:en".into(),
            digest: digest.into(),
            device: "metal".into(),
            precision: "fp32".into(),
            size,
            loaded_at: t0 + Duration::from_secs(loaded_s),
            expires_at: expires_s.map(|s| t0 + Duration::from_secs(s)),
        };

        // Two plain runners with different digests (e.g. a re-pulled tag) stay two rows.
        let got = merge_rows(vec![
            (row("sha256:old", 5000, 10, Some(100)), false),
            (row("sha256:new", 5000, 20, Some(200)), false),
        ]);
        assert_eq!(got.len(), 2);

        // Two plain rows with the same name and digest also stay separate.
        let got = merge_rows(vec![
            (row("sha256:d1", 5000, 10, Some(100)), false),
            (row("sha256:d1", 5000, 20, Some(200)), false),
        ]);
        assert_eq!(got.len(), 2);

        // A plain runner and a shared adapter base with the same name and digest merge.
        let got = merge_rows(vec![
            (row("sha256:d1", 5000, 20, Some(100)), false),
            (row("sha256:d1", 5001, 10, None), true),
        ]);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].size, 10001);
        assert_eq!(got[0].loaded_at, t0 + Duration::from_secs(10));
        assert_eq!(got[0].expires_at, None);

        // A plain runner and a shared adapter base with different digests stay two rows.
        let got = merge_rows(vec![
            (row("sha256:old", 5000, 10, Some(100)), false),
            (row("sha256:new", 5000, 20, Some(200)), true),
        ]);
        assert_eq!(got.len(), 2);
    }
}
