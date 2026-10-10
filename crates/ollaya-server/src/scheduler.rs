//! The runner scheduler: one child process per loaded model.
//!
//! Requests lease a runner. When the last lease ends, the runner's keep-alive clock starts; a
//! reaper unloads runners whose clock ran out. Loads happen one at a time. When `max_loaded` is
//! reached, the least recently used idle runner is unloaded first. Killing a runner process
//! returns all of its memory, including GPU memory.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use serde::Deserialize;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

use ollaya_decision::{cygnet, jebadiah, jevk5, llm_logits, snap, winnow};
use ollaya_registry::pull::RunCheck;

use crate::Error;
use crate::models::{EngineFiles, Loadable};

/// How long a model stays loaded after its last request (the API's `keep_alive`).
pub use ollaya_api::KeepAlive;

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub keep_alive: KeepAlive,
    pub max_loaded: usize,
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
    child: tokio::sync::Mutex<Child>,
    http: reqwest::Client,
    leases: AtomicUsize,
    /// When an idle runner may be unloaded; `None` while leased or when kept forever.
    expires: Mutex<Option<Instant>>,
    keep_forever: Mutex<bool>,
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
    /// Forward a request to the runner process. `images` are base64 (vision models).
    pub async fn decide(
        &self,
        state: &Value,
        questions: &Value,
        images: &[String],
    ) -> Result<Value, Error> {
        let mut body = serde_json::json!({"state": state, "questions": questions});
        if !images.is_empty() {
            body["images"] = serde_json::json!(images);
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
}

/// A runner in use; dropping it starts the runner's keep-alive clock.
pub struct Lease {
    pub runner: Arc<Runner>,
    keep_alive: KeepAlive,
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
            "" | "onnx" | "router" => Ok(()),
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
    pub async fn acquire(
        &self,
        model: &Loadable,
        keep_alive: Option<KeepAlive>,
    ) -> Result<(Lease, Duration), Error> {
        let keep_alive = keep_alive.unwrap_or(self.config.keep_alive);
        if let Some(r) = self.lease_existing(&model.digest) {
            return Ok((
                Lease {
                    runner: r,
                    keep_alive,
                },
                Duration::ZERO,
            ));
        }
        let _guard = self.load_lock.lock().await;
        if let Some(r) = self.lease_existing(&model.digest) {
            return Ok((
                Lease {
                    runner: r,
                    keep_alive,
                },
                Duration::ZERO,
            ));
        }
        self.make_room().await;
        let started = Instant::now();
        let runner = Arc::new(self.spawn(model).await?);
        runner.leases.fetch_add(1, Ordering::SeqCst);
        self.runners
            .lock()
            .unwrap()
            .insert(model.digest.clone(), runner.clone());
        tracing::info!(model = %model.name, device = %runner.device, precision = %runner.precision,
            engine = %runner.engine, threads = self.config.threads,
            load_ms = started.elapsed().as_millis() as u64, "loaded model");
        Ok((Lease { runner, keep_alive }, started.elapsed()))
    }

    fn lease_existing(&self, digest: &str) -> Option<Arc<Runner>> {
        let runners = self.runners.lock().unwrap();
        let r = runners.get(digest)?.clone();
        r.leases.fetch_add(1, Ordering::SeqCst);
        *r.expires.lock().unwrap() = None;
        Some(r)
    }

    /// Unload a model now (the runner finishes in-flight requests first).
    pub async fn unload(&self, digest: &str) -> bool {
        let runner = self.runners.lock().unwrap().remove(digest);
        match runner {
            Some(r) => {
                while r.leases.load(Ordering::SeqCst) > 0 {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                r.kill().await;
                tracing::info!(model = %r.name, "unloaded model");
                true
            }
            None => false,
        }
    }

    pub fn running(&self) -> Vec<RunningInfo> {
        let now_i = Instant::now();
        let now_s = SystemTime::now();
        let mut out: Vec<RunningInfo> = self
            .runners
            .lock()
            .unwrap()
            .values()
            .map(|r| {
                let busy = r.leases.load(Ordering::SeqCst) > 0;
                let expires_at = if *r.keep_forever.lock().unwrap() {
                    None
                } else if busy {
                    Some(now_s + Duration::from_secs(0))
                } else {
                    r.expires
                        .lock()
                        .unwrap()
                        .map(|e| now_s + e.saturating_duration_since(now_i))
                };
                RunningInfo {
                    name: r.name.clone(),
                    digest: r.digest.clone(),
                    device: r.device.clone(),
                    precision: r.precision.clone(),
                    size: r.size,
                    loaded_at: r.loaded_at,
                    expires_at,
                }
            })
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
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
        let mut expired = Vec::new();
        for (digest, r) in self.runners.lock().unwrap().iter() {
            let idle = r.leases.load(Ordering::SeqCst) == 0;
            let due = r.expires.lock().unwrap().is_some_and(|e| e <= now);
            if idle && due && !*r.keep_forever.lock().unwrap() {
                expired.push(digest.clone());
            }
        }
        for digest in expired {
            self.unload(&digest).await;
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
            runners
                .iter()
                .filter(|(_, r)| r.leases.load(Ordering::SeqCst) == 0)
                .min_by_key(|(_, r)| *r.last_used.lock().unwrap())
                .map(|(d, _)| d.clone())
        };
        if let Some(digest) = victim {
            self.unload(&digest).await;
        }
    }

    async fn spawn(&self, model: &Loadable) -> Result<Runner, Error> {
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
                if let (Some(arch), Some(weights)) = (&f.arch, &f.weights) {
                    cmd.arg("--arch").arg(arch).arg("--weights").arg(weights);
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
        Ok(Runner {
            name: model.name.to_string(),
            digest: model.digest.clone(),
            device: hello.device,
            precision: hello.precision,
            engine: hello.engine,
            size: model.size,
            loaded_at: SystemTime::now(),
            port: hello.port,
            child: tokio::sync::Mutex::new(child),
            // Plain HTTP to 127.0.0.1: no TLS roots, so a host without CA certificates works.
            http: reqwest::Client::builder()
                .no_proxy()
                .tls_certs_only(std::iter::empty())
                .build()
                .map_err(|e| Error::LoadFailed(e.to_string()))?,
            leases: AtomicUsize::new(0),
            expires: Mutex::new(None),
            keep_forever: Mutex::new(false),
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
    use super::{Plan, plan, runner_warns};

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
}
