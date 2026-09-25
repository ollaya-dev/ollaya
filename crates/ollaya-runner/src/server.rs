//! The runner process: one loaded model behind a localhost HTTP endpoint.
//!
//! The daemon spawns `ollaya runner ...` per loaded model, as Ollama spawns a runner per model:
//! a crash or an out-of-memory in native code takes down one model, not the server, and killing
//! the process returns all of its GPU memory.
//!
//! Protocol (JSON over HTTP on 127.0.0.1, port chosen by the OS):
//! * On startup, after the model is loaded, the runner prints one JSON line to stdout:
//!   `{"port":<u16>,"device":"cuda:0"|"cpu"|"metal","precision":"fp16"|"fp32","engine":"onnx"|"mlx"}`.
//! * `GET /health` -> the same object plus `"status":"ok"`.
//! * `POST /decide` `{state, questions}` ->
//!   `{questions:[{logits, act_logits}], input_tokens, state_tokens, state_truncated}`.
//!   Errors are `{"error":{"code","message"}}` with status 400 (bad request) or 500.

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::engine::{self, Engine};
use crate::onnx::{Device, ModelFiles};
use crate::{Error, QuestionOutput};

/// Which device to try. `Auto` prefers MLX on the Metal GPU (builds with the `mlx` feature, models
/// with an arch layer), then CUDA, then the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceRequest {
    Auto,
    Cpu,
    Cuda(i32),
    Metal,
}

impl std::str::FromStr for DeviceRequest {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(DeviceRequest::Auto),
            "cpu" => Ok(DeviceRequest::Cpu),
            "cuda" => Ok(DeviceRequest::Cuda(0)),
            "metal" => Ok(DeviceRequest::Metal),
            s => s
                .strip_prefix("cuda:")
                .and_then(|n| n.parse().ok())
                .map(DeviceRequest::Cuda)
                .ok_or_else(|| {
                    format!("unknown device {s:?}; use auto, cpu, cuda, cuda:<n> or metal")
                }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// fp32 graph: used on CPU, and on GPU when no fp16 graph exists.
    pub graph_fp32: Option<PathBuf>,
    /// fp16 graph: preferred on GPU.
    pub graph_fp16: Option<PathBuf>,
    pub tokenizer: PathBuf,
    pub decision: PathBuf,
    /// The `arch` layer, for MLX.
    pub arch: Option<PathBuf>,
    /// The author's weights file, for MLX.
    pub weights: Option<PathBuf>,
    pub device: DeviceRequest,
    pub threads: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Loaded {
    pub device: String,
    pub precision: &'static str,
    /// `onnx` (ONNX Runtime) or `mlx`.
    pub engine: &'static str,
}

#[derive(Debug, Deserialize)]
struct DecideRequest {
    state: Value,
    questions: Value,
}

#[derive(Debug, Serialize)]
struct QuestionLogits {
    logits: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    act_logits: Option<Vec<f32>>,
}

struct AppState {
    model: Box<dyn Engine>,
    loaded: Loaded,
}

/// Load the model on the best available device.
pub fn load(config: &RunnerConfig) -> Result<(Box<dyn Engine>, Loaded), Error> {
    let files = |graph: &PathBuf| ModelFiles {
        graph: graph.clone(),
        tokenizer: config.tokenizer.clone(),
        decision: config.decision.clone(),
        calibration: None,
        arch: config.arch.clone(),
        weights: config.weights.clone(),
    };
    if config.device == DeviceRequest::Metal
        || (cfg!(feature = "mlx") && config.device == DeviceRequest::Auto)
    {
        let graph = config.graph_fp32.clone().unwrap_or_default();
        match metal(config, &files(&graph)) {
            // Warmed up here, as on CUDA: a failure on the first request moves `auto` on to the
            // next device instead of failing every request.
            Ok(model) => match warm_up(model.as_ref()) {
                Err(e) if config.device == DeviceRequest::Auto => {
                    tracing::warn!("MLX failed its first request, not using it: {e}");
                }
                warmed => {
                    if let Err(e) = warmed {
                        tracing::warn!("warm-up failed: {e}");
                    }
                    return Ok((
                        model,
                        Loaded {
                            device: "metal".into(),
                            precision: "fp32",
                            engine: "mlx",
                        },
                    ));
                }
            },
            Err(e) if config.device == DeviceRequest::Auto => {
                tracing::info!("not using MLX: {e}");
            }
            Err(e) => return Err(e),
        }
    }
    let gpu = match config.device {
        DeviceRequest::Cpu | DeviceRequest::Metal => None,
        DeviceRequest::Auto => Some(0),
        DeviceRequest::Cuda(id) => Some(id),
    };
    if let Some(id) = gpu {
        match cuda_providers_present() {
            Err(why) if config.device == DeviceRequest::Auto => {
                tracing::info!("GPU runtime not installed, using CPU: {why}");
            }
            Err(why) => return Err(Error::Model(why)),
            Ok(()) => {
                let (graph, precision) = match (&config.graph_fp16, &config.graph_fp32) {
                    (Some(g), _) => (g, "fp16"),
                    (None, Some(g)) => (g, "fp32"),
                    (None, None) => return Err(Error::Model("no graph given".into())),
                };
                match engine::load(&files(graph), Device::Cuda(id), config.threads) {
                    // A session can load on a GPU whose architecture this build has no CUDA
                    // kernels for (RTX 50-series with an ONNX Runtime built before sm_120), and
                    // then fail on its first request. Warm up here so `auto` can fall back to
                    // the CPU instead of failing every request.
                    Ok(model) => {
                        let warmed = warm_up(model.as_ref());
                        match after_gpu_warm_up(&warmed, config.device) {
                            AfterWarmUp::FallBackToCpu => {
                                if let Err(e) = warmed {
                                    tracing::warn!(
                                        "GPU {id} failed its first request, using CPU: {e}"
                                    );
                                }
                            }
                            AfterWarmUp::Fail => {
                                let e = warmed.err().map(|e| e.to_string()).unwrap_or_default();
                                return Err(Error::Model(format!(
                                    "GPU {id} failed its first request: {e}"
                                )));
                            }
                            AfterWarmUp::KeepGpu => {
                                if let Err(e) = warmed {
                                    tracing::warn!("warm-up failed: {e}");
                                }
                                return Ok((
                                    model,
                                    Loaded {
                                        device: format!("cuda:{id}"),
                                        precision,
                                        engine: "onnx",
                                    },
                                ));
                            }
                        }
                    }
                    Err(e) if config.device == DeviceRequest::Auto => {
                        tracing::info!("CUDA unavailable, using CPU: {e}");
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    }
    let graph = config
        .graph_fp32
        .as_ref()
        .or(config.graph_fp16.as_ref())
        .ok_or_else(|| Error::Model("no graph given".into()))?;
    let precision = if config.graph_fp32.is_some() {
        "fp32"
    } else {
        "fp16"
    };
    let model = engine::load(&files(graph), Device::Cpu, config.threads)?;
    Ok((
        model,
        Loaded {
            device: "cpu".into(),
            precision,
            engine: "onnx",
        },
    ))
}

/// The model on the Metal GPU through MLX: it needs an arch layer, a layout MLX implements and
/// the Metal library. Loading starts the MLX thread and its self-check.
#[cfg(feature = "mlx")]
fn metal(config: &RunnerConfig, files: &ModelFiles) -> Result<Box<dyn Engine>, Error> {
    let layout = engine::layout_of(&config.decision)?;
    crate::mlx::usable(config.arch.as_deref(), &layout).map_err(Error::Model)?;
    engine::load(files, Device::Metal, config.threads)
}

#[cfg(not(feature = "mlx"))]
fn metal(_config: &RunnerConfig, _files: &ModelFiles) -> Result<Box<dyn Engine>, Error> {
    Err(Error::Model(
        "this build of ollaya has no MLX support".into(),
    ))
}

/// ONNX Runtime loads its CUDA provider from its runtime path (see [`ort_runtime_dir`]), which
/// the daemon points at the CUDA runtime pack. Check before asking ORT, for a clear message.
fn cuda_providers_present() -> Result<(), String> {
    if !cfg!(feature = "cuda") {
        return Err("this build of ollaya has no CUDA support".into());
    }
    let dir = ort_runtime_dir()?;
    if dir.join(PROVIDERS_SHARED).is_file() {
        Ok(())
    } else {
        Err(format!(
            "no CUDA runtime in {} (install the GPU pack)",
            dir.display()
        ))
    }
}

/// An error that came from the CUDA provider rather than from the request: ONNX Runtime names
/// CUDA in those messages ("CUDA error cudaErrorNoKernelImageForDevice ...").
fn is_cuda_failure(e: &Error) -> bool {
    e.to_string().contains("CUDA")
}

/// What a model that loaded on a GPU does after its warm-up request.
#[derive(Debug, PartialEq)]
enum AfterWarmUp {
    KeepGpu,
    FallBackToCpu,
    Fail,
}

/// A CUDA failure on the first request moves an `auto` model to the CPU and fails an explicit
/// `cuda` one; anything else keeps the GPU, as before.
fn after_gpu_warm_up(warmed: &Result<(), Error>, device: DeviceRequest) -> AfterWarmUp {
    match warmed {
        Err(e) if is_cuda_failure(e) && device == DeviceRequest::Auto => AfterWarmUp::FallBackToCpu,
        Err(e) if is_cuda_failure(e) => AfterWarmUp::Fail,
        _ => AfterWarmUp::KeepGpu,
    }
}

#[cfg(windows)]
const PROVIDERS_SHARED: &str = "onnxruntime_providers_shared.dll";
#[cfg(not(windows))]
const PROVIDERS_SHARED: &str = "libonnxruntime_providers_shared.so";

/// The directory ONNX Runtime loads its provider libraries from (`Env::GetRuntimePath`). On
/// Windows that is the executable's own directory (`GetModuleFileNameW`); the daemon starts GPU
/// runners from a copy of the executable inside the pack.
#[cfg(windows)]
fn ort_runtime_dir() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate ollaya.exe: {e}"))?;
    exe.parent()
        .map(PathBuf::from)
        .ok_or_else(|| format!("{} has no parent directory", exe.display()))
}

/// The directory ONNX Runtime loads its provider libraries from (`Env::GetRuntimePath`). With ORT
/// linked into the executable, glibc's `dladdr` reports `argv[0]` for it, so that is the
/// directory of `argv[0]`, which the daemon sets for GPU runners.
#[cfg(not(windows))]
fn ort_runtime_dir() -> Result<PathBuf, String> {
    let arg0 = std::env::args_os()
        .next()
        .map(PathBuf::from)
        .unwrap_or_default();
    arg0.parent()
        .filter(|d| d.is_absolute())
        .map(PathBuf::from)
        .ok_or_else(|| "argv[0] is not an absolute path".into())
}

/// Run one small request before announcing readiness. The first run on a device pays one-off
/// costs (CUDA/cuDNN handles, kernel selection, arena growth) that would otherwise land on the
/// caller's first request.
fn warm_up(model: &dyn Engine) -> Result<(), Error> {
    let questions = serde_json::json!({
        "warm_up": {"type": "choice", "instructions": "Pick one.", "criteria": {"a": "first", "b": "second", "c": "third"}},
        "check": {"type": "noul", "instructions": "Is this a warm-up?"},
    });
    // A fixed-preset model answers only its own questions.
    let questions = match model.preset() {
        Some(preset) => Ok(preset.clone()),
        None => ollaya_decision::parse_questions(&questions),
    };
    match questions {
        Ok(q) => model
            .run(&Value::String("Warm-up request for the runner.".into()), &q)
            .map(drop),
        Err(_) => Ok(()),
    }
}

/// Load, bind, announce the port on stdout, and serve until killed.
pub async fn run(config: RunnerConfig) -> Result<(), Error> {
    let (model, loaded) = tokio::task::spawn_blocking(move || {
        let (model, loaded) = load(&config)?;
        // A GPU model was warmed up while it was chosen.
        if loaded.device == "cpu"
            && let Err(e) = warm_up(model.as_ref())
        {
            tracing::warn!("warm-up failed: {e}");
        }
        Ok::<_, Error>((model, loaded))
    })
    .await
    .map_err(|e| Error::Model(format!("load task failed: {e}")))??;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| Error::Model(e.to_string()))?;
    let port = listener
        .local_addr()
        .map_err(|e| Error::Model(e.to_string()))?
        .port();
    let hello = json!({
        "port": port,
        "device": loaded.device,
        "precision": loaded.precision,
        "engine": loaded.engine,
    });
    {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{hello}");
        let _ = out.flush();
    }
    let state = Arc::new(AppState { model, loaded });
    let app = Router::new()
        .route("/health", get(health))
        .route("/decide", post(decide))
        .with_state(state);
    axum::serve(listener, app)
        .await
        .map_err(|e| Error::Model(e.to_string()))
}

async fn health(State(s): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "device": s.loaded.device,
        "precision": s.loaded.precision,
        "engine": s.loaded.engine,
    }))
}

async fn decide(State(s): State<Arc<AppState>>, Json(req): Json<DecideRequest>) -> Response {
    let result = tokio::task::spawn_blocking(move || -> Result<Value, Error> {
        let questions = ollaya_decision::parse_questions(&req.questions)?;
        let out = s.model.run(&req.state, &questions)?;
        let questions: Vec<QuestionLogits> = out
            .questions
            .into_iter()
            .map(|QuestionOutput { logits, act_logits }| QuestionLogits { logits, act_logits })
            .collect();
        Ok(json!({
            "questions": questions,
            "input_tokens": out.input_tokens,
            "state_tokens": out.state_tokens,
            "state_truncated": out.state_truncated,
        }))
    })
    .await;
    match result {
        Ok(Ok(body)) => Json(body).into_response(),
        Ok(Err(Error::Decision(ollaya_decision::Error::TooManyOptions { question, options, head_max_len }))) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": {
                "code": "TOO_MANY_OPTIONS",
                "message": format!("question {question:?}: {options} options exceed the model's option budget"),
                "question": question, "options": options, "head_max_len": head_max_len,
            }})),
        )
            .into_response(),
        Ok(Err(Error::Decision(e))) => error(StatusCode::BAD_REQUEST, "INVALID_REQUEST", &e.to_string()),
        Ok(Err(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, "RUNNER_ERROR", &e.to_string()),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "RUNNER_ERROR", &e.to_string()),
    }
}

fn error(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(json!({"error": {"code": code, "message": message}})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_cuda_failure_on_the_first_request_moves_auto_to_the_cpu() {
        use super::{AfterWarmUp, DeviceRequest, after_gpu_warm_up};
        // The message ONNX Runtime gives on an RTX 5090 with kernels only up to sm_90 (issue #10).
        let no_kernel = || {
            Err(crate::Error::Model(
                "Non-zero status code returned while running Cast node. Status Message: CUDA error \
                 cudaErrorNoKernelImageForDevice:no kernel image is available for execution on the device"
                    .into(),
            ))
        };
        assert_eq!(
            after_gpu_warm_up(&no_kernel(), DeviceRequest::Auto),
            AfterWarmUp::FallBackToCpu
        );
        assert_eq!(
            after_gpu_warm_up(&no_kernel(), DeviceRequest::Cuda(0)),
            AfterWarmUp::Fail
        );
        assert_eq!(
            after_gpu_warm_up(&Ok(()), DeviceRequest::Auto),
            AfterWarmUp::KeepGpu
        );
        // A failure that has nothing to do with the GPU keeps the model where it is.
        let other = Err(crate::Error::Model("unexpected question type".into()));
        assert_eq!(
            after_gpu_warm_up(&other, DeviceRequest::Auto),
            AfterWarmUp::KeepGpu
        );
    }

    use std::sync::Mutex;

    use ollaya_decision::{Questions, parse_questions};
    use serde_json::{Value, json};

    use super::{DeviceRequest, warm_up};
    use crate::{Engine, Error, Output};

    #[test]
    fn devices_parse() {
        for (s, want) in [
            ("auto", DeviceRequest::Auto),
            ("cpu", DeviceRequest::Cpu),
            ("cuda", DeviceRequest::Cuda(0)),
            ("cuda:2", DeviceRequest::Cuda(2)),
            ("metal", DeviceRequest::Metal),
        ] {
            assert_eq!(s.parse::<DeviceRequest>(), Ok(want));
        }
        assert!("mps".parse::<DeviceRequest>().is_err());
    }

    /// Records the question ids of every run.
    struct Recorder {
        preset: Option<Questions>,
        asked: Mutex<Vec<Vec<String>>>,
    }

    impl Engine for Recorder {
        fn run(&self, _state: &Value, questions: &Questions) -> Result<Output, Error> {
            self.asked
                .lock()
                .unwrap()
                .push(questions.keys().cloned().collect());
            Ok(Output {
                questions: Vec::new(),
                input_tokens: 0,
                state_tokens: 0,
                state_truncated: false,
            })
        }

        fn preset(&self) -> Option<&Questions> {
            self.preset.as_ref()
        }
    }

    #[test]
    fn warm_up_asks_a_fixed_preset_model_its_own_questions() {
        let open = Recorder {
            preset: None,
            asked: Mutex::default(),
        };
        warm_up(&open).unwrap();
        assert_eq!(*open.asked.lock().unwrap(), [["warm_up", "check"]]);

        let preset = json!({"unsafe": {"type": "noul", "instructions": "Is it unsafe?"}});
        let guard = Recorder {
            preset: Some(parse_questions(&preset).unwrap()),
            asked: Mutex::default(),
        };
        warm_up(&guard).unwrap();
        assert_eq!(*guard.asked.lock().unwrap(), [["unsafe"]]);
    }
}
