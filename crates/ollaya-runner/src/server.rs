//! The runner process: one loaded model behind a localhost HTTP endpoint.
//!
//! The daemon spawns `ollaya runner ...` per loaded model, as Ollama spawns a runner per model:
//! a crash or an out-of-memory in native code takes down one model, not the server, and killing
//! the process returns all of its GPU memory.
//!
//! Protocol (JSON over HTTP on 127.0.0.1, port chosen by the OS):
//! * On startup, after the model is loaded, the runner prints one JSON line to stdout:
//!   `{"port":<u16>,"device":"cuda:0"|"cpu","precision":"fp16"|"fp32"}`.
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

/// Which device to try. `Auto` prefers CUDA and falls back to CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceRequest {
    Auto,
    Cpu,
    Cuda(i32),
}

impl std::str::FromStr for DeviceRequest {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(DeviceRequest::Auto),
            "cpu" => Ok(DeviceRequest::Cpu),
            "cuda" => Ok(DeviceRequest::Cuda(0)),
            s => s
                .strip_prefix("cuda:")
                .and_then(|n| n.parse().ok())
                .map(DeviceRequest::Cuda)
                .ok_or_else(|| format!("unknown device {s:?}; use auto, cpu, cuda or cuda:<n>")),
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
    pub device: DeviceRequest,
    pub threads: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Loaded {
    pub device: String,
    pub precision: &'static str,
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
    };
    let gpu = match config.device {
        DeviceRequest::Cpu => None,
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
                    Ok(model) => {
                        return Ok((
                            model,
                            Loaded {
                                device: format!("cuda:{id}"),
                                precision,
                            },
                        ));
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
        },
    ))
}

/// ONNX Runtime loads its CUDA provider from the directory of `argv[0]`; the daemon points
/// `argv[0]` at the CUDA runtime pack. Check before asking ORT, for a clear message.
fn cuda_providers_present() -> Result<(), String> {
    if !cfg!(feature = "cuda") {
        return Err("this build of ollaya has no CUDA support".into());
    }
    let arg0 = std::env::args_os()
        .next()
        .map(PathBuf::from)
        .unwrap_or_default();
    let dir = arg0
        .parent()
        .filter(|d| d.is_absolute())
        .ok_or("argv[0] is not an absolute path")?;
    if dir.join("libonnxruntime_providers_shared.so").is_file() {
        Ok(())
    } else {
        Err(format!(
            "no CUDA runtime in {} (install the GPU pack)",
            dir.display()
        ))
    }
}

/// Run one small request before announcing readiness. The first run on a device pays one-off
/// costs (CUDA/cuDNN handles, kernel selection, arena growth) that would otherwise land on the
/// caller's first request.
fn warm_up(model: &dyn Engine) {
    let questions = serde_json::json!({
        "warm_up": {"type": "choice", "instructions": "Pick one.", "criteria": {"a": "first", "b": "second", "c": "third"}},
        "check": {"type": "noul", "instructions": "Is this a warm-up?"},
    });
    // A fixed-preset model answers only its own questions.
    let questions = match model.preset() {
        Some(preset) => Ok(preset.clone()),
        None => ollaya_decision::parse_questions(&questions),
    };
    if let Ok(q) = questions
        && let Err(e) = model.run(&Value::String("Warm-up request for the runner.".into()), &q)
    {
        tracing::warn!("warm-up failed: {e}");
    }
}

/// Load, bind, announce the port on stdout, and serve until killed.
pub async fn run(config: RunnerConfig) -> Result<(), Error> {
    let (model, loaded) = tokio::task::spawn_blocking(move || {
        let (model, loaded) = load(&config)?;
        warm_up(model.as_ref());
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
    let hello = json!({"port": port, "device": loaded.device, "precision": loaded.precision});
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
    Json(json!({"status": "ok", "device": s.loaded.device, "precision": s.loaded.precision}))
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
    use std::sync::Mutex;

    use ollaya_decision::{Questions, parse_questions};
    use serde_json::{Value, json};

    use super::warm_up;
    use crate::{Engine, Error, Output};

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
        warm_up(&open);
        assert_eq!(*open.asked.lock().unwrap(), [["warm_up", "check"]]);

        let preset = json!({"unsafe": {"type": "noul", "instructions": "Is it unsafe?"}});
        let guard = Recorder {
            preset: Some(parse_questions(&preset).unwrap()),
            asked: Mutex::default(),
        };
        warm_up(&guard);
        assert_eq!(*guard.asked.lock().unwrap(), [["unsafe"]]);
    }
}
