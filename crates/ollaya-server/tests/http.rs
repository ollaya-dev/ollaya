//! The HTTP API end to end: a real daemon on a temp store, real runner processes, real sockets.
//!
//! The runner is this test binary itself: the scheduler spawns `<test exe> runner ...`, and `main`
//! answers the runner protocol (`crates/ollaya-runner/src/server.rs`) with deterministic logits
//! instead of a network. Everything else (routing, scheduling, validation, errors, streaming) is
//! the production code. Hence `harness = false`: this file brings its own `main`.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Duration;

use ollaya_api::{Client, ClientError, DecideRequest, ErrorCode, KeepAlive, PullRequest};
use ollaya_registry::manifest::{
    ANNOTATION_GRAPH, ANNOTATION_PRECISION, Descriptor, MANIFEST_V2, Manifest, media,
};
use ollaya_registry::store::sha256_hex;
use ollaya_registry::{ModelName, Store};
use ollaya_server::config::ServerConfig;
use ollaya_server::http::{RunnerLaunch, build, serve_on};
use serde_json::{Value, json};

// ------------------------------------------------------------------------------------------------
// The fake runner

mod fake_runner {
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};

    use axum::Json;
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};
    use axum::routing::{delete, get, post};
    use serde_json::{Value, json};

    /// `runner --decision <file> ... --device <device>`: behaviour comes from the decision
    /// layer's `fake` object.
    pub fn main(args: &[String]) {
        let arg = |name: &str| {
            args.iter()
                .position(|a| a == name)
                .and_then(|i| args.get(i + 1))
                .cloned()
        };
        let decision: Value =
            serde_json::from_str(&std::fs::read_to_string(arg("--decision").unwrap()).unwrap())
                .unwrap();
        let device = arg("--device").unwrap_or_else(|| "cpu".into());
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(serve(Arc::new(decision), device));
    }

    async fn serve(decision: Arc<Value>, device: String) {
        let fake = &decision["fake"];
        if fake["fail_load"].as_bool() == Some(true) {
            eprintln!("fake runner: cannot load this model");
            std::process::exit(1);
        }
        // A GPU backend that takes the process down while it loads (a driver fault, an abort in
        // ggml), before the runner can fall back to the CPU itself.
        if fake["abort_unless_cpu"].as_bool() == Some(true) && device != "cpu" {
            eprintln!("fake runner: the GPU backend aborted");
            std::process::exit(134);
        }
        // Log more than any pipe buffer holds before announcing, as a verbose runner does.
        for i in 0..fake["stderr_lines"].as_u64().unwrap_or(0) {
            eprintln!("fake runner: loading, log line {i} {}", "x".repeat(100));
        }
        tokio::time::sleep(std::time::Duration::from_millis(
            fake["load_ms"].as_u64().unwrap_or(0),
        ))
        .await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let device = if device == "auto" {
            "cpu".into()
        } else {
            device
        };
        // `bytes` is what a runner reports its base holds; an adapter's is 1000 (below).
        println!(
            "{}",
            json!({"port": port, "device": device, "precision": "fp32", "engine": "mlx", "bytes": 5000})
        );
        let d = decision.clone();
        // A runner whose queue is stuck: it takes the request and never answers.
        let (hang_post, hang_delete) = (
            fake["hang_adapter_post"].as_bool() == Some(true),
            fake["hang_adapter_delete"].as_bool() == Some(true),
        );
        let d_log: Option<&'static str> = fake["delete_log"]
            .as_str()
            .map(|s| &*Box::leak(s.to_owned().into_boxed_str()));
        let slow_ms = fake["slow_ms"].as_u64().unwrap_or(0);
        let slow_log: Option<&'static str> = fake["slow_log"]
            .as_str()
            .map(|s| &*Box::leak(s.to_owned().into_boxed_str()));
        let delete_ms = fake["delete_ms"].as_u64().unwrap_or(0);
        let loaded: Arc<Mutex<HashSet<String>>> = Default::default();
        let (on_decide, on_add, on_drop) = (loaded.clone(), loaded.clone(), loaded);
        let app = axum::Router::new()
            .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
            .route(
                "/decide",
                post(move |Json(req): Json<Value>| async move {
                    // A request for an adapter the runner does not hold is a 404, as the real one's.
                    if let Some(a) = req["adapter"].as_str()
                        && !on_decide.lock().unwrap().contains(a)
                    {
                        return error(
                            StatusCode::NOT_FOUND,
                            json!({"code": "ADAPTER_NOT_FOUND", "message": format!("no adapter {a}")}),
                        );
                    }
                    if slow_ms > 0 && req["state"].as_str().is_some_and(|s| s.contains("SLOW")) {
                        if let Some(log) = slow_log {
                            use std::io::Write as _;
                            let mut f = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(log)
                                .unwrap();
                            writeln!(f, "slow").unwrap();
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(slow_ms)).await;
                    }
                    decide(&d, &req)
                }),
            )
            .route(
                "/adapters",
                post(move |Json(req): Json<Value>| async move {
                    if hang_post {
                        std::future::pending::<()>().await;
                    }
                    // The runner gets the adapter's own files, which exist in the store.
                    for key in ["spec", "weights", "decision", "tokenizer"] {
                        let ok = req[key].as_str().is_some_and(|p| std::path::Path::new(p).is_file());
                        if !ok {
                            return error(
                                StatusCode::BAD_REQUEST,
                                json!({"code": "ADAPTER_LOAD_FAILED", "message": format!("no file for {key}")}),
                            );
                        }
                    }
                    let own: Value = serde_json::from_str(
                        &std::fs::read_to_string(req["decision"].as_str().unwrap()).unwrap(),
                    )
                    .unwrap();
                    if own["fake"]["fail_adapter"].as_bool() == Some(true) {
                        return error(
                            StatusCode::BAD_REQUEST,
                            json!({"code": "ADAPTER_LOAD_FAILED", "message": "fake adapter load failure"}),
                        );
                    }
                    on_add.lock().unwrap().insert(req["digest"].as_str().unwrap().into());
                    Json(json!({"bytes": 1000})).into_response()
                }),
            )
            .route(
                "/adapters/{digest}",
                delete(move |axum::extract::Path(digest): axum::extract::Path<String>| async move {
                    // The deletes the runner got, for a test to read.
                    if let Some(log) = d_log {
                        use std::io::Write as _;
                        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(log).unwrap();
                        writeln!(f, "{digest}").unwrap();
                    }
                    if hang_delete {
                        std::future::pending::<()>().await;
                    }
                    if delete_ms > 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(delete_ms)).await;
                    }
                    if on_drop.lock().unwrap().remove(&digest) {
                        Json(json!({"status": "ok"})).into_response()
                    } else {
                        error(StatusCode::NOT_FOUND, json!({"code": "ADAPTER_NOT_FOUND", "message": "no"}))
                    }
                }),
            );
        axum::serve(listener, app).await.unwrap();
    }

    fn error(status: StatusCode, err: Value) -> Response {
        (status, Json(json!({"error": err}))).into_response()
    }

    /// Option 0 always wins, with a margin that makes confidences non-trivial.
    fn decide(decision: &Value, req: &Value) -> Response {
        let max_options = decision["fake"]["max_options"].as_u64().unwrap_or(255) as usize;
        let questions = match ollaya_decision::parse_questions(&req["questions"]) {
            Ok(q) => q,
            Err(e) => {
                return error(
                    StatusCode::BAD_REQUEST,
                    json!({"code": "INVALID_REQUEST", "message": e.to_string()}),
                );
            }
        };
        let state = match &req["state"] {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        if state.contains("CRASH") {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"code": "RUNNER_ERROR", "message": "fake runner crashed"}),
            );
        }
        let act = decision["outputs"]
            .as_array()
            .is_some_and(|o| o.iter().any(|x| x == "act_logits"));
        let mut rows = Vec::new();
        let mut input_tokens = 0;
        for (qid, q) in &questions {
            let n = q.num_options();
            if n > max_options {
                return error(
                    StatusCode::BAD_REQUEST,
                    json!({"code": "TOO_MANY_OPTIONS", "message": "too many", "question": qid,
                           "options": n, "head_max_len": 192}),
                );
            }
            input_tokens += 10 + n;
            // The base's answer is option 0; an adapter's is option 1: the two are told apart.
            let win = usize::from(req["adapter"].is_string() && n > 1);
            let logits: Vec<f32> = (0..n)
                .map(|i| if i == win { 2.0 } else { -(i as f32) / 4.0 })
                .collect();
            let mut row = json!({"logits": logits});
            if act {
                row["act_logits"] = json!([1.0, 0.0]);
            }
            rows.push(row);
        }
        let state_tokens = state.split_whitespace().count();
        Json(json!({
            "questions": rows,
            "input_tokens": input_tokens + state_tokens.min(100) * questions.len(),
            "state_tokens": state_tokens,
            "state_truncated": state_tokens > 100,
        }))
        .into_response()
    }
}

// ------------------------------------------------------------------------------------------------
// Fixtures

fn blob(store: &Store, media_type: &str, bytes: &[u8], precision: Option<&str>) -> Descriptor {
    Descriptor {
        media_type: media_type.into(),
        digest: store.write_blob(bytes).unwrap(),
        size: bytes.len() as u64,
        urls: vec![],
        annotations: precision
            .map(|p| [(ANNOTATION_PRECISION.to_owned(), p.to_owned())].into())
            .unwrap_or_default(),
    }
}

fn decision_json(fake: Value) -> Value {
    json!({"engine": "onnx", "family": "laya", "layout": "laya-markers-v1",
           "encoder": "answerdotai/ModernBERT-large", "max_len": 512, "head_max_len": 192,
           "outputs": ["logits", "act_logits"], "fake": fake})
}

/// A runnable model: fp32 + fp16 graphs, tokenizer, decision layer, license.
fn add_model(store: &Store, name: &str, languages: &[&str], fake: Value) {
    let config = json!({"model_format": "onnx", "family": "laya", "parameter_size": "421M",
        "context_length": 512, "languages": languages, "description": format!("{name} for tests"),
        "release_date": "2026-09-20", "source": "huggingface.co/test/laya@abc", "license": "Apache-2.0"});
    let manifest = Manifest {
        schema_version: 2,
        media_type: MANIFEST_V2.into(),
        config: blob(store, media::CONFIG, config.to_string().as_bytes(), None),
        layers: vec![
            blob(
                store,
                media::GRAPH_ONNX,
                format!("{name} fp32").as_bytes(),
                Some("fp32"),
            ),
            blob(
                store,
                media::GRAPH_ONNX,
                format!("{name} fp16").as_bytes(),
                Some("fp16"),
            ),
            blob(store, media::TOKENIZER, b"{}", None),
            blob(
                store,
                media::DECISION,
                decision_json(fake).to_string().as_bytes(),
                None,
            ),
            blob(
                store,
                media::LICENSE,
                b"Apache License\nVersion 2.0\n",
                None,
            ),
        ],
    };
    let name = ModelName::parse(name).unwrap();
    store
        .write_manifest(&name, &serde_json::to_vec(&manifest).unwrap())
        .unwrap();
}

/// A safetensors file of float32 tensors `(name, shape)`, zero-filled.
fn safetensors(tensors: &[(&str, &[usize])]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut at = 0u64;
    for (name, shape) in tensors {
        let bytes = shape.iter().product::<usize>() as u64 * 4;
        header.insert(
            (*name).into(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [at, at + bytes]}),
        );
        at += bytes;
    }
    let header = Value::Object(header).to_string().into_bytes();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header);
    out.extend(vec![0u8; at as usize]);
    out
}

/// An encoder the MLX engine runs: an arch layer and a safetensors weights layer next to the
/// graph, as `laya:en` has. Its linears: `encoder.layers.0.attn.Wo` [8, 8] and
/// `encoder.layers.0.mlp.Wi` [32, 8].
fn add_mlx_model(store: &Store, name: &str, fake: Value) {
    add_mlx_encoder(store, name, fake, "modernbert");
}

/// `add_mlx_model` with another backbone type in the arch layer.
fn add_mlx_encoder(store: &Store, name: &str, fake: Value, backbone: &str) {
    let config = json!({"model_format": "onnx", "family": "laya", "parameter_size": "421M",
        "context_length": 512, "languages": ["en"], "description": format!("{name} for tests"),
        "license": "Apache-2.0"});
    let arch = json!({"schema": 1, "backbone": {"type": backbone, "prefix": "encoder."},
        "head": {"type": "laya"}, "weights": {"format": "safetensors", "file": "model.safetensors"}});
    let weights = safetensors(&[
        ("encoder.layers.0.attn.Wo.weight", &[8, 8]),
        ("encoder.layers.0.mlp.Wi.weight", &[32, 8]),
        ("encoder.norm.weight", &[8]),
    ]);
    let manifest = Manifest {
        schema_version: 2,
        media_type: MANIFEST_V2.into(),
        config: blob(store, media::CONFIG, config.to_string().as_bytes(), None),
        layers: vec![
            blob(
                store,
                media::GRAPH_ONNX,
                format!("{name} fp32").as_bytes(),
                Some("fp32"),
            ),
            blob(store, media::TOKENIZER, b"{}", None),
            blob(
                store,
                media::DECISION,
                decision_json(fake).to_string().as_bytes(),
                None,
            ),
            blob(store, media::ARCH, arch.to_string().as_bytes(), None),
            blob(store, media::WEIGHTS, &weights, None),
        ],
    };
    store
        .write_manifest(
            &ModelName::parse(name).unwrap(),
            &serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
}

/// A peft adapter on `add_mlx_model`'s linears, rank 4.
fn adapter_weights() -> Vec<u8> {
    safetensors(&[
        ("base_model.model.layers.0.attn.Wo.lora_A.weight", &[4, 8]),
        ("base_model.model.layers.0.attn.Wo.lora_B.weight", &[8, 4]),
        ("base_model.model.layers.0.mlp.Wi.lora_A.weight", &[4, 8]),
        ("base_model.model.layers.0.mlp.Wi.lora_B.weight", &[32, 4]),
    ])
}

fn adapter_config(alpha: u32) -> Value {
    json!({"peft_type": "LORA", "r": 4, "lora_alpha": alpha, "use_rslora": false,
           "use_dora": false, "base_model_name_or_path": "answerdotai/ModernBERT-large"})
}

fn sha(bytes: &[u8]) -> String {
    format!("sha256:{}", sha256_hex(bytes))
}

/// `POST /api/blobs/:digest`, as the CLI uploads an adapter.
async fn upload(d: &Daemon, bytes: &[u8]) -> String {
    let digest = sha(bytes);
    let r = http()
        .post(d.url(&format!("/api/blobs/{digest}")))
        .body(bytes.to_vec())
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "{}", r.text().await.unwrap());
    digest
}

async fn create_adapter(
    d: &Daemon,
    name: &str,
    from: &str,
    config: Value,
    weights: &str,
) -> (u16, Value) {
    let (status, _, body) = post_raw(
        &d.url("/api/create"),
        json!({"model": name, "from": from, "stream": false,
               "adapter": {"config": config, "weights": weights}})
        .to_string(),
    )
    .await;
    (status, body)
}

/// A daemon with an MLX encoder `laya:en` and the adapter blob uploaded (device `metal`: adapter
/// models run on MLX only; the fake runner takes any device).
async fn adapter_daemon(tweak: impl FnOnce(&mut ServerConfig)) -> (Daemon, String) {
    let d = Daemon::start(
        |dir| add_mlx_model(&Store::open(dir).unwrap(), "laya:en", json!({})),
        |c| {
            c.device = "metal".into();
            tweak(c);
        },
    )
    .await;
    let weights = upload(&d, &adapter_weights()).await;
    (d, weights)
}

/// A GGUF model, which runs on llama.cpp: the GGUF file and its decision layer.
fn add_gguf_model(store: &Store, name: &str, fake: Value) {
    add_gguf_media_model(store, name, fake, false, false);
}

/// A GGUF model with a projector (`vision`), and audio opted in (`audio`).
fn add_gguf_media_model(store: &Store, name: &str, fake: Value, vision: bool, audio: bool) {
    let config = json!({"model_format": "gguf", "family": "winnow", "layout": "winnow-v1",
        "parameter_size": "4B", "description": format!("{name} for tests"), "license": "Apache-2.0"});
    let decision = json!({"engine": "llama", "family": "winnow", "layout": "winnow-v1",
        "llama": {"audio": audio}, "fake": fake});
    let mut layers = vec![
        blob(store, media::GGUF, format!("{name} gguf").as_bytes(), None),
        blob(
            store,
            media::DECISION,
            decision.to_string().as_bytes(),
            None,
        ),
    ];
    if vision {
        layers.push(blob(store, media::MMPROJ, b"projector", None));
    }
    let manifest = Manifest {
        schema_version: 2,
        media_type: MANIFEST_V2.into(),
        config: blob(store, media::CONFIG, config.to_string().as_bytes(), None),
        layers,
    };
    let name = ModelName::parse(name).unwrap();
    store
        .write_manifest(&name, &serde_json::to_vec(&manifest).unwrap())
        .unwrap();
}

fn add_router(store: &Store) {
    let config = json!({"model_format": "router", "family": "laya", "languages": ["en", "multilingual"],
        "description": "laya router"});
    let router = json!({"strategy": "script", "default": "english",
        "routes": {"english": "laya:en", "multilingual": "laya:multilingual"}});
    let manifest = Manifest {
        schema_version: 2,
        media_type: MANIFEST_V2.into(),
        config: blob(store, media::CONFIG, config.to_string().as_bytes(), None),
        layers: vec![blob(
            store,
            media::ROUTER,
            router.to_string().as_bytes(),
            None,
        )],
    };
    let name = ModelName::parse("laya").unwrap();
    store
        .write_manifest(&name, &serde_json::to_vec(&manifest).unwrap())
        .unwrap();
}

/// laya (router), laya:en, laya:multilingual.
fn laya_store(dir: &Path) {
    let store = Store::open(dir).unwrap();
    add_model(&store, "laya:en", &["en"], json!({}));
    add_model(&store, "laya:multilingual", &["multilingual"], json!({}));
    add_router(&store);
}

struct Daemon {
    base: String,
    client: Client,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    handle: tokio::task::JoinHandle<()>,
    _dir: tempfile::TempDir,
}

impl Daemon {
    async fn start(setup: impl FnOnce(&Path), tweak: impl FnOnce(&mut ServerConfig)) -> Daemon {
        let dir = tempfile::tempdir().unwrap();
        setup(dir.path());
        let models = dir.path().display().to_string();
        let mut config =
            ServerConfig::from_lookup(|k| (k == "OLLAYA_MODELS").then(|| models.clone())).unwrap();
        config.device = "cpu".into();
        tweak(&mut config);
        let runner = RunnerLaunch {
            exe: std::env::current_exe().unwrap(),
            arg0: None,
            env: vec![],
            // The fake runner needs no llama.cpp; GGUF models need the directory to be set.
            llama_dir: Some(dir.path().join("llama")),
            cpu_exe: None,
        };
        let state = build(config.clone(), runner).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            serve_on(listener, state, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });
        let mut client = Client::new(&base).unwrap();
        if let Some(key) = &config.api_key {
            client = client.with_api_key(key);
        }
        Daemon {
            base,
            client,
            stop: Some(tx),
            handle,
            _dir: dir,
        }
    }

    async fn laya() -> Daemon {
        Daemon::start(laya_store, |_| {}).await
    }

    async fn stop(mut self) {
        let _ = self.stop.take().unwrap().send(());
        let _ = (&mut self.handle).await;
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }
}

fn http() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

async fn post_raw(
    url: &str,
    body: impl Into<reqwest::Body>,
) -> (u16, reqwest::header::HeaderMap, Value) {
    let r = http().post(url).body(body).send().await.unwrap();
    let (status, headers) = (r.status().as_u16(), r.headers().clone());
    let text = r.text().await.unwrap();
    (
        status,
        headers,
        serde_json::from_str(&text).unwrap_or(Value::String(text)),
    )
}

fn api_err(e: ClientError) -> (u16, ollaya_api::ErrorBody) {
    match e {
        ClientError::Api { status, body } => (status, body),
        other => panic!("expected an API error, got {other:?}"),
    }
}

fn triage() -> Value {
    json!({
        "department": {"type": "choice", "instructions": "Which team?",
                       "criteria": {"billing": "Payments", "technical": "Bugs", "account": "Login"}},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "normal", "high"]},
        "refund": {"type": "noul", "instructions": "Asks for a refund."}
    })
}

// ------------------------------------------------------------------------------------------------
// Tests

async fn liveness_version_and_request_ids() {
    let d = Daemon::laya().await;
    let r = http().get(d.url("/")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert!(
        r.headers()["x-request-id"]
            .to_str()
            .unwrap()
            .starts_with("req_")
    );
    assert_eq!(r.text().await.unwrap(), "Ollaya is running");
    let r = http().head(d.url("/")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    d.client.heartbeat().await.unwrap();
    assert_eq!(
        d.client.version().await.unwrap().version,
        ollaya_server::config::VERSION
    );
    let r = http()
        .get(d.url("/v1/models"))
        .header("x-request-id", "trace-42")
        .send()
        .await
        .unwrap();
    assert_eq!(r.headers()["x-request-id"], "trace-42");
    assert_eq!(r.headers()["x-typesafe-request-id"], "trace-42");
    d.stop().await;
}

async fn adapter_models_create_decide_share_a_runner_and_delete() {
    let (d, weights) = adapter_daemon(|_| {}).await;
    let base = d.client.show("laya:en").await.unwrap();
    for (name, alpha) in [("ft-a", 8), ("ft-b", 16)] {
        let (status, body) =
            create_adapter(&d, name, "laya:en", adapter_config(alpha), &weights).await;
        assert_eq!(
            (status, &body["status"]),
            (200, &json!("success")),
            "{body}"
        );
    }
    let shown = d.client.show("ft-a").await.unwrap();
    assert_eq!(shown.details.format, "adapter");
    assert_eq!(shown.details.parent_model, "laya:en");
    // The manifest keeps the base's layers and adds the adapter's two.
    let store = Store::open(d._dir.path()).unwrap();
    let m = store
        .read_manifest(&ModelName::parse("ft-a").unwrap())
        .unwrap()
        .unwrap()
        .manifest;
    assert!(m.layer(media::ARCH).is_some() && m.layer(media::ADAPTER).is_some());
    assert_eq!(m.layer(media::ADAPTER_WEIGHTS).unwrap().digest, weights);
    let spec: ollaya_registry::adapter::AdapterSpec = store
        .read_blob_json(m.layer(media::ADAPTER).unwrap())
        .unwrap();
    assert_eq!((spec.rank, spec.scaling), (4, 2.0));
    assert_eq!(
        spec.targets,
        ["encoder.layers.0.attn.Wo", "encoder.layers.0.mlp.Wi"]
    );
    // The same adapter on the same base writes the same spec blob (canonical form).
    let (status, _) = create_adapter(&d, "ft-a2", "laya:en", adapter_config(8), &weights).await;
    assert_eq!(status, 200);
    let m2 = store
        .read_manifest(&ModelName::parse("ft-a2").unwrap())
        .unwrap()
        .unwrap()
        .manifest;
    assert_eq!(
        m2.layer(media::ADAPTER).unwrap().digest,
        m.layer(media::ADAPTER).unwrap().digest
    );

    // Decide: each adapter answers as an adapter (option 1 in the fake), under its own name.
    let client = &d.client;
    let ask = |model: &'static str| async move {
        client
            .decide(&DecideRequest::new(
                model,
                json!("An invoice question."),
                serde_json::from_value(triage()).ok(),
            ))
            .await
            .unwrap()
    };
    let a = ask("ft-a").await;
    assert_eq!(a.model, "ft-a:latest");
    let b = ask("ft-b").await;
    assert_eq!(b.model, "ft-b:latest");
    assert_eq!(
        a.answers["department"], b.answers["department"],
        "the fake answers alike"
    );

    // One runner for both: the base's row, and one row for each adapter, which add up.
    let ps = d.client.ps().await.unwrap().models;
    let names: Vec<_> = ps.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        ["ft-a:latest", "ft-b:latest", "laya:en"],
        "{names:?}"
    );
    let row = |n: &str| ps.iter().find(|m| m.name == n).unwrap();
    assert_eq!(
        (
            row("laya:en").size,
            row("ft-a:latest").size,
            row("ft-b:latest").size
        ),
        (5000, 1000, 1000)
    );
    assert_eq!(row("ft-a:latest").details.parent_model, "laya:en");
    assert_eq!(
        row("laya:en").digest,
        base.digest.trim_start_matches("sha256:")
    );
    assert_ne!(row("ft-a:latest").digest, row("ft-b:latest").digest);

    // A plain model stays keyed by its manifest digest: asking for the base itself loads a
    // second runner, with its own copy of the base.
    let plain = ask("laya:en").await;
    assert_eq!(plain.model, "laya:en");
    // Two runners, but `ps` shows one row per name: `laya:en` once, with both copies counted.
    let ps = d.client.ps().await.unwrap().models;
    let names: Vec<_> = ps.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        ["ft-a:latest", "ft-b:latest", "laya:en"],
        "{names:?}"
    );
    assert!(
        ps.iter().find(|m| m.name == "laya:en").unwrap().size > 5000,
        "the plain runner's copy and the shared base's"
    );
    // `stop laya:en` reaches both runners, and the adapters that were on the shared one go.
    d.client.unload("laya:en").await.unwrap();
    assert!(d.client.ps().await.unwrap().models.is_empty());
    ask("ft-a").await;
    ask("ft-b").await;
    ask("laya:en").await;
    assert_eq!(d.client.ps().await.unwrap().models.len(), 3);

    // Deleting an adapter model drops it from its runner; the base and the other adapter stay.
    d.client.delete("ft-a").await.unwrap();
    let names: Vec<_> = d
        .client
        .ps()
        .await
        .unwrap()
        .models
        .into_iter()
        .map(|m| m.name)
        .collect();
    assert!(
        !names.contains(&"ft-a:latest".to_owned()) && names.contains(&"ft-b:latest".to_owned()),
        "{names:?}"
    );
    ask("ft-b").await;
    // Unloading the last adapter frees the runner.
    d.client.unload("ft-b").await.unwrap();
    let names: Vec<_> = d
        .client
        .ps()
        .await
        .unwrap()
        .models
        .into_iter()
        .map(|m| m.name)
        .collect();
    assert_eq!(
        names,
        ["laya:en"],
        "only the plain base's own runner is left"
    );
    d.stop().await;
}

async fn adapter_refusals_name_the_problem_and_write_nothing() {
    let (d, weights) = adapter_daemon(|_| {}).await;
    let store = Store::open(d._dir.path()).unwrap();
    let before = store.list().unwrap().len();
    let refuse = |status: u16, body: &Value, what: &str| {
        assert_eq!(status, 422, "{body}");
        let text = body.to_string();
        assert!(text.contains(what), "{what:?} not in {text}");
    };
    // modules_to_save in the config.
    let mut config = adapter_config(8);
    config["modules_to_save"] = json!(["classifier"]);
    let (status, body) = create_adapter(&d, "x1", "laya:en", config, &weights).await;
    refuse(status, &body, "modules_to_save");
    // A trained head among the tensors.
    let head = safetensors(&[
        ("base_model.model.layers.0.attn.Wo.lora_A.weight", &[4, 8]),
        ("base_model.model.layers.0.attn.Wo.lora_B.weight", &[8, 4]),
        (
            "base_model.model.classifier.modules_to_save.default.weight",
            &[2, 8],
        ),
    ]);
    let head = upload(&d, &head).await;
    let (status, body) = create_adapter(&d, "x2", "laya:en", adapter_config(8), &head).await;
    refuse(status, &body, "not a LoRA pair");
    // A target module the base does not have, and a shape that is not the base's.
    let unknown = upload(
        &d,
        &safetensors(&[
            ("base_model.model.layers.7.attn.Wo.lora_A.weight", &[4, 8]),
            ("base_model.model.layers.7.attn.Wo.lora_B.weight", &[8, 4]),
        ]),
    )
    .await;
    let (status, body) = create_adapter(&d, "x3", "laya:en", adapter_config(8), &unknown).await;
    refuse(status, &body, "layers.7.attn.Wo");
    let wide = upload(
        &d,
        &safetensors(&[
            ("base_model.model.layers.0.mlp.Wi.lora_A.weight", &[4, 16]),
            ("base_model.model.layers.0.mlp.Wi.lora_B.weight", &[32, 4]),
        ]),
    )
    .await;
    let (status, body) = create_adapter(&d, "x4", "laya:en", adapter_config(8), &wide).await;
    refuse(status, &body, "[4, 8]");
    // Offsets that run past the file, run backwards, or overlap (peft files are trusted by
    // nobody: the runner subtracts them).
    let lora = |at: [(u64, u64); 2]| {
        let mut header = serde_json::Map::new();
        for (name, shape, (b, e)) in [
            (
                "base_model.model.layers.0.attn.Wo.lora_A.weight",
                [4, 8],
                at[0],
            ),
            (
                "base_model.model.layers.0.attn.Wo.lora_B.weight",
                [8, 4],
                at[1],
            ),
        ] {
            header.insert(
                name.into(),
                json!({"dtype": "F32", "shape": shape, "data_offsets": [b, e]}),
            );
        }
        let header = Value::Object(header).to_string().into_bytes();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend(header);
        out.extend(vec![0u8; 256]);
        out
    };
    for (i, (at, what)) in [
        ([(128, 0), (128, 256)], "reversed"),
        ([(0, 128), (128, 9999)], "run past"),
        ([(0, 128), (64, 192)], "overlap"),
        ([(0, 100), (128, 256)], "bytes for shape"),
    ]
    .into_iter()
    .enumerate()
    {
        let blob = upload(&d, &lora(at)).await;
        let (status, body) =
            create_adapter(&d, &format!("off{i}"), "laya:en", adapter_config(8), &blob).await;
        refuse(status, &body, what);
    }
    // Options the engine does not run, and weights that were never uploaded.
    let mut config = adapter_config(8);
    config["use_dora"] = json!(true);
    let (status, body) = create_adapter(&d, "x5", "laya:en", config, &weights).await;
    refuse(status, &body, "use_dora");
    let (status, body) = create_adapter(
        &d,
        "x6",
        "laya:en",
        adapter_config(8),
        &sha(b"never uploaded"),
    )
    .await;
    refuse(status, &body, "upload them first");
    // A base with no arch layer, a router, an adapter model; PARAMETER precision.
    let (status, body) = create_adapter(&d, "x7", "laya:en", adapter_config(8), &weights).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = create_adapter(&d, "x8", "x7", adapter_config(8), &weights).await;
    refuse(status, &body, "do not stack");
    let (status, _, body) = post_raw(
        &d.url("/api/create"),
        json!({"model": "x9", "from": "laya:en", "parameters": {"precision": "fp32"}, "stream": false,
               "adapter": {"config": adapter_config(8), "weights": weights}}).to_string(),
    ).await;
    refuse(status, &body, "precision");
    // Nothing was written by any refusal: only `x7`, and the blobs uploaded for them.
    assert_eq!(store.list().unwrap().len(), before + 1);
    d.stop().await;

    // A DeBERTa-v2 base: its position projections ignore the adapter, so it is refused.
    let d = Daemon::start(
        |dir| add_mlx_encoder(&Store::open(dir).unwrap(), "deb", json!({}), "deberta-v2"),
        |c| c.device = "metal".into(),
    )
    .await;
    let weights = upload(&d, &adapter_weights()).await;
    let (status, body) = create_adapter(&d, "z", "deb", adapter_config(8), &weights).await;
    refuse(status, &body, "not supported on DeBERTa-v2");
    d.stop().await;

    // A base with no arch layer (ONNX only).
    let d = Daemon::start(
        |dir| add_model(&Store::open(dir).unwrap(), "onnx-only", &["en"], json!({})),
        |c| c.device = "metal".into(),
    )
    .await;
    let weights = upload(&d, &adapter_weights()).await;
    let (status, body) = create_adapter(&d, "y", "onnx-only", adapter_config(8), &weights).await;
    refuse(status, &body, "no arch layer");
    d.stop().await;
}

async fn adapter_models_run_on_mlx_only_and_fail_alone() {
    // On the CPU: a message that says why, no runner, the base untouched.
    let (d, weights) = adapter_daemon(|c| c.device = "cpu".into()).await;
    let (status, _) = create_adapter(&d, "ft", "laya:en", adapter_config(8), &weights).await;
    assert_eq!(status, 200);
    let err = d
        .client
        .decide(&DecideRequest::new(
            "ft",
            json!("x"),
            serde_json::from_value(triage()).ok(),
        ))
        .await
        .unwrap_err();
    let (status, body) = api_err(err);
    assert_eq!((status, body.code), (500, ErrorCode::ModelLoadFailed));
    assert!(body.error.contains("MLX"), "{}", body.error);
    assert!(d.client.ps().await.unwrap().models.is_empty());
    d.stop().await;

    // A load that fails because of one adapter leaves the base and the other adapters loaded.
    let d = Daemon::start(
        |dir| add_mlx_model(&Store::open(dir).unwrap(), "laya:en", json!({})),
        |c| c.device = "metal".into(),
    )
    .await;
    let weights = upload(&d, &adapter_weights()).await;
    create_adapter(&d, "good", "laya:en", adapter_config(8), &weights).await;
    // A third adapter model whose own decision.json makes the runner refuse it.
    let store = Store::open(d._dir.path()).unwrap();
    create_adapter(&d, "bad", "laya:en", adapter_config(16), &weights).await;
    let name = ModelName::parse("bad").unwrap();
    let mut m = store.read_manifest(&name).unwrap().unwrap().manifest;
    m.layers.retain(|l| l.media_type != media::DECISION);
    m.layers.push(blob(
        &store,
        media::DECISION,
        decision_json(json!({"fail_adapter": true}))
            .to_string()
            .as_bytes(),
        None,
    ));
    store
        .write_manifest(&name, &serde_json::to_vec(&m).unwrap())
        .unwrap();
    let client = &d.client;
    let ask = |model: &'static str| async move {
        client
            .decide(&DecideRequest::new(
                model,
                json!("x"),
                serde_json::from_value(triage()).ok(),
            ))
            .await
    };
    ask("good").await.unwrap();
    let (status, body) = api_err(ask("bad").await.unwrap_err());
    assert_eq!((status, body.code), (500, ErrorCode::ModelLoadFailed));
    assert!(
        body.error.contains("fake adapter load failure"),
        "{}",
        body.error
    );
    let names: Vec<_> = d
        .client
        .ps()
        .await
        .unwrap()
        .models
        .into_iter()
        .map(|m| m.name)
        .collect();
    assert_eq!(names, ["good:latest", "laya:en"], "{names:?}");
    ask("good").await.unwrap();
    d.stop().await;
}

async fn the_adapter_limit_drops_the_least_recently_used_idle_adapter() {
    let (d, weights) = adapter_daemon(|c| c.max_adapters = 2).await;
    for (i, name) in ["a", "b", "c"].into_iter().enumerate() {
        create_adapter(&d, name, "laya:en", adapter_config(8 + i as u32), &weights).await;
    }
    let client = &d.client;
    let ask = |model: &'static str| async move {
        client
            .decide(&DecideRequest::new(
                model,
                json!("x"),
                serde_json::from_value(triage()).ok(),
            ))
            .await
    };
    ask("a").await.unwrap();
    ask("b").await.unwrap();
    ask("a").await.unwrap();
    // `b` is the least recently used: it goes, nothing else, and no process dies.
    ask("c").await.unwrap();
    let names: Vec<_> = d
        .client
        .ps()
        .await
        .unwrap()
        .models
        .into_iter()
        .map(|m| m.name)
        .collect();
    assert_eq!(names, ["a:latest", "c:latest", "laya:en"], "{names:?}");
    // It loads again when asked for, dropping the then least recently used (`a`).
    ask("b").await.unwrap();
    let names: Vec<_> = d
        .client
        .ps()
        .await
        .unwrap()
        .models
        .into_iter()
        .map(|m| m.name)
        .collect();
    assert_eq!(names, ["b:latest", "c:latest", "laya:en"], "{names:?}");
    d.stop().await;
}

/// A daemon whose `laya:en` has `fake` in its decision layer; adapters inherit it.
async fn fake_adapter_daemon(fake: Value, load_timeout: Duration) -> (Daemon, String) {
    let d = Daemon::start(
        |dir| add_mlx_model(&Store::open(dir).unwrap(), "laya:en", fake),
        |c| {
            c.device = "metal".into();
            c.load_timeout = load_timeout;
        },
    )
    .await;
    let weights = upload(&d, &adapter_weights()).await;
    (d, weights)
}

async fn a_runner_that_never_answers_does_not_hold_the_load_lock_for_ever() {
    // Registering an adapter: the runner takes the request and never answers.
    let (d, weights) = fake_adapter_daemon(
        json!({"hang_adapter_post": true}),
        Duration::from_millis(800),
    )
    .await;
    create_adapter(&d, "a", "laya:en", adapter_config(8), &weights).await;
    let ask = |model: &'static str| {
        let client = &d.client;
        async move {
            tokio::time::timeout(
                Duration::from_secs(20),
                client.decide(&DecideRequest::new(
                    model,
                    json!("x"),
                    serde_json::from_value(triage()).ok(),
                )),
            )
            .await
            .expect("the load lock was held with no timeout")
        }
    };
    let (status, body) = api_err(ask("a").await.unwrap_err());
    assert_eq!((status, body.code), (500, ErrorCode::ModelLoadFailed));
    assert!(body.error.contains("load timeout"), "{}", body.error);
    // Every other load still goes: a plain model needs the same lock.
    ask("laya:en").await.unwrap();
    d.stop().await;

    // After the timeout the daemon tells the runner to drop the adapter, in case it finishes
    // loading it later and would serve it outside the count.
    let log = std::env::temp_dir().join(format!("ollaya-deletes-{}", std::process::id()));
    let _ = std::fs::remove_file(&log);
    let (d, weights) = fake_adapter_daemon(
        json!({"hang_adapter_post": true, "delete_log": log.display().to_string()}),
        Duration::from_millis(800),
    )
    .await;
    create_adapter(&d, "a", "laya:en", adapter_config(8), &weights).await;
    let req = DecideRequest::new("a", json!("x"), serde_json::from_value(triage()).ok());
    d.client.decide(&req).await.unwrap_err();
    let digest = Store::open(d._dir.path())
        .unwrap()
        .read_manifest(&ModelName::parse("a").unwrap())
        .unwrap()
        .unwrap()
        .digest;
    let deleted = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_file(&log);
    assert!(
        deleted.contains(digest.trim_start_matches("sha256:")) || deleted.contains(&digest),
        "no DELETE /adapters/<digest> after the timeout: {deleted:?}"
    );
    d.stop().await;

    // Dropping an adapter: the unload gives up on the runner, forgets the adapter, and returns.
    let (d, weights) = fake_adapter_daemon(
        json!({"hang_adapter_delete": true}),
        Duration::from_millis(800),
    )
    .await;
    create_adapter(&d, "a", "laya:en", adapter_config(8), &weights).await;
    let req = DecideRequest::new("a", json!("x"), serde_json::from_value(triage()).ok());
    d.client.decide(&req).await.unwrap();
    tokio::time::timeout(Duration::from_secs(20), d.client.unload("a"))
        .await
        .expect("the unload waited for the runner with no timeout")
        .unwrap();
    let names: Vec<_> = d
        .client
        .ps()
        .await
        .unwrap()
        .models
        .into_iter()
        .map(|m| m.name)
        .collect();
    assert!(!names.contains(&"a:latest".to_owned()), "{names:?}");
    d.stop().await;
}

async fn the_base_stops_being_kept_when_its_keep_forever_adapter_goes() {
    let (d, weights) = adapter_daemon(|_| {}).await;
    // Different alphas, so that the two are two models (one digest each).
    for (name, alpha) in [("a", 8), ("b", 16)] {
        create_adapter(&d, name, "laya:en", adapter_config(alpha), &weights).await;
    }
    let mut keep = DecideRequest::new("a", json!("x"), serde_json::from_value(triage()).ok());
    keep.keep_alive = Some(KeepAlive::Forever);
    d.client.decide(&keep).await.unwrap();
    let req = DecideRequest::new("b", json!("x"), serde_json::from_value(triage()).ok());
    d.client.decide(&req).await.unwrap();
    let expires = |ps: Vec<ollaya_api::RunningModel>, name: &str| {
        ps.into_iter().find(|m| m.name == name).unwrap().expires_at
    };
    let ps = d.client.ps().await.unwrap().models;
    assert_eq!(
        expires(ps.clone(), "laya:en"),
        None,
        "-1 on `a` keeps the base"
    );
    assert!(expires(ps, "b:latest").is_some());
    // `a` goes; the base's clock is now `b`'s.
    d.client.unload("a").await.unwrap();
    let ps = d.client.ps().await.unwrap().models;
    assert!(
        expires(ps, "laya:en").is_some(),
        "the base was kept for ever by an adapter that is gone"
    );
    d.stop().await;
}

async fn an_expiring_plain_runner_leaves_the_shared_base_alone_and_stop_takes_it() {
    let (d, weights) = adapter_daemon(|_| {}).await;
    create_adapter(&d, "a", "laya:en", adapter_config(8), &weights).await;
    let mut keep = DecideRequest::new("a", json!("x"), serde_json::from_value(triage()).ok());
    keep.keep_alive = Some(KeepAlive::Forever);
    d.client.decide(&keep).await.unwrap();
    // The plain `laya:en` is a second runner of the same name, and expires after a second.
    let mut brief =
        DecideRequest::new("laya:en", json!("x"), serde_json::from_value(triage()).ok());
    brief.keep_alive = Some(KeepAlive::For(Duration::from_secs(1)));
    d.client.decide(&brief).await.unwrap();
    let size =
        |ps: &[ollaya_api::RunningModel]| ps.iter().find(|m| m.name == "laya:en").map(|m| m.size);
    assert!(size(&d.client.ps().await.unwrap().models) > Some(5000));
    // Wait for the reaper to take the plain runner (its copy no longer counted).
    let mut waited = 0;
    while size(&d.client.ps().await.unwrap().models) != Some(5000) {
        waited += 1;
        assert!(waited < 100, "the plain runner never expired");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let names: Vec<_> = d
        .client
        .ps()
        .await
        .unwrap()
        .models
        .into_iter()
        .map(|m| m.name)
        .collect();
    assert_eq!(
        names,
        ["a:latest", "laya:en"],
        "the shared base went with it: {names:?}"
    );
    // The user's `stop laya:en` does take the shared base, with its adapters.
    d.client.unload("laya:en").await.unwrap();
    assert!(d.client.ps().await.unwrap().models.is_empty());
    d.stop().await;
}

async fn blobs_follow_the_api_key_are_verified_and_pruned() {
    let d = Daemon::start(laya_store, |c| c.api_key = Some("secret".into())).await;
    let bytes = b"some adapter bytes".to_vec();
    let digest = sha(&bytes);
    let url = d.url(&format!("/api/blobs/{digest}"));
    let anon = http();
    assert_eq!(anon.head(&url).send().await.unwrap().status(), 401);
    assert_eq!(
        anon.post(&url)
            .body(bytes.clone())
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
    let keyed = |r: reqwest::RequestBuilder| r.bearer_auth("secret");
    assert_eq!(keyed(anon.head(&url)).send().await.unwrap().status(), 404);
    // A body that is not the digest's is refused and leaves nothing behind.
    let r = keyed(anon.post(&url))
        .body(b"other bytes".to_vec())
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 422);
    assert!(r.text().await.unwrap().contains("digest mismatch"));
    let blobs = Store::open(d._dir.path()).unwrap().blobs_dir();
    let leftovers = |blobs: &Path| {
        std::fs::read_dir(blobs)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains("partial")
            })
            .count()
    };
    assert_eq!(leftovers(&blobs), 0);
    assert_eq!(keyed(anon.head(&url)).send().await.unwrap().status(), 404);
    // Not a digest at all.
    let r = keyed(anon.post(d.url("/api/blobs/nonsense")))
        .body("x")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 422);
    // The right bytes are stored, also when sent twice.
    for _ in 0..2 {
        assert_eq!(
            keyed(anon.post(&url))
                .body(bytes.clone())
                .send()
                .await
                .unwrap()
                .status(),
            201
        );
    }
    assert_eq!(keyed(anon.head(&url)).send().await.unwrap().status(), 200);
    // No manifest references it: it falls under the pruning that every removal runs.
    d.client.copy("laya:en", "copy").await.unwrap();
    d.client.delete("copy").await.unwrap();
    assert_eq!(keyed(anon.head(&url)).send().await.unwrap().status(), 404);

    // The client the CLI uses: check, upload, check again; a wrong digest is an API error.
    assert!(!d.client.blob_exists(&digest).await.unwrap());
    d.client
        .upload_blob(&digest, reqwest::Body::from(bytes.clone()))
        .await
        .unwrap();
    assert!(d.client.blob_exists(&digest).await.unwrap());
    let (status, _) = api_err(
        d.client
            .upload_blob(&sha(b"x"), reqwest::Body::from(bytes))
            .await
            .unwrap_err(),
    );
    assert_eq!(status, 422);
    d.stop().await;
}

async fn a_cancelled_delete_finishes_unloading_and_does_not_block_later_loads() {
    let log_dir = tempfile::tempdir().unwrap();
    let slow_log = log_dir.path().join("slow.log");
    let (d, weights) = fake_adapter_daemon(
        json!({"slow_ms": 600, "slow_log": slow_log.display().to_string()}),
        Duration::from_millis(800),
    )
    .await;
    create_adapter(&d, "a", "laya:en", adapter_config(8), &weights).await;
    create_adapter(&d, "b", "laya:en", adapter_config(16), &weights).await;
    d.client
        .decide(&DecideRequest::new(
            "a",
            json!("x"),
            serde_json::from_value(triage()).ok(),
        ))
        .await
        .unwrap();

    // Hold a lease on `a` inside `/decide`, then send a DELETE that disconnects early.
    let slow = tokio::spawn({
        let client = Client::new(&d.base).unwrap();
        async move {
            client
                .decide(&DecideRequest::new(
                    "a",
                    json!("SLOW"),
                    serde_json::from_value(triage()).ok(),
                ))
                .await
        }
    });
    for _ in 0..300 {
        if std::fs::read_to_string(&slow_log)
            .unwrap_or_default()
            .contains("slow")
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        std::fs::read_to_string(&slow_log)
            .unwrap_or_default()
            .contains("slow"),
        "slow decision never reached the runner"
    );

    let impatient = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_millis(150))
        .build()
        .unwrap();
    let res = impatient
        .delete(d.url("/api/delete"))
        .json(&json!({"model": "a"}))
        .send()
        .await;
    assert!(res.is_err(), "impatient delete should have timed out");

    slow.await.unwrap().unwrap();
    // Wait until `store.remove` finishes in the spawned delete task.
    let mut removed = false;
    for _ in 0..40 {
        let tags = d.client.tags().await.unwrap().models;
        if !tags.iter().any(|m| m.name == "a:latest") {
            removed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(removed, "cancelled delete left `a:latest` in the store");

    // Later loads (both `b` and re-creating/loading `a`) must not hang on `Slot::Unloading`.
    tokio::time::timeout(
        Duration::from_secs(2),
        d.client.decide(&DecideRequest::new(
            "b",
            json!("x"),
            serde_json::from_value(triage()).ok(),
        )),
    )
    .await
    .expect("load of `b` hung after cancelled delete")
    .unwrap();
    let (status, _) = create_adapter(&d, "a", "laya:en", adapter_config(8), &weights).await;
    assert_eq!(status, 200);
    tokio::time::timeout(
        Duration::from_secs(2),
        d.client.decide(&DecideRequest::new(
            "a",
            json!("x"),
            serde_json::from_value(triage()).ok(),
        )),
    )
    .await
    .expect("reload of `a` hung after cancelled delete")
    .unwrap();
    d.stop().await;
}

async fn unknown_paths_methods_and_reserved() {
    let d = Daemon::laya().await;
    let r = http().get(d.url("/api/nope")).send().await.unwrap();
    assert_eq!(r.status(), 404);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["code"], "NOT_FOUND");
    let (status, _, body) = post_raw(&d.url("/api/generate"), "{}").await;
    assert_eq!(status, 404);
    assert!(
        body["error"].as_str().unwrap().contains("/api/decide"),
        "{body}"
    );
    let r = http().get(d.url("/api/decide")).send().await.unwrap();
    assert_eq!(r.status(), 405);
    assert!(r.headers().contains_key("allow"), "{:?}", r.headers());
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["code"], "METHOD_NOT_ALLOWED");
    let (status, _, body) = post_raw(&d.url("/api/push"), "{}").await;
    assert_eq!(
        (status, body["code"].as_str()),
        (501, Some("NOT_IMPLEMENTED"))
    );
    d.stop().await;
}

async fn bodies_are_json_whatever_the_content_type() {
    let d = Daemon::laya().await;
    // curl -d sends application/x-www-form-urlencoded: still parsed as JSON.
    let r = http()
        .post(d.url("/api/show"))
        .header("content-type", "application/x-www-form-urlencoded")
        .body(r#"{"model": "laya:en"}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let (status, _, body) = post_raw(&d.url("/api/decide"), "{not json").await;
    assert_eq!((status, body["code"].as_str()), (400, Some("INVALID_JSON")));
    let (status, _, body) = post_raw(&d.url("/api/decide"), "").await;
    assert_eq!(
        (status, body["error"].as_str()),
        (400, Some("missing request body"))
    );
    let big = vec![b' '; ollaya_api::MAX_BODY_BYTES + 1];
    let (status, _, body) = post_raw(&d.url("/api/decide"), big).await;
    assert_eq!(
        (status, body["code"].as_str()),
        (413, Some("REQUEST_TOO_LARGE"))
    );
    // TypeSafe-shaped 422 with every issue.
    let (status, headers, body) = post_raw(
        &d.url("/v1/systemone"),
        r#"{"model": "laya", "questions": {"urgency": {"type": "score", "criteria": ["only"]}}}"#,
    )
    .await;
    assert_eq!(status, 422);
    assert!(headers.contains_key("x-typesafe-request-id"));
    assert_eq!(
        body,
        json!({
            "error": "state: Field required; questions.urgency.score.criteria: List should have at least 2 items after validation, not 1",
            "code": "INVALID_REQUEST",
            "detail": [
                {"loc": ["body", "state"], "msg": "Field required", "type": "missing"},
                {"loc": ["body", "questions", "urgency", "score", "criteria"],
                 "msg": "List should have at least 2 items after validation, not 1", "type": "too_short",
                 "ctx": {"field_type": "List", "min_length": 2, "actual_length": 1}}
            ]
        })
    );
    let (status, _, body) = post_raw(&d.url("/api/show"), r#"{"model": "a/b/c/d"}"#).await;
    assert_eq!(status, 422);
    assert_eq!(body["detail"][0]["type"], "model_name");
    d.stop().await;
}

async fn unknown_models_are_404_with_ollama_text() {
    let d = Daemon::laya().await;
    let (status, body) = api_err(d.client.show("nope").await.unwrap_err());
    assert_eq!((status, &body.code), (404, &ErrorCode::ModelNotFound));
    assert_eq!(
        body.error,
        "model \"nope:latest\" not found, try pulling it first"
    );
    let (status, _, body) = post_raw(
        &d.url("/v1/systemone"),
        json!({"model": "jev-latest", "state": "x", "questions": triage()}).to_string(),
    )
    .await;
    assert_eq!(status, 404);
    assert_eq!(
        body["error"],
        "model \"jev-latest:latest\" not found, try pulling it first"
    );
    // A router whose target is missing names the target.
    d.client.delete("laya:en").await.unwrap();
    let english = DecideRequest::new(
        "laya",
        json!("Hello there, my friend."),
        serde_json::from_value(triage()).ok(),
    );
    let (status, body) = api_err(d.client.decide(&english).await.unwrap_err());
    assert_eq!(status, 404);
    assert_eq!(
        body.error,
        "model \"laya:en\" not found, try pulling it first (routed from \"laya:latest\")"
    );
    let (status, body) = api_err(d.client.delete("laya:en").await.unwrap_err());
    assert_eq!((status, body.code), (404, ErrorCode::ModelNotFound));
    d.stop().await;
}

async fn tags_show_ps_and_v1_models() {
    let d = Daemon::laya().await;
    let tags = d.client.tags().await.unwrap();
    let names: Vec<&str> = tags.models.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names.len(), 3);
    assert!(
        tags.models
            .windows(2)
            .all(|w| w[0].modified_at >= w[1].modified_at)
    );
    let en = tags.models.iter().find(|m| m.name == "laya:en").unwrap();
    assert_eq!(en.details.quantization_level, "F16/F32");
    assert_eq!(en.digest.len(), 64);
    let router = tags
        .models
        .iter()
        .find(|m| m.name == "laya:latest")
        .unwrap();
    assert_eq!(
        (
            router.details.format.as_str(),
            router.details.quantization_level.as_str()
        ),
        ("router", "")
    );

    let show = d.client.show("laya:en").await.unwrap();
    assert_eq!(show.capabilities, ["choice", "score", "noul", "act"]);
    assert_eq!(show.model_info["laya.context_length"], 512);
    assert_eq!(show.model_info["general.languages"], json!(["en"]));
    assert!(show.license.starts_with("Apache License"));
    assert!(show.modelfile.contains("FROM laya:en"));
    let show = d.client.show("laya").await.unwrap();
    let r = show.router.unwrap();
    assert_eq!(
        (r.strategy.as_str(), r.default.as_str()),
        ("script", "english")
    );
    assert_eq!(r.routes["multilingual"], "laya:multilingual");

    let models = d.client.models().await.unwrap();
    let names: Vec<&str> = models.models.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["laya:en", "laya:latest", "laya:multilingual"]);
    assert_eq!(models.models[0].release_date, "2026-09-20");
    assert_eq!(
        models.models[1].release_date.len(),
        10,
        "falls back to the pull date"
    );
    assert!(d.client.ps().await.unwrap().models.is_empty());
    d.stop().await;
}

async fn decide_routes_answers_and_unloads() {
    let d = Daemon::laya().await;
    let questions = serde_json::from_value(triage()).unwrap();
    let en = d
        .client
        .decide(&DecideRequest::new(
            "laya",
            json!(
                "Hi, how much would we save by switching to the annual plan? No rush, just curious."
            ),
            Some(questions),
        ))
        .await
        .unwrap();
    assert_eq!(en.model, "laya:en");
    let routing = en.routing.as_ref().unwrap();
    assert_eq!(
        (
            routing.router.as_str(),
            routing.model.as_str(),
            routing.route.as_str(),
            routing.reason.as_str()
        ),
        ("laya:latest", "laya:en", "english", "English Latin text")
    );
    assert!(en.load_duration > 0 && en.total_duration >= en.eval_duration);
    let a = serde_json::to_value(&en.answers).unwrap();
    assert_eq!(a["department"]["choice"], "billing");
    assert_eq!(
        a["department"]["probabilities"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        ["billing", "technical", "account"]
    );
    assert_eq!(a["urgency"]["legend"]["2"], "high");
    assert!(a["refund"].get("confidence").is_none());
    assert!(a["refund"].get("laya").is_none());

    let mut req = DecideRequest::new(
        "laya",
        json!(
            "Mart faturasında iki kez ücret alınmış. Bugün iade edilmezse aboneliğimizi iptal edeceğiz!"
        ),
        serde_json::from_value(triage()).ok(),
    );
    req.extras = vec![ollaya_api::Extra::Laya, ollaya_api::Extra::Logits];
    req.keep_alive = Some(KeepAlive::Forever);
    let tr = d.client.decide(&req).await.unwrap();
    assert_eq!(tr.model, "laya:multilingual");
    assert_eq!(tr.routing.as_ref().unwrap().route, "multilingual");
    let laya = tr.answers["department"].laya.as_ref().unwrap();
    assert_eq!(laya.act_probability.map(|p| p > 0.5), Some(true));
    // The option logits before the temperature, one per option.
    for (qid, a) in &tr.answers {
        let logits = a.logits.as_ref().unwrap();
        let options = req.questions.as_ref().unwrap()[qid].num_options();
        assert_eq!(logits.len(), options, "{qid}");
    }

    let ps = d.client.ps().await.unwrap();
    let names: Vec<&str> = ps.models.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["laya:en", "laya:multilingual"]);
    let ml = &ps.models[1];
    assert_eq!(
        (
            ml.device.as_str(),
            ml.details.quantization_level.as_str(),
            ml.size_vram
        ),
        ("cpu", "F32", 0)
    );
    assert_eq!(ml.expires_at, None, "keep_alive -1 keeps it forever");
    assert!(ps.models[0].expires_at.is_some());

    // Unload the router: both targets go.
    let r = d.client.unload("laya").await.unwrap();
    assert_eq!(
        (r.model.as_str(), r.done_reason),
        ("laya:latest", ollaya_api::DoneReason::Unload)
    );
    assert!(d.client.ps().await.unwrap().models.is_empty());
    // Load without deciding.
    let r = d.client.load("laya:en", None).await.unwrap();
    assert_eq!(r.done_reason, ollaya_api::DoneReason::Load);
    assert!(r.load_duration > 0 && r.answers.is_empty());
    d.stop().await;
}

async fn preload_loads_models_before_ready() {
    let ready = |d: &Daemon| {
        let url = d.url("/api/ready");
        async move { http().get(url).send().await.unwrap().status().as_u16() }
    };
    let plain = Daemon::laya().await;
    assert_eq!(
        ready(&plain).await,
        200,
        "nothing to preload: ready at once"
    );
    plain.stop().await;

    let d = Daemon::start(
        |dir| {
            let store = Store::open(dir).unwrap();
            add_model(&store, "slow:latest", &["en"], json!({"load_ms": 500}));
        },
        |c| c.preload = vec!["slow".into(), "nope:missing".into()],
    )
    .await;
    let mut statuses = vec![ready(&d).await];
    while statuses.last() != Some(&200) && statuses.len() < 300 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        statuses.push(ready(&d).await);
    }
    assert_eq!(statuses.first(), Some(&503), "{statuses:?}");
    assert_eq!(
        statuses.last(),
        Some(&200),
        "a failed preload does not block readiness"
    );
    let loaded = d.client.ps().await.unwrap();
    assert_eq!(loaded.models.len(), 1, "{loaded:?}");
    let r = d
        .client
        .decide(&DecideRequest::new(
            "slow",
            json!("Refund me, please."),
            Some(serde_json::from_value(triage()).unwrap()),
        ))
        .await
        .unwrap();
    assert_eq!(r.load_duration, 0, "warm right after ready");
    let kept = d.client.ps().await.unwrap();
    assert_eq!(
        kept.models[0].expires_at, None,
        "a request without keep_alive leaves a preloaded model loaded: {kept:?}"
    );
    d.stop().await;
}

async fn a_passed_deadline_skips_the_decision() {
    let d = Daemon::start(
        |dir| {
            let store = Store::open(dir).unwrap();
            add_model(&store, "slow:latest", &["en"], json!({"load_ms": 400}));
        },
        |_| {},
    )
    .await;
    let body = json!({"model": "slow", "state": "Refund me.",
                      "questions": {"refund": {"type": "noul", "instructions": "Refund?"}}})
    .to_string();
    let send = |timeout: &str| {
        http()
            .post(d.url("/v1/systemone"))
            .header("x-ollaya-timeout-ms", timeout)
            .body(body.clone())
            .send()
    };
    let late = send("100").await.unwrap();
    assert_eq!(late.status().as_u16(), 504);
    let late: Value = late.json().await.unwrap();
    assert_eq!(late["code"], "DEADLINE_EXCEEDED", "{late}");
    let bad = send("soon").await.unwrap();
    assert_eq!(bad.status().as_u16(), 422);
    let ok = send("10000").await.unwrap();
    assert_eq!(
        ok.status().as_u16(),
        200,
        "loaded now, well within the deadline"
    );
    d.stop().await;
}

async fn systemone_is_typesafe_shaped() {
    let d = Daemon::laya().await;
    let (status, headers, body) = post_raw(
        &d.url("/v1/decisions"),
        json!({"model": "laya:en", "state": {"subject": "Invoice", "message": "Need an invoice"},
               "questions": {"tone": {"type": "choice", "criteria": {"calm": null, "angry": "Upset"}}},
               "keep_alive": "bogus-ignored-on-v1"})
        .to_string(),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert!(headers.contains_key("x-typesafe-request-id"));
    let nanos = |name: &str| -> u64 { headers[name].to_str().unwrap().parse().unwrap() };
    let (eval, load, total) = (
        nanos("x-ollaya-eval-duration"),
        nanos("x-ollaya-load-duration"),
        nanos("x-ollaya-total-duration"),
    );
    assert!(eval > 0 && total >= eval + load, "{eval} {load} {total}");
    let keys: Vec<&String> = body.as_object().unwrap().keys().collect();
    assert_eq!(keys, ["model", "answers", "usage"]);
    assert_eq!(body["model"], "laya:en");
    assert_eq!(body["answers"]["tone"]["choice"], "calm");
    let answer_keys: Vec<&String> = body["answers"]["tone"]
        .as_object()
        .unwrap()
        .keys()
        .collect();
    assert_eq!(
        answer_keys,
        ["type", "choice", "confidence", "probabilities"]
    );
    assert_eq!(body["usage"]["output_tokens"], 0);

    // The request shows in the activity log, newest first, and so does a failed one.
    let _ = d
        .client
        .decide(&DecideRequest::new("nope", json!("hi"), None))
        .await;
    let log = d.client.activity().await.unwrap();
    assert_eq!(log.total, 2);
    let [failed, answered] = &log.requests[..] else {
        panic!("{log:?}")
    };
    assert_eq!(failed.error, Some(ErrorCode::ModelNotFound));
    assert_eq!(
        (answered.endpoint.as_str(), answered.model.as_str()),
        ("/v1/decisions", "laya:en")
    );
    assert_eq!((answered.questions, answered.error.clone()), (1, None));
    d.stop().await;
}

async fn systemone_rejects_truncated_state() {
    let d = Daemon::laya().await;
    let request = |state: String| {
        json!({"model": "laya:en", "state": state, "questions": triage()}).to_string()
    };
    for path in ["/v1/systemone", "/v1/decisions"] {
        let (status, _, body) = post_raw(&d.url(path), request("word ".repeat(100))).await;
        assert_eq!(status, 200, "{path}: {body}");
        assert_eq!(body["model"], "laya:en");
        assert!(body.get("state_truncated").is_none());

        let (status, headers, body) = post_raw(
            &d.url(path),
            request(format!("{}refund", "word ".repeat(100))),
        )
        .await;
        assert_eq!(status, 422, "{path}: {body}");
        assert_eq!(headers["x-typesafe-request-id"], headers["x-request-id"]);
        assert_eq!(
            body,
            json!({
                "error": "state: part of state was dropped to fit the context of laya:en",
                "code": "STATE_TRUNCATED",
                "detail": [{
                    "loc": ["body", "state"],
                    "msg": "part of state was dropped to fit the context of laya:en",
                    "type": "state_truncated",
                    "ctx": {"model": "laya:en"}
                }]
            }),
            "{path}"
        );
    }

    let (status, _, body) = post_raw(
        &d.url("/api/decide"),
        request(format!("{}refund", "word ".repeat(100))),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["state_truncated"], true);
    d.stop().await;
}

async fn model_specific_errors() {
    let d = Daemon::start(
        |dir| {
            let store = Store::open(dir).unwrap();
            add_model(&store, "small:latest", &["en"], json!({"max_options": 100}));
            add_model(&store, "broken:latest", &["en"], json!({"fail_load": true}));
            add_model(
                &store,
                "chatty:latest",
                &["en"],
                json!({"stderr_lines": 10_000}),
            );
        },
        |c| c.load_timeout = Duration::from_secs(30),
    )
    .await;
    // Media the model cannot read are refused before it loads: `broken` would fail to load.
    for (field, what) in [("images", "images"), ("audio", "audio")] {
        let (status, _, body) = post_raw(
            &d.url("/api/decide"),
            json!({"model": "broken", "state": "x", "questions": triage(), field: ["AAAA"]})
                .to_string(),
        )
        .await;
        assert_eq!(status, 422, "{body}");
        assert_eq!(
            body["detail"][0]["msg"],
            format!("broken:latest does not read {what}")
        );
    }
    // About 1 MB of stderr before the runner announces itself: the load must not stall on a
    // full pipe.
    d.client.load("chatty", None).await.unwrap();
    let options: serde_json::Map<String, Value> =
        (0..120).map(|i| (format!("o{i}"), Value::Null)).collect();
    let (status, _, body) = post_raw(
        &d.url("/api/decide"),
        json!({"model": "small", "state": "x", "questions": {"intent": {"type": "choice", "criteria": options}}}).to_string(),
    )
    .await;
    assert_eq!(status, 422, "{body}");
    assert_eq!(body["code"], "TOO_MANY_OPTIONS");
    assert_eq!(
        body["detail"][0]["loc"],
        json!(["body", "questions", "intent", "choice", "criteria"])
    );
    assert_eq!(
        body["detail"][0]["ctx"],
        json!({"options": 120, "model": "small:latest"})
    );

    let long = "word ".repeat(ollaya_api::MAX_STATE_TOKENS + 1);
    let (status, _, body) = post_raw(
        &d.url("/v1/systemone"),
        json!({"model": "small", "state": long, "questions": triage()}).to_string(),
    )
    .await;
    assert_eq!(
        (status, body["code"].as_str()),
        (422, Some("INPUT_TOO_LONG"))
    );

    let (status, _, body) = post_raw(
        &d.url("/v1/systemone"),
        json!({"model": "small", "state": "x"}).to_string(),
    )
    .await;
    assert_eq!(status, 422);
    assert_eq!(
        body["detail"],
        json!([{"loc": ["body", "questions"], "msg": "Field required", "type": "missing"}])
    );

    let (status, _, body) = post_raw(
        &d.url("/api/decide"),
        json!({"model": "small", "state": "CRASH", "questions": triage()}).to_string(),
    )
    .await;
    assert_eq!(
        (status, body["code"].as_str()),
        (500, Some("INFERENCE_FAILED"))
    );

    let r = d
        .client
        .decide(&DecideRequest::new(
            "small",
            json!("fine"),
            serde_json::from_value(triage()).ok(),
        ))
        .await
        .unwrap();
    assert!(!r.state_truncated);
    let r = d
        .client
        .decide(&DecideRequest::new(
            "small",
            json!("w ".repeat(150)),
            serde_json::from_value(triage()).ok(),
        ))
        .await
        .unwrap();
    assert!(r.state_truncated);

    let (status, body) = api_err(d.client.load("broken", None).await.unwrap_err());
    assert_eq!((status, body.code), (500, ErrorCode::ModelLoadFailed));
    assert!(
        body.error.contains("cannot load this model"),
        "{}",
        body.error
    );
    d.stop().await;
}

async fn auth_origin_and_host() {
    let d = Daemon::start(laya_store, |c| {
        c.api_key = Some("s3cret".into());
        c.origins = vec!["https://*.example.com".into()];
    })
    .await;
    // Liveness needs no key.
    assert_eq!(http().get(d.url("/")).send().await.unwrap().status(), 200);
    assert_eq!(
        http()
            .get(d.url("/api/ready"))
            .send()
            .await
            .unwrap()
            .status(),
        200,
        "a health check reads readiness without the key"
    );
    let r = http().get(d.url("/api/tags")).send().await.unwrap();
    assert_eq!(r.status(), 401);
    assert_eq!(r.headers()["www-authenticate"], "Bearer");
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["code"], "UNAUTHORIZED");
    d.client.tags().await.unwrap(); // the client sends the key
    let bearer = |r: reqwest::RequestBuilder| r.header("authorization", "Bearer s3cret");

    let r = bearer(
        http()
            .get(d.url("/api/tags"))
            .header("origin", "http://localhost:5173"),
    )
    .send()
    .await
    .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(
        r.headers()["access-control-allow-origin"],
        "http://localhost:5173"
    );
    let r = bearer(
        http()
            .get(d.url("/api/tags"))
            .header("origin", "https://app.example.com"),
    )
    .send()
    .await
    .unwrap();
    assert_eq!(r.status(), 200);
    let r = bearer(
        http()
            .get(d.url("/api/tags"))
            .header("origin", "https://evil.com"),
    )
    .send()
    .await
    .unwrap();
    assert_eq!(r.status(), 403);
    let r = http()
        .request(reqwest::Method::OPTIONS, d.url("/api/decide"))
        .header("origin", "http://127.0.0.1:3000")
        .header("access-control-request-method", "POST")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 204, "preflight needs no key");
    assert!(
        r.headers()["access-control-allow-headers"]
            .to_str()
            .unwrap()
            .contains("Authorization")
    );

    // DNS rebinding: a foreign Host on a loopback-bound server.
    let addr = d.base.trim_start_matches("http://").to_owned();
    let mut sock = tokio::net::TcpStream::connect(&addr).await.unwrap();
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    sock.write_all(b"GET /api/version HTTP/1.1\r\nHost: attacker.example:11435\r\nAuthorization: Bearer s3cret\r\nConnection: close\r\n\r\n").await.unwrap();
    let mut resp = String::new();
    sock.read_to_string(&mut resp).await.unwrap();
    assert!(resp.starts_with("HTTP/1.1 403"), "{resp}");
    d.stop().await;
}

async fn queue_bound_and_cancellation() {
    let d = Daemon::start(
        |dir| {
            let store = Store::open(dir).unwrap();
            add_model(&store, "slow:latest", &["en"], json!({"load_ms": 1500}));
        },
        |c| c.max_queue = 1,
    )
    .await;
    // A client that gives up while the model loads.
    let impatient = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_millis(300))
        .build()
        .unwrap();
    let body = json!({"model": "slow", "state": "x", "questions": triage()}).to_string();
    let url = d.url("/api/decide");
    let first = tokio::spawn({
        let (impatient, url, body) = (impatient.clone(), url.clone(), body.clone());
        async move { impatient.post(url).body(body).send().await }
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    // The single slot is taken: the next request is refused at once.
    let (status, headers, body_json) = post_raw(&url, body.clone()).await;
    assert_eq!(status, 503, "{body_json}");
    assert_eq!(body_json["code"], "QUEUE_FULL");
    assert_eq!(headers["retry-after"], "1");
    assert!(
        first.await.unwrap().is_err(),
        "the impatient client timed out"
    );
    // The load went on without the client: the model ends up loaded.
    let mut loaded = false;
    for _ in 0..40 {
        if !d.client.ps().await.unwrap().models.is_empty() {
            loaded = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(loaded, "the load continued after the client left");
    let r = d
        .client
        .decide(&DecideRequest::new(
            "slow",
            json!("x"),
            serde_json::from_value(triage()).ok(),
        ))
        .await
        .unwrap();
    assert_eq!(r.load_duration, 0, "warm");
    d.stop().await;
}

/// A static registry (the website's `/v2/...` files) on a random port, with the website's
/// rewrite for manifests by digest (`site/public/_redirects`): `.../manifests/sha256:<hex>` is the
/// file `sha256-<hex>`. Missing files are an HTML 404 page, as there.
async fn registry(dir: PathBuf) -> String {
    let app = axum::Router::new().route(
        "/{*path}",
        axum::routing::get(
            move |axum::extract::Path(path): axum::extract::Path<String>| {
                let file = dir.join(path.replacen("/manifests/sha256:", "/manifests/sha256-", 1));
                async move {
                    match std::fs::read(&file) {
                        Ok(b) => (axum::http::StatusCode::OK, b),
                        Err(_) => (
                            axum::http::StatusCode::NOT_FOUND,
                            b"<!DOCTYPE html><title>Page not found</title>".to_vec(),
                        ),
                    }
                }
            },
        ),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

fn publish(root: &Path, base: &str, name: &str) {
    publish_tag(root, base, name, "latest", 7);
}

/// Publish `name:tag`, whose graph is 4 KiB of `graph`, and keep its manifest by digest as
/// `package.py` does (`manifests/sha256-<hex>`). Returns the digest.
fn publish_tag(root: &Path, base: &str, name: &str, tag: &str, graph: u8) -> String {
    let blobs = root.join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    let put = |media_type: &str, bytes: &[u8], precision: Option<&str>| {
        let hex = sha256_hex(bytes);
        std::fs::write(blobs.join(format!("sha256-{hex}")), bytes).unwrap();
        Descriptor {
            media_type: media_type.into(),
            digest: format!("sha256:{hex}"),
            size: bytes.len() as u64,
            urls: vec![format!("{base}/blobs/sha256-{hex}")],
            annotations: precision
                .map(|p| [(ANNOTATION_PRECISION.to_owned(), p.to_owned())].into())
                .unwrap_or_default(),
        }
    };
    let manifest = Manifest {
        schema_version: 2,
        media_type: MANIFEST_V2.into(),
        config: put(
            media::CONFIG,
            br#"{"model_format":"onnx","family":"tiny","parameter_size":"1M"}"#,
            None,
        ),
        layers: vec![
            put(media::GRAPH_ONNX, &[graph; 4096], Some("fp32")),
            put(media::TOKENIZER, b"{}", None),
            put(
                media::DECISION,
                decision_json(json!({})).to_string().as_bytes(),
                None,
            ),
        ],
    };
    let dir = root.join(format!("v2/library/{name}/manifests"));
    std::fs::create_dir_all(&dir).unwrap();
    let bytes = serde_json::to_vec(&manifest).unwrap();
    std::fs::write(dir.join(tag), &bytes).unwrap();
    let hex = sha256_hex(&bytes);
    std::fs::write(dir.join(format!("sha256-{hex}")), &bytes).unwrap();
    format!("sha256:{hex}")
}

async fn pull_streams_ndjson() {
    let remote = tempfile::tempdir().unwrap();
    let base = registry(remote.path().to_owned()).await;
    publish(remote.path(), &base, "tiny");
    let d = Daemon::start(|_| {}, |_| {}).await;
    let name = format!("{base}/library/tiny");

    // Before the stream starts: an unknown model is a real 404.
    let (status, _, body) = post_raw(
        &d.url("/api/pull"),
        json!({"model": format!("{base}/library/nope")}).to_string(),
    )
    .await;
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["code"], "MODEL_NOT_FOUND");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("not found in registry"),
        "{body}"
    );

    let r = http()
        .post(d.url("/api/pull"))
        .body(json!({"model": name}).to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "application/x-ndjson");
    let text = r.text().await.unwrap();
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let statuses: Vec<&str> = lines
        .iter()
        .map(|l| l["status"].as_str().unwrap())
        .collect();
    assert_eq!(statuses.first(), Some(&"pulling manifest"));
    assert_eq!(statuses.last(), Some(&"success"));
    assert_eq!(statuses.iter().filter(|s| **s == "success").count(), 1);
    assert!(
        statuses.contains(&"verifying sha256 digest") && statuses.contains(&"writing manifest")
    );
    let layer = lines.iter().find(|l| l["digest"].is_string()).unwrap();
    assert!(layer["status"].as_str().unwrap().starts_with("pulling ") && layer["total"].is_u64());

    // Present now; a second pull is idempotent, here through the client and `stream: false`.
    let tags = d.client.tags().await.unwrap();
    assert!(
        tags.models
            .iter()
            .any(|m| m.name.ends_with("/library/tiny:latest")),
        "{tags:?}"
    );
    let r = http()
        .post(d.url("/api/pull"))
        .body(json!({"model": name, "stream": false}).to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.json::<Value>().await.unwrap(),
        json!({"status": "success"})
    );
    let mut seen = Vec::new();
    d.client
        .pull(&PullRequest::new(&name), |p| seen.push(p.status.clone()))
        .await
        .unwrap();
    assert_eq!(seen.last().map(String::as_str), Some("success"));

    // The pulled model answers.
    let r = d
        .client
        .decide(&DecideRequest::new(
            &name,
            json!("hello"),
            serde_json::from_value(triage()).ok(),
        ))
        .await
        .unwrap();
    assert!(r.model.ends_with("/library/tiny:latest"));
    d.stop().await;
}

async fn pull_by_digest() {
    let remote = tempfile::tempdir().unwrap();
    let base = registry(remote.path().to_owned()).await;
    let d1 = publish_tag(remote.path(), &base, "tiny", "latest", 7);
    let d2 = publish_tag(remote.path(), &base, "tiny", "latest", 8); // the tag moves
    let d = Daemon::start(|_| {}, |_| {}).await;
    let name = format!("{base}/library/tiny");
    let latest = |m: &str| format!("{name}:latest@{m}");
    let local_digest = |tags: &ollaya_api::TagsResponse, suffix: &str| {
        tags.models
            .iter()
            .find(|m| m.name.ends_with(suffix))
            .map(|m| format!("sha256:{}", m.digest))
    };

    // A digest the registry doesn't have: 404 before the stream starts, saying so.
    let unknown = format!("sha256:{}", "0".repeat(64));
    let (status, _, body) = post_raw(
        &d.url("/api/pull"),
        json!({"model": latest(&unknown)}).to_string(),
    )
    .await;
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["code"], "MODEL_NOT_FOUND");
    let error = body["error"].as_str().unwrap();
    assert!(
        error.contains(&format!("/library/tiny:latest@{unknown}\""))
            && error.ends_with("it has no manifest with that digest"),
        "{body}"
    );

    // The old version, by digest, under the tag that has moved on since.
    let mut seen = Vec::new();
    d.client
        .pull(&PullRequest::new(latest(&d1)), |p| {
            seen.push(p.status.clone())
        })
        .await
        .unwrap();
    assert_eq!(seen.first().map(String::as_str), Some("pulling manifest"));
    assert_eq!(seen.last().map(String::as_str), Some("success"));
    let tags = d.client.tags().await.unwrap();
    assert_eq!(
        local_digest(&tags, "/library/tiny:latest"),
        Some(d1.clone())
    );

    // A pinned name answers only while the tag holds that manifest.
    let r = d
        .client
        .decide(&DecideRequest::new(
            latest(&d1),
            json!("hello"),
            serde_json::from_value(triage()).ok(),
        ))
        .await
        .unwrap();
    assert!(
        r.model.ends_with(&format!("/library/tiny:latest@{d1}")),
        "{}",
        r.model
    );
    let (status, body) = api_err(
        d.client
            .decide(&DecideRequest::new(
                latest(&d2),
                json!("hello"),
                serde_json::from_value(triage()).ok(),
            ))
            .await
            .unwrap_err(),
    );
    assert_eq!((status, body.code), (404, ErrorCode::ModelNotFound));

    // By digest alone: its own name, next to the tag.
    let r = http()
        .post(d.url("/api/pull"))
        .body(json!({"model": format!("{name}@{d2}"), "stream": false}).to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.json::<Value>().await.unwrap(),
        json!({"status": "success"})
    );
    let tags = d.client.tags().await.unwrap();
    assert_eq!(
        local_digest(&tags, &format!("/library/tiny@{d2}")),
        Some(d2.clone())
    );
    assert_eq!(
        local_digest(&tags, "/library/tiny:latest"),
        Some(d1.clone())
    );

    // A digest can't name a model being written.
    for (path, body, field) in [
        (
            "/api/copy",
            json!({"source": latest(&d1), "destination": format!("{name}:copy@{d1}")}),
            "destination",
        ),
        (
            "/api/create",
            json!({"model": format!("{name}:mine@{d1}"), "from": latest(&d1)}),
            "model",
        ),
    ] {
        let (status, _, body) = post_raw(&d.url(path), body.to_string()).await;
        assert_eq!(status, 422, "{path}: {body}");
        assert_eq!(body["detail"][0]["loc"], json!(["body", field]), "{body}");
        assert_eq!(body["detail"][0]["type"], "model_name", "{body}");
    }

    // A registry that serves another manifest under d1's digest: 502, and nothing changes.
    let dir = remote.path().join("v2/library/tiny/manifests");
    std::fs::copy(
        dir.join(format!("sha256-{}", &d2[7..])),
        dir.join(format!("sha256-{}", &d1[7..])),
    )
    .unwrap();
    let (status, _, body) = post_raw(
        &d.url("/api/pull"),
        json!({"model": latest(&d1)}).to_string(),
    )
    .await;
    assert_eq!(status, 502, "{body}");
    assert_eq!(body["code"], "DIGEST_MISMATCH");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("nothing was stored"),
        "{body}"
    );
    let tags = d.client.tags().await.unwrap();
    assert_eq!(local_digest(&tags, "/library/tiny:latest"), Some(d1));
    d.stop().await;
}

async fn create_copy_delete() {
    let d = Daemon::start(
        |dir| {
            laya_store(dir);
            let store = Store::open(dir).unwrap();
            let name = ModelName::parse("laya:en").unwrap();
            let mut entry = store.read_manifest(&name).unwrap().unwrap();
            let calibration = json!({
                "temperature": [6.08, 3.0, 40.0],
                "temperature_map": {"kind": "von-entropy-length-v1", "bias": 0.2056},
                "temperature_range": [0.5, 50.0]
            });
            entry.manifest.layers.push(blob(
                &store,
                media::CALIBRATION,
                calibration.to_string().as_bytes(),
                None,
            ));
            store
                .write_manifest(&name, &serde_json::to_vec(&entry.manifest).unwrap())
                .unwrap();
        },
        |_| {},
    )
    .await;
    d.client
        .create(
            &serde_json::from_value(json!({"model": "inherited", "from": "laya:en"})).unwrap(),
            |_| {},
        )
        .await
        .unwrap();
    let inherited = d.client.show("inherited").await.unwrap();
    assert!(inherited.modelfile.contains("FROM laya:en"));
    assert!(
        !inherited.modelfile.contains("CALIBRATION"),
        "calibration outside the create API must stay inherited"
    );
    let calibration = json!({
        "temperature": [1.6, 1.25, 1.98],
        "temperature_by_options": {"choice:2": 1.9, "score:3-5": 1.25}
    });
    let r = http()
        .post(d.url("/api/create"))
        .body(
            json!({"model": "triage", "from": "laya:en",
                   "questions": {"department": {"type": "choice", "criteria": ["billing", "technical"]}},
                   "calibration": calibration,
                   "parameters": {"precision": "fp32"}, "license": ["MIT", "Apache-2.0"],
                   "description": "Ticket triage"})
            .to_string(),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let text = r.text().await.unwrap();
    let statuses: Vec<String> = text
        .lines()
        .map(|l| {
            serde_json::from_str::<Value>(l).unwrap()["status"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect();
    assert!(
        statuses
            .iter()
            .any(|s| s.starts_with("using existing layer sha256:")),
        "{statuses:?}"
    );
    assert!(
        statuses
            .iter()
            .any(|s| s.starts_with("creating new layer sha256:")),
        "{statuses:?}"
    );
    assert_eq!(
        &statuses[statuses.len() - 2..],
        ["writing manifest", "success"]
    );

    let show = d.client.show("triage").await.unwrap();
    assert_eq!(show.details.parent_model, "laya:en");
    assert_eq!(show.parameters, "precision fp32");
    assert_eq!(show.details.quantization_level, "F32");
    assert_eq!(show.license, "MIT\n\nApache-2.0");
    assert!(show.questions.as_ref().unwrap().contains_key("department"));
    assert!(show.modelfile.contains("FROM laya:en") && show.modelfile.contains("QUESTIONS"));
    let exported_calibration = show
        .modelfile
        .split_once("CALIBRATION \"\"\"\n")
        .expect("the Modelfile must preserve custom calibration")
        .1
        .split_once("\n\"\"\"")
        .unwrap()
        .0;
    assert_eq!(
        serde_json::from_str::<Value>(exported_calibration).unwrap(),
        calibration
    );
    // A model derived from triage without its own calibration shares triage's layer: FROM brings
    // it back, so the Modelfile leaves it out, even though it would round-trip.
    d.client
        .create(
            &serde_json::from_value(json!({"model": "triage-child", "from": "triage"})).unwrap(),
            |_| {},
        )
        .await
        .unwrap();
    let child = d.client.show("triage-child").await.unwrap();
    assert!(child.modelfile.contains("FROM triage"));
    assert!(
        !child.modelfile.contains("CALIBRATION"),
        "an inherited calibration stays inherited: {}",
        child.modelfile
    );
    let models = d.client.models().await.unwrap();
    assert_eq!(
        models
            .models
            .iter()
            .find(|m| m.name == "triage:latest")
            .unwrap()
            .description,
        "Ticket triage"
    );

    // Baked questions answer when the request brings none; the id stands in for instructions.
    let r = d
        .client
        .decide(&DecideRequest::new(
            "triage",
            json!("I was charged twice"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(r.answers.keys().collect::<Vec<_>>(), ["department"]);

    let (status, body) = api_err(
        d.client
            .create(
                &serde_json::from_value(json!({"model": "x", "from": "missing:tag"})).unwrap(),
                |_| {},
            )
            .await
            .unwrap_err(),
    );
    assert_eq!((status, body.code), (404, ErrorCode::ModelNotFound));

    d.client.copy("triage", "triage-copy").await.unwrap();
    d.client.copy("triage", "triage-copy").await.unwrap(); // overwrite: idempotent
    assert!(d.client.show("triage-copy").await.is_ok());
    d.client.delete("triage-copy").await.unwrap();
    let (status, body) = api_err(d.client.delete("triage-copy").await.unwrap_err());
    assert_eq!((status, body.code), (404, ErrorCode::ModelNotFound));
    let (status, _, body) = post_raw(
        &d.url("/api/copy"),
        json!({"source": "nope", "destination": "x"}).to_string(),
    )
    .await;
    assert_eq!(
        (status, body["code"].as_str()),
        (404, Some("MODEL_NOT_FOUND"))
    );
    d.stop().await;
}

async fn show_modelfile_keeps_a_custom_license() {
    let d = Daemon::laya().await;
    for (model, license) in [("short", "MIT"), ("long", "MIT\n\nApache-2.0")] {
        d.client
            .create(
                &serde_json::from_value(
                    json!({"model": model, "from": "laya:en", "license": license}),
                )
                .unwrap(),
                |_| {},
            )
            .await
            .unwrap();
        let show = d.client.show(model).await.unwrap();
        // Even a one-line license goes in a `"""` block: a bare LICENSE value is read as a path.
        assert!(
            show.modelfile
                .contains(&format!("\nLICENSE \"\"\"\n{license}\n\"\"\"\n")),
            "{}",
            show.modelfile
        );
    }

    // A license the model shares with its parent is inherited: FROM brings it back.
    d.client
        .create(
            &serde_json::from_value(json!({"model": "child", "from": "long"})).unwrap(),
            |_| {},
        )
        .await
        .unwrap();
    let show = d.client.show("child").await.unwrap();
    assert_eq!(show.license, "MIT\n\nApache-2.0");
    assert!(!show.modelfile.contains("LICENSE"), "{}", show.modelfile);
    d.stop().await;
}

async fn create_rejects_an_unavailable_precision() {
    // Most of the library is fp32-only. `solo` is such a base; `solo:vision` adds an fp16 image
    // graph, which is not a main graph and so cannot be pinned.
    let d = Daemon::start(
        |dir| {
            laya_store(dir);
            let store = Store::open(dir).unwrap();
            let laya = ModelName::parse("laya:en").unwrap();
            let mut manifest = store.read_manifest(&laya).unwrap().unwrap().manifest;
            manifest.layers.retain(|l| {
                l.annotations.get(ANNOTATION_PRECISION).map(String::as_str) != Some("fp16")
            });
            let write = |name: &str, manifest: &Manifest| {
                let name = ModelName::parse(name).unwrap();
                let bytes = serde_json::to_vec(manifest).unwrap();
                store.write_manifest(&name, &bytes).unwrap();
            };
            write("solo", &manifest);
            let mut vision = blob(&store, media::GRAPH_ONNX, b"image graph", Some("fp16"));
            vision
                .annotations
                .insert(ANNOTATION_GRAPH.into(), "vision".into());
            manifest.layers.push(vision);
            write("solo:vision", &manifest);
        },
        |_| {},
    )
    .await;
    let create = |model: &str, from: &str, precision: &str| {
        let req = json!({"model": model, "from": from, "parameters": {"precision": precision}});
        let client = &d.client;
        async move {
            client
                .create(&serde_json::from_value(req).unwrap(), |_| {})
                .await
        }
    };
    let digest = async |model: &str| {
        let tags = d.client.tags().await.unwrap();
        tags.models
            .into_iter()
            .find(|m| m.name == model)
            .map(|m| m.digest)
    };

    // A precision the base carries pins as before, on either kind of base.
    create("pinned", "solo", "fp32").await.unwrap();
    create("laya-fp16", "laya:en", "fp16").await.unwrap();
    assert_eq!(
        d.client.show("laya-fp16").await.unwrap().parameters,
        "precision fp16"
    );
    let before = digest("pinned:latest").await.unwrap();

    for (model, from) in [
        ("unpinned", "solo"),
        ("pinned", "solo"),
        ("unpinned", "solo:vision"),
    ] {
        let (status, body) = api_err(create(model, from, "fp16").await.unwrap_err());
        assert_eq!(
            (status, &body.code),
            (422, &ErrorCode::InvalidRequest),
            "{model} from {from}"
        );
        assert!(
            body.error.contains("has no fp16 graph"),
            "{model} from {from}: {}",
            body.error
        );
    }
    // Nothing was written: no new model, and the existing one is untouched.
    let (status, _) = api_err(d.client.show("unpinned").await.unwrap_err());
    assert_eq!(status, 404);
    assert_eq!(digest("pinned:latest").await.unwrap(), before);
    d.client.show("pinned").await.unwrap();
    d.stop().await;
}

async fn show_modelfile_keeps_a_multi_line_description() {
    let d = Daemon::laya().await;
    let description = "Ticket triage\nRoutes billing and technical tickets.";
    d.client
        .create(
            &serde_json::from_value(
                json!({"model": "triage", "from": "laya:en", "description": description}),
            )
            .unwrap(),
            |_| {},
        )
        .await
        .unwrap();
    let show = d.client.show("triage").await.unwrap();
    // A Modelfile reads a multi-line value only from a `"""` block; on one line, the second
    // line would be parsed as a directive.
    assert!(
        show.modelfile
            .contains(&format!("\nDESCRIPTION \"\"\"\n{description}\n\"\"\"\n")),
        "{}",
        show.modelfile
    );

    d.client
        .create(
            &serde_json::from_value(
                json!({"model": "short", "from": "laya:en", "description": "Ticket triage"}),
            )
            .unwrap(),
            |_| {},
        )
        .await
        .unwrap();
    let show = d.client.show("short").await.unwrap();
    assert!(
        show.modelfile.contains("\nDESCRIPTION Ticket triage\n"),
        "{}",
        show.modelfile
    );
    d.stop().await;
}

// ------------------------------------------------------------------------------------------------
// Harness

type Test = fn() -> Pin<Box<dyn Future<Output = ()>>>;

macro_rules! tests {
    ($($name:ident),* $(,)?) => {
        vec![$((stringify!($name), (|| Box::pin($name()) as Pin<Box<dyn Future<Output = ()>>>) as Test)),*]
    };
}

async fn presets_create_show_decide_delete() {
    let d = Daemon::laya().await;
    let builtin = d.client.presets().await.unwrap().presets;
    assert_eq!(builtin.len(), ollaya_api::presets::NAMES.len());
    assert!(builtin.iter().all(|p| p.builtin));

    let create = |name: &str| ollaya_api::CreatePresetRequest {
        name: name.into(),
        questions: serde_json::from_value(triage()).unwrap(),
        description: Some("Ticket routing".into()),
        state: None,
    };
    d.client.create_preset(&create("my-triage")).await.unwrap();
    // A built-in name, or one that is not a file name, is a 422 on `name`.
    for bad in ["triage", "../x", "Mine"] {
        let (status, body) = api_err(d.client.create_preset(&create(bad)).await.unwrap_err());
        assert_eq!(
            (status, body.detail.unwrap()[0].path()),
            (422, "name".into()),
            "{bad}"
        );
    }
    let list = d.client.presets().await.unwrap().presets;
    let mine = list.last().unwrap();
    assert_eq!((mine.name.as_str(), mine.builtin), ("my-triage", false));
    assert_eq!(mine.questions, ["department", "urgency", "refund"]);
    assert!(mine.modified_at.is_some());
    let shown = d.client.show_preset("my-triage").await.unwrap();
    assert_eq!(shown.description.as_deref(), Some("Ticket routing"));
    assert!(d.client.show_preset("triage").await.unwrap().builtin);

    // A preset stands in for `questions`, and answers the same.
    let state = json!("Hi, how much would we save by switching to the annual plan?");
    let mut req = DecideRequest::new("laya", state.clone(), None);
    req.preset = Some("my-triage".into());
    let by_preset = d.client.decide(&req).await.unwrap();
    let by_questions = d
        .client
        .decide(&DecideRequest::new(
            "laya",
            state.clone(),
            serde_json::from_value(triage()).ok(),
        ))
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(&by_preset.answers).unwrap(),
        serde_json::to_value(&by_questions.answers).unwrap()
    );
    let mut both = DecideRequest::new("laya", state.clone(), serde_json::from_value(triage()).ok());
    both.preset = Some("my-triage".into());
    assert_eq!(api_err(d.client.decide(&both).await.unwrap_err()).0, 422);

    // A state template takes plain text; `requires` picks the questions a state can answer.
    d.client
        .create_preset(&ollaya_api::CreatePresetRequest {
            name: "shaped".into(),
            questions: serde_json::from_value(json!({
                "refund": {"type": "noul", "instructions": "Asks for a refund.", "requires": ["ticket"]},
                "urgency": {"type": "score", "criteria": ["low", "high"], "requires": ["ticket", "plan"]},
            }))
            .unwrap(),
            description: None,
            state: Some(json!({"ticket": "{text}"})),
        })
        .await
        .unwrap();
    assert_eq!(
        d.client.show_preset("shaped").await.unwrap().state,
        Some(json!({"ticket": "{text}"}))
    );
    let mut shaped = DecideRequest::new("laya", json!("I want my money back"), None);
    shaped.preset = Some("shaped".into());
    let r = d.client.decide(&shaped).await.unwrap();
    assert_eq!(r.answers.keys().collect::<Vec<_>>(), ["refund"]);
    shaped.state = Some(json!({"ticket": "refund please", "plan": "pro"}));
    assert_eq!(d.client.decide(&shaped).await.unwrap().answers.len(), 2);
    shaped.state = Some(json!({"other": 1}));
    let (status, body) = api_err(d.client.decide(&shaped).await.unwrap_err());
    assert_eq!(
        (status, body.detail.unwrap()[0].path()),
        (422, "state".into())
    );
    d.client.delete_preset("shaped").await.unwrap();

    req.preset = Some("nope".into());
    let (status, body) = api_err(d.client.decide(&req).await.unwrap_err());
    assert_eq!((status, body.code), (404, ollaya_api::ErrorCode::NotFound));

    let (status, _) = api_err(d.client.delete_preset("triage").await.unwrap_err());
    assert_eq!(status, 403);
    d.client.delete_preset("my-triage").await.unwrap();
    assert_eq!(
        api_err(d.client.delete_preset("my-triage").await.unwrap_err()).0,
        404
    );
    assert_eq!(
        d.client.presets().await.unwrap().presets.len(),
        ollaya_api::presets::NAMES.len()
    );
    d.stop().await;
}

async fn gguf_runner_that_dies_on_the_gpu_restarts_on_the_cpu() {
    let d = Daemon::start(
        |dir| {
            let store = Store::open(dir).unwrap();
            add_gguf_model(&store, "fragile:latest", json!({"abort_unless_cpu": true}));
            add_gguf_model(&store, "broken:latest", json!({"fail_load": true}));
            add_gguf_media_model(&store, "sees:latest", json!({}), true, false);
            add_gguf_media_model(&store, "hears:latest", json!({}), true, true);
        },
        |c| {
            c.device = "auto".into();
            c.load_timeout = Duration::from_secs(30);
        },
    )
    .await;
    d.client.load("fragile", None).await.unwrap();
    let ps = d.client.ps().await.unwrap();
    let devices: Vec<(&str, &str)> = ps
        .models
        .iter()
        .map(|m| (m.name.as_str(), m.device.as_str()))
        .collect();
    assert_eq!(devices, [("fragile:latest", "cpu")]);
    // A model that fails everywhere still reports its error.
    let (status, body) = api_err(d.client.load("broken", None).await.unwrap_err());
    assert_eq!(status, 500, "{body:?}");
    assert!(body.error.contains("cannot load this model"), "{body:?}");
    // The media a model reads, as clients discover them.
    for (name, caps) in [
        ("broken", &["choice", "score", "noul"][..]),
        ("sees", &["choice", "score", "noul", "vision"]),
        ("hears", &["choice", "score", "noul", "vision", "audio"]),
    ] {
        assert_eq!(d.client.show(name).await.unwrap().capabilities, caps);
    }
    d.stop().await;
}

async fn read_only_refuses_store_changes() {
    let d = Daemon::start(laya_store, |c| c.read_only = true).await;
    let read_only = |e: ClientError, what: &str| {
        let (status, body) = api_err(e);
        assert_eq!((status, &body.code), (403, &ErrorCode::ReadOnly), "{what}");
        assert!(body.error.contains("read-only"), "{what}: {}", body.error);
    };
    read_only(
        d.client
            .pull(&PullRequest::new("laya:multilingual"), |_| {})
            .await
            .unwrap_err(),
        "pull",
    );
    let create = serde_json::from_value(json!({"model": "mine", "from": "laya:en"})).unwrap();
    read_only(
        d.client.create(&create, |_| {}).await.unwrap_err(),
        "create",
    );
    read_only(d.client.copy("laya:en", "mine").await.unwrap_err(), "copy");
    read_only(d.client.delete("laya:en").await.unwrap_err(), "delete");
    let preset = ollaya_api::CreatePresetRequest {
        name: "my-triage".into(),
        questions: serde_json::from_value(triage()).unwrap(),
        description: None,
        state: None,
    };
    read_only(
        d.client.create_preset(&preset).await.unwrap_err(),
        "create preset",
    );
    read_only(
        d.client.delete_preset("my-triage").await.unwrap_err(),
        "delete preset",
    );
    for path in ["/api/push", "/api/blobs/sha256:00"] {
        let (status, _, body) = post_raw(&d.url(path), "{}").await;
        assert_eq!(
            (status, body["code"].as_str()),
            (403, Some("READ_ONLY")),
            "{path}"
        );
    }

    // Nothing changed, and everything that only reads or decides still works.
    let tags = d.client.tags().await.unwrap();
    let mut names: Vec<_> = tags.models.iter().map(|m| m.name.as_str()).collect();
    names.sort_unstable();
    assert!(
        names.contains(&"laya:en") && !names.contains(&"mine:latest"),
        "{names:?}"
    );
    d.client.show("laya:en").await.unwrap();
    d.client.presets().await.unwrap();
    d.client.version().await.unwrap();
    let answer = d
        .client
        .decide(&DecideRequest::new(
            "laya:en",
            json!("Hi, how much would we save by switching to the annual plan?"),
            serde_json::from_value(triage()).ok(),
        ))
        .await
        .unwrap();
    assert_eq!(answer.model, "laya:en");
    assert!(!d.client.ps().await.unwrap().models.is_empty());
    d.stop().await;
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("runner") {
        fake_runner::main(&args[2..]);
        return;
    }
    let filter = args.iter().skip(1).find(|a| !a.starts_with('-')).cloned();
    let all: Vec<(&str, Test)> = tests![
        liveness_version_and_request_ids,
        unknown_paths_methods_and_reserved,
        bodies_are_json_whatever_the_content_type,
        unknown_models_are_404_with_ollama_text,
        tags_show_ps_and_v1_models,
        decide_routes_answers_and_unloads,
        preload_loads_models_before_ready,
        a_passed_deadline_skips_the_decision,
        systemone_is_typesafe_shaped,
        systemone_rejects_truncated_state,
        model_specific_errors,
        auth_origin_and_host,
        queue_bound_and_cancellation,
        pull_streams_ndjson,
        pull_by_digest,
        create_copy_delete,
        show_modelfile_keeps_a_custom_license,
        create_rejects_an_unavailable_precision,
        show_modelfile_keeps_a_multi_line_description,
        gguf_runner_that_dies_on_the_gpu_restarts_on_the_cpu,
        presets_create_show_decide_delete,
        read_only_refuses_store_changes,
        adapter_models_create_decide_share_a_runner_and_delete,
        adapter_refusals_name_the_problem_and_write_nothing,
        adapter_models_run_on_mlx_only_and_fail_alone,
        the_adapter_limit_drops_the_least_recently_used_idle_adapter,
        a_runner_that_never_answers_does_not_hold_the_load_lock_for_ever,
        the_base_stops_being_kept_when_its_keep_forever_adapter_goes,
        an_expiring_plain_runner_leaves_the_shared_base_alone_and_stop_takes_it,
        blobs_follow_the_api_key_are_verified_and_pruned,
        a_cancelled_delete_finishes_unloading_and_does_not_block_later_loads,
    ];
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (mut passed, mut failed) = (0, Vec::new());
    for (name, test) in all {
        if filter.as_deref().is_some_and(|f| !name.contains(f)) {
            continue;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.block_on(test())));
        match result {
            Ok(()) => {
                println!("test {name} ... ok");
                passed += 1;
            }
            Err(_) => {
                println!("test {name} ... FAILED");
                failed.push(name);
            }
        }
    }
    let verdict = if failed.is_empty() { "ok" } else { "FAILED" };
    println!(
        "\ntest result: {verdict}. {passed} passed; {} failed",
        failed.len()
    );
    if !failed.is_empty() {
        println!("failures: {failed:?}");
        std::process::exit(1);
    }
}
