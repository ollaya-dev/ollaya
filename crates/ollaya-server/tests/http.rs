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
use ollaya_registry::manifest::{ANNOTATION_PRECISION, Descriptor, MANIFEST_V2, Manifest, media};
use ollaya_registry::store::sha256_hex;
use ollaya_registry::{ModelName, Store};
use ollaya_server::config::ServerConfig;
use ollaya_server::http::{RunnerLaunch, build, serve_on};
use serde_json::{Value, json};

// ------------------------------------------------------------------------------------------------
// The fake runner

mod fake_runner {
    use std::sync::Arc;

    use axum::Json;
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};
    use serde_json::{Value, json};

    /// `runner --decision <file> ...`: behaviour comes from the decision layer's `fake` object.
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
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(serve(Arc::new(decision)));
    }

    async fn serve(decision: Arc<Value>) {
        let fake = &decision["fake"];
        if fake["fail_load"].as_bool() == Some(true) {
            eprintln!("fake runner: cannot load this model");
            std::process::exit(1);
        }
        tokio::time::sleep(std::time::Duration::from_millis(
            fake["load_ms"].as_u64().unwrap_or(0),
        ))
        .await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        println!(
            "{}",
            json!({"port": port, "device": "cpu", "precision": "fp32"})
        );
        let d = decision.clone();
        let app = axum::Router::new()
            .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
            .route(
                "/decide",
                post(move |Json(req): Json<Value>| async move { decide(&d, &req) }),
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
            let logits: Vec<f32> = (0..n)
                .map(|i| if i == 0 { 2.0 } else { -(i as f32) / 4.0 })
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
    req.extras = vec![ollaya_api::Extra::Laya];
    req.keep_alive = Some(KeepAlive::Forever);
    let tr = d.client.decide(&req).await.unwrap();
    assert_eq!(tr.model, "laya:multilingual");
    assert_eq!(tr.routing.as_ref().unwrap().route, "multilingual");
    let laya = tr.answers["department"].laya.as_ref().unwrap();
    assert_eq!(laya.act_probability.map(|p| p > 0.5), Some(true));

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
    d.stop().await;
}

async fn model_specific_errors() {
    let d = Daemon::start(
        |dir| {
            let store = Store::open(dir).unwrap();
            add_model(&store, "small:latest", &["en"], json!({"max_options": 100}));
            add_model(&store, "broken:latest", &["en"], json!({"fail_load": true}));
        },
        |_| {},
    )
    .await;
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

/// A static registry (the website's `/v2/...` files) on a random port.
async fn registry(dir: PathBuf) -> String {
    let app = axum::Router::new().route(
        "/{*path}",
        axum::routing::get(
            move |axum::extract::Path(path): axum::extract::Path<String>| {
                let file = dir.join(path);
                async move {
                    match std::fs::read(&file) {
                        Ok(b) => (axum::http::StatusCode::OK, b),
                        Err(_) => (axum::http::StatusCode::NOT_FOUND, Vec::new()),
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
            put(media::GRAPH_ONNX, &[7u8; 4096], Some("fp32")),
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
    std::fs::write(dir.join("latest"), serde_json::to_vec(&manifest).unwrap()).unwrap();
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

async fn create_copy_delete() {
    let d = Daemon::laya().await;
    let r = http()
        .post(d.url("/api/create"))
        .body(
            json!({"model": "triage", "from": "laya:en",
                   "questions": {"department": {"type": "choice", "criteria": ["billing", "technical"]}},
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

// ------------------------------------------------------------------------------------------------
// Harness

type Test = fn() -> Pin<Box<dyn Future<Output = ()>>>;

macro_rules! tests {
    ($($name:ident),* $(,)?) => {
        vec![$((stringify!($name), (|| Box::pin($name()) as Pin<Box<dyn Future<Output = ()>>>) as Test)),*]
    };
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
        systemone_is_typesafe_shaped,
        model_specific_errors,
        auth_origin_and_host,
        queue_bound_and_cancellation,
        pull_streams_ndjson,
        create_copy_delete,
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
