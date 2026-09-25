//! The HTTP API of `docs/api.md`: axum routes over [`Ollaya`], plus the cross-cutting parts
//! (request IDs, `Host`/`Origin` checks and CORS, bearer auth, body limits, error bodies, the
//! decision queue bound, client cancellation and graceful shutdown).
//!
//! Handlers only translate: validation lives in `ollaya_api::validate`, behaviour in
//! [`crate::service`], and response shapes in `ollaya_api`'s types.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::future::{Future, IntoFuture};
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use indexmap::IndexMap;
use ollaya_api::error::{Loc, ValidationIssue, body_loc};
use ollaya_api::host::Host;
use ollaya_api::{
    DecideResponse, DoneReason, ErrorBody, ErrorCode, Extra, KeepAlive, ModelList,
    ProgressResponse, PsResponse, Question, Questions, TagsResponse, Usage, VersionResponse,
    validate,
};
use ollaya_registry::{ModelName, Store};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast::error::RecvError;

use crate::config::{ServerConfig, VERSION};
use crate::scheduler::{Scheduler, SchedulerConfig};
use crate::service::{CreateSpec, DecideInput, Ollaya, PullEvent};
use crate::{Error, views};

/// Shared state of the HTTP layer.
pub struct AppState {
    pub ollaya: Arc<Ollaya>,
    pub config: ServerConfig,
    /// Allowed browser origin patterns (`*` wildcards).
    origins: Vec<String>,
    /// Bound to a loopback address: `Host` headers are checked (DNS-rebinding defence).
    loopback: bool,
    hostname: Option<String>,
    /// Decision requests in flight, bounded by `max_queue`.
    in_flight: AtomicUsize,
}

impl AppState {
    pub fn new(ollaya: Arc<Ollaya>, config: ServerConfig) -> Arc<Self> {
        let mut origins = crate::config::default_origins();
        origins.extend(config.origins.iter().cloned());
        Arc::new(AppState {
            loopback: binds_loopback(&config),
            ollaya,
            config,
            origins,
            hostname: machine_hostname(),
            in_flight: AtomicUsize::new(0),
        })
    }

    fn store(&self) -> &Store {
        &self.ollaya.store
    }
}

fn binds_loopback(config: &ServerConfig) -> bool {
    let h = config
        .host
        .host
        .trim_start_matches('[')
        .trim_end_matches(']');
    h.eq_ignore_ascii_case("localhost") || h.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

fn machine_hostname() -> Option<String> {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .or_else(|| std::env::var("HOSTNAME").ok())
        .map(|h| h.trim().to_ascii_lowercase())
        .filter(|h| !h.is_empty())
}

/// The router for every endpoint of `docs/api.md` §1.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/api/version", get(version))
        .route("/api/tags", get(tags))
        .route("/api/ps", get(ps))
        .route("/api/show", post(show))
        .route("/api/pull", post(pull))
        .route("/api/delete", delete(delete_model))
        .route("/api/copy", post(copy))
        .route("/api/create", post(create))
        .route("/api/decide", post(decide))
        .route("/api/push", post(reserved))
        .route("/api/blobs/{digest}", post(reserved).head(reserved))
        .route("/v1/systemone", post(systemone))
        .route("/v1/decisions", post(systemone))
        .route("/v1/models", get(v1_models))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

// ---------------------------------------------------------------------------------------------
// Errors

/// An error response: the contract's body plus any headers it needs.
pub struct ApiError {
    body: ErrorBody,
    headers: Vec<(HeaderName, HeaderValue)>,
}

impl ApiError {
    fn with_header(mut self, name: HeaderName, value: &'static str) -> Self {
        self.headers.push((name, HeaderValue::from_static(value)));
        self
    }
}

impl From<ErrorBody> for ApiError {
    fn from(body: ErrorBody) -> Self {
        ApiError {
            body,
            headers: Vec::new(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.body.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let mut resp = (status, Json(self.body)).into_response();
        for (k, v) in self.headers {
            resp.headers_mut().insert(k, v);
        }
        resp
    }
}

type ApiResult<T> = Result<T, ApiError>;

/// The error body for a service error. `pulling` selects registry wording for store errors.
pub fn error_body(e: &Error, pulling: bool) -> ErrorBody {
    use ollaya_registry::Error as R;
    match e {
        Error::ModelNotFound(n) => ErrorBody::model_not_found(n),
        Error::RoutedModelNotFound { target, router } => {
            ErrorBody::routed_model_not_found(target, router)
        }
        Error::NoQuestions(_) => {
            ErrorBody::invalid_request(vec![ValidationIssue::missing(body_loc(&["questions"]))])
        }
        Error::InvalidRequest(m) => ErrorBody::invalid_request(vec![ValidationIssue::new(
            vec![Loc::key("body")],
            "value_error",
            m.clone(),
        )]),
        Error::TooManyOptions {
            question,
            options,
            model,
        } => ErrorBody::too_many_options(question, "choice", *options, model),
        Error::InputTooLong(n) => ErrorBody::input_too_long(*n),
        Error::LoadFailed(m) => ErrorBody::new(ErrorCode::ModelLoadFailed, m.clone()),
        Error::Runner(m) => ErrorBody::new(ErrorCode::InferenceFailed, m.clone()),
        Error::Unsupported(m) => ErrorBody::new(ErrorCode::UnsupportedModel, m.clone()),
        Error::Busy(n) => ErrorBody::operation_in_progress(n),
        Error::Cancelled => ErrorBody::new(ErrorCode::InferenceFailed, "request cancelled"),
        Error::Corrupt(m) => ErrorBody::new(ErrorCode::StorageError, format!("corrupt model: {m}")),
        Error::Registry(r) => match r {
            R::InvalidName(s) => ErrorBody::invalid_request(vec![ValidationIssue::model_name(
                body_loc(&["model"]),
                s,
            )]),
            R::NotFound(n) if pulling => {
                let host = ModelName::parse(n).map_or_else(|_| n.clone(), |m| m.host);
                ErrorBody::not_in_registry(n, &host)
            }
            R::NotFound(n) => ErrorBody::model_not_found(n),
            R::DigestMismatch { expected, .. } => ErrorBody::digest_mismatch(expected),
            R::Unsupported(m) => ErrorBody::new(ErrorCode::UnsupportedModel, m.clone()),
            R::Stalled(_) | R::Http(_) => ErrorBody::new(ErrorCode::RegistryError, r.to_string()),
            _ if pulling => ErrorBody::new(ErrorCode::RegistryError, r.to_string()),
            _ => ErrorBody::new(ErrorCode::StorageError, r.to_string()),
        },
    }
}

fn api_error(e: &Error) -> ApiError {
    error_body(e, false).into()
}

/// Parse `name` fields; a bad one is a `model_name` issue on that field.
fn names<const N: usize>(fields: [(&str, &str); N]) -> ApiResult<[ModelName; N]> {
    let mut issues = Vec::new();
    let parsed = fields.map(|(field, value)| {
        ModelName::parse(value)
            .map_err(|_| issues.push(ValidationIssue::model_name(body_loc(&[field]), value)))
            .ok()
    });
    if !issues.is_empty() {
        return Err(ErrorBody::invalid_request(issues).into());
    }
    Ok(parsed.map(|n| n.expect("every name parsed")))
}

// ---------------------------------------------------------------------------------------------
// Bodies

/// Read a request body up to [`ollaya_api::MAX_BODY_BYTES`], whatever its `Content-Type`.
async fn read_body(body: Body) -> ApiResult<Bytes> {
    let mut stream = body.into_data_stream();
    let mut buf = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|e| ErrorBody::invalid_json(format!("reading the request body: {e}")))?;
        if buf.len() + chunk.len() > ollaya_api::MAX_BODY_BYTES {
            return Err(ErrorBody::request_too_large().into());
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf.into())
}

async fn parse<T>(
    body: Body,
    validator: fn(validate::Body) -> Result<T, validate::Issues>,
) -> ApiResult<T> {
    let bytes = read_body(body).await?;
    Ok(validate::body(&bytes, validator)?)
}

fn ndjson(lines: impl futures_util::Stream<Item = Vec<u8>> + Send + 'static) -> Response {
    let body = Body::from_stream(lines.map(Ok::<_, Infallible>));
    (
        [(header::CONTENT_TYPE, ollaya_api::NDJSON_CONTENT_TYPE)],
        body,
    )
        .into_response()
}

fn line<T: serde::Serialize>(v: &T) -> Vec<u8> {
    let mut out = serde_json::to_vec(v).expect("response types serialize");
    out.push(b'\n');
    out
}

// ---------------------------------------------------------------------------------------------
// Handlers

async fn root() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        ollaya_api::RUNNING_MESSAGE,
    )
}

async fn version() -> Json<VersionResponse> {
    Json(VersionResponse {
        version: VERSION.to_owned(),
    })
}

async fn tags(State(s): State<Arc<AppState>>) -> ApiResult<Json<TagsResponse>> {
    let mut models: Vec<_> = s
        .ollaya
        .list()
        .map_err(|e| api_error(&e))?
        .iter()
        .map(|e| views::local_model(s.store(), e))
        .collect();
    models.sort_by(|a, b| {
        b.modified_at
            .cmp(&a.modified_at)
            .then_with(|| a.name.cmp(&b.name))
    });
    Ok(Json(TagsResponse { models }))
}

async fn ps(State(s): State<Arc<AppState>>) -> Json<PsResponse> {
    let mut models: Vec<_> = s
        .ollaya
        .running()
        .iter()
        .map(|r| views::running_model(s.store(), r))
        .collect();
    models.sort_by(|a, b| a.name.cmp(&b.name));
    Json(PsResponse { models })
}

async fn v1_models(State(s): State<Arc<AppState>>) -> ApiResult<Json<ModelList>> {
    let mut models: Vec<_> = s
        .ollaya
        .list()
        .map_err(|e| api_error(&e))?
        .iter()
        .map(|e| views::model_metadata(s.store(), e))
        .collect();
    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Json(ModelList { models }))
}

async fn show(State(s): State<Arc<AppState>>, body: Body) -> ApiResult<Response> {
    let req = parse(body, validate::show_request).await?;
    names([("model", &req.model)])?;
    let info = s.ollaya.show(&req.model).map_err(|e| api_error(&e))?;
    Ok(Json(views::show(s.store(), &info)).into_response())
}

async fn delete_model(State(s): State<Arc<AppState>>, body: Body) -> ApiResult<StatusCode> {
    let req = parse(body, validate::delete_request).await?;
    let [name] = names([("model", &req.model)])?;
    match s.ollaya.delete(&req.model).await {
        Ok(true) => Ok(StatusCode::OK),
        Ok(false) => Err(ErrorBody::model_not_found(&name.to_string()).into()),
        Err(e) => Err(api_error(&e)),
    }
}

async fn copy(State(s): State<Arc<AppState>>, body: Body) -> ApiResult<StatusCode> {
    let req = parse(body, validate::copy_request).await?;
    names([("source", &req.source), ("destination", &req.destination)])?;
    s.ollaya
        .copy(&req.source, &req.destination)
        .map_err(|e| api_error(&e))?;
    Ok(StatusCode::OK)
}

async fn reserved(uri: Uri) -> ApiError {
    ErrorBody::not_implemented(uri.path()).into()
}

async fn not_found(uri: Uri) -> ApiError {
    ErrorBody::path_not_found(uri.path()).into()
}

async fn method_not_allowed(method: Method, uri: Uri) -> ApiError {
    ErrorBody::method_not_allowed(method.as_str(), uri.path()).into()
}

fn progress(p: ollaya_registry::Progress) -> ProgressResponse {
    ProgressResponse {
        status: p.status,
        digest: p.digest,
        total: p.total,
        completed: p.completed,
    }
}

/// `POST /api/pull`. Failures before the first layer line are HTTP errors; later ones are a
/// final error line.
async fn pull(State(s): State<Arc<AppState>>, body: Body) -> ApiResult<Response> {
    let req = parse(body, validate::pull_request).await?;
    names([("model", &req.model)])?;
    let pull_error = |e: &Error| -> ApiError { error_body(e, true).into() };
    let mut rx = s.ollaya.pull(&req.model).map_err(|e| pull_error(&e))?;

    let mut head = VecDeque::new();
    let mut finished = loop {
        match rx.recv().await {
            Ok(PullEvent::Progress(p)) => {
                let (layer, success) = (p.digest.is_some(), p.status == "success");
                head.push_back(progress(p));
                if layer || success {
                    break success;
                }
            }
            Ok(PullEvent::Done) => {
                head.push_back(ProgressResponse::success());
                break true;
            }
            Ok(PullEvent::Failed(e)) => return Err(pull_error(&e)),
            Err(RecvError::Lagged(_)) => continue,
            Err(RecvError::Closed) => {
                return Err(
                    ErrorBody::new(ErrorCode::Internal, "the pull stopped unexpectedly").into(),
                );
            }
        }
    };

    if req.stream == Some(false) {
        while !finished {
            match rx.recv().await {
                Ok(PullEvent::Progress(p)) => finished = p.status == "success",
                Ok(PullEvent::Done) => finished = true,
                Ok(PullEvent::Failed(e)) => return Err(pull_error(&e)),
                Err(RecvError::Lagged(_)) => {}
                Err(RecvError::Closed) => {
                    return Err(ErrorBody::new(
                        ErrorCode::Internal,
                        "the pull stopped unexpectedly",
                    )
                    .into());
                }
            }
        }
        return Ok(Json(ProgressResponse::success()).into_response());
    }

    struct Lines {
        head: VecDeque<ProgressResponse>,
        rx: tokio::sync::broadcast::Receiver<PullEvent>,
        done: bool,
    }
    let lines = futures_util::stream::unfold(
        Lines {
            head,
            rx,
            done: finished,
        },
        |mut st| async move {
            if let Some(p) = st.head.pop_front() {
                return Some((line(&p), st));
            }
            if st.done {
                return None;
            }
            loop {
                match st.rx.recv().await {
                    Ok(PullEvent::Progress(p)) => {
                        st.done = p.status == "success";
                        return Some((line(&progress(p)), st));
                    }
                    // `success` already went out as a progress line.
                    Ok(PullEvent::Done) | Err(RecvError::Closed) => return None,
                    Ok(PullEvent::Failed(e)) => {
                        st.done = true;
                        return Some((line(&error_body(&e, true)), st));
                    }
                    Err(RecvError::Lagged(_)) => continue,
                }
            }
        },
    );
    Ok(ndjson(lines))
}

/// `POST /api/create`. Creating writes a few small blobs, so the whole operation runs before
/// the response starts: every failure is an ordinary HTTP error.
async fn create(State(s): State<Arc<AppState>>, body: Body) -> ApiResult<Response> {
    let req = parse(body, validate::create_request).await?;
    names([("model", &req.model), ("from", &req.from)])?;
    let spec = CreateSpec {
        from: req.from.clone(),
        questions: req
            .questions
            .as_ref()
            .map(ollaya_api::decide::engine_questions),
        calibration: req
            .calibration
            .as_ref()
            .map(|c| serde_json::to_value(c).expect("calibration serializes")),
        precision: req.parameters.as_ref().and_then(|p| p.precision.clone()),
        license: req.license.as_ref().map(|l| l.text()),
        description: req.description.clone(),
    };
    let lines = Arc::new(Mutex::new(Vec::new()));
    let sink = lines.clone();
    let report = move |status: String| sink.lock().unwrap().push(ProgressResponse::status(status));
    s.ollaya
        .create(&req.model, spec, &report)
        .await
        .map_err(|e| api_error(&e))?;
    if req.stream == Some(false) {
        return Ok(Json(ProgressResponse::success()).into_response());
    }
    let mut out: Vec<Vec<u8>> = lines.lock().unwrap().iter().map(line).collect();
    out.push(line(&ProgressResponse::success()));
    Ok(ndjson(futures_util::stream::iter(out)))
}

/// Holds one of the `max_queue` decision slots.
struct Slot(Arc<AppState>);

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

fn slot(s: &Arc<AppState>) -> ApiResult<Slot> {
    let before = s.in_flight.fetch_add(1, Ordering::SeqCst);
    if before >= s.config.max_queue {
        s.in_flight.fetch_sub(1, Ordering::SeqCst);
        return Err(ApiError::from(ErrorBody::queue_full(s.config.max_queue))
            .with_header(header::RETRY_AFTER, "1"));
    }
    Ok(Slot(s.clone()))
}

/// Marks the request cancelled when the handler is dropped (the client went away).
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Run a decision detached from the connection: a client that leaves does not abort a model
/// load (the next request finds the model warm), and the decision itself is skipped.
async fn run_decision(
    s: &Arc<AppState>,
    mut input: DecideInput,
) -> Result<crate::service::DecideOutput, Error> {
    let flag = Arc::new(AtomicBool::new(false));
    let _cancel = CancelOnDrop(flag.clone());
    input.cancel = Some(flag);
    let ollaya = s.ollaya.clone();
    tokio::spawn(async move { ollaya.decide(input).await })
        .await
        .map_err(|e| Error::Runner(format!("decision task failed: {e}")))?
}

/// The `type` of question `id` in a request, for `TOO_MANY_OPTIONS` locations.
fn qtype(questions: Option<&Questions>, id: &str) -> &'static str {
    questions
        .and_then(|q| q.get(id))
        .map_or("choice", Question::type_name)
}

fn decision_error(e: &Error, questions: Option<&Questions>) -> ApiError {
    match e {
        Error::TooManyOptions {
            question,
            options,
            model,
        } => ErrorBody::too_many_options(question, qtype(questions, question), *options, model)
            .into(),
        e => api_error(e),
    }
}

async fn decide(State(s): State<Arc<AppState>>, body: Body) -> ApiResult<Json<DecideResponse>> {
    let req = parse(body, validate::decide_request).await?;
    let [name] = names([("model", &req.model)])?;
    let _slot = slot(&s)?;
    let started = Instant::now();

    let Some(state) = req.state.clone() else {
        // No state: load or unload the model (Ollama's `generate` without a prompt).
        let unload = req.keep_alive == Some(KeepAlive::UNLOAD);
        let (ollaya, model, keep_alive) = (s.ollaya.clone(), req.model.clone(), req.keep_alive);
        let loading = tokio::spawn(async move { ollaya.preload(&model, keep_alive).await })
            .await
            .map_err(|e| api_error(&Error::LoadFailed(format!("load task failed: {e}"))))?
            .map_err(|e| api_error(&e))?;
        return Ok(Json(DecideResponse {
            model: name.to_string(),
            answers: IndexMap::new(),
            usage: Usage::default(),
            routing: None,
            state_truncated: false,
            done_reason: if unload {
                DoneReason::Unload
            } else {
                DoneReason::Load
            },
            created_at: chrono::Utc::now(),
            total_duration: started.elapsed().as_nanos() as u64,
            load_duration: loading.as_nanos() as u64,
            eval_duration: 0,
        }));
    };

    let input = DecideInput {
        model: req.model.clone(),
        state,
        questions: req
            .questions
            .as_ref()
            .map(ollaya_api::decide::engine_questions),
        keep_alive: req.keep_alive,
        cancel: None,
    };
    let out = run_decision(&s, input)
        .await
        .map_err(|e| decision_error(&e, req.questions.as_ref()))?;
    let resp = views::decide_response(out, req.wants(Extra::Laya)).map_err(|e| api_error(&e))?;
    Ok(Json(resp))
}

/// `POST /v1/systemone` and `/v1/decisions`: TypeSafe's shape, nothing else.
async fn systemone(
    State(s): State<Arc<AppState>>,
    body: Body,
) -> ApiResult<Json<ollaya_api::SystemOneResponse>> {
    let req = parse(body, validate::system_one_request).await?;
    names([("model", &req.model)])?;
    let _slot = slot(&s)?;
    let input = DecideInput {
        model: req.model.clone(),
        state: req.state.clone(),
        questions: req
            .questions
            .as_ref()
            .map(ollaya_api::decide::engine_questions),
        keep_alive: None,
        cancel: None,
    };
    let out = run_decision(&s, input)
        .await
        .map_err(|e| decision_error(&e, req.questions.as_ref()))?;
    let resp = views::decide_response(out, false).map_err(|e| api_error(&e))?;
    Ok(Json(resp.into_system_one()))
}

// ---------------------------------------------------------------------------------------------
// Middleware: request IDs, Host and Origin checks, CORS, auth

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn new_request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let n = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("req_{nanos:x}{n:04x}")
}

fn valid_request_id(id: &str) -> bool {
    (1..=128).contains(&id.len())
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// `*` matches any run of characters.
fn glob(pattern: &str, text: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == text,
        Some((prefix, rest)) => {
            let Some(text) = text.strip_prefix(prefix) else {
                return false;
            };
            (0..=text.len())
                .filter(|&i| text.is_char_boundary(i))
                .any(|i| glob(rest, &text[i..]))
        }
    }
}

/// A `Host` header allowed on a loopback-bound server (as Ollama allows them).
fn allowed_host(host: &str, hostname: Option<&str>) -> bool {
    let host = host.trim().to_ascii_lowercase();
    let bare = if let Some(v6) = host.strip_prefix('[') {
        v6.split(']').next().unwrap_or_default().to_owned()
    } else {
        match host.rsplit_once(':') {
            Some((h, port)) if port.bytes().all(|b| b.is_ascii_digit()) => h.to_owned(),
            _ => host.clone(),
        }
    };
    if bare.is_empty()
        || bare == "localhost"
        || hostname == Some(bare.as_str())
        || [".localhost", ".local", ".internal"]
            .iter()
            .any(|tld| bare.ends_with(tld))
    {
        return true;
    }
    match bare.parse::<IpAddr>() {
        Ok(IpAddr::V4(ip)) => {
            ip.is_loopback() || ip.is_private() || ip.is_unspecified() || ip.is_link_local()
        }
        Ok(IpAddr::V6(ip)) => ip.is_loopback() || ip.is_unspecified() || ip.is_unique_local(),
        Err(_) => false,
    }
}

fn bearer_ok(headers: &HeaderMap, key: &str) -> bool {
    let Some(value) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    else {
        return false;
    };
    let Some((scheme, token)) = value.trim().split_once(' ') else {
        return false;
    };
    let (a, b) = (token.trim().as_bytes(), key.as_bytes());
    // Constant time in the key's content.
    scheme.eq_ignore_ascii_case("bearer")
        && a.len() == b.len()
        && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

enum Admit {
    Pass,
    Preflight,
}

fn admit(s: &AppState, req: &Request, origin: Option<&str>) -> ApiResult<Admit> {
    if s.loopback
        && let Some(host) = req
            .headers()
            .get(header::HOST)
            .and_then(|v| v.to_str().ok())
        && !allowed_host(host, s.hostname.as_deref())
    {
        return Err(ErrorBody::forbidden(&format!("Host {host:?}")).into());
    }
    if let Some(o) = origin
        && !s.origins.iter().any(|p| glob(p, o))
    {
        return Err(ErrorBody::forbidden(&format!("origin {o:?}")).into());
    }
    if req.method() == Method::OPTIONS
        && origin.is_some()
        && req
            .headers()
            .contains_key(header::ACCESS_CONTROL_REQUEST_METHOD)
    {
        return Ok(Admit::Preflight);
    }
    if let Some(key) = &s.config.api_key {
        let liveness =
            req.uri().path() == "/" && matches!(*req.method(), Method::GET | Method::HEAD);
        if !liveness && !bearer_ok(req.headers(), key) {
            return Err(ApiError::from(ErrorBody::unauthorized())
                .with_header(header::WWW_AUTHENTICATE, "Bearer"));
        }
    }
    Ok(Admit::Pass)
}

const CORS_ALLOW_HEADERS: &str = "Authorization, Content-Type, Accept, User-Agent, X-Requested-With, X-Request-Id, X-TypeSafe-SDK, X-TypeSafe-Runtime, X-TypeSafe-Retry-Count";
const CORS_EXPOSE_HEADERS: &str = "X-Request-Id, x-typesafe-request-id, Retry-After";

async fn guard(State(s): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let started = Instant::now();
    let id = req
        .headers()
        .get(ollaya_api::REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|v| valid_request_id(v))
        .map_or_else(new_request_id, str::to_owned);
    let (method, path) = (req.method().clone(), req.uri().path().to_owned());
    let origin = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let (mut resp, cors) = match admit(&s, &req, origin.as_deref()) {
        Err(e) => (e.into_response(), false),
        Ok(Admit::Preflight) => {
            let mut r = StatusCode::NO_CONTENT.into_response();
            let h = r.headers_mut();
            h.insert(
                header::ACCESS_CONTROL_ALLOW_METHODS,
                HeaderValue::from_static("GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS"),
            );
            h.insert(
                header::ACCESS_CONTROL_ALLOW_HEADERS,
                HeaderValue::from_static(CORS_ALLOW_HEADERS),
            );
            h.insert(
                header::ACCESS_CONTROL_MAX_AGE,
                HeaderValue::from_static("86400"),
            );
            (r, true)
        }
        Ok(Admit::Pass) => (next.run(req).await, origin.is_some()),
    };
    let h = resp.headers_mut();
    if let Ok(v) = HeaderValue::from_str(&id) {
        if path.starts_with("/v1/") {
            h.insert(ollaya_api::TYPESAFE_REQUEST_ID_HEADER, v.clone());
        }
        h.insert(ollaya_api::REQUEST_ID_HEADER, v);
    }
    if cors
        && let Some(v) = origin
            .as_deref()
            .and_then(|o| HeaderValue::from_str(o).ok())
    {
        h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, v);
        h.insert(header::VARY, HeaderValue::from_static("Origin"));
        h.insert(
            header::ACCESS_CONTROL_EXPOSE_HEADERS,
            HeaderValue::from_static(CORS_EXPOSE_HEADERS),
        );
    }
    tracing::debug!(request_id = %id, %method, %path, status = resp.status().as_u16(),
        ms = started.elapsed().as_millis() as u64, "request");
    resp
}

// ---------------------------------------------------------------------------------------------
// Serving

#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Service(#[from] Error),
}

/// How runner processes are started (see [`crate::launch`]).
pub use crate::launch::RunnerLaunch;

/// The store, scheduler and service for `config`.
pub fn build(config: ServerConfig, runner: RunnerLaunch) -> Result<Arc<AppState>, ServeError> {
    let store = Store::open(&config.models).map_err(Error::from)?;
    if ollaya_registry::name::is_default_registry() {
        for legacy in ollaya_registry::name::LEGACY_REGISTRIES {
            let to = ollaya_registry::name::DEFAULT_REGISTRY;
            match store.migrate_host(legacy, to) {
                Ok(0) => {}
                Ok(n) => tracing::info!("moved {n} manifests pulled from {legacy} to {to}"),
                Err(e) => tracing::warn!("could not move the models pulled from {legacy}: {e}"),
            }
        }
    }
    let scheduler = Scheduler::new(SchedulerConfig {
        keep_alive: config.keep_alive,
        max_loaded: config.max_loaded,
        device: config.device.clone(),
        load_timeout: config.load_timeout,
        exe: runner.exe,
        arg0: runner.arg0,
        env: runner.env,
        llama_dir: runner.llama_dir,
    });
    let ollaya = Ollaya::new(store, scheduler)?;
    Ok(AppState::new(ollaya, config))
}

/// The daemon's sockets: the bound address, plus `[::1]` next to a `127.0.0.1` or `localhost`
/// host (see [`Listeners::bind`]).
pub struct Listeners(Vec<TcpListener>);

impl Listeners {
    /// Binds `host`. A loopback host (`127.0.0.1`, `localhost`) also gets `[::1]` on the same
    /// port when the system has IPv6. Windows resolves `localhost` to `::1` first, and WSL only
    /// forwards it to a Linux process that listens there; without it, every new connection from a
    /// Windows program to `localhost:11435` waits about 200 ms for the fallback to IPv4.
    pub async fn bind(host: &Host) -> std::io::Result<Listeners> {
        if !(host.host.eq_ignore_ascii_case("localhost") || host.host == "127.0.0.1") {
            return Ok(Listeners(vec![TcpListener::bind(host.bind_addr()).await?]));
        }
        let v4 = TcpListener::bind(("127.0.0.1", host.port)).await?;
        let port = v4.local_addr()?.port();
        let mut all = vec![v4];
        match TcpListener::bind(("::1", port)).await {
            Ok(v6) => all.push(v6),
            Err(e) => tracing::debug!("not listening on [::1]:{port}: {e}"),
        }
        Ok(Listeners(all))
    }

    pub fn local_addrs(&self) -> Vec<SocketAddr> {
        self.0.iter().filter_map(|l| l.local_addr().ok()).collect()
    }
}

impl From<TcpListener> for Listeners {
    fn from(listener: TcpListener) -> Listeners {
        Listeners(vec![listener])
    }
}

impl axum::serve::Listener for Listeners {
    type Io = TcpStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (TcpStream, SocketAddr) {
        loop {
            let accepted = std::future::poll_fn(|cx| {
                for listener in &self.0 {
                    if let Poll::Ready(r) = listener.poll_accept(cx) {
                        return Poll::Ready(r);
                    }
                }
                Poll::Pending
            })
            .await;
            match accepted {
                Ok(conn) => return conn,
                // The peer gave up before we accepted: nothing to do.
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::ConnectionRefused
                            | std::io::ErrorKind::ConnectionAborted
                            | std::io::ErrorKind::ConnectionReset
                    ) => {}
                // Out of file descriptors and the like, as axum handles it: wait for some to close.
                Err(e) => {
                    tracing::error!("accept error: {e}");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.0[0].local_addr()
    }
}

/// Serve until `shutdown` resolves, then give open connections 5 s and stop every runner.
pub async fn serve_on(
    listeners: impl Into<Listeners>,
    state: Arc<AppState>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let scheduler = state.ollaya.scheduler.clone();
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let server = axum::serve(listeners.into(), router(state))
        .with_graceful_shutdown(async {
            let _ = stop_rx.await;
        })
        .into_future();
    tokio::pin!(server);
    tokio::select! {
        r = &mut server => r?,
        () = shutdown => {
            tracing::info!("shutting down");
            let _ = stop_tx.send(());
            if tokio::time::timeout(Duration::from_secs(5), &mut server).await.is_err() {
                tracing::warn!("connections still open after 5 s; closing them");
            }
        }
    }
    scheduler.shutdown().await;
    Ok(())
}

/// Ctrl-C or SIGTERM.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = term => {}
    }
}

/// `ollaya serve`: bind `OLLAYA_HOST`, serve until Ctrl-C or SIGTERM, stop every runner.
pub async fn serve(config: ServerConfig) -> Result<(), ServeError> {
    let listeners = Listeners::bind(&config.host).await?;
    serve_bound(config, listeners).await
}

/// [`serve`] on sockets the caller has already bound with [`Listeners::bind`].
pub async fn serve_bound(config: ServerConfig, listeners: Listeners) -> Result<(), ServeError> {
    let state = build(config, RunnerLaunch::current()?)?;
    let local = listeners
        .local_addrs()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    if !state.loopback && state.config.api_key.is_none() {
        tracing::warn!(
            "listening on {local}, which is reachable from other machines, without OLLAYA_API_KEY: \
             anyone who can reach it can run, pull and delete models"
        );
    }
    tracing::info!(address = %local, version = VERSION, models = %state.config.models.display(),
        "Ollaya is running");
    serve_on(listeners, state, shutdown_signal()).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn loopback_hosts_listen_on_both_loopbacks() {
        let has_v6 = TcpListener::bind("[::1]:0").await.is_ok();
        for name in ["127.0.0.1:0", "localhost:0"] {
            let listeners = Listeners::bind(&Host::parse(name).unwrap()).await.unwrap();
            let addrs = listeners.local_addrs();
            assert!(
                addrs[0].ip().is_loopback() && addrs[0].is_ipv4(),
                "{name}: {addrs:?}"
            );
            if has_v6 {
                assert_eq!(addrs.len(), 2, "{name}: {addrs:?}");
                assert!(addrs[1].is_ipv6() && addrs[1].ip().is_loopback());
                assert_eq!(addrs[0].port(), addrs[1].port());
            }
        }
        let listeners = Listeners::bind(&Host::parse("0.0.0.0:0").unwrap())
            .await
            .unwrap();
        assert_eq!(listeners.local_addrs().len(), 1);
    }

    #[test]
    fn globs() {
        assert!(glob("http://localhost:*", "http://localhost:3000"));
        assert!(glob("https://*.example.com", "https://a.b.example.com"));
        assert!(glob("*", "anything"));
        assert!(!glob("http://localhost", "http://localhost:3000"));
        assert!(!glob("https://*.example.com", "https://example.org"));
    }

    #[test]
    fn hosts() {
        for ok in [
            "localhost:11435",
            "127.0.0.1:11435",
            "[::1]:11435",
            "10.1.2.3",
            "box.local",
            "my-pc",
            "app.localhost",
        ] {
            assert!(allowed_host(ok, Some("my-pc")), "{ok}");
        }
        for bad in ["evil.com", "evil.com:11435", "8.8.8.8", "[2001:db8::1]:80"] {
            assert!(!allowed_host(bad, Some("my-pc")), "{bad}");
        }
    }

    #[test]
    fn bearer() {
        let mut h = HeaderMap::new();
        assert!(!bearer_ok(&h, "k"));
        h.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer k"));
        assert!(bearer_ok(&h, "k"));
        h.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("bearer  k "),
        );
        assert!(bearer_ok(&h, "k"));
        h.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer kk"));
        assert!(!bearer_ok(&h, "k"));
        h.insert(header::AUTHORIZATION, HeaderValue::from_static("Basic k"));
        assert!(!bearer_ok(&h, "k"));
    }

    #[test]
    fn request_ids() {
        assert!(valid_request_id("abc-123_x.y"));
        assert!(
            !valid_request_id("")
                && !valid_request_id("a b")
                && !valid_request_id(&"x".repeat(129))
        );
        assert_ne!(new_request_id(), new_request_id());
    }
}
