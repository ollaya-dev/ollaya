//! Async HTTP client for the Ollaya daemon, used by the CLI.
//!
//! The base URL comes from `OLLAYA_HOST` (default `127.0.0.1:11435`) and the optional bearer key
//! from `OLLAYA_API_KEY`. Non-2xx responses become [`ClientError::Api`] with the server's
//! [`ErrorBody`]; a failure inside a stream becomes the same error.

use std::pin::Pin;
use std::time::Duration;

use futures_util::{Stream, StreamExt};
use reqwest::{Method, RequestBuilder, Response};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::decide::{DecideRequest, DecideResponse, SystemOneRequest, SystemOneResponse};
use crate::error::{ErrorBody, ErrorCode};
use crate::host::{Host, InvalidHost};
use crate::models::{
    CopyRequest, CreateRequest, DeleteRequest, ModelList, ProgressResponse, PsResponse,
    PullRequest, ShowRequest, ShowResponse, TagsResponse, VersionResponse,
};

pub const API_KEY_ENV: &str = "OLLAYA_API_KEY";

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// The server answered with an error body (non-2xx, or an error line inside a stream).
    /// `status` is the HTTP status; for a stream error it is the code's status.
    #[error("{body}")]
    Api { status: u16, body: ErrorBody },
    /// No connection: the daemon is probably not running.
    #[error("could not connect to ollaya at {url}; is `ollaya serve` running? ({source})")]
    Connect {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("request to {url} failed: {source}")]
    Http {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    /// A 2xx body that is not what the contract says.
    #[error("unexpected response from {url}: {message}")]
    Decode { url: String, message: String },
    /// A stream ended without `success` or an error line.
    #[error("the stream from {url} ended before it finished")]
    Incomplete { url: String },
    #[error(transparent)]
    InvalidHost(#[from] InvalidHost),
}

impl ClientError {
    /// The error code, for API errors.
    pub fn code(&self) -> Option<&ErrorCode> {
        match self {
            ClientError::Api { body, .. } => Some(&body.code),
            _ => None,
        }
    }

    /// Whether the model is not on this machine (`ollaya run` pulls it then).
    pub fn is_model_not_found(&self) -> bool {
        self.code() == Some(&ErrorCode::ModelNotFound)
    }
}

pub type Result<T> = std::result::Result<T, ClientError>;

/// Progress lines of `/api/pull` and `/api/create`.
pub type ProgressStream = Pin<Box<dyn Stream<Item = Result<ProgressResponse>> + Send>>;

#[derive(Debug, Clone)]
pub struct Client {
    base: String,
    http: reqwest::Client,
    api_key: Option<String>,
}

impl Client {
    /// A client for `host`, in any `OLLAYA_HOST` form.
    pub fn new(host: &str) -> Result<Self> {
        let host = Host::parse(host)?;
        let mut builder = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .user_agent(concat!("ollaya/", env!("CARGO_PKG_VERSION")));
        if host.is_local() {
            // A system proxy must never see traffic to the local daemon.
            builder = builder.no_proxy();
        }
        let http = builder.build().map_err(|source| ClientError::Http {
            url: host.base_url(),
            source,
        })?;
        Ok(Client {
            base: host.base_url(),
            http,
            api_key: None,
        })
    }

    /// `OLLAYA_HOST` and `OLLAYA_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let client = Client::new(&std::env::var(crate::host::HOST_ENV).unwrap_or_default())?;
        Ok(match std::env::var(API_KEY_ENV) {
            Ok(key) if !key.trim().is_empty() => client.with_api_key(key.trim()),
            _ => client,
        })
    }

    /// Send `Authorization: Bearer <key>` on every request.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base
    }

    /// `HEAD /`: is the daemon up?
    pub async fn heartbeat(&self) -> Result<()> {
        self.send(Method::HEAD, "/", None::<&()>).await.map(drop)
    }

    /// `GET /api/version`.
    pub async fn version(&self) -> Result<VersionResponse> {
        self.get_json("/api/version").await
    }

    /// `GET /api/tags`.
    pub async fn tags(&self) -> Result<TagsResponse> {
        self.get_json("/api/tags").await
    }

    /// `GET /api/ps`.
    pub async fn ps(&self) -> Result<PsResponse> {
        self.get_json("/api/ps").await
    }

    /// `POST /api/show`.
    pub async fn show(&self, model: &str) -> Result<ShowResponse> {
        let req = ShowRequest {
            model: model.to_owned(),
        };
        self.post_json("/api/show", &req).await
    }

    /// `DELETE /api/delete`. A `404 MODEL_NOT_FOUND` is returned as an error; retrying callers
    /// treat it as success.
    pub async fn delete(&self, model: &str) -> Result<()> {
        let req = DeleteRequest {
            model: model.to_owned(),
        };
        self.send(Method::DELETE, "/api/delete", Some(&req))
            .await
            .map(drop)
    }

    /// `POST /api/copy`.
    pub async fn copy(&self, source: &str, destination: &str) -> Result<()> {
        let req = CopyRequest {
            source: source.to_owned(),
            destination: destination.to_owned(),
        };
        self.send(Method::POST, "/api/copy", Some(&req))
            .await
            .map(drop)
    }

    /// `POST /api/decide`.
    pub async fn decide(&self, req: &DecideRequest) -> Result<DecideResponse> {
        self.post_json("/api/decide", req).await
    }

    /// Load `model` now (`/api/decide` without a state).
    pub async fn load(
        &self,
        model: &str,
        keep_alive: Option<crate::KeepAlive>,
    ) -> Result<DecideResponse> {
        self.decide(&DecideRequest::load(model, keep_alive)).await
    }

    /// Unload `model` (`/api/decide` with `keep_alive: 0`), as `ollaya stop` does.
    pub async fn unload(&self, model: &str) -> Result<DecideResponse> {
        self.decide(&DecideRequest::unload(model)).await
    }

    /// `POST /v1/systemone`.
    pub async fn systemone(&self, req: &SystemOneRequest) -> Result<SystemOneResponse> {
        self.post_json("/v1/systemone", req).await
    }

    /// `GET /v1/models`.
    pub async fn models(&self) -> Result<ModelList> {
        self.get_json("/v1/models").await
    }

    /// `POST /api/pull`, streaming. The stream ends after `success` or yields the error.
    pub async fn pull_stream(&self, req: &PullRequest) -> Result<ProgressStream> {
        let req = PullRequest {
            stream: Some(true),
            ..req.clone()
        };
        self.progress_stream("/api/pull", &req).await
    }

    /// `POST /api/pull`, calling `on_progress` for every line until `success`.
    pub async fn pull(
        &self,
        req: &PullRequest,
        on_progress: impl FnMut(&ProgressResponse),
    ) -> Result<()> {
        drain(self.pull_stream(req).await?, on_progress).await
    }

    /// `POST /api/create`, streaming.
    pub async fn create_stream(&self, req: &CreateRequest) -> Result<ProgressStream> {
        let req = CreateRequest {
            stream: Some(true),
            ..req.clone()
        };
        self.progress_stream("/api/create", &req).await
    }

    /// `POST /api/create`, calling `on_progress` for every line until `success`.
    pub async fn create(
        &self,
        req: &CreateRequest,
        on_progress: impl FnMut(&ProgressResponse),
    ) -> Result<()> {
        drain(self.create_stream(req).await?, on_progress).await
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn request(&self, method: Method, url: &str) -> RequestBuilder {
        let rb = self.http.request(method, url);
        match &self.api_key {
            Some(key) => rb.bearer_auth(key),
            None => rb,
        }
    }

    /// Send a request; non-2xx responses become [`ClientError::Api`].
    async fn send<B: Serialize + ?Sized>(
        &self,
        method: Method,
        path: &str,
        body: Option<&B>,
    ) -> Result<Response> {
        let url = self.url(path);
        let mut rb = self.request(method, &url);
        if let Some(body) = body {
            rb = rb.json(body);
        }
        let resp = rb.send().await.map_err(|source| {
            if source.is_connect() {
                ClientError::Connect {
                    url: url.clone(),
                    source,
                }
            } else {
                ClientError::Http {
                    url: url.clone(),
                    source,
                }
            }
        })?;
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|source| ClientError::Http { url, source })?;
        Err(ClientError::Api {
            status: status.as_u16(),
            body: ErrorBody::from_response(status.as_u16(), &bytes),
        })
    }

    async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let resp = self.send(Method::GET, path, None::<&()>).await?;
        decode(resp, self.url(path)).await
    }

    async fn post_json<B: Serialize + ?Sized, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let resp = self.send(Method::POST, path, Some(body)).await?;
        decode(resp, self.url(path)).await
    }

    async fn progress_stream<B: Serialize + ?Sized>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<ProgressStream> {
        let resp = self.send(Method::POST, path, Some(body)).await?;
        Ok(ndjson(resp, self.url(path)))
    }
}

async fn decode<T: DeserializeOwned>(resp: Response, url: String) -> Result<T> {
    let bytes = resp.bytes().await.map_err(|source| ClientError::Http {
        url: url.clone(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|e| ClientError::Decode {
        url,
        message: e.to_string(),
    })
}

async fn drain(
    mut stream: ProgressStream,
    mut on_progress: impl FnMut(&ProgressResponse),
) -> Result<()> {
    while let Some(line) = stream.next().await {
        let line = line?;
        on_progress(&line);
        if line.is_success() {
            return Ok(());
        }
    }
    Ok(())
}

/// One NDJSON line: a progress update or an error body.
fn parse_line(line: &[u8], url: &str) -> Result<ProgressResponse> {
    let value: Value = serde_json::from_slice(line).map_err(|e| ClientError::Decode {
        url: url.to_owned(),
        message: format!("bad stream line: {e}"),
    })?;
    if value.get("error").is_some() {
        let body = ErrorBody::from_response(500, line);
        return Err(ClientError::Api {
            status: body.code.status(),
            body,
        });
    }
    serde_json::from_value(value).map_err(|e| ClientError::Decode {
        url: url.to_owned(),
        message: format!("bad progress line: {e}"),
    })
}

struct LineReader<S> {
    bytes: Pin<Box<S>>,
    buf: Vec<u8>,
    url: String,
    finished: bool,
}

/// Turn an NDJSON response into a stream of progress lines. The stream ends after the terminal
/// line (`success` or an error); ending without one yields [`ClientError::Incomplete`].
fn ndjson(resp: Response, url: String) -> ProgressStream {
    let reader = LineReader {
        bytes: Box::pin(resp.bytes_stream()),
        buf: Vec::new(),
        url,
        finished: false,
    };
    Box::pin(futures_util::stream::unfold(reader, |mut r| async move {
        loop {
            if r.finished {
                return None;
            }
            if let Some(pos) = r.buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = r.buf.drain(..=pos).collect();
                if line.iter().all(u8::is_ascii_whitespace) {
                    continue;
                }
                let item = parse_line(&line, &r.url);
                r.finished = !matches!(&item, Ok(p) if !p.is_success());
                return Some((item, r));
            }
            match r.bytes.next().await {
                Some(Ok(chunk)) => r.buf.extend_from_slice(&chunk),
                Some(Err(source)) => {
                    r.finished = true;
                    let url = r.url.clone();
                    return Some((Err(ClientError::Http { url, source }), r));
                }
                None if r.buf.iter().any(|b| !b.is_ascii_whitespace()) => {
                    // A last line without a trailing newline (e.g. a `"stream": false` body).
                    let line = std::mem::take(&mut r.buf);
                    let item = parse_line(&line, &r.url);
                    r.finished = true;
                    let item = match item {
                        Ok(p) if !p.is_success() => {
                            Err(ClientError::Incomplete { url: r.url.clone() })
                        }
                        other => other,
                    };
                    return Some((item, r));
                }
                None => {
                    r.finished = true;
                    let url = r.url.clone();
                    return Some((Err(ClientError::Incomplete { url }), r));
                }
            }
        }
    }))
}
