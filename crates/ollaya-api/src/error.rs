//! The error body every endpoint returns (`docs/api.md` §4).
//!
//! ```json
//! {"error": "<message>", "code": "MODEL_NOT_FOUND", "detail": [ ... ]}
//! ```
//!
//! `error` stays a string so Ollama clients and the TypeSafe SDK's `extract_message` read it
//! directly; `code` is the machine-readable addition; `detail` is TypeSafe's (FastAPI's)
//! `ValidationError` list, present only for validation failures.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value, json};

use crate::{MAX_STATE_TOKENS, QUEUE_FULL_RETRY_AFTER_SECS};

/// Machine-readable error code. The set is open: unknown codes deserialize to [`ErrorCode::Other`]
/// and clients fall back to the HTTP status.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    InvalidJson,
    InvalidRequest,
    TooManyOptions,
    InputTooLong,
    Unauthorized,
    Forbidden,
    ModelNotFound,
    NotFound,
    MethodNotAllowed,
    OperationInProgress,
    RequestTooLarge,
    QueueFull,
    ModelLoadFailed,
    InferenceFailed,
    StorageError,
    Internal,
    UnsupportedModel,
    NotImplemented,
    RegistryError,
    DigestMismatch,
    /// A code this version of the contract does not know.
    Other(String),
}

impl ErrorCode {
    /// Every code this version defines, in the order of the table in `docs/api.md` §4.2.
    pub const ALL: [ErrorCode; 20] = [
        ErrorCode::InvalidJson,
        ErrorCode::InvalidRequest,
        ErrorCode::TooManyOptions,
        ErrorCode::InputTooLong,
        ErrorCode::Unauthorized,
        ErrorCode::Forbidden,
        ErrorCode::ModelNotFound,
        ErrorCode::NotFound,
        ErrorCode::MethodNotAllowed,
        ErrorCode::OperationInProgress,
        ErrorCode::RequestTooLarge,
        ErrorCode::QueueFull,
        ErrorCode::ModelLoadFailed,
        ErrorCode::InferenceFailed,
        ErrorCode::StorageError,
        ErrorCode::Internal,
        ErrorCode::UnsupportedModel,
        ErrorCode::NotImplemented,
        ErrorCode::RegistryError,
        ErrorCode::DigestMismatch,
    ];

    pub fn as_str(&self) -> &str {
        match self {
            ErrorCode::InvalidJson => "INVALID_JSON",
            ErrorCode::InvalidRequest => "INVALID_REQUEST",
            ErrorCode::TooManyOptions => "TOO_MANY_OPTIONS",
            ErrorCode::InputTooLong => "INPUT_TOO_LONG",
            ErrorCode::Unauthorized => "UNAUTHORIZED",
            ErrorCode::Forbidden => "FORBIDDEN",
            ErrorCode::ModelNotFound => "MODEL_NOT_FOUND",
            ErrorCode::NotFound => "NOT_FOUND",
            ErrorCode::MethodNotAllowed => "METHOD_NOT_ALLOWED",
            ErrorCode::OperationInProgress => "OPERATION_IN_PROGRESS",
            ErrorCode::RequestTooLarge => "REQUEST_TOO_LARGE",
            ErrorCode::QueueFull => "QUEUE_FULL",
            ErrorCode::ModelLoadFailed => "MODEL_LOAD_FAILED",
            ErrorCode::InferenceFailed => "INFERENCE_FAILED",
            ErrorCode::StorageError => "STORAGE_ERROR",
            ErrorCode::Internal => "INTERNAL",
            ErrorCode::UnsupportedModel => "UNSUPPORTED_MODEL",
            ErrorCode::NotImplemented => "NOT_IMPLEMENTED",
            ErrorCode::RegistryError => "REGISTRY_ERROR",
            ErrorCode::DigestMismatch => "DIGEST_MISMATCH",
            ErrorCode::Other(code) => code,
        }
    }

    pub fn parse(code: &str) -> ErrorCode {
        ErrorCode::ALL
            .into_iter()
            .find(|c| c.as_str() == code)
            .unwrap_or_else(|| ErrorCode::Other(code.to_owned()))
    }

    /// The HTTP status this code is sent with. Unknown codes report 500.
    pub fn status(&self) -> u16 {
        match self {
            ErrorCode::InvalidJson => 400,
            ErrorCode::Unauthorized => 401,
            ErrorCode::Forbidden => 403,
            ErrorCode::ModelNotFound | ErrorCode::NotFound => 404,
            ErrorCode::MethodNotAllowed => 405,
            ErrorCode::OperationInProgress => 409,
            ErrorCode::RequestTooLarge => 413,
            ErrorCode::InvalidRequest | ErrorCode::TooManyOptions | ErrorCode::InputTooLong => 422,
            ErrorCode::ModelLoadFailed
            | ErrorCode::InferenceFailed
            | ErrorCode::StorageError
            | ErrorCode::Internal
            | ErrorCode::Other(_) => 500,
            ErrorCode::UnsupportedModel | ErrorCode::NotImplemented => 501,
            ErrorCode::RegistryError | ErrorCode::DigestMismatch => 502,
            ErrorCode::QueueFull => 503,
        }
    }

    /// Best code for a response that carried none (another server, a proxy).
    pub fn from_status(status: u16) -> ErrorCode {
        match status {
            400 => ErrorCode::InvalidJson,
            401 => ErrorCode::Unauthorized,
            403 => ErrorCode::Forbidden,
            404 => ErrorCode::NotFound,
            405 => ErrorCode::MethodNotAllowed,
            409 => ErrorCode::OperationInProgress,
            413 => ErrorCode::RequestTooLarge,
            422 => ErrorCode::InvalidRequest,
            501 => ErrorCode::NotImplemented,
            502 => ErrorCode::RegistryError,
            503 => ErrorCode::QueueFull,
            _ => ErrorCode::Internal,
        }
    }

    /// Whether `detail` accompanies this code.
    pub fn has_detail(&self) -> bool {
        matches!(
            self,
            ErrorCode::InvalidRequest | ErrorCode::TooManyOptions | ErrorCode::InputTooLong
        )
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for ErrorCode {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ErrorCode {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(ErrorCode::parse(&String::deserialize(d)?))
    }
}

/// One element of an issue's `loc`: an object key or an array index.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Loc {
    Index(u64),
    Key(String),
}

impl Loc {
    pub fn key(k: impl Into<String>) -> Loc {
        Loc::Key(k.into())
    }
}

impl From<&str> for Loc {
    fn from(k: &str) -> Loc {
        Loc::Key(k.to_owned())
    }
}

impl From<String> for Loc {
    fn from(k: String) -> Loc {
        Loc::Key(k)
    }
}

impl From<usize> for Loc {
    fn from(i: usize) -> Loc {
        Loc::Index(i as u64)
    }
}

impl fmt::Display for Loc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Loc::Index(i) => write!(f, "{i}"),
            Loc::Key(k) => f.write_str(k),
        }
    }
}

/// One validation problem, in TypeSafe's (FastAPI's) `ValidationError` shape. Ollaya never sends
/// `input`, so user data is not echoed back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub loc: Vec<Loc>,
    pub msg: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx: Option<Map<String, Value>>,
}

fn ctx(value: Value) -> Option<Map<String, Value>> {
    match value {
        Value::Object(m) => Some(m),
        _ => None,
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

impl ValidationIssue {
    pub fn new(loc: Vec<Loc>, kind: impl Into<String>, msg: impl Into<String>) -> Self {
        ValidationIssue {
            loc,
            msg: msg.into(),
            kind: kind.into(),
            ctx: None,
        }
    }

    pub fn with_ctx(mut self, value: Value) -> Self {
        self.ctx = ctx(value);
        self
    }

    /// `Field required`.
    pub fn missing(loc: Vec<Loc>) -> Self {
        ValidationIssue::new(loc, "missing", "Field required")
    }

    /// Wrong JSON type: `string_type`, `bool_type`, `dict_type`, `list_type`, `int_type`.
    pub fn wrong_type(loc: Vec<Loc>, kind: &str) -> Self {
        let what = match kind {
            "string_type" => "a valid string",
            "bool_type" => "a valid boolean",
            "dict_type" => "a valid dictionary",
            "list_type" => "a valid list",
            "int_type" => "a valid integer",
            "float_type" => "a valid number",
            _ => "valid",
        };
        ValidationIssue::new(loc, kind, format!("Input should be {what}"))
    }

    /// `state` is not a string, object or array.
    pub fn state_type(loc: Vec<Loc>) -> Self {
        ValidationIssue::new(
            loc,
            "state_type",
            "Input should be a string, an object or an array",
        )
    }

    /// `instructions` or a criterion has a type TypeSafe does not accept.
    pub fn json_type(loc: Vec<Loc>, null_allowed: bool) -> Self {
        let msg = if null_allowed {
            "Input should be a string, an object, an array or null"
        } else {
            "Input should be a string, an object or an array"
        };
        ValidationIssue::new(loc, "json_type", msg)
    }

    /// A list or dictionary with too few items (`field_type` is `List` or `Dictionary`).
    pub fn too_short(loc: Vec<Loc>, field_type: &str, min: usize, actual: usize) -> Self {
        ValidationIssue::new(
            loc,
            "too_short",
            format!(
                "{field_type} should have at least {min} item{} after validation, not {actual}",
                plural(min)
            ),
        )
        .with_ctx(json!({"field_type": field_type, "min_length": min, "actual_length": actual}))
    }

    /// A list or dictionary with too many items.
    pub fn too_long(loc: Vec<Loc>, field_type: &str, max: usize, actual: usize) -> Self {
        ValidationIssue::new(
            loc,
            "too_long",
            format!(
                "{field_type} should have at most {max} item{} after validation, not {actual}",
                plural(max)
            ),
        )
        .with_ctx(json!({"field_type": field_type, "max_length": max, "actual_length": actual}))
    }

    /// An empty string where a name is required.
    pub fn string_too_short(loc: Vec<Loc>) -> Self {
        ValidationIssue::new(
            loc,
            "string_too_short",
            "String should have at least 1 character",
        )
        .with_ctx(json!({"min_length": 1}))
    }

    /// A question without a string `type`.
    pub fn union_tag_not_found(loc: Vec<Loc>) -> Self {
        ValidationIssue::new(
            loc,
            "union_tag_not_found",
            "Unable to extract tag using discriminator 'type'",
        )
        .with_ctx(json!({"discriminator": "'type'"}))
    }

    /// A question whose `type` is not `choice`, `score` or `noul`.
    pub fn union_tag_invalid(loc: Vec<Loc>, tag: &str) -> Self {
        let expected = "'choice', 'score', 'noul'";
        ValidationIssue::new(
            loc,
            "union_tag_invalid",
            format!(
                "Input tag '{tag}' found using 'type' does not match any of the expected tags: {expected}"
            ),
        )
        .with_ctx(json!({"discriminator": "'type'", "tag": tag, "expected_tags": expected}))
    }

    /// A model name that does not parse as `[host/][namespace/]model[:tag]`. The daemon builds
    /// this from `ollaya-registry`'s parser, which owns the grammar.
    pub fn model_name(loc: Vec<Loc>, input: &str) -> Self {
        ValidationIssue::new(
            loc,
            "model_name",
            format!("invalid model name {input:?}; expected [host/][namespace/]model[:tag]"),
        )
    }

    /// An unparseable `keep_alive`.
    pub fn keep_alive(loc: Vec<Loc>, msg: impl Into<String>) -> Self {
        ValidationIssue::new(loc, "keep_alive", msg)
    }

    /// A value outside a closed set; `expected` lists the allowed values, quoted.
    pub fn enumeration(loc: Vec<Loc>, expected: &str) -> Self {
        ValidationIssue::new(loc, "enum", format!("Input should be {expected}"))
            .with_ctx(json!({"expected": expected}))
    }

    /// `"stream": true` on `/api/decide`.
    pub fn stream_unsupported(loc: Vec<Loc>) -> Self {
        ValidationIssue::new(
            loc,
            "stream_unsupported",
            "/api/decide does not stream; omit \"stream\" or set it to false",
        )
    }

    /// An unknown or invalid `/api/create` parameter.
    pub fn parameter(loc: Vec<Loc>, msg: impl Into<String>) -> Self {
        ValidationIssue::new(loc, "parameter", msg)
    }

    /// An invalid `/api/create` calibration entry.
    pub fn calibration(loc: Vec<Loc>, msg: impl Into<String>) -> Self {
        ValidationIssue::new(loc, "calibration", msg)
    }

    /// Options that do not fit the answering model's option budget.
    pub fn too_many_options(question_id: &str, qtype: &str, options: usize, model: &str) -> Self {
        ValidationIssue::new(
            body_loc(&["questions", question_id, qtype, "criteria"]),
            "too_many_options",
            format!("{options} options do not fit the option budget of {model}"),
        )
        .with_ctx(json!({"options": options, "model": model}))
    }

    /// A state longer than [`MAX_STATE_TOKENS`].
    pub fn input_too_long(tokens: usize) -> Self {
        ValidationIssue::new(
            body_loc(&["state"]),
            "input_too_long",
            format!("state is {tokens} tokens long; the limit is {MAX_STATE_TOKENS}"),
        )
        .with_ctx(json!({"max_tokens": MAX_STATE_TOKENS, "tokens": tokens}))
    }

    /// `loc` as the TypeSafe SDK prints it: keys joined by `.`, without the leading `body`.
    pub fn path(&self) -> String {
        self.loc
            .iter()
            .filter(|l| !matches!(l, Loc::Key(k) if k == "body"))
            .map(Loc::to_string)
            .collect::<Vec<_>>()
            .join(".")
    }
}

/// `["body", parts...]`.
pub fn body_loc(parts: &[&str]) -> Vec<Loc> {
    std::iter::once("body")
        .chain(parts.iter().copied())
        .map(Loc::from)
        .collect()
}

/// The message the TypeSafe SDK's `extract_message` builds from a `detail` list:
/// `"<path>: <msg>"` per issue, joined with `"; "`.
pub fn issues_message(issues: &[ValidationIssue]) -> String {
    issues
        .iter()
        .map(|i| match i.path() {
            p if p.is_empty() => i.msg.clone(),
            p => format!("{p}: {}", i.msg),
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The error body of `docs/api.md` §4.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorBody {
    /// Human-readable message; not stable.
    pub error: String,
    /// Machine-readable code; branch on this.
    pub code: ErrorCode,
    /// Validation issues, only for `INVALID_REQUEST`, `TOO_MANY_OPTIONS` and `INPUT_TOO_LONG`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<Vec<ValidationIssue>>,
}

impl fmt::Display for ErrorBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.error)
    }
}

impl std::error::Error for ErrorBody {}

impl ErrorBody {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        ErrorBody {
            error: message.into(),
            code,
            detail: None,
        }
    }

    /// An error carrying validation issues; the message is built from them.
    pub fn with_issues(code: ErrorCode, issues: Vec<ValidationIssue>) -> Self {
        ErrorBody {
            error: issues_message(&issues),
            code,
            detail: Some(issues),
        }
    }

    /// `422 INVALID_REQUEST` from boundary validation.
    pub fn invalid_request(issues: Vec<ValidationIssue>) -> Self {
        ErrorBody::with_issues(ErrorCode::InvalidRequest, issues)
    }

    /// The HTTP status to send this body with.
    pub fn status(&self) -> u16 {
        self.code.status()
    }

    /// `400 INVALID_JSON`.
    pub fn invalid_json(message: impl Into<String>) -> Self {
        ErrorBody::new(ErrorCode::InvalidJson, message)
    }

    /// `404 MODEL_NOT_FOUND` for a local lookup. The sentence is Ollama's, verbatim: tools match
    /// on "not found, try pulling it first".
    pub fn model_not_found(canonical: &str) -> Self {
        ErrorBody::new(
            ErrorCode::ModelNotFound,
            format!("model {canonical:?} not found, try pulling it first"),
        )
    }

    /// `404 MODEL_NOT_FOUND` for a router's missing target.
    pub fn routed_model_not_found(target: &str, router: &str) -> Self {
        ErrorBody::new(
            ErrorCode::ModelNotFound,
            format!("model {target:?} not found, try pulling it first (routed from {router:?})"),
        )
    }

    /// `404 MODEL_NOT_FOUND` from `/api/pull`: the registry has no such model.
    pub fn not_in_registry(canonical: &str, registry: &str) -> Self {
        ErrorBody::new(
            ErrorCode::ModelNotFound,
            format!("model {canonical:?} not found in registry {registry}"),
        )
    }

    /// `422 TOO_MANY_OPTIONS`.
    pub fn too_many_options(question_id: &str, qtype: &str, options: usize, model: &str) -> Self {
        ErrorBody::with_issues(
            ErrorCode::TooManyOptions,
            vec![ValidationIssue::too_many_options(
                question_id,
                qtype,
                options,
                model,
            )],
        )
    }

    /// `422 INPUT_TOO_LONG`.
    pub fn input_too_long(tokens: usize) -> Self {
        ErrorBody::with_issues(
            ErrorCode::InputTooLong,
            vec![ValidationIssue::input_too_long(tokens)],
        )
    }

    /// `503 QUEUE_FULL`; send with `Retry-After: 1`.
    pub fn queue_full(max_queue: usize) -> Self {
        ErrorBody::new(
            ErrorCode::QueueFull,
            format!(
                "server busy: {max_queue} requests are already waiting; retry in {QUEUE_FULL_RETRY_AFTER_SECS} s"
            ),
        )
    }

    /// `409 OPERATION_IN_PROGRESS`.
    pub fn operation_in_progress(canonical: &str) -> Self {
        ErrorBody::new(
            ErrorCode::OperationInProgress,
            format!("model {canonical:?} is being pulled or created; retry when it finishes"),
        )
    }

    /// `413 REQUEST_TOO_LARGE`.
    pub fn request_too_large() -> Self {
        ErrorBody::new(
            ErrorCode::RequestTooLarge,
            format!(
                "request body is larger than {} bytes",
                crate::MAX_BODY_BYTES
            ),
        )
    }

    /// `401 UNAUTHORIZED`; send with `WWW-Authenticate: Bearer`.
    pub fn unauthorized() -> Self {
        ErrorBody::new(
            ErrorCode::Unauthorized,
            "missing or invalid API key; send Authorization: Bearer <OLLAYA_API_KEY>",
        )
    }

    /// `403 FORBIDDEN` for a disallowed `Origin` or `Host`.
    pub fn forbidden(what: &str) -> Self {
        ErrorBody::new(ErrorCode::Forbidden, format!("{what} is not allowed"))
    }

    /// `404 NOT_FOUND` for an unknown path. Ollama's text endpoints get a pointer to
    /// `/api/decide`.
    pub fn path_not_found(path: &str) -> Self {
        let text_endpoint = matches!(
            path,
            "/api/generate" | "/api/chat" | "/api/embed" | "/api/embeddings"
        );
        let message = if text_endpoint {
            format!(
                "{path} is not served by ollaya: decision models do not generate text; use POST /api/decide"
            )
        } else {
            format!("{path} not found")
        };
        ErrorBody::new(ErrorCode::NotFound, message)
    }

    /// `405 METHOD_NOT_ALLOWED`; send with an `Allow` header.
    pub fn method_not_allowed(method: &str, path: &str) -> Self {
        ErrorBody::new(
            ErrorCode::MethodNotAllowed,
            format!("{method} is not allowed on {path}"),
        )
    }

    /// `501 NOT_IMPLEMENTED` for reserved endpoints.
    pub fn not_implemented(path: &str) -> Self {
        ErrorBody::new(
            ErrorCode::NotImplemented,
            format!("{path} is reserved and not implemented yet"),
        )
    }

    /// `502 DIGEST_MISMATCH`.
    pub fn digest_mismatch(digest: &str) -> Self {
        ErrorBody::new(
            ErrorCode::DigestMismatch,
            format!("blob {digest} does not match its digest; the download was discarded"),
        )
    }

    /// `500 INTERNAL`. The message never carries internals; they go to the log.
    pub fn internal(request_id: &str) -> Self {
        ErrorBody::new(
            ErrorCode::Internal,
            format!("internal error; see the server log for request {request_id}"),
        )
    }

    /// Read any error response leniently: this contract's body, TypeSafe/FastAPI bodies
    /// (`detail`), `{"error": {"message"}}`, `{"message"}`, or plain text. The message follows
    /// TypeSafe's `extract_message` precedence; a missing code comes from the status.
    pub fn from_response(status: u16, body: &[u8]) -> ErrorBody {
        let fallback = || ErrorCode::from_status(status);
        let parsed: Option<Value> = serde_json::from_slice(body).ok();
        let Some(Value::Object(obj)) = parsed else {
            let text = String::from_utf8_lossy(body).trim().to_owned();
            let error = if text.is_empty() {
                format!("HTTP {status}")
            } else {
                text
            };
            return ErrorBody {
                error,
                code: fallback(),
                detail: None,
            };
        };
        let detail: Option<Vec<ValidationIssue>> = obj
            .get("detail")
            .and_then(|d| serde_json::from_value(d.clone()).ok());
        let message = extract_message(&obj).unwrap_or_else(|| format!("HTTP {status}"));
        let code = obj
            .get("code")
            .and_then(Value::as_str)
            .or_else(|| obj.get("error")?.get("code")?.as_str())
            .map(ErrorCode::parse)
            .unwrap_or_else(fallback);
        ErrorBody {
            error: message,
            code,
            detail,
        }
    }
}

/// TypeSafe's `extract_message`, for JSON objects.
fn extract_message(obj: &Map<String, Value>) -> Option<String> {
    let text = |v: Option<&Value>| v.and_then(Value::as_str).map(str::to_owned);
    if let Some(s) = text(obj.get("error")) {
        return Some(s);
    }
    if let Some(s) = text(obj.get("error").and_then(|e| e.get("message"))) {
        return Some(s);
    }
    if let Some(s) = text(obj.get("message")) {
        return Some(s);
    }
    match obj.get("detail") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(d)) => text(d.get("message")),
        Some(Value::Array(entries)) => {
            let parts: Vec<String> = entries
                .iter()
                .filter_map(|e| {
                    let msg = e.get("msg")?.as_str()?;
                    let path = match e.get("loc") {
                        Some(Value::Array(loc)) => loc
                            .iter()
                            .filter(|l| l.as_str() != Some("body"))
                            .map(|l| match l {
                                Value::String(s) => s.clone(),
                                other => other.to_string(),
                            })
                            .collect::<Vec<_>>()
                            .join("."),
                        _ => String::new(),
                    };
                    Some(if path.is_empty() {
                        msg.to_owned()
                    } else {
                        format!("{path}: {msg}")
                    })
                })
                .collect();
            (!parts.is_empty()).then(|| parts.join("; "))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip_and_map_to_statuses() {
        for code in ErrorCode::ALL {
            assert_eq!(ErrorCode::parse(code.as_str()), code);
            let json = serde_json::to_string(&code).unwrap();
            assert_eq!(serde_json::from_str::<ErrorCode>(&json).unwrap(), code);
            assert!((400..600).contains(&code.status()), "{code}");
        }
        assert_eq!(
            ErrorCode::parse("SOMETHING_NEW"),
            ErrorCode::Other("SOMETHING_NEW".into())
        );
        assert_eq!(ErrorCode::QueueFull.status(), 503);
        assert_eq!(ErrorCode::InvalidRequest.status(), 422);
        assert_eq!(ErrorCode::InvalidJson.status(), 400);
    }

    #[test]
    fn message_matches_typesafe_extract_message() {
        let issues = vec![
            ValidationIssue::missing(body_loc(&["state"])),
            ValidationIssue::too_short(
                body_loc(&["questions", "urgency", "score", "criteria"]),
                "List",
                2,
                1,
            ),
        ];
        let body = ErrorBody::invalid_request(issues);
        assert_eq!(
            body.error,
            "state: Field required; questions.urgency.score.criteria: List should have at least 2 items after validation, not 1"
        );
        // Re-reading our own body with the SDK's precedence gives the same message.
        let bytes = serde_json::to_vec(&body).unwrap();
        assert_eq!(ErrorBody::from_response(422, &bytes), body);
    }

    #[test]
    fn lenient_parse_follows_sdk_precedence() {
        let cases: [(&str, &str); 6] = [
            (r#"{"error":"plain"}"#, "plain"),
            (r#"{"error":{"code":"X","message":"nested"}}"#, "nested"),
            (r#"{"message":"top"}"#, "top"),
            (r#"{"detail":"fastapi"}"#, "fastapi"),
            (r#"{"detail":{"message":"dict"}}"#, "dict"),
            (
                r#"{"detail":[{"loc":["body","questions",0],"msg":"bad","type":"x"}]}"#,
                "questions.0: bad",
            ),
        ];
        for (body, message) in cases {
            assert_eq!(
                ErrorBody::from_response(400, body.as_bytes()).error,
                message,
                "{body}"
            );
        }
        let nested = ErrorBody::from_response(404, br#"{"error":{"code":"X","message":"m"}}"#);
        assert_eq!(nested.code, ErrorCode::Other("X".into()));
        let text = ErrorBody::from_response(502, b"Bad Gateway");
        assert_eq!(
            (text.error.as_str(), text.code),
            ("Bad Gateway", ErrorCode::RegistryError)
        );
        assert_eq!(ErrorBody::from_response(500, b"").error, "HTTP 500");
    }
}
