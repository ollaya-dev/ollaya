//! Ollaya's HTTP contract, as specified in `docs/api.md`.
//!
//! * [`decide`]: questions, answers and the `/api/decide` + `/v1/systemone` bodies. Answers keep
//!   TypeSafe's wire shapes exactly.
//! * [`models`]: model management (`/api/tags`, `show`, `ps`, `pull`, `delete`, `copy`, `create`,
//!   `version`) and TypeSafe's `/v1/models`.
//! * [`error`]: the one error body every endpoint returns, and its codes.
//! * [`validate`]: boundary validation that turns a JSON body into a typed request, or into the
//!   `detail` list of a `422`.
//! * [`KeepAlive`]: Ollama's `keep_alive` values.
//! * [`Client`]: the async HTTP client the CLI uses.
//!
//! The server and the client both build on these types, so the contract lives in one place.

pub mod client;
pub mod decide;
pub mod error;
pub mod host;
pub mod keep_alive;
pub mod models;
pub mod presets;
pub mod validate;

pub use client::{Client, ClientError};
pub use decide::{
    Answer, ChoiceAnswer, ChoiceCriteria, ChoiceQuestion, DecideAnswer, DecideRequest,
    DecideResponse, DoneReason, Extra, LayaExtra, NoulAnswer, NoulCriteria, NoulQuestion, Question,
    Questions, Routing, ScoreAnswer, ScoreQuestion, SystemOneRequest, SystemOneResponse, Usage,
};
pub use error::{ErrorBody, ErrorCode, Loc, ValidationIssue};
pub use keep_alive::{KeepAlive, KeepAliveError};
pub use models::{
    CalibrationSpec, CopyRequest, CreateParameters, CreateRequest, DeleteRequest, License,
    LocalModel, ModelDetails, ModelList, ModelMetadata, ProgressResponse, PsResponse, PullRequest,
    RouterInfo, RunningModel, ShowRequest, ShowResponse, TagsResponse, VersionResponse,
};

/// Port the daemon listens on by default (Ollama's 11434, plus one).
pub const DEFAULT_PORT: u16 = 11435;
/// `OLLAYA_HOST` when unset.
pub const DEFAULT_HOST: &str = "127.0.0.1:11435";

/// Largest request body, in bytes (`413 REQUEST_TOO_LARGE` above it).
pub const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
/// Questions per request.
pub const MIN_QUESTIONS: usize = 1;
pub const MAX_QUESTIONS: usize = 256;
/// Options of a `choice` question.
pub const MIN_CHOICE_OPTIONS: usize = 2;
pub const MAX_CHOICE_OPTIONS: usize = 255;
/// Levels of a `score` question.
pub const MIN_SCORE_LEVELS: usize = 2;
pub const MAX_SCORE_LEVELS: usize = 10;
/// Longest `state`, in tokens of the answering model (TypeSafe's limit; `422 INPUT_TOO_LONG`).
pub const MAX_STATE_TOKENS: usize = 65_536;
/// Waiting requests before `503 QUEUE_FULL` (`OLLAYA_MAX_QUEUE`).
pub const DEFAULT_MAX_QUEUE: usize = 512;
/// Seconds sent in `Retry-After` with `QUEUE_FULL`.
pub const QUEUE_FULL_RETRY_AFTER_SECS: u64 = 1;

/// `GET /` body.
pub const RUNNING_MESSAGE: &str = "Ollaya is running";
/// Content type of streaming responses.
pub const NDJSON_CONTENT_TYPE: &str = "application/x-ndjson";
/// Request ID header on every response.
pub const REQUEST_ID_HEADER: &str = "x-request-id";
/// The same ID on `/v1/*` responses, where the TypeSafe SDK reads it.
pub const TYPESAFE_REQUEST_ID_HEADER: &str = "x-typesafe-request-id";
