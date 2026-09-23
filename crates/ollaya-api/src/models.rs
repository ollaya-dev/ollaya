//! Model management (`docs/api.md` §7.2, §7.4–§7.10) and TypeSafe's `GET /v1/models` (§8.3).
//!
//! Shapes and names mirror Ollama's native API.

use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::decide::Questions;

/// `GET /api/version`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionResponse {
    pub version: String,
}

/// `details` of `/api/tags`, `/api/show` and `/api/ps` entries.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDetails {
    /// `from` of a created model; `""` otherwise.
    pub parent_model: String,
    /// Open set: `onnx`, `gguf`, `router`.
    pub format: String,
    pub family: String,
    #[serde(default)]
    pub families: Vec<String>,
    /// e.g. `421M`; `""` for routers.
    pub parameter_size: String,
    /// `F32`, `F16`, `INT8`; `""` for routers.
    pub quantization_level: String,
}

/// One `GET /api/tags` entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalModel {
    pub name: String,
    /// Same as `name`.
    pub model: String,
    pub modified_at: DateTime<Utc>,
    /// Bytes of the model's own blobs.
    pub size: u64,
    /// sha256 of the manifest, bare hex.
    pub digest: String,
    pub details: ModelDetails,
}

/// `GET /api/tags`: newest `modified_at` first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TagsResponse {
    pub models: Vec<LocalModel>,
}

/// `POST /api/show`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShowRequest {
    pub model: String,
}

/// A router's rules, as `/api/show` reports them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouterInfo {
    /// e.g. `script`.
    pub strategy: String,
    /// Route used when detection is inconclusive.
    pub default: String,
    /// Route name -> canonical model name.
    pub routes: IndexMap<String, String>,
}

/// `POST /api/show` response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShowResponse {
    pub license: String,
    /// Informative Modelfile text; not for parsing.
    pub modelfile: String,
    /// `name value` per line.
    pub parameters: String,
    /// Embedded question schema.
    pub questions: Option<Questions>,
    pub router: Option<RouterInfo>,
    pub details: ModelDetails,
    /// `general.*` keys are stable; `<family>.*` keys are family-specific.
    pub model_info: IndexMap<String, Value>,
    /// Open set: `choice`, `score`, `noul`, `act`.
    pub capabilities: Vec<String>,
    pub modified_at: DateTime<Utc>,
}

/// One `GET /api/ps` entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunningModel {
    pub name: String,
    pub model: String,
    /// Estimated memory, RAM plus VRAM, in bytes.
    pub size: u64,
    pub digest: String,
    pub details: ModelDetails,
    /// `None` (JSON `null`) when kept loaded indefinitely.
    pub expires_at: Option<DateTime<Utc>>,
    pub size_vram: u64,
    /// The model's `max_len`, in tokens.
    pub context_length: u64,
    /// Where the runner computes: `cpu`, `cuda:0`, …
    pub device: String,
}

/// `GET /api/ps`: sorted by name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsResponse {
    pub models: Vec<RunningModel>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// `POST /api/pull`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PullRequest {
    pub model: String,
    /// Allow `http://` registries and skip TLS verification.
    #[serde(default, skip_serializing_if = "is_false")]
    pub insecure: bool,
    /// Defaults to `true` on the server.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

impl PullRequest {
    pub fn new(model: impl Into<String>) -> Self {
        PullRequest {
            model: model.into(),
            insecure: false,
            stream: None,
        }
    }
}

/// One line of a pull or create stream (and the whole `"stream": false` response).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgressResponse {
    /// Open set; see `docs/api.md` §7.6 and §7.9.
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}

pub mod status {
    pub const PULLING_MANIFEST: &str = "pulling manifest";
    pub const VERIFYING: &str = "verifying sha256 digest";
    pub const WRITING_MANIFEST: &str = "writing manifest";
    pub const SUCCESS: &str = "success";
}

/// First 12 hex characters of a digest, as Ollama prints them.
pub fn short_digest(digest: &str) -> &str {
    let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
    &hex[..hex.len().min(12)]
}

impl ProgressResponse {
    pub fn status(status: impl Into<String>) -> Self {
        ProgressResponse {
            status: status.into(),
            ..Default::default()
        }
    }

    pub fn success() -> Self {
        ProgressResponse::status(status::SUCCESS)
    }

    /// `pulling <short digest>` with byte counts.
    pub fn layer(digest: &str, total: u64, completed: u64) -> Self {
        ProgressResponse {
            status: format!("pulling {}", short_digest(digest)),
            digest: Some(digest.to_owned()),
            total: Some(total),
            completed: Some(completed),
        }
    }

    /// `using existing layer <digest>` (create).
    pub fn existing_layer(digest: &str) -> Self {
        ProgressResponse::status(format!("using existing layer {digest}"))
    }

    /// `creating new layer <digest>` (create).
    pub fn new_layer(digest: &str) -> Self {
        ProgressResponse::status(format!("creating new layer {digest}"))
    }

    pub fn is_success(&self) -> bool {
        self.status == status::SUCCESS
    }
}

/// `DELETE /api/delete`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeleteRequest {
    pub model: String,
}

/// `POST /api/copy`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CopyRequest {
    pub source: String,
    pub destination: String,
}

/// `calibration` of `/api/create`: the calibration layer's format.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationSpec {
    /// One temperature per question type: choice, score, noul.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub temperature: Vec<f64>,
    /// `"<type>:<2|3-5|6-10|11+>"` -> temperature.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub temperature_by_options: IndexMap<String, f64>,
}

/// `parameters` of `/api/create` (closed set; Modelfile `PARAMETER`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateParameters {
    /// Pin the graph precision: `fp16` or `fp32`. Unset, the device decides (fp16 on CUDA).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<String>,
}

/// One license text or several.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum License {
    One(String),
    Many(Vec<String>),
}

/// `POST /api/create`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateRequest {
    pub model: String,
    pub from: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub questions: Option<Questions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<CreateParameters>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<License>,
    /// One line for `/v1/models` and `ollaya show`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Defaults to `true` on the server.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

impl License {
    /// All texts as one, separated by a blank line.
    pub fn text(&self) -> String {
        match self {
            License::One(s) => s.clone(),
            License::Many(v) => v.join("\n\n"),
        }
    }
}

/// One entry of TypeSafe's `ModelMetadataList`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub description: String,
    /// `YYYY-MM-DD`.
    pub release_date: String,
}

/// `GET /v1/models`: sorted by name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelList {
    pub models: Vec<ModelMetadata>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_lines_match_ollama() {
        let d = "sha256:8d32a80bb199bcd4ff10abc28d651fe576fb59f86b24039402e49be9e01578c2";
        assert_eq!(
            serde_json::to_string(&ProgressResponse::layer(d, 10, 0)).unwrap(),
            format!(
                r#"{{"status":"pulling 8d32a80bb199","digest":"{d}","total":10,"completed":0}}"#
            )
        );
        assert_eq!(
            serde_json::to_string(&ProgressResponse::success()).unwrap(),
            r#"{"status":"success"}"#
        );
        assert_eq!(short_digest("abc"), "abc");
    }
}
