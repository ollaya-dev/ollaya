//! Manifests: Docker distribution v2 (schema 2), with Ollaya's own layer media types.
//!
//! Every layer is content-addressed by sha256. A descriptor may list `urls` (the OCI "foreign
//! layer" field): Ollaya uses them to fetch weights straight from the author's Hugging Face
//! repository, so the registry only ever serves manifests and small derived files.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const MANIFEST_V2: &str = "application/vnd.docker.distribution.manifest.v2+json";

pub mod media {
    pub const CONFIG: &str = "application/vnd.ollaya.config.v1+json";
    /// ONNX graph. `org.ollaya.precision` says which precision it computes in.
    pub const GRAPH_ONNX: &str = "application/vnd.ollaya.graph.onnx";
    /// Weights file a graph references by its blob file name (ONNX external data).
    pub const WEIGHTS: &str = "application/vnd.ollaya.weights";
    pub const TOKENIZER: &str = "application/vnd.ollaya.tokenizer";
    pub const DECISION: &str = "application/vnd.ollaya.decision";
    pub const CALIBRATION: &str = "application/vnd.ollaya.calibration";
    /// A question schema baked into a derived model (Modelfile `QUESTIONS`).
    pub const QUESTIONS: &str = "application/vnd.ollaya.questions";
    /// Routing rules: this model dispatches each request to one of several other models.
    pub const ROUTER: &str = "application/vnd.ollaya.router";
    pub const PARAMS: &str = "application/vnd.ollaya.params";
    pub const LICENSE: &str = "application/vnd.ollaya.license";
    pub const README: &str = "application/vnd.ollaya.readme";
}

pub const ANNOTATION_PRECISION: &str = "org.ollaya.precision";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Descriptor {
    pub media_type: String,
    pub digest: String,
    pub size: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub urls: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub annotations: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub schema_version: u32,
    pub media_type: String,
    pub config: Descriptor,
    pub layers: Vec<Descriptor>,
}

/// The config blob: what `ollaya list` / `show` report about a model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// `onnx`, `gguf`, or `router` for a model that only dispatches.
    pub model_format: String,
    pub family: String,
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub context_length: u32,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub description: String,
    /// Where the weights come from, e.g. `huggingface.co/convaiinnovations/laya@<commit>`.
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub license: String,
    /// When the upstream checkpoint was published (YYYY-MM-DD), for `/v1/models`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_date: Option<String>,
}

impl Manifest {
    pub fn layers_of<'a>(
        &'a self,
        media_type: &'a str,
    ) -> impl Iterator<Item = &'a Descriptor> + 'a {
        self.layers
            .iter()
            .filter(move |l| l.media_type == media_type)
    }

    pub fn layer(&self, media_type: &str) -> Option<&Descriptor> {
        self.layers.iter().find(|l| l.media_type == media_type)
    }

    /// Every blob the manifest needs locally, config included.
    pub fn blobs(&self) -> impl Iterator<Item = &Descriptor> {
        std::iter::once(&self.config).chain(self.layers.iter())
    }

    /// Total bytes of all blobs (what `ollaya list` shows as SIZE).
    pub fn total_size(&self) -> u64 {
        self.blobs().map(|d| d.size).sum()
    }
}

/// `laya:latest`-style routing: which model answers, decided per request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Router {
    /// Routing strategy; `script` routes on the state's script and language.
    pub strategy: String,
    /// Route name -> model name, e.g. `english` -> `laya:en`.
    pub routes: BTreeMap<String, String>,
    pub default: String,
}
