//! Model resolution: a name in the local store to what a runner needs to load it.

use std::path::PathBuf;

use ollaya_decision::{Calibration, CalibrationFile};
use ollaya_registry::manifest::{ANNOTATION_PRECISION, media};
use ollaya_registry::{Entry, ModelConfig, ModelName, Router, Store};
use serde_json::Value;

use crate::Error;

/// Files a runner loads, all inside the blob store.
#[derive(Debug, Clone, PartialEq)]
pub struct RunnerFiles {
    pub graph_fp32: Option<PathBuf>,
    pub graph_fp16: Option<PathBuf>,
    pub tokenizer: PathBuf,
    pub decision: PathBuf,
}

/// A model that answers requests itself.
#[derive(Debug, Clone)]
pub struct Loadable {
    pub name: ModelName,
    /// Manifest digest: runners are keyed by content, so two tags of one model share a runner.
    pub digest: String,
    pub files: RunnerFiles,
    pub calibration: Calibration,
    pub config: ModelConfig,
    /// A question schema baked in with a Modelfile, used when a request brings none.
    pub questions: Option<Value>,
    /// Bytes on disk, for `ps` and scheduling.
    pub size: u64,
}

#[derive(Debug, Clone)]
pub enum Resolved {
    Model(Box<Loadable>),
    /// A router: the target is chosen per request.
    Router {
        name: ModelName,
        digest: String,
        router: Box<Router>,
        config: Box<ModelConfig>,
        questions: Option<Value>,
    },
}

pub fn resolve(store: &Store, name: &ModelName) -> Result<Resolved, Error> {
    let entry = store
        .read_manifest(name)?
        .ok_or_else(|| Error::ModelNotFound(name.to_string()))?;
    resolve_entry(store, entry)
}

pub fn resolve_entry(store: &Store, entry: Entry) -> Result<Resolved, Error> {
    let Entry {
        name,
        manifest,
        digest,
        ..
    } = entry;
    let config: ModelConfig = store.read_blob_json(&manifest.config)?;
    let questions = match manifest.layer(media::QUESTIONS) {
        Some(d) => Some(store.read_blob_json::<Value>(d)?),
        None => None,
    };
    if let Some(router) = manifest.layer(media::ROUTER) {
        let router = store.read_blob_json(router)?;
        return Ok(Resolved::Router {
            name,
            digest,
            router: Box::new(router),
            config: Box::new(config),
            questions,
        });
    }
    if !matches!(config.model_format.as_str(), "" | "onnx") {
        return Err(Error::Unsupported(format!(
            "{name} is a {} model, which this build of ollaya cannot run",
            config.model_format
        )));
    }
    let blob = |media_type: &str| -> Result<PathBuf, Error> {
        let d = manifest
            .layer(media_type)
            .ok_or_else(|| Error::Corrupt(format!("{name}: manifest has no {media_type} layer")))?;
        Ok(store.blob_path(&d.digest)?)
    };
    let mut files = RunnerFiles {
        graph_fp32: None,
        graph_fp16: None,
        tokenizer: blob(media::TOKENIZER)?,
        decision: blob(media::DECISION)?,
    };
    for g in manifest.layers_of(media::GRAPH_ONNX) {
        let path = store.blob_path(&g.digest)?;
        match g.annotations.get(ANNOTATION_PRECISION).map(String::as_str) {
            Some("fp16") => files.graph_fp16 = Some(path),
            _ => files.graph_fp32 = Some(path),
        }
    }
    // A precision-pinned tag (`laya:en-fp32`) keeps only the graph it names.
    if let Some(params) = manifest.layer(media::PARAMS) {
        let params: Value = store.read_blob_json(params)?;
        match params["precision"].as_str() {
            Some("fp32") => files.graph_fp16 = None,
            Some("fp16") => files.graph_fp32 = None,
            _ => {}
        }
    }
    if files.graph_fp32.is_none() && files.graph_fp16.is_none() {
        return Err(Error::Corrupt(format!("{name}: manifest has no graph")));
    }
    let calibration = match manifest.layer(media::CALIBRATION) {
        Some(d) => Calibration::from_file(&store.read_blob_json::<CalibrationFile>(d)?),
        None => Calibration::default(),
    };
    Ok(Resolved::Model(Box::new(Loadable {
        name,
        digest,
        files,
        calibration,
        config,
        questions,
        size: manifest.total_size(),
    })))
}
