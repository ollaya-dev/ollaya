//! Engines: a loaded model that turns a request into per-question option logits.
//!
//! Each model family has its own sequence layout (how state and questions become network
//! inputs) and graph contract (which tensors go in and come out). The `decision` layer names
//! the layout; [`load`] picks the engine for it. Engines only produce raw option logits:
//! calibration and answer rendering stay in the daemon, identical for every family.

use std::path::Path;

use ollaya_decision::Questions;
use serde_json::Value;

use crate::onnx::{Device, ModelFiles, OnnxModel};
use crate::{Error, Output};

pub trait Engine: Send + Sync {
    /// Answer every question about `state`, returning one logit per option per question.
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error>;
}

impl Engine for OnnxModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        OnnxModel::run(self, state, questions)
    }
}

/// Layouts this build can run.
pub const LAYOUTS: &[&str] = &["laya-markers-v1"];

/// The layout a `decision` layer declares.
pub fn layout_of(decision: &Path) -> Result<String, Error> {
    let text = std::fs::read_to_string(decision)
        .map_err(|e| Error::Model(format!("{}: {e}", decision.display())))?;
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| Error::Model(format!("{}: {e}", decision.display())))?;
    v["layout"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| Error::Model("decision layer has no layout".into()))
}

/// Load the engine for a model's layout on `device`.
pub fn load(
    files: &ModelFiles,
    device: Device,
    threads: Option<usize>,
) -> Result<Box<dyn Engine>, Error> {
    match layout_of(&files.decision)?.as_str() {
        "laya-markers-v1" => Ok(Box::new(OnnxModel::load_files(files, device, threads)?)),
        other => Err(Error::Model(format!(
            "this version of ollaya cannot run layout {other:?} (supported: {}); upgrade ollaya",
            LAYOUTS.join(", ")
        ))),
    }
}
