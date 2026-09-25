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

    /// Answer a request whose questions are the JSON the daemon sent. Engines whose layouts
    /// validate the definitions themselves (llama.cpp's `winnow-v1`, `llm-logits-v1`) override
    /// it; the rest parse the typed questions first.
    fn run_json(&self, state: &Value, questions: &Value) -> Result<Output, Error> {
        let questions = ollaya_decision::parse_questions(questions)?;
        self.run(state, &questions)
    }

    /// The only questions a fixed-preset model answers (its built-in set); `None` for models
    /// that answer any typed question.
    fn preset(&self) -> Option<&Questions> {
        None
    }
}

impl Engine for OnnxModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        OnnxModel::run(self, state, questions)
    }
}

/// Padded tokens per `session.run`. Rows are independent, so splitting a request only bounds
/// peak memory: 64 rows of 512 tokens fit comfortably even for large encoders on the GPU.
pub const TOKEN_BUDGET: usize = 32_768;

/// Split `lens` (row lengths, in order) into consecutive batches whose padded size
/// (rows × longest row) stays within `budget`, and at most `max_rows` rows each.
pub fn batches(lens: &[usize], budget: usize, max_rows: usize) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let (mut start, mut longest) = (0, 0);
    for (i, &len) in lens.iter().enumerate() {
        let next_longest = longest.max(len);
        let rows = i - start + 1;
        if rows > 1 && (rows * next_longest > budget || rows > max_rows) {
            out.push(start..i);
            start = i;
            longest = len;
        } else {
            longest = next_longest;
        }
    }
    if start < lens.len() {
        out.push(start..lens.len());
    }
    out
}

/// Layouts this build can run.
pub const LAYOUTS: &[&str] = &[
    "laya-markers-v1",
    "gliclass-uni-v1",
    "nli-pairs-v1",
    "decider-slots-v1",
    "kev-pointer-v1",
    "qwen3guard-gen-v1",
    "von-option-marker-v1",
    "decision-endpoint-v1",
];

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
        "gliclass-uni-v1" => Ok(Box::new(crate::gliclass::GliclassModel::load_files(
            files, device, threads,
        )?)),
        "nli-pairs-v1" => Ok(Box::new(crate::nli::NliModel::load_files(
            files, device, threads,
        )?)),
        "decider-slots-v1" => Ok(Box::new(crate::decider::DeciderModel::load_files(
            files, device, threads,
        )?)),
        "kev-pointer-v1" => Ok(Box::new(crate::kev::KevModel::load_files(
            files, device, threads,
        )?)),
        "qwen3guard-gen-v1" => Ok(Box::new(crate::qwen3guard::GuardModel::load_files(
            files, device, threads,
        )?)),
        "von-option-marker-v1" => Ok(Box::new(crate::von::VonModel::load_files(
            files, device, threads,
        )?)),
        "decision-endpoint-v1" => Ok(Box::new(crate::decision::DecisionModel::load_files(
            files, device, threads,
        )?)),
        other => Err(Error::Model(format!(
            "this version of ollaya cannot run layout {other:?} (supported: {}); upgrade ollaya",
            LAYOUTS.join(", ")
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::batches;

    #[test]
    fn batches_respect_the_token_budget() {
        // 512-token rows, budget 2048 -> 4 rows per batch.
        assert_eq!(
            batches(&[512; 10], 2048, usize::MAX),
            vec![0..4, 4..8, 8..10]
        );
        // A longer row later on shrinks its batch; a single row over budget still runs alone.
        assert_eq!(
            batches(&[100, 100, 900, 5000, 10], 2000, usize::MAX),
            vec![0..2, 2..3, 3..4, 4..5]
        );
        assert_eq!(batches(&[10; 5], 1_000_000, 2), vec![0..2, 2..4, 4..5]);
        assert!(batches(&[], 10, 10).is_empty());
    }
}
