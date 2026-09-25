//! ONNX Runtime engine for Qwen3Guard-Gen (layout `qwen3guard-gen-v1`), a fixed-preset model.
//!
//! A request is one or two rows (the safety line, and the categories line when `category` is
//! asked). The graph maps `input_ids` [rows, seq] (right-padded, positions implicit, no mask:
//! every layer is causal) and `last_pos` [rows] to `cand_logits` [rows, 13]: the next-token logits
//! of the label first tokens. It answers only its preset (`ollaya_decision::qwen3guard`); any
//! other question is rejected.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Ix2};
use ollaya_decision::qwen3guard::{GuardConfig, GuardLayout, GuardRows};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder};
use ort::session::Session;
use serde::Deserialize;
use serde_json::Value;

use crate::engine::Engine;
use crate::onnx::{Device, ModelFiles, load_tokenizer, session_with};
use crate::{Error, Output, QuestionOutput};

/// Padded tokens per `session.run`. Attention is full, so long rows run one at a time.
const TOKEN_BUDGET: usize = 8192;
const INPUTS: [&str; 2] = ["input_ids", "last_pos"];
const OUTPUT: &str = "cand_logits";

/// The fields of the `decision` layer this engine reads.
#[derive(Debug, Clone, Deserialize)]
struct DecisionConfig {
    engine: String,
    layout: String,
    #[serde(flatten)]
    guard: GuardConfig,
}

pub struct GuardModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    /// Tokens the prompt adds around the state (reported state tokens exclude them).
    overhead: usize,
    pub layout: GuardLayout,
    pub calibration: Calibration,
    pub device: Device,
}

struct Tokenizer(tokenizers::Tokenizer);

impl TokenEncoder for Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, ollaya_decision::Error> {
        self.0
            .encode_fast(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| ollaya_decision::Error::Tokenizer(e.to_string()))
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::Model(format!("{}: {e}", path.display())))
}

impl Engine for GuardModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        GuardModel::run(self, state, questions)
    }

    fn preset(&self) -> Option<&Questions> {
        Some(&self.layout.preset)
    }
}

impl GuardModel {
    /// Load a model exported to one directory (development and parity tooling).
    pub fn load(dir: &Path, device: Device, intra_threads: Option<usize>) -> Result<Self, Error> {
        Self::load_files(&ModelFiles::dir(dir), device, intra_threads)
    }

    pub fn load_files(
        files: &ModelFiles,
        device: Device,
        intra_threads: Option<usize>,
    ) -> Result<Self, Error> {
        let config: DecisionConfig = read_json(&files.decision)?;
        if config.engine != "onnx" || config.layout != "qwen3guard-gen-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/qwen3guard-gen-v1",
                config.engine, config.layout
            )));
        }
        let layout = GuardLayout::new(config.guard)
            .map_err(|e| Error::Model(format!("{}: {e}", files.decision.display())))?;
        let calibration = match &files.calibration {
            Some(path) => Calibration::from_file(&read_json::<CalibrationFile>(path)?),
            None => Calibration::default(),
        };
        let tokenizer = Tokenizer(load_tokenizer(&files.tokenizer)?);
        let overhead = layout.overhead(&tokenizer)?;

        let session = session_with(&files.graph, device, intra_threads, |b| {
            crate::decider::configure(b, device)
        })?;
        let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        if inputs.len() != INPUTS.len()
            || !INPUTS.iter().all(|n| inputs.contains(n))
            || !session.outputs().iter().any(|o| o.name() == OUTPUT)
        {
            return Err(Error::Model(format!(
                "graph inputs {inputs:?} do not match the qwen3guard contract ({INPUTS:?} -> {OUTPUT:?})"
            )));
        }

        Ok(GuardModel {
            session: Mutex::new(session),
            tokenizer,
            overhead,
            layout,
            calibration,
            device,
        })
    }

    /// The rows a request needs; questions other than the preset's are rejected.
    pub fn encode(&self, state: &Value, questions: &Questions) -> Result<GuardRows, Error> {
        Ok(self.layout.encode(&self.tokenizer, state, questions)?)
    }

    /// Answer every question.
    pub fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let rows = self.encode(state, questions)?;
        let cand = self.cand_logits(&rows)?;
        Ok(self.output(&rows, &cand))
    }

    /// Each row's candidate logits (`[safety..., categories...]`).
    pub fn cand_logits(&self, rows: &GuardRows) -> Result<Vec<Vec<f32>>, Error> {
        let width = self.layout.candidate_ids().count();
        let lens: Vec<usize> = rows.rows.iter().map(Vec::len).collect();
        let pad = i64::from(self.layout.config.special_tokens.pad);
        let mut cand = vec![Vec::new(); lens.len()];
        let mut session = self.session.lock().expect("session mutex poisoned");
        for range in crate::engine::batches(&lens, TOKEN_BUDGET, usize::MAX) {
            let batch = &rows.rows[range.clone()];
            let seq = lens[range.clone()].iter().copied().max().unwrap_or(0);
            let mut input_ids = Array2::<i64>::from_elem((batch.len(), seq), pad);
            let mut last_pos = Array1::<i64>::zeros(batch.len());
            for (r, row) in batch.iter().enumerate() {
                for (c, &id) in row.iter().enumerate() {
                    input_ids[[r, c]] = i64::from(id);
                }
                last_pos[r] = row.len() as i64 - 1;
            }
            let outputs = session.run(ort::inputs![
                "input_ids" => ort::value::Tensor::from_array(input_ids)?,
                "last_pos" => ort::value::Tensor::from_array(last_pos)?,
            ])?;
            let out = outputs[OUTPUT]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix2>()
                .map_err(|e| Error::Model(format!("{OUTPUT}: {e}")))?;
            if out.nrows() != batch.len() || out.ncols() != width {
                return Err(Error::Model(format!(
                    "{OUTPUT} has shape {:?} for {} rows of {width} candidates",
                    out.shape(),
                    batch.len()
                )));
            }
            for (slot, row) in cand[range].iter_mut().zip(out.rows()) {
                *slot = row.to_vec();
            }
        }
        Ok(cand)
    }

    /// Option logits per question from the candidate logits of each row.
    pub fn output(&self, rows: &GuardRows, cand: &[Vec<f32>]) -> Output {
        Output {
            questions: rows
                .readouts
                .iter()
                .map(|&r| QuestionOutput {
                    logits: self.layout.option_logits(r, &cand[r.row()]),
                    act_logits: None,
                })
                .collect(),
            input_tokens: rows.rows.iter().map(Vec::len).sum(),
            state_tokens: rows.rows[0].len().saturating_sub(self.overhead),
            state_truncated: false,
        }
    }
}
