//! ONNX Runtime engine for GLiClass v4's decoder-KV models (layout `gliclass-kv-v1`).
//!
//! Every question is one row (`ollaya_decision::gliclass_kv`): the context, then the label section.
//! The graph takes `input_ids` [rows, seq] (right-padded to a multiple of 64, positions implicit, the
//! decoder is causal), the label section's positions and mask [rows, len], each `<<LABEL>>`'s index in
//! the section [rows, k] with its mask, and the final `<<SEP>>`'s index [rows]; it returns the
//! calibrated label `logits` [rows, k] (the model's own calibrator runs inside the graph) and the
//! recurrent `steps` it took. A row's first `k_row` logits are its labels'; the option logits are
//! those, or `[0, z]` for a one-label noul (`Mode`).

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Ix2};
use ollaya_decision::gliclass_kv::{GliclassKvLayout, LAYOUT, Row};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder};
use ort::session::Session;
use serde::Deserialize;
use serde_json::Value;

use crate::decider::WeightsInMemory;
use crate::engine::Engine;
use crate::kev::configure;
use crate::onnx::{CudaArena, Device, ModelFiles, load_tokenizer, session_for};
use crate::{Error, Output, QuestionOutput};

/// Rows per `session.run`: the export's row axis is 1..=1024.
const MAX_ROWS: usize = 1024;
/// Padded tokens per `session.run`, as for the other Qwen3.5 decoders.
const TOKEN_BUDGET: usize = 8192;
/// The export's label-section axis starts at 2.
const MIN_SECTION: usize = 2;
const INPUTS: [&str; 6] = [
    "input_ids",
    "section_pos",
    "section_mask",
    "label_pos",
    "label_mask",
    "sep_pos",
];
const OUTPUT: &str = "logits";

#[derive(Debug, Clone, Deserialize)]
struct Contract {
    seq_multiple: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct DecisionConfig {
    engine: String,
    layout: String,
    contract: Contract,
    #[serde(default)]
    weights_in_memory: WeightsInMemory,
    #[serde(flatten)]
    rows: GliclassKvLayout,
}

/// Rows for one request, one per question.
#[derive(Debug, Clone)]
pub struct GliclassKvEncoding {
    pub rows: Vec<Row>,
    pub state_tokens: usize,
}

pub struct GliclassKvModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    seq_multiple: usize,
    pub layout: GliclassKvLayout,
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

impl Engine for GliclassKvModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        GliclassKvModel::run(self, state, questions)
    }
}

impl GliclassKvModel {
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
        if config.engine != "onnx" || config.layout != LAYOUT {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/{LAYOUT}",
                config.engine, config.layout
            )));
        }
        let bad = |e: String| Error::Model(format!("{}: {e}", files.decision.display()));
        config.rows.validate().map_err(|e| bad(e.to_string()))?;
        if config.contract.seq_multiple == 0 {
            return Err(bad("contract.seq_multiple must be positive".into()));
        }
        let calibration = match &files.calibration {
            Some(path) => Calibration::from_file(&read_json::<CalibrationFile>(path)?),
            None => Calibration::default(),
        };
        let tokenizer = load_tokenizer(&files.tokenizer)?;
        let weights = config.weights_in_memory;
        let session = session_for(
            &files.graph,
            device,
            intra_threads,
            CudaArena::SameAsRequested,
            |b| weights.configure(configure(b, device)?),
        )?;
        let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        if inputs.len() != INPUTS.len()
            || !INPUTS.iter().all(|n| inputs.contains(n))
            || !session.outputs().iter().any(|o| o.name() == OUTPUT)
        {
            return Err(Error::Model(format!(
                "graph inputs {inputs:?} do not match the gliclass-kv contract ({INPUTS:?} -> {OUTPUT:?})"
            )));
        }
        Ok(GliclassKvModel {
            session: Mutex::new(session),
            tokenizer: Tokenizer(tokenizer),
            seq_multiple: config.contract.seq_multiple,
            layout: config.rows,
            calibration,
            device,
        })
    }

    /// Encode every question against the shared state.
    pub fn encode(
        &self,
        state: &Value,
        questions: &Questions,
    ) -> Result<GliclassKvEncoding, Error> {
        let state_text = self.layout.state_text(state);
        let rows = questions
            .iter()
            .map(|(qid, q)| self.layout.encode(&self.tokenizer, &state_text, qid, q))
            .collect::<Result<_, _>>()?;
        Ok(GliclassKvEncoding {
            rows,
            state_tokens: self.tokenizer.encode(&state_text)?.len(),
        })
    }

    /// Answer every question.
    pub fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let encoding = self.encode(state, questions)?;
        let labels = self.label_logits(&encoding)?;
        Ok(Output {
            questions: encoding
                .rows
                .iter()
                .zip(&labels)
                .map(|(row, z)| QuestionOutput {
                    logits: row.mode.option_logits(z),
                    act_logits: None,
                })
                .collect(),
            input_tokens: encoding.rows.iter().map(|r| r.ids.len()).sum(),
            state_tokens: encoding.state_tokens,
            state_truncated: encoding.rows.iter().any(|r| r.state_truncated),
        })
    }

    /// Each row's calibrated label logits, in question order.
    pub fn label_logits(&self, encoding: &GliclassKvEncoding) -> Result<Vec<Vec<f32>>, Error> {
        let rows = &encoding.rows;
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].ids.len());
        let padded: Vec<usize> = order
            .iter()
            .map(|&i| rows[i].ids.len().div_ceil(self.seq_multiple) * self.seq_multiple)
            .collect();
        let pad = i64::from(self.layout.special_tokens.pad);

        let mut logits = vec![Vec::new(); rows.len()];
        for range in crate::engine::batches(&padded, TOKEN_BUDGET, MAX_ROWS) {
            let batch = &order[range.clone()];
            let n = batch.len();
            let seq = padded[range].iter().copied().max().unwrap_or(0);
            let section_len = |r: &Row| r.sep + 1 - r.section_start;
            let len = batch
                .iter()
                .map(|&i| section_len(&rows[i]))
                .max()
                .unwrap_or(0)
                .max(MIN_SECTION);
            let k = batch
                .iter()
                .map(|&i| rows[i].markers.len())
                .max()
                .unwrap_or(0)
                .max(1);
            let mut input_ids = Array2::<i64>::from_elem((n, seq), pad);
            let mut section_pos = Array2::<i64>::zeros((n, len));
            let mut section_mask = Array2::<bool>::from_elem((n, len), false);
            let mut label_pos = Array2::<i64>::zeros((n, k));
            let mut label_mask = Array2::<bool>::from_elem((n, k), false);
            let mut sep_pos = Array1::<i64>::zeros(n);
            for (r, &i) in batch.iter().enumerate() {
                let row = &rows[i];
                for (c, &id) in row.ids.iter().enumerate() {
                    input_ids[[r, c]] = i64::from(id);
                }
                for c in 0..len {
                    section_pos[[r, c]] = (row.section_start + c).min(row.sep) as i64;
                    section_mask[[r, c]] = c < section_len(row);
                }
                for (c, &m) in row.markers.iter().enumerate() {
                    label_pos[[r, c]] = (m - row.section_start) as i64;
                    label_mask[[r, c]] = true;
                }
                sep_pos[r] = (row.sep - row.section_start) as i64;
            }
            let options = crate::engine::run_options(self.device, n * seq, TOKEN_BUDGET)?;
            let mut session = self.session.lock().expect("session mutex poisoned");
            let outputs = session.run_with_options(
                ort::inputs![
                    "input_ids" => ort::value::Tensor::from_array(input_ids)?,
                    "section_pos" => ort::value::Tensor::from_array(section_pos)?,
                    "section_mask" => ort::value::Tensor::from_array(section_mask)?,
                    "label_pos" => ort::value::Tensor::from_array(label_pos)?,
                    "label_mask" => ort::value::Tensor::from_array(label_mask)?,
                    "sep_pos" => ort::value::Tensor::from_array(sep_pos)?,
                ],
                &options,
            )?;
            let out = outputs[OUTPUT]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix2>()
                .map_err(|e| Error::Model(format!("{OUTPUT}: {e}")))?
                .to_owned();
            if out.nrows() != n || out.ncols() != k {
                return Err(Error::Model(format!(
                    "{OUTPUT} has shape {:?} for {n} rows of up to {k} labels",
                    out.shape()
                )));
            }
            for (&i, row) in batch.iter().zip(out.rows()) {
                logits[i] = row.iter().take(rows[i].markers.len()).copied().collect();
            }
        }
        Ok(logits)
    }
}
