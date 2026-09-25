//! ONNX Runtime engine for llm-semantic-router's Decision 1.0 models (layout
//! `decision-endpoint-v1`).
//!
//! Every question is one row: the context, question and options, then the decision suffix. The
//! graph maps `input_ids` [rows, seq] (right-padded to a multiple of 64, positions implicit, no
//! mask: every layer is causal), `query_pos` [rows] and `cand_pos` [rows, k] (shorter rows padded
//! with 0) to the endpoint head's raw `logits` [rows, k]. A row's first `k_row` logits are its
//! option logits. Rows come from `ollaya_decision::decision`.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Ix2};
use ollaya_decision::decision::{DecisionLayout, DecisionRow};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder};
use ort::session::Session;
use serde::Deserialize;
use serde_json::Value;

use crate::engine::Engine;
use crate::onnx::{Device, ModelFiles, load_tokenizer, session_with};
use crate::{Error, Output, QuestionOutput};

/// Rows per `session.run`: the export's row axis is 1..=4096.
const MAX_ROWS: usize = 4096;
/// Padded tokens per `session.run`, as for the other Qwen3.5 decoders (`decider`, `kev`). A row
/// longer than this runs alone.
const TOKEN_BUDGET: usize = 8192;
const INPUTS: [&str; 3] = ["input_ids", "query_pos", "cand_pos"];
const OUTPUT: &str = "logits";

#[derive(Debug, Clone, Deserialize)]
struct Contract {
    /// Rows are padded to a multiple of this (one `Scan` step of the DeltaNet layers).
    seq_multiple: usize,
}

/// The fields of the `decision` layer this engine reads.
#[derive(Debug, Clone, Deserialize)]
struct DecisionConfig {
    engine: String,
    layout: String,
    contract: Contract,
    #[serde(flatten)]
    decision: DecisionLayout,
}

/// Rows for one request, one per question.
#[derive(Debug, Clone)]
pub struct DecisionEncoding {
    pub rows: Vec<DecisionRow>,
    /// Tokens of the rendered state on its own (reported as the request's state length).
    pub state_tokens: usize,
}

pub struct DecisionModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    seq_multiple: usize,
    pub layout: DecisionLayout,
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

impl Engine for DecisionModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        DecisionModel::run(self, state, questions)
    }
}

impl DecisionModel {
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
        if config.engine != "onnx" || config.layout != "decision-endpoint-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/decision-endpoint-v1",
                config.engine, config.layout
            )));
        }
        let bad = |e: String| Error::Model(format!("{}: {e}", files.decision.display()));
        config.decision.validate().map_err(|e| bad(e.to_string()))?;
        if config.contract.seq_multiple == 0 {
            return Err(bad("contract.seq_multiple must be positive".into()));
        }
        let calibration = match &files.calibration {
            Some(path) => Calibration::from_file(&read_json::<CalibrationFile>(path)?),
            None => Calibration::default(),
        };
        let tokenizer = load_tokenizer(&files.tokenizer)?;

        let session = session_with(&files.graph, device, intra_threads, |b| {
            crate::decider::configure(b, device)
        })?;
        let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        if inputs.len() != INPUTS.len()
            || !INPUTS.iter().all(|n| inputs.contains(n))
            || !session.outputs().iter().any(|o| o.name() == OUTPUT)
        {
            return Err(Error::Model(format!(
                "graph inputs {inputs:?} do not match the decision contract ({INPUTS:?} -> {OUTPUT:?})"
            )));
        }

        Ok(DecisionModel {
            session: Mutex::new(session),
            tokenizer: Tokenizer(tokenizer),
            seq_multiple: config.contract.seq_multiple,
            layout: config.decision,
            calibration,
            device,
        })
    }

    /// Encode every question's row. Any invalid question rejects the request, as upstream does.
    pub fn encode(&self, state: &Value, questions: &Questions) -> Result<DecisionEncoding, Error> {
        let rows = questions
            .iter()
            .map(|(qid, q)| self.layout.encode(&self.tokenizer, state, qid, q))
            .collect::<Result<_, _>>()?;
        let state_tokens = self
            .tokenizer
            .encode(&ollaya_decision::decision::payload(state))?
            .len();
        Ok(DecisionEncoding { rows, state_tokens })
    }

    /// Answer every question.
    pub fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let encoding = self.encode(state, questions)?;
        let logits = self.logits(&encoding)?;
        Ok(self.output(&encoding, logits))
    }

    /// Each row's option logits (its first `k_row` graph outputs). Rows run shortest first, in
    /// batches padded to a multiple of `seq_multiple`.
    pub fn logits(&self, encoding: &DecisionEncoding) -> Result<Vec<Vec<f32>>, Error> {
        let rows = &encoding.rows;
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].ids.len());
        let padded: Vec<usize> = order
            .iter()
            .map(|&i| rows[i].ids.len().div_ceil(self.seq_multiple) * self.seq_multiple)
            .collect();
        let pad = i64::from(self.layout.pad);

        let mut logits = vec![Vec::new(); rows.len()];
        let mut session = self.session.lock().expect("session mutex poisoned");
        for range in crate::engine::batches(&padded, TOKEN_BUDGET, MAX_ROWS) {
            let batch = &order[range.clone()];
            let seq = padded[range].iter().copied().max().unwrap_or(0);
            let k = batch
                .iter()
                .map(|&i| rows[i].cands.len())
                .max()
                .unwrap_or(0);
            let mut input_ids = Array2::<i64>::from_elem((batch.len(), seq), pad);
            let mut query_pos = Array1::<i64>::zeros(batch.len());
            let mut cand_pos = Array2::<i64>::zeros((batch.len(), k));
            for (r, &i) in batch.iter().enumerate() {
                let row = &rows[i];
                for (c, &id) in row.ids.iter().enumerate() {
                    input_ids[[r, c]] = i64::from(id);
                }
                query_pos[r] = row.query as i64;
                for (c, &p) in row.cands.iter().enumerate() {
                    cand_pos[[r, c]] = p as i64;
                }
            }
            let outputs = session.run(ort::inputs![
                "input_ids" => ort::value::Tensor::from_array(input_ids)?,
                "query_pos" => ort::value::Tensor::from_array(query_pos)?,
                "cand_pos" => ort::value::Tensor::from_array(cand_pos)?,
            ])?;
            let out = outputs[OUTPUT]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix2>()
                .map_err(|e| Error::Model(format!("{OUTPUT}: {e}")))?;
            if out.nrows() != batch.len() || out.ncols() != k {
                return Err(Error::Model(format!(
                    "{OUTPUT} has shape {:?} for {} rows of up to {k} options",
                    out.shape(),
                    batch.len()
                )));
            }
            for (&i, row) in batch.iter().zip(out.rows()) {
                logits[i] = row.iter().take(rows[i].cands.len()).copied().collect();
            }
        }
        Ok(logits)
    }

    /// The request's output from each row's option logits.
    pub fn output(&self, encoding: &DecisionEncoding, logits: Vec<Vec<f32>>) -> Output {
        Output {
            questions: logits
                .into_iter()
                .map(|logits| QuestionOutput {
                    logits,
                    act_logits: None,
                })
                .collect(),
            input_tokens: encoding.rows.iter().map(|r| r.ids.len()).sum(),
            state_tokens: encoding.state_tokens,
            state_truncated: false,
        }
    }
}
