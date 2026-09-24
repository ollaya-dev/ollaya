//! ONNX Runtime engine for zero-shot NLI cross-encoders (layout `nli-pairs-v1`, contract P).
//!
//! Every (state, hypothesis) pair is one row. The graph maps `input_ids` and `attention_mask`
//! [rows, seq] to `scores` [rows, classes]. All rows of a request run as one padded batch, split
//! only past the graph's row limit. Rows and their readout come from `ollaya_decision::nli`.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array2, Ix2};
use ollaya_decision::nli::{NliLayout, Pairs};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder, serialize_state};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use serde_json::Value;

use crate::engine::Engine;
use crate::onnx::{Device, ModelFiles, load_tokenizer, session};
use crate::{Error, Output, QuestionOutput};

/// Rows per `session.run`: the export's row axis is 1..4096.
const MAX_ROWS: usize = 4096;
/// The export's sequence axis starts at 8; shorter batches are padded up to it.
const MIN_SEQ: usize = 8;
const INPUTS: [&str; 2] = ["input_ids", "attention_mask"];
const OUTPUT: &str = "scores";

/// The `decision` layer.
#[derive(Debug, Clone, Deserialize)]
struct DecisionConfig {
    engine: String,
    layout: String,
    #[serde(flatten)]
    nli: NliLayout,
}

/// Encoder rows for one request: every question against the shared state.
#[derive(Debug, Clone)]
pub struct NliEncoding {
    pub questions: Vec<Pairs>,
    /// Tokens in the serialized state, before any truncation.
    pub state_tokens: usize,
}

pub struct NliModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    pub layout: NliLayout,
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

impl Engine for NliModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        NliModel::run(self, state, questions)
    }
}

impl NliModel {
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
        if config.engine != "onnx" || config.layout != "nli-pairs-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/nli-pairs-v1",
                config.engine, config.layout
            )));
        }
        config
            .nli
            .validate()
            .map_err(|e| Error::Model(format!("{}: {e}", files.decision.display())))?;
        let calibration = match &files.calibration {
            Some(path) => Calibration::from_file(&read_json::<CalibrationFile>(path)?),
            None => Calibration::default(),
        };
        let tokenizer = load_tokenizer(&files.tokenizer)?;

        let session = session(&files.graph, device, intra_threads)?;
        let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        if inputs.len() != INPUTS.len()
            || !INPUTS.iter().all(|n| inputs.contains(n))
            || !session.outputs().iter().any(|o| o.name() == OUTPUT)
        {
            return Err(Error::Model(format!(
                "graph inputs {inputs:?} do not match contract P ({INPUTS:?} -> {OUTPUT:?})"
            )));
        }

        Ok(NliModel {
            session: Mutex::new(session),
            tokenizer: Tokenizer(tokenizer),
            layout: config.nli,
            calibration,
            device,
        })
    }

    /// Encode every question's rows against the shared state.
    pub fn encode(&self, state: &Value, questions: &Questions) -> Result<NliEncoding, Error> {
        let premise = self.tokenizer.encode(&serialize_state(state))?;
        let questions = questions
            .iter()
            .map(|(qid, q)| self.layout.encode(&self.tokenizer, &premise, qid, q))
            .collect::<Result<_, _>>()?;
        Ok(NliEncoding {
            questions,
            state_tokens: premise.len(),
        })
    }

    /// Answer every question, all rows batched together.
    pub fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let encoding = self.encode(state, questions)?;
        let scores = self.scores(&encoding)?;
        Ok(self.output(&encoding, &scores))
    }

    /// Class logits of every row, question by question.
    pub fn scores(&self, encoding: &NliEncoding) -> Result<Vec<Vec<f32>>, Error> {
        let rows: Vec<&[u32]> = encoding
            .questions
            .iter()
            .flat_map(|p| p.rows.iter().map(Vec::as_slice))
            .collect();
        let classes = &self.layout.classes;
        let width = classes.entailment.max(classes.not_entailment) + 1;
        let pad = i64::from(self.layout.special_tokens.pad);

        let mut scores = Vec::with_capacity(rows.len());
        let mut session = self.session.lock().expect("session mutex poisoned");
        let lens: Vec<usize> = rows.iter().map(|r| r.len()).collect();
        for range in crate::engine::batches(&lens, crate::engine::TOKEN_BUDGET, MAX_ROWS) {
            let chunk = &rows[range];
            let seq = chunk
                .iter()
                .map(|r| r.len())
                .max()
                .unwrap_or(0)
                .max(MIN_SEQ);
            let mut input_ids = Array2::<i64>::from_elem((chunk.len(), seq), pad);
            let mut attention = Array2::<i64>::zeros((chunk.len(), seq));
            for (r, ids) in chunk.iter().enumerate() {
                for (c, &id) in ids.iter().enumerate() {
                    input_ids[[r, c]] = i64::from(id);
                    attention[[r, c]] = 1;
                }
            }
            let outputs = session.run(ort::inputs![
                "input_ids" => Tensor::from_array(input_ids)?,
                "attention_mask" => Tensor::from_array(attention)?,
            ])?;
            let out = outputs[OUTPUT]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix2>()
                .map_err(|e| Error::Model(format!("{OUTPUT}: {e}")))?;
            if out.nrows() != chunk.len() || out.ncols() < width {
                return Err(Error::Model(format!(
                    "{OUTPUT} has shape {:?} for {} rows; the classes need {width} columns",
                    out.shape(),
                    chunk.len()
                )));
            }
            scores.extend(out.rows().into_iter().map(|r| r.to_vec()));
        }
        Ok(scores)
    }

    /// Option logits per question from the scores of every row.
    pub fn output(&self, encoding: &NliEncoding, scores: &[Vec<f32>]) -> Output {
        let mut next = 0;
        let questions = encoding
            .questions
            .iter()
            .map(|p| {
                let rows = &scores[next..next + p.rows.len()];
                next += p.rows.len();
                QuestionOutput {
                    logits: self
                        .layout
                        .option_logits(p.mode, rows.iter().map(Vec::as_slice)),
                    act_logits: None,
                }
            })
            .collect();
        let rows = encoding.questions.iter().flat_map(|p| &p.rows);
        Output {
            questions,
            input_tokens: rows.map(Vec::len).sum(),
            state_tokens: encoding.state_tokens,
            state_truncated: encoding.questions.iter().any(|p| p.state_truncated),
        }
    }
}
