//! ONNX Runtime engine for GLiClass uni-encoders (layout `gliclass-uni-v1`, contract `markers`).
//!
//! One row per question (`ollaya_decision::gliclass`), and a request's rows run as one padded
//! batch. The graph returns one logit per label; [`Mode::option_logits`] turns them into option
//! logits whose softmax at T = 1 is upstream's score (softmax over labels, or a label's sigmoid).
//!
//! [`Mode::option_logits`]: ollaya_decision::gliclass::Mode::option_logits

use std::path::Path;

use ndarray::Array2;
use ollaya_decision::gliclass::{GliclassLayout, GliclassTokens, Row};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder};
use serde::Deserialize;
use serde_json::Value;

use crate::engine::Engine;
use crate::net::{Batch, Head, Net};
use crate::onnx::{Device, ModelFiles, load_tokenizer};
use crate::{Error, Output, QuestionOutput};

/// Rows per `session.run`: the graph's row axis was exported for 1..=1024.
const MAX_ROWS: usize = 1024;
/// The graph's sequence axis starts at 8; shorter batches get masked padding.
const MIN_SEQ: usize = 8;

/// The fields of the `decision` layer this engine reads.
#[derive(Debug, Clone, Deserialize)]
struct DecisionConfig {
    engine: String,
    layout: String,
    max_len: usize,
    special_tokens: GliclassTokens,
    sanitize: Vec<String>,
    #[serde(default = "one")]
    min_markers: usize,
}

fn one() -> usize {
    1
}

/// Encoder rows for one request.
#[derive(Debug, Clone)]
pub struct GliclassEncoding {
    pub rows: Vec<Row>,
    /// Tokens in the serialized state, before any truncation.
    pub state_tokens: usize,
}

pub struct GliclassModel {
    net: Net,
    tokenizer: Tokenizer,
    layout: GliclassLayout,
    min_markers: usize,
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

/// Name the question in an encoding error.
fn for_question(e: ollaya_decision::Error, qid: &str) -> ollaya_decision::Error {
    match e {
        ollaya_decision::Error::Invalid(m) => {
            ollaya_decision::Error::invalid(format!("question {qid:?}: {m}"))
        }
        e => e.for_question(qid),
    }
}

impl Engine for GliclassModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        GliclassModel::run(self, state, questions)
    }
}

impl GliclassModel {
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
        if config.engine != "onnx" || config.layout != "gliclass-uni-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/gliclass-uni-v1",
                config.engine, config.layout
            )));
        }
        let calibration = match &files.calibration {
            Some(path) => Calibration::from_file(&read_json::<CalibrationFile>(path)?),
            None => Calibration::default(),
        };
        let tokenizer = load_tokenizer(&files.tokenizer)?;
        let net = crate::net::load(files, device, intra_threads, Head::Gliclass)?;
        Ok(GliclassModel {
            net,
            tokenizer: Tokenizer(tokenizer),
            layout: GliclassLayout {
                max_len: config.max_len,
                special: config.special_tokens,
                sanitize: config.sanitize,
            },
            min_markers: config.min_markers,
            calibration,
            device,
        })
    }

    /// Encode every question against the shared state (token ids and `<<LABEL>>` positions).
    pub fn encode(&self, state: &Value, questions: &Questions) -> Result<GliclassEncoding, Error> {
        let state_text = self.layout.state_text(state);
        let rows = questions
            .iter()
            .map(|(qid, q)| {
                self.layout
                    .encode(&self.tokenizer, &state_text, q)
                    .map_err(|e| Error::Decision(for_question(e, qid)))
            })
            .collect::<Result<_, _>>()?;
        Ok(GliclassEncoding {
            rows,
            state_tokens: self.tokenizer.encode(&state_text)?.len(),
        })
    }

    /// Answer every question, batching all rows into one forward pass.
    pub fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let encoding = self.encode(state, questions)?;
        self.run_encoded(&encoding, questions)
    }

    pub fn run_encoded(
        &self,
        encoding: &GliclassEncoding,
        questions: &Questions,
    ) -> Result<Output, Error> {
        let qtypes: Vec<i64> = questions.values().map(|q| q.qtype.index() as i64).collect();
        let mut outputs = Vec::with_capacity(encoding.rows.len());
        let lens: Vec<usize> = encoding.rows.iter().map(|r| r.ids.len()).collect();
        for range in crate::engine::batches(&lens, crate::engine::TOKEN_BUDGET, MAX_ROWS) {
            outputs.extend(self.run_batch(&encoding.rows[range.clone()], &qtypes[range])?);
        }
        Ok(Output {
            questions: outputs,
            input_tokens: encoding.rows.iter().map(|r| r.ids.len()).sum(),
            state_tokens: encoding.state_tokens,
            state_truncated: encoding.rows.iter().any(|r| r.state_truncated),
        })
    }

    fn run_batch(&self, rows: &[Row], qtypes: &[i64]) -> Result<Vec<QuestionOutput>, Error> {
        let n = rows.len();
        let seq = rows
            .iter()
            .map(|r| r.ids.len())
            .max()
            .unwrap_or(0)
            .max(MIN_SEQ);
        let k = rows
            .iter()
            .map(|r| r.markers.len())
            .max()
            .unwrap_or(0)
            .max(self.min_markers);

        let pad = i64::from(self.layout.special.pad);
        let mut input_ids = Array2::<i64>::from_elem((n, seq), pad);
        let mut attention = Array2::<i64>::zeros((n, seq));
        let mut marker_pos = Array2::<i64>::zeros((n, k));
        let mut marker_mask = Array2::<bool>::from_elem((n, k), false);
        for (r, row) in rows.iter().enumerate() {
            for (c, &id) in row.ids.iter().enumerate() {
                input_ids[[r, c]] = i64::from(id);
                attention[[r, c]] = 1;
            }
            for (c, &m) in row.markers.iter().enumerate() {
                marker_pos[[r, c]] = m as i64;
                marker_mask[[r, c]] = true;
            }
        }

        let logits = self
            .net
            .run(
                Batch {
                    input_ids,
                    attention,
                    markers: Some((marker_pos, marker_mask)),
                    qtype: Some(qtypes.to_vec()),
                },
                &["logits"],
            )?
            .remove(0);
        Ok(rows
            .iter()
            .enumerate()
            .map(|(r, row)| {
                let labels: Vec<f32> = (0..row.markers.len()).map(|c| logits[[r, c]]).collect();
                QuestionOutput {
                    logits: row.mode.option_logits(&labels),
                    act_logits: None,
                }
            })
            .collect())
    }
}
