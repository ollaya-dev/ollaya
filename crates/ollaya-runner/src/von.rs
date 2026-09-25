//! ONNX Runtime engine for Von (layout `von-option-marker-v1`, contract M).
//!
//! Every question is one row (a zero-shot noul question two): the question's instructions and
//! the state, then one `[MASK]` per option. The graph maps `input_ids`, `attention_mask` [rows,
//! seq], `marker_pos`, `marker_mask` [rows, markers] and `qtype` [rows] (ignored) to `logits`
//! [rows, markers], the scorer's output at each marker. All rows of a request run together,
//! shortest first, in batches bounded by a token budget. Rows and their readout come from
//! `ollaya_decision::von`.

use std::path::Path;

use ndarray::Array2;
use ollaya_decision::von::{CharOffsetEncoder, Scoring, VonLayout};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder};
use serde::Deserialize;
use serde_json::Value;

use crate::engine::Engine;
use crate::net::{Batch, Head, Net};
use crate::onnx::{Device, ModelFiles, load_tokenizer};
use crate::{Error, Output, QuestionOutput};

/// Rows per `session.run`: the export's row axis is 1..=1024.
const MAX_ROWS: usize = 1024;
/// The export's sequence axis starts at 8; shorter batches are padded up to it.
const MIN_SEQ: usize = 8;
/// Padded tokens per `session.run`. The global-attention layers materialise
/// [rows, heads, seq, seq] scores, so rows up to 8192 tokens take a smaller budget than
/// [`crate::engine::TOKEN_BUDGET`]: one 8192-token row runs alone.
const TOKEN_BUDGET: usize = 8192;
const INPUTS: [&str; 5] = [
    "input_ids",
    "attention_mask",
    "marker_pos",
    "marker_mask",
    "qtype",
];
const OUTPUT: &str = "logits";

/// The fields of the `decision` layer this engine reads.
#[derive(Debug, Clone, Deserialize)]
struct DecisionConfig {
    engine: String,
    layout: String,
    /// The graph takes at least this many marker slots (extra slots are masked).
    #[serde(default = "one")]
    min_markers: usize,
    #[serde(flatten)]
    von: VonLayout,
}

fn one() -> usize {
    1
}

/// Encoder rows for one request.
#[derive(Debug, Clone)]
pub struct VonEncoding {
    pub questions: Vec<Scoring>,
    /// Tokens in the rendered state, before sanitising and truncation.
    pub state_tokens: usize,
}

pub struct VonModel {
    net: Net,
    tokenizer: Tokenizer,
    min_markers: usize,
    pub layout: VonLayout,
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

impl CharOffsetEncoder for Tokenizer {
    fn char_ends(&self, text: &str) -> Result<Vec<usize>, ollaya_decision::Error> {
        self.0
            .encode_char_offsets(text, false)
            .map(|e| e.get_offsets().iter().map(|&(_, end)| end).collect())
            .map_err(|e| ollaya_decision::Error::Tokenizer(e.to_string()))
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::Model(format!("{}: {e}", path.display())))
}

impl Engine for VonModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        VonModel::run(self, state, questions)
    }
}

impl VonModel {
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
        if config.engine != "onnx" || config.layout != "von-option-marker-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/von-option-marker-v1",
                config.engine, config.layout
            )));
        }
        config
            .von
            .validate()
            .map_err(|e| Error::Model(format!("{}: {e}", files.decision.display())))?;
        let calibration = match &files.calibration {
            Some(path) => Calibration::from_file(&read_json::<CalibrationFile>(path)?),
            None => Calibration::default(),
        };
        let tokenizer = load_tokenizer(&files.tokenizer)?;

        let net = crate::net::load(files, device, intra_threads, Head::OptionMarker)?;
        if let Some(session) = net.session() {
            let session = session.lock().expect("session mutex poisoned");
            let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
            if inputs.len() != INPUTS.len()
                || !INPUTS.iter().all(|n| inputs.contains(n))
                || !session.outputs().iter().any(|o| o.name() == OUTPUT)
            {
                return Err(Error::Model(format!(
                    "graph inputs {inputs:?} do not match contract M ({INPUTS:?} -> {OUTPUT:?})"
                )));
            }
        }

        Ok(VonModel {
            net,
            tokenizer: Tokenizer(tokenizer),
            min_markers: config.min_markers.max(1),
            layout: config.von,
            calibration,
            device,
        })
    }

    /// Encode every question's rows against the shared state.
    pub fn encode(&self, state: &Value, questions: &Questions) -> Result<VonEncoding, Error> {
        let state = self.layout.encode_state(&self.tokenizer, state)?;
        let questions = questions
            .iter()
            .map(|(qid, q)| self.layout.encode(&self.tokenizer, &state, qid, q))
            .collect::<Result<_, _>>()?;
        Ok(VonEncoding {
            questions,
            state_tokens: state.tokens,
        })
    }

    /// Answer every question.
    pub fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let encoding = self.encode(state, questions)?;
        let logits = self.row_logits(&encoding)?;
        Ok(self.output(&encoding, &logits))
    }

    /// The scorer's logit at every marker of every row, question by question.
    pub fn row_logits(&self, encoding: &VonEncoding) -> Result<Vec<Vec<f32>>, Error> {
        let rows: Vec<(&ollaya_decision::von::Row, i64)> = encoding
            .questions
            .iter()
            .flat_map(|s| s.rows.iter().map(|r| (r, s.qtype.index() as i64)))
            .collect();
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].0.ids.len());
        let lens: Vec<usize> = order
            .iter()
            .map(|&i| rows[i].0.ids.len().max(MIN_SEQ))
            .collect();
        let pad = i64::from(self.layout.special_tokens.pad);

        let mut logits = vec![Vec::new(); rows.len()];
        for range in crate::engine::batches(&lens, TOKEN_BUDGET, MAX_ROWS) {
            let batch = &order[range.clone()];
            let seq = lens[range].iter().copied().max().unwrap_or(MIN_SEQ);
            let k = batch
                .iter()
                .map(|&i| rows[i].0.markers.len())
                .max()
                .unwrap_or(0)
                .max(self.min_markers);
            let n = batch.len();
            let mut input_ids = Array2::<i64>::from_elem((n, seq), pad);
            let mut attention = Array2::<i64>::zeros((n, seq));
            let mut marker_pos = Array2::<i64>::zeros((n, k));
            let mut marker_mask = Array2::<bool>::from_elem((n, k), false);
            let mut qtype = Vec::with_capacity(n);
            for (r, &i) in batch.iter().enumerate() {
                let (row, qt) = rows[i];
                for (c, &id) in row.ids.iter().enumerate() {
                    input_ids[[r, c]] = i64::from(id);
                    attention[[r, c]] = 1;
                }
                for (c, &m) in row.markers.iter().enumerate() {
                    marker_pos[[r, c]] = m as i64;
                    marker_mask[[r, c]] = true;
                }
                qtype.push(qt);
            }
            let out = self
                .net
                .run(
                    Batch {
                        input_ids,
                        attention,
                        markers: Some((marker_pos, marker_mask)),
                        qtype: Some(qtype),
                    },
                    &[OUTPUT],
                )?
                .remove(0);
            if out.nrows() != n || out.ncols() < k {
                return Err(Error::Model(format!(
                    "{OUTPUT} has shape {:?} for {n} rows of up to {k} markers",
                    out.shape()
                )));
            }
            for (&i, row) in batch.iter().zip(out.rows()) {
                let markers = rows[i].0.markers.len();
                logits[i] = row.iter().take(markers).copied().collect();
            }
        }
        Ok(logits)
    }

    /// Option logits per question from the logits of every row.
    pub fn output(&self, encoding: &VonEncoding, logits: &[Vec<f32>]) -> Output {
        let mut next = 0;
        let questions = encoding
            .questions
            .iter()
            .map(|s| {
                let rows: Vec<&[f32]> = logits[next..next + s.rows.len()]
                    .iter()
                    .map(Vec::as_slice)
                    .collect();
                next += s.rows.len();
                QuestionOutput {
                    logits: self.layout.option_logits(s, &rows),
                    act_logits: None,
                }
            })
            .collect();
        Output {
            questions,
            input_tokens: encoding
                .questions
                .iter()
                .flat_map(|s| &s.rows)
                .map(|r| r.ids.len())
                .sum(),
            state_tokens: encoding.state_tokens,
            state_truncated: encoding.questions.iter().any(|s| s.truncated),
        }
    }
}
