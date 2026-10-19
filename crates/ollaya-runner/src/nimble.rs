//! ONNX Runtime engine for Bespoke Labs' Nimble (layout `nimble-codes-v1`).
//!
//! Every question is one chat row holding the whole request (`ollaya_decision::nimble`). The graph
//! maps `input_ids` [rows, seq] (right-padded to a multiple of 64, positions implicit, no mask:
//! every layer is causal) and `last_pos` [rows] to `cand_logits` [rows, 255]: the next-token logits
//! of the option codes at each row's last token. A question's option logits are its first k.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Ix2};
use ollaya_decision::nimble::{NimbleLayout, NimbleRow};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder};
use ort::session::Session;
use serde::Deserialize;
use serde_json::Value;

use crate::decider::WeightsInMemory;
use crate::engine::Engine;
use crate::onnx::{CudaArena, Device, ModelFiles, load_tokenizer, session_for};
use crate::{Error, Output, QuestionOutput};

/// Rows per `session.run`: the export's row axis is 1..=4096.
const MAX_ROWS: usize = 4096;
/// Padded tokens per `session.run`. Half the other Qwen3.5 decoders' budget: a 9B's weights take
/// 18 GB of a 24 GB GPU, and every Nimble row repeats the whole request, so rows run long.
const TOKEN_BUDGET: usize = 2048;
const INPUTS: [&str; 2] = ["input_ids", "last_pos"];
const OUTPUT: &str = "cand_logits";

#[derive(Debug, Clone, Deserialize)]
struct Contract {
    /// Rows are padded to a multiple of this (one `Scan` step of the DeltaNet layers).
    seq_multiple: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct SpecialTokens {
    pad: u32,
}

/// The fields of the `decision` layer this engine reads.
#[derive(Debug, Clone, Deserialize)]
struct DecisionConfig {
    engine: String,
    layout: String,
    contract: Contract,
    special_tokens: SpecialTokens,
    #[serde(default)]
    weights_in_memory: WeightsInMemory,
    #[serde(flatten)]
    nimble: NimbleLayout,
}

pub struct NimbleModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    seq_multiple: usize,
    pad: u32,
    pub layout: NimbleLayout,
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

impl Engine for NimbleModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let defs: Vec<(&str, &Value)> = questions
            .iter()
            .map(|(qid, q)| (qid.as_str(), &q.definition))
            .collect();
        self.answer(state, &defs)
    }
}

impl NimbleModel {
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
        if config.engine != "onnx" || config.layout != "nimble-codes-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/nimble-codes-v1",
                config.engine, config.layout
            )));
        }
        let bad = |e: String| Error::Model(format!("{}: {e}", files.decision.display()));
        config.nimble.validate().map_err(|e| bad(e.to_string()))?;
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
            |b| weights.configure(crate::decider::configure(b, device)?),
        )?;
        let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        if inputs.len() != INPUTS.len()
            || !INPUTS.iter().all(|n| inputs.contains(n))
            || !session.outputs().iter().any(|o| o.name() == OUTPUT)
        {
            return Err(Error::Model(format!(
                "graph inputs {inputs:?} do not match the nimble contract ({INPUTS:?} -> {OUTPUT:?})"
            )));
        }
        Ok(NimbleModel {
            session: Mutex::new(session),
            tokenizer: Tokenizer(tokenizer),
            seq_multiple: config.contract.seq_multiple,
            pad: config.special_tokens.pad,
            layout: config.nimble,
            calibration,
            device,
        })
    }

    /// Every question's row, in request order.
    pub fn rows(&self, state: &Value, defs: &[(&str, &Value)]) -> Result<Vec<NimbleRow>, Error> {
        Ok(self.layout.rows(&self.tokenizer, state, defs)?)
    }

    /// Answer the questions: one logit per option.
    pub fn answer(&self, state: &Value, defs: &[(&str, &Value)]) -> Result<Output, Error> {
        let rows = self.rows(state, defs)?;
        let logits = self.option_logits(&rows)?;
        Ok(Output {
            questions: logits
                .into_iter()
                .map(|logits| QuestionOutput {
                    logits,
                    act_logits: None,
                })
                .collect(),
            input_tokens: rows.iter().map(|r| r.ids.len()).sum(),
            state_tokens: 0,
            state_truncated: false,
        })
    }

    /// Each row's option logits (its first `options` code logits). Rows run shortest first, in
    /// batches padded to a multiple of `seq_multiple`.
    pub fn option_logits(&self, rows: &[NimbleRow]) -> Result<Vec<Vec<f32>>, Error> {
        let width = self.layout.code_ids().len();
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].ids.len());
        let padded: Vec<usize> = order
            .iter()
            .map(|&i| rows[i].ids.len().div_ceil(self.seq_multiple) * self.seq_multiple)
            .collect();
        let pad = i64::from(self.pad);
        let mut logits = vec![Vec::new(); rows.len()];
        let mut session = self.session.lock().expect("session mutex poisoned");
        for range in crate::engine::batches(&padded, TOKEN_BUDGET, MAX_ROWS) {
            let batch = &order[range.clone()];
            let seq = padded[range].iter().copied().max().unwrap_or(0);
            let mut input_ids = Array2::<i64>::from_elem((batch.len(), seq), pad);
            let mut last_pos = Array1::<i64>::zeros(batch.len());
            for (r, &i) in batch.iter().enumerate() {
                for (c, &id) in rows[i].ids.iter().enumerate() {
                    input_ids[[r, c]] = i64::from(id);
                }
                last_pos[r] = rows[i].ids.len() as i64 - 1;
            }
            let options = crate::engine::run_options(self.device, batch.len() * seq, TOKEN_BUDGET)?;
            let outputs = session.run_with_options(
                ort::inputs![
                    "input_ids" => ort::value::Tensor::from_array(input_ids)?,
                    "last_pos" => ort::value::Tensor::from_array(last_pos)?,
                ],
                &options,
            )?;
            let out = outputs[OUTPUT]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix2>()
                .map_err(|e| Error::Model(format!("{OUTPUT}: {e}")))?;
            if out.nrows() != batch.len() || out.ncols() != width {
                return Err(Error::Model(format!(
                    "{OUTPUT} has shape {:?} for {} rows of {width} codes",
                    out.shape(),
                    batch.len()
                )));
            }
            for (&i, row) in batch.iter().zip(out.rows()) {
                logits[i] = row.iter().take(rows[i].options).copied().collect();
            }
        }
        Ok(logits)
    }
}
