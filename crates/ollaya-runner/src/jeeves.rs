//! ONNX Runtime engine for PostHog's Jeeves (layout `jeeves-markers-v1`), no-thinking mode.
//!
//! Every question is one row (`ollaya_decision::jeeves`). The graph has Kev's contract: it maps
//! `input_ids` [rows, seq] (right-padded to a multiple of 64, positions implicit, no mask: every
//! layer is causal), `decide_pos` [rows] and `opt_pos` [rows, k] (shorter rows padded with 0) to
//! the pointer head's raw `scores` [rows, k]; a row's first `k_row` scores are its option logits.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Ix2};
use ollaya_decision::jeeves::{JeevesLayout, JeevesRow};
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
/// Padded tokens per `session.run`: a 9B's BF16 weights take 18 GB of a 24 GB GPU.
const TOKEN_BUDGET: usize = 4096;
const INPUTS: [&str; 3] = ["input_ids", "decide_pos", "opt_pos"];
const OUTPUT: &str = "scores";

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
    jeeves: JeevesLayout,
}

pub struct JeevesModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    seq_multiple: usize,
    pub layout: JeevesLayout,
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

impl Engine for JeevesModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let defs: Vec<(&str, &Value)> = questions
            .iter()
            .map(|(qid, q)| (qid.as_str(), &q.definition))
            .collect();
        self.answer(state, &defs)
    }
}

impl JeevesModel {
    pub fn load(dir: &Path, device: Device, intra_threads: Option<usize>) -> Result<Self, Error> {
        Self::load_files(&ModelFiles::dir(dir), device, intra_threads)
    }

    pub fn load_files(
        files: &ModelFiles,
        device: Device,
        intra_threads: Option<usize>,
    ) -> Result<Self, Error> {
        let config: DecisionConfig = read_json(&files.decision)?;
        if config.engine != "onnx" || config.layout != "jeeves-markers-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/jeeves-markers-v1",
                config.engine, config.layout
            )));
        }
        let bad = |e: String| Error::Model(format!("{}: {e}", files.decision.display()));
        config.jeeves.validate().map_err(|e| bad(e.to_string()))?;
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
            |b| weights.configure(crate::kev::configure(b, device)?),
        )?;
        let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        if inputs.len() != INPUTS.len()
            || !INPUTS.iter().all(|n| inputs.contains(n))
            || !session.outputs().iter().any(|o| o.name() == OUTPUT)
        {
            return Err(Error::Model(format!(
                "graph inputs {inputs:?} do not match the jeeves contract ({INPUTS:?} -> {OUTPUT:?})"
            )));
        }
        Ok(JeevesModel {
            session: Mutex::new(session),
            tokenizer: Tokenizer(tokenizer),
            seq_multiple: config.contract.seq_multiple,
            layout: config.jeeves,
            calibration,
            device,
        })
    }

    /// Every question's row, in request order.
    pub fn rows(&self, state: &Value, defs: &[(&str, &Value)]) -> Result<Vec<JeevesRow>, Error> {
        Ok(self.layout.rows(&self.tokenizer, state, defs)?)
    }

    /// Answer the questions: one logit per option.
    pub fn answer(&self, state: &Value, defs: &[(&str, &Value)]) -> Result<Output, Error> {
        let rows = self.rows(state, defs)?;
        let scores = self.scores(&rows)?;
        Ok(Output {
            questions: scores
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

    /// Each row's option scores (its first `opts.len()` graph outputs). Rows run shortest first, in
    /// batches padded to a multiple of `seq_multiple`.
    pub fn scores(&self, rows: &[JeevesRow]) -> Result<Vec<Vec<f32>>, Error> {
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].ids.len());
        let padded: Vec<usize> = order
            .iter()
            .map(|&i| rows[i].ids.len().div_ceil(self.seq_multiple) * self.seq_multiple)
            .collect();
        let pad = i64::from(self.layout.pad);
        let mut scores = vec![Vec::new(); rows.len()];
        let mut session = self.session.lock().expect("session mutex poisoned");
        for range in crate::engine::batches(&padded, TOKEN_BUDGET, MAX_ROWS) {
            let batch = &order[range.clone()];
            let seq = padded[range].iter().copied().max().unwrap_or(0);
            let k = batch.iter().map(|&i| rows[i].opts.len()).max().unwrap_or(0);
            let mut input_ids = Array2::<i64>::from_elem((batch.len(), seq), pad);
            let mut decide_pos = Array1::<i64>::zeros(batch.len());
            let mut opt_pos = Array2::<i64>::zeros((batch.len(), k));
            for (r, &i) in batch.iter().enumerate() {
                let row = &rows[i];
                for (c, &id) in row.ids.iter().enumerate() {
                    input_ids[[r, c]] = i64::from(id);
                }
                decide_pos[r] = row.decide as i64;
                for (c, &p) in row.opts.iter().enumerate() {
                    opt_pos[[r, c]] = p as i64;
                }
            }
            let options = crate::engine::run_options(self.device, batch.len() * seq, TOKEN_BUDGET)?;
            let outputs = session.run_with_options(
                ort::inputs![
                    "input_ids" => ort::value::Tensor::from_array(input_ids)?,
                    "decide_pos" => ort::value::Tensor::from_array(decide_pos)?,
                    "opt_pos" => ort::value::Tensor::from_array(opt_pos)?,
                ],
                &options,
            )?;
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
                scores[i] = row.iter().take(rows[i].opts.len()).copied().collect();
            }
        }
        Ok(scores)
    }
}
