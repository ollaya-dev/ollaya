//! ONNX Runtime engine for Perplexity's pplx-decider (layout `pplx-joint-v1`) and vllm-sr's d3
//! (layout `d3-code-readout-v1`), which share the graph.
//!
//! A pplx request is one sequence (or, over the window, a few; `ollaya_decision::pplx`); a d3
//! request is one sequence per question, read at its last token (`ollaya_decision::d3`). The graph maps
//! `input_ids` [rows, seq] (right-padded to a multiple of 64, positions implicit), `lengths` [rows]
//! (the full-attention layers read only a row's real tokens; the linear-attention layers are
//! causal) and `markers` [rows, q] to the readout `logits` [rows, q, 255]. A question's option
//! logits are its first k.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Ix3};
use ollaya_decision::d3::D3Layout;
use ollaya_decision::pplx::{PplxLayout, PplxSequence};
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
/// Padded tokens per `session.run`: one full window. Attention is materialized, so its memory
/// grows with rows × seq².
const TOKEN_BUDGET: usize = 8192;
const INPUTS: [&str; 3] = ["input_ids", "lengths", "markers"];
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
    #[serde(default)]
    weights_in_memory: WeightsInMemory,
}

/// How a request becomes the graph's rows.
pub enum Layout {
    /// `pplx-joint-v1`: every question in one sequence, read at its marker.
    Joint(PplxLayout),
    /// `d3-code-readout-v1`: one sequence per question, read at its last token.
    PerQuestion(D3Layout),
}

impl Layout {
    fn parse(layout: &str, config: Value) -> Result<Self, String> {
        let layout = match layout {
            "pplx-joint-v1" => {
                Layout::Joint(serde_json::from_value(config).map_err(|e| e.to_string())?)
            }
            "d3-code-readout-v1" => {
                Layout::PerQuestion(serde_json::from_value(config).map_err(|e| e.to_string())?)
            }
            other => return Err(format!("unsupported layout {other}")),
        };
        match &layout {
            Layout::Joint(l) => l.validate(),
            Layout::PerQuestion(l) => l.validate(),
        }
        .map_err(|e| e.to_string())?;
        Ok(layout)
    }

    fn encode(
        &self,
        tok: &Tokenizer,
        state: &Value,
        questions: &Questions,
    ) -> Result<Vec<PplxSequence>, ollaya_decision::Error> {
        match self {
            Layout::Joint(l) => l.encode(tok, state, questions),
            Layout::PerQuestion(l) => l.encode(tok, state, questions),
        }
    }

    fn pad(&self) -> u32 {
        match self {
            Layout::Joint(l) => l.special_tokens.pad,
            Layout::PerQuestion(l) => l.special_tokens.pad,
        }
    }

    fn max_options(&self) -> usize {
        match self {
            Layout::Joint(l) => l.max_options,
            Layout::PerQuestion(l) => l.max_options,
        }
    }
}

pub struct PplxModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    seq_multiple: usize,
    pub layout: Layout,
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

impl Engine for PplxModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        PplxModel::run(self, state, questions)
    }
}

impl PplxModel {
    /// Load a model exported to one directory (development and parity tooling).
    pub fn load(dir: &Path, device: Device, intra_threads: Option<usize>) -> Result<Self, Error> {
        Self::load_files(&ModelFiles::dir(dir), device, intra_threads)
    }

    pub fn load_files(
        files: &ModelFiles,
        device: Device,
        intra_threads: Option<usize>,
    ) -> Result<Self, Error> {
        let raw: Value = read_json(&files.decision)?;
        let bad = |e: String| Error::Model(format!("{}: {e}", files.decision.display()));
        let config: DecisionConfig =
            serde_json::from_value(raw.clone()).map_err(|e| bad(e.to_string()))?;
        if config.engine != "onnx" {
            return Err(bad(format!(
                "unsupported engine {}; this engine serves onnx",
                config.engine
            )));
        }
        let layout = Layout::parse(&config.layout, raw).map_err(bad)?;
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
                "graph inputs {inputs:?} do not match the pplx contract ({INPUTS:?} -> {OUTPUT:?})"
            )));
        }
        Ok(PplxModel {
            session: Mutex::new(session),
            tokenizer: Tokenizer(tokenizer),
            seq_multiple: config.contract.seq_multiple,
            layout,
            calibration,
            device,
        })
    }

    /// The request's sequences.
    pub fn encode(&self, state: &Value, questions: &Questions) -> Result<Vec<PplxSequence>, Error> {
        Ok(self.layout.encode(&self.tokenizer, state, questions)?)
    }

    /// Every question's option logits, in request order (`n` questions in all).
    pub fn option_logits(&self, seqs: &[PplxSequence], n: usize) -> Result<Vec<Vec<f32>>, Error> {
        let mut order: Vec<usize> = (0..seqs.len()).collect();
        order.sort_by_key(|&i| seqs[i].ids.len());
        let padded: Vec<usize> = order
            .iter()
            .map(|&i| seqs[i].ids.len().div_ceil(self.seq_multiple) * self.seq_multiple)
            .collect();
        let pad = i64::from(self.layout.pad());
        let mut logits = vec![Vec::new(); n];
        let mut session = self.session.lock().expect("session mutex poisoned");
        for range in crate::engine::batches(&padded, TOKEN_BUDGET, MAX_ROWS) {
            let batch = &order[range.clone()];
            let seq = padded[range].iter().copied().max().unwrap_or(0);
            let width = batch
                .iter()
                .map(|&i| seqs[i].markers.len())
                .max()
                .unwrap_or(0);
            let mut input_ids = Array2::<i64>::from_elem((batch.len(), seq), pad);
            let mut lengths = Array1::<i64>::zeros(batch.len());
            let mut markers = Array2::<i64>::zeros((batch.len(), width));
            for (r, &i) in batch.iter().enumerate() {
                for (c, &id) in seqs[i].ids.iter().enumerate() {
                    input_ids[[r, c]] = i64::from(id);
                }
                lengths[r] = seqs[i].ids.len() as i64;
                for (c, &m) in seqs[i].markers.iter().enumerate() {
                    markers[[r, c]] = m as i64;
                }
            }
            let options = crate::engine::run_options(self.device, batch.len() * seq, TOKEN_BUDGET)?;
            let outputs = session.run_with_options(
                ort::inputs![
                    "input_ids" => ort::value::Tensor::from_array(input_ids)?,
                    "lengths" => ort::value::Tensor::from_array(lengths)?,
                    "markers" => ort::value::Tensor::from_array(markers)?,
                ],
                &options,
            )?;
            let out = outputs[OUTPUT]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix3>()
                .map_err(|e| Error::Model(format!("{OUTPUT}: {e}")))?;
            if out.shape()[..2] != [batch.len(), width]
                || out.shape()[2] < self.layout.max_options()
            {
                return Err(Error::Model(format!(
                    "{OUTPUT} has shape {:?} for {} rows of {width} questions",
                    out.shape(),
                    batch.len()
                )));
            }
            for (r, &i) in batch.iter().enumerate() {
                let s = &seqs[i];
                for (j, (&q, &k)) in s.questions.iter().zip(&s.counts).enumerate() {
                    logits[q] = (0..k).map(|c| out[[r, j, c]]).collect();
                }
            }
        }
        Ok(logits)
    }

    /// Answer every question.
    pub fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let seqs = self.encode(state, questions)?;
        let logits = self.option_logits(&seqs, questions.len())?;
        Ok(Output {
            questions: logits
                .into_iter()
                .map(|logits| QuestionOutput {
                    logits,
                    act_logits: None,
                })
                .collect(),
            input_tokens: seqs.iter().map(|s| s.ids.len()).sum(),
            state_tokens: 0,
            state_truncated: false,
        })
    }
}
