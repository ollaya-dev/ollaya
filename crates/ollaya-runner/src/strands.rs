//! ONNX Runtime engine for Strands Agents' Strands Decider (layout `strands-pointer-v1`).
//!
//! Every question is one row: the state, then the question with one line per option. The graph has
//! kev's contract: `input_ids` [rows, seq] (right-padded to a multiple of 64, positions implicit,
//! no mask: every layer is causal), `decide_pos` [rows] and `opt_pos` [rows, k] to the pointer
//! head's raw `scores` [rows, k]. Rows come from `ollaya_decision::strands`.

use std::path::Path;
use std::sync::Mutex;

use ollaya_decision::strands::{Encoded, OffsetEncoder, StrandsEncoding, StrandsLayout};
use ollaya_decision::{Calibration, CalibrationFile, Questions};
use serde::Deserialize;
use serde_json::Value;

use crate::decider::WeightsInMemory;
use crate::engine::Engine;
use crate::kev::{configure, pointer_scores};
use crate::net::Decoder;
use crate::onnx::{CudaArena, Device, ModelFiles, load_tokenizer, session_for};
use crate::{Error, Output, QuestionOutput};

const INPUTS: [&str; 3] = ["input_ids", "decide_pos", "opt_pos"];
const OUTPUT: &str = "scores";

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
    #[serde(flatten)]
    strands: StrandsLayout,
}

pub struct StrandsModel {
    net: Decoder,
    tokenizer: Tokenizer,
    seq_multiple: usize,
    pub layout: StrandsLayout,
    pub calibration: Calibration,
    pub device: Device,
}

struct Tokenizer(tokenizers::Tokenizer);

impl OffsetEncoder for Tokenizer {
    fn encode_offsets(&self, text: &str) -> Result<Encoded, ollaya_decision::Error> {
        self.0
            .encode_char_offsets(text, false)
            .map(|e| (e.get_ids().to_vec(), e.get_offsets().to_vec()))
            .map_err(|e| ollaya_decision::Error::Tokenizer(e.to_string()))
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::Model(format!("{}: {e}", path.display())))
}

impl Engine for StrandsModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        StrandsModel::run(self, state, questions)
    }
}

impl StrandsModel {
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
        if config.engine != "onnx" || config.layout != "strands-pointer-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/strands-pointer-v1",
                config.engine, config.layout
            )));
        }
        let bad = |e: String| Error::Model(format!("{}: {e}", files.decision.display()));
        config.strands.validate().map_err(|e| bad(e.to_string()))?;
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
                "graph inputs {inputs:?} do not match the strands contract ({INPUTS:?} -> {OUTPUT:?})"
            )));
        }
        Ok(StrandsModel {
            net: Decoder::Ort(Mutex::new(session)),
            tokenizer: Tokenizer(tokenizer),
            seq_multiple: config.contract.seq_multiple,
            layout: config.strands,
            calibration,
            device,
        })
    }

    /// Every question's row, fitted to the window.
    pub fn encode(&self, state: &Value, questions: &Questions) -> Result<StrandsEncoding, Error> {
        Ok(self.layout.encode(&self.tokenizer, state, questions)?)
    }

    /// Each row's option scores.
    pub fn scores(&self, encoding: &StrandsEncoding) -> Result<Vec<Vec<f32>>, Error> {
        pointer_scores(
            &self.net,
            &encoding.rows,
            self.seq_multiple,
            self.layout.special_tokens.pad,
            self.device,
        )
    }

    /// Answer every question.
    pub fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let encoding = self.encode(state, questions)?;
        let scores = self.scores(&encoding)?;
        Ok(Output {
            questions: scores
                .into_iter()
                .map(|logits| QuestionOutput {
                    logits,
                    act_logits: None,
                })
                .collect(),
            input_tokens: encoding.rows.iter().map(|r| r.ids.len()).sum(),
            state_tokens: encoding.state_tokens,
            state_truncated: encoding.state_truncated,
        })
    }
}
