//! ONNX Runtime engine for Cloudflare's Clef (layout `clef-joint-v1`), text only.
//!
//! A request is one sequence holding every question (`ollaya_decision::clef`). The graph maps
//! `input_ids` [1, seq] (right-padded to a multiple of 64; positions implicit, no mask: every
//! backbone layer is causal), `token_positions` [n] (0..n), `question_spans` [q, 2],
//! `question_types` [q], `option_spans` [o, 2] and `option_question` [o] to the joint schema
//! head's raw `logits` [o], every question's options in turn.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Ix1};
use ollaya_decision::clef::{ClefLayout, ClefRow};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder};
use ort::session::Session;
use serde::Deserialize;
use serde_json::Value;

use crate::decider::WeightsInMemory;
use crate::engine::Engine;
use crate::onnx::{CudaArena, Device, ModelFiles, load_tokenizer, session_for};
use crate::{Error, Output, QuestionOutput};

const INPUTS: [&str; 6] = [
    "input_ids",
    "token_positions",
    "question_spans",
    "question_types",
    "option_spans",
    "option_question",
];
const OUTPUT: &str = "logits";
/// A request longer than this gives the CUDA arena back after its run (`engine::run_options`).
const TOKEN_BUDGET: usize = 8192;

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
    clef: ClefLayout,
}

pub struct ClefModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    seq_multiple: usize,
    pub layout: ClefLayout,
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

impl Engine for ClefModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let defs: Vec<(&str, &Value)> = questions
            .iter()
            .map(|(qid, q)| (qid.as_str(), &q.definition))
            .collect();
        let row = self.row(state, &defs)?;
        let logits = self.logits(&row)?;
        let questions = questions
            .values()
            .zip(&row.questions)
            .zip(logits)
            .map(|((q, cq), z)| {
                Ok(QuestionOutput {
                    logits: cq.in_question_order(q, &z)?,
                    act_logits: None,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Output {
            questions,
            input_tokens: row.ids.len(),
            state_tokens: row.state_tokens,
            state_truncated: row.state_truncated,
        })
    }
}

impl ClefModel {
    pub fn load(dir: &Path, device: Device, intra_threads: Option<usize>) -> Result<Self, Error> {
        Self::load_files(&ModelFiles::dir(dir), device, intra_threads)
    }

    pub fn load_files(
        files: &ModelFiles,
        device: Device,
        intra_threads: Option<usize>,
    ) -> Result<Self, Error> {
        let config: DecisionConfig = read_json(&files.decision)?;
        if config.engine != "onnx" || config.layout != ollaya_decision::clef::LAYOUT {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/{}",
                config.engine,
                config.layout,
                ollaya_decision::clef::LAYOUT
            )));
        }
        let bad = |e: String| Error::Model(format!("{}: {e}", files.decision.display()));
        config.clef.validate().map_err(|e| bad(e.to_string()))?;
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
                "graph inputs {inputs:?} do not match the clef contract ({INPUTS:?} -> {OUTPUT:?})"
            )));
        }
        Ok(ClefModel {
            session: Mutex::new(session),
            tokenizer: Tokenizer(tokenizer),
            seq_multiple: config.contract.seq_multiple,
            layout: config.clef,
            calibration,
            device,
        })
    }

    /// The request's sequence.
    pub fn row(&self, state: &Value, defs: &[(&str, &Value)]) -> Result<ClefRow, Error> {
        Ok(self.layout.encode(&self.tokenizer, state, defs)?)
    }

    /// Each question's raw logits, in the layout's option order.
    pub fn logits(&self, row: &ClefRow) -> Result<Vec<Vec<f32>>, Error> {
        let n = row.ids.len();
        let seq = n.div_ceil(self.seq_multiple) * self.seq_multiple;
        let mut input_ids = Array2::<i64>::from_elem((1, seq), i64::from(self.layout.pad));
        for (c, &id) in row.ids.iter().enumerate() {
            input_ids[[0, c]] = i64::from(id);
        }
        let positions = Array1::from_iter(0..n as i64);
        let qs = &row.questions;
        let mut question_spans = Array2::<i64>::zeros((qs.len(), 2));
        let mut question_types = Array1::<i64>::zeros(qs.len());
        let count: usize = qs.iter().map(|q| q.options.len()).sum();
        let mut option_spans = Array2::<i64>::zeros((count, 2));
        let mut option_question = Array1::<i64>::zeros(count);
        let mut o = 0;
        for (i, q) in qs.iter().enumerate() {
            question_spans[[i, 0]] = q.span.0 as i64;
            question_spans[[i, 1]] = q.span.1 as i64;
            question_types[i] = q.type_index;
            for &(a, b) in &q.options {
                option_spans[[o, 0]] = a as i64;
                option_spans[[o, 1]] = b as i64;
                option_question[o] = i as i64;
                o += 1;
            }
        }
        let mut session = self.session.lock().expect("session mutex poisoned");
        let options = crate::engine::run_options(self.device, seq, TOKEN_BUDGET)?;
        let outputs = session.run_with_options(
            ort::inputs![
                "input_ids" => ort::value::Tensor::from_array(input_ids)?,
                "token_positions" => ort::value::Tensor::from_array(positions)?,
                "question_spans" => ort::value::Tensor::from_array(question_spans)?,
                "question_types" => ort::value::Tensor::from_array(question_types)?,
                "option_spans" => ort::value::Tensor::from_array(option_spans)?,
                "option_question" => ort::value::Tensor::from_array(option_question)?,
            ],
            &options,
        )?;
        let out = outputs[OUTPUT]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix1>()
            .map_err(|e| Error::Model(format!("{OUTPUT}: {e}")))?;
        if out.len() != count {
            return Err(Error::Model(format!(
                "{OUTPUT} has {} values for {count} options",
                out.len()
            )));
        }
        let mut it = out.iter().copied();
        Ok(qs
            .iter()
            .map(|q| it.by_ref().take(q.options.len()).collect())
            .collect())
    }
}
