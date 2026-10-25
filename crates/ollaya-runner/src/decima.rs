//! The `decima-late-interaction-v1` and `-v2` engine: Decima's encoder and late-interaction scorer
//! in one ONNX graph (`ollaya_decision::decima` builds the rows; the two layouts differ only in how
//! option rows are rendered).
//!
//! The graph maps `state_ids` / `state_mask` [q, ls] (one state row per question), `option_ids` /
//! `option_mask` [o, lo] (every option row, questions in order) and `option_state` [o] (each
//! option's question) to three values per option: `scores` (raw, before the temperature),
//! `ordinal_g` and `ordinal_gap` (the ordinal head's projections of the pooled option). Rows are
//! right-padded with the pad id; the masks keep padding out of every attention and mean.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Ix1};
use ollaya_decision::decima::{
    DecimaEncoder, DecimaLayout, DecimaQuestion, DecimaRequest, LAYOUT, LAYOUT_FIT,
};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder};
use ort::session::Session;
use serde::Deserialize;
use serde_json::Value;

use crate::engine::{Engine, TOKEN_BUDGET};
use crate::onnx::{Device, ModelFiles, load_tokenizer, session};
use crate::{Error, Output, QuestionOutput};

const INPUTS: [&str; 5] = [
    "state_ids",
    "state_mask",
    "option_ids",
    "option_mask",
    "option_state",
];
const OUTPUTS: [&str; 3] = ["scores", "ordinal_g", "ordinal_gap"];

#[derive(Debug, Clone, Deserialize)]
struct DecisionConfig {
    engine: String,
    layout: String,
    #[serde(flatten)]
    decima: DecimaLayout,
}

/// The graph's outputs for one question, in upstream's option order.
#[derive(Debug, Clone, PartialEq)]
pub struct Scores {
    pub scores: Vec<f32>,
    pub ordinal_g: Vec<f32>,
    pub ordinal_gap: Vec<f32>,
}

pub struct DecimaModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    pub layout: DecimaLayout,
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

impl DecimaEncoder for Tokenizer {
    fn nfc(&self, text: &str) -> String {
        tokenizers::NormalizedString::from(text)
            .nfc()
            .get()
            .to_owned()
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::Model(format!("{}: {e}", path.display())))
}

/// Consecutive questions per `session.run`, so that the padded tokens (state rows × the longest
/// state row, plus option rows × the longest option row) stay within `budget`. A question alone
/// over the budget still runs, alone.
fn batches(questions: &[DecimaQuestion], budget: usize) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    let (mut ls, mut lo, mut options) = (0, 0, 0);
    for (i, q) in questions.iter().enumerate() {
        let next_ls = ls.max(q.state_ids.len());
        let next_lo = q.option_ids.iter().map(Vec::len).fold(lo, usize::max);
        let next_options = options + q.option_ids.len();
        let cost = (i - start + 1) * next_ls + next_options * next_lo;
        if i > start && cost > budget {
            out.push(start..i);
            start = i;
            ls = q.state_ids.len();
            lo = q.option_ids.iter().map(Vec::len).max().unwrap_or(0);
            options = q.option_ids.len();
        } else {
            (ls, lo, options) = (next_ls, next_lo, next_options);
        }
    }
    if start < questions.len() {
        out.push(start..questions.len());
    }
    out
}

impl Engine for DecimaModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let request = self.encode(state, questions)?;
        let scores = self.scores(&request.questions)?;
        let outputs = request
            .questions
            .iter()
            .zip(&scores)
            .map(|(q, s)| QuestionOutput {
                logits: self
                    .layout
                    .answer_logits(q.qtype, &s.scores, &s.ordinal_g, &s.ordinal_gap),
                act_logits: None,
            })
            .collect();
        let input_tokens = request
            .questions
            .iter()
            .map(|q| q.state_ids.len() + q.option_ids.iter().map(Vec::len).sum::<usize>())
            .sum();
        Ok(Output {
            questions: outputs,
            input_tokens,
            state_tokens: request.state_tokens,
            state_truncated: request.questions.iter().any(|q| q.state_truncated),
        })
    }

    fn ordinal(&self) -> bool {
        true
    }
}

impl DecimaModel {
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
        if config.engine != "onnx" || ![LAYOUT, LAYOUT_FIT].contains(&config.layout.as_str()) {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/{LAYOUT} and onnx/{LAYOUT_FIT}",
                config.engine, config.layout
            )));
        }
        // option_fit changes the rows, so it comes with its own layout id: an older runtime refuses
        // the model instead of rendering its options the v1 way.
        if config.layout != config.decima.layout_id() {
            return Err(Error::Model(format!(
                "{}: layout {} with option_fit {}; option_fit models are {LAYOUT_FIT}",
                files.decision.display(),
                config.layout,
                config.decima.option_fit
            )));
        }
        config
            .decima
            .validate()
            .map_err(|e| Error::Model(format!("{}: {e}", files.decision.display())))?;
        let calibration = match &files.calibration {
            Some(path) => Calibration::from_file(&read_json::<CalibrationFile>(path)?),
            None => Calibration::default(),
        };
        let tokenizer = load_tokenizer(&files.tokenizer)?;
        let session = session(&files.graph, device, intra_threads)?;
        let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        let outputs: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
        if inputs.len() != INPUTS.len()
            || !INPUTS.iter().all(|n| inputs.contains(n))
            || !OUTPUTS.iter().all(|n| outputs.contains(n))
        {
            return Err(Error::Model(format!(
                "graph inputs {inputs:?} and outputs {outputs:?} do not match the decima contract \
                 ({INPUTS:?} -> {OUTPUTS:?})"
            )));
        }
        Ok(DecimaModel {
            session: Mutex::new(session),
            tokenizer: Tokenizer(tokenizer),
            layout: config.decima,
            calibration,
            device,
        })
    }

    /// Every question's rows.
    pub fn encode(&self, state: &Value, questions: &Questions) -> Result<DecimaRequest, Error> {
        Ok(self.layout.encode(&self.tokenizer, state, questions)?)
    }

    /// The graph's outputs for every question, in upstream's option order.
    pub fn scores(&self, questions: &[DecimaQuestion]) -> Result<Vec<Scores>, Error> {
        let mut out = Vec::with_capacity(questions.len());
        for range in batches(questions, TOKEN_BUDGET) {
            out.extend(self.run_batch(&questions[range])?);
        }
        Ok(out)
    }

    fn run_batch(&self, questions: &[DecimaQuestion]) -> Result<Vec<Scores>, Error> {
        let pad = i64::from(self.layout.special_tokens.pad);
        let ls = questions
            .iter()
            .map(|q| q.state_ids.len())
            .max()
            .unwrap_or(0);
        let lo = questions
            .iter()
            .flat_map(|q| q.option_ids.iter().map(Vec::len))
            .max()
            .unwrap_or(0);
        let count: usize = questions.iter().map(|q| q.option_ids.len()).sum();
        let mut state_ids = Array2::<i64>::from_elem((questions.len(), ls), pad);
        let mut state_mask = Array2::<i64>::zeros((questions.len(), ls));
        let mut option_ids = Array2::<i64>::from_elem((count, lo), pad);
        let mut option_mask = Array2::<i64>::zeros((count, lo));
        let mut option_state = Array1::<i64>::zeros(count);
        let mut k = 0;
        for (i, q) in questions.iter().enumerate() {
            for (c, &id) in q.state_ids.iter().enumerate() {
                state_ids[[i, c]] = i64::from(id);
                state_mask[[i, c]] = 1;
            }
            for row in &q.option_ids {
                for (c, &id) in row.iter().enumerate() {
                    option_ids[[k, c]] = i64::from(id);
                    option_mask[[k, c]] = 1;
                }
                option_state[k] = i as i64;
                k += 1;
            }
        }
        let mut session = self.session.lock().expect("session mutex poisoned");
        let outputs = session.run(ort::inputs![
            "state_ids" => ort::value::Tensor::from_array(state_ids)?,
            "state_mask" => ort::value::Tensor::from_array(state_mask)?,
            "option_ids" => ort::value::Tensor::from_array(option_ids)?,
            "option_mask" => ort::value::Tensor::from_array(option_mask)?,
            "option_state" => ort::value::Tensor::from_array(option_state)?,
        ])?;
        let mut values = Vec::with_capacity(OUTPUTS.len());
        for name in OUTPUTS {
            let v = outputs[name]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix1>()
                .map_err(|e| Error::Model(format!("{name}: {e}")))?
                .to_vec();
            if v.len() != count {
                return Err(Error::Model(format!(
                    "{name} has {} values for {count} options",
                    v.len()
                )));
            }
            values.push(v);
        }
        let mut start = 0;
        Ok(questions
            .iter()
            .map(|q| {
                let r = start..start + q.option_ids.len();
                start = r.end;
                Scores {
                    scores: values[0][r.clone()].to_vec(),
                    ordinal_g: values[1][r.clone()].to_vec(),
                    ordinal_gap: values[2][r].to_vec(),
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ollaya_decision::QType;

    fn q(state: usize, options: &[usize]) -> DecimaQuestion {
        DecimaQuestion {
            qtype: QType::Choice,
            state_ids: vec![0; state],
            option_ids: options.iter().map(|&n| vec![0; n]).collect(),
            state_truncated: false,
        }
    }

    #[test]
    fn batches_respect_the_token_budget() {
        // 2 * 100 + 4 * 10 = 240 fits 250; a third question would make 3 * 100 + 6 * 10 = 360.
        let qs = [q(100, &[10, 10]), q(100, &[10, 10]), q(100, &[10, 10])];
        assert_eq!(batches(&qs, 250), vec![0..2, 2..3]);
        // A longer option row later on counts for every option row of its batch.
        let qs = [q(10, &[5, 5]), q(10, &[50])];
        assert_eq!(batches(&qs, 100), vec![0..1, 1..2]);
        // One question over the budget still runs alone.
        assert_eq!(batches(&[q(512, &[64; 10])], 100), vec![0..1]);
        assert!(batches(&[], 100).is_empty());
        assert_eq!(batches(&qs, 1_000_000), vec![0..2]);
    }
}
