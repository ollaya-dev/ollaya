//! ONNX Runtime engine for jaredpalmer's Kev models (layout `kev-pointer-v1`).
//!
//! Every question is one row: the shared state, the question, its option spans and the decide
//! token. The graph maps `input_ids` [rows, seq] (right-padded to a multiple of 64, positions
//! implicit, no mask: every layer is causal), `decide_pos` [rows] and `opt_pos` [rows, k] (shorter
//! rows padded with 0) to the pointer head's raw `scores` [rows, k]. A row's first `k_row` scores
//! are its option logits. Rows come from `ollaya_decision::kev`.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Ix2};
use ollaya_decision::kev::{KevLayout, KevRow, KevState};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder};
use ort::session::Session;
use ort::session::builder::SessionBuilder;
use serde::Deserialize;
use serde_json::Value;

use crate::decider::WeightsInMemory;
use crate::engine::Engine;
use crate::onnx::{CudaArena, Device, ModelFiles, load_tokenizer, session_for};
use crate::{Error, Output, QuestionOutput};

/// Rows per `session.run`: the export's row axis is 1..=4096.
const MAX_ROWS: usize = 4096;
/// Padded tokens per `session.run`, as for the other Qwen3.5 decoders (`decider`).
const TOKEN_BUDGET: usize = 8192;
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
    kev: KevLayout,
}

/// Rows for one request, one per question.
#[derive(Debug, Clone)]
pub struct KevEncoding {
    pub rows: Vec<KevRow>,
    pub state: KevState,
}

pub struct KevModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    seq_multiple: usize,
    pub layout: KevLayout,
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

impl Engine for KevModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        KevModel::run(self, state, questions)
    }
}

impl KevModel {
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
        if config.engine != "onnx" || config.layout != "kev-pointer-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/kev-pointer-v1",
                config.engine, config.layout
            )));
        }
        let bad = |e: String| Error::Model(format!("{}: {e}", files.decision.display()));
        config.kev.validate().map_err(|e| bad(e.to_string()))?;
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
                "graph inputs {inputs:?} do not match the kev contract ({INPUTS:?} -> {OUTPUT:?})"
            )));
        }

        Ok(KevModel {
            session: Mutex::new(session),
            tokenizer: Tokenizer(tokenizer),
            seq_multiple: config.contract.seq_multiple,
            layout: config.kev,
            calibration,
            device,
        })
    }

    /// Encode every question's row after the shared state.
    pub fn encode(&self, state: &Value, questions: &Questions) -> Result<KevEncoding, Error> {
        let state = self.layout.encode_state(&self.tokenizer, state)?;
        let rows = questions
            .iter()
            .map(|(qid, q)| self.layout.encode(&self.tokenizer, &state.ids, qid, q))
            .collect::<Result<_, _>>()?;
        Ok(KevEncoding { rows, state })
    }

    /// Answer every question.
    pub fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let encoding = self.encode(state, questions)?;
        let scores = self.scores(&encoding)?;
        Ok(self.output(&encoding, scores))
    }

    /// Each row's option scores (its first `k_row` graph outputs). Rows run shortest first, in
    /// batches padded to a multiple of `seq_multiple`.
    pub fn scores(&self, encoding: &KevEncoding) -> Result<Vec<Vec<f32>>, Error> {
        let rows = &encoding.rows;
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].ids.len());
        let padded: Vec<usize> = order
            .iter()
            .map(|&i| rows[i].ids.len().div_ceil(self.seq_multiple) * self.seq_multiple)
            .collect();
        let pad = i64::from(self.layout.special_tokens.pad);

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
            let outputs = session.run(ort::inputs![
                "input_ids" => ort::value::Tensor::from_array(input_ids)?,
                "decide_pos" => ort::value::Tensor::from_array(decide_pos)?,
                "opt_pos" => ort::value::Tensor::from_array(opt_pos)?,
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
                scores[i] = row.iter().take(rows[i].opts.len()).copied().collect();
            }
        }
        Ok(scores)
    }

    /// The request's output from each row's option scores.
    pub fn output(&self, encoding: &KevEncoding, scores: Vec<Vec<f32>>) -> Output {
        Output {
            questions: scores
                .into_iter()
                .map(|logits| QuestionOutput {
                    logits,
                    act_logits: None,
                })
                .collect(),
            input_tokens: encoding.rows.iter().map(|r| r.ids.len()).sum(),
            state_tokens: encoding.state.tokens,
            state_truncated: encoding.state.truncated,
        }
    }
}

/// Session options for the kev graphs on `device`: the decoder options (see
/// [`crate::decider::configure`]), and no `GemmTransposeFusion`.
///
/// This works around an ONNX Runtime bug; it is not a precision setting. The graph's only `Gemm`
/// is the pointer head's query projection (`head.q`, 1,024 -> 256). Its input, the decide-token
/// hidden states `[rows, 1024]`, comes from the export's row gather through an identity
/// `Transpose` (perm `[0, 1]`). ONNX Runtime 1.28, which this build links, folds a `Transpose`
/// that feeds only `Gemm`s into them by flipping `transA`, without looking at `perm`, so the
/// rewritten graph multiplies the transposed input: runs fail with "GEMM: Dimension mismatch",
/// and a batch of exactly 1,024 rows would pass the shape check and return wrong scores. ONNX
/// Runtime 1.30 fuses only a real matrix transpose (perm `[1, 0]`, microsoft/onnxruntime#32435).
/// With the pass off, ONNX Runtime runs the graph as exported, which matches the goldens on CPU
/// and CUDA. The head is two small matmuls, so the fusion saves nothing here. The setting can go
/// once `ort` links ONNX Runtime 1.30 or newer.
fn configure(builder: SessionBuilder, device: Device) -> Result<SessionBuilder, Error> {
    Ok(crate::decider::configure(builder, device)?
        .with_disabled_optimizers("GemmTransposeFusion")?)
}
