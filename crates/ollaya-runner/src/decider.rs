//! ONNX Runtime engine for Mapika's decider models (layout `decider-slots-v1`).
//!
//! Every question is one row (with isolated levels, every score level is one): the shared
//! context, then the question, ending at the `(` answer slot. The graph maps `input_ids`
//! [rows, seq] (right-padded to a multiple of 64, positions implicit, no mask: every layer is
//! causal) and `slot_pos` [rows] to `label_logits` [rows, labels], the tied LM head at the slot
//! restricted to the option-label tokens. Rows and their readout come from
//! `ollaya_decision::decider`.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Ix2};
use ollaya_decision::decider::{Context, DeciderLayout, Scoring};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder};
use ort::session::Session;
use ort::session::builder::SessionBuilder;
use serde::Deserialize;
use serde_json::Value;

use crate::engine::Engine;
use crate::onnx::{Device, ModelFiles, load_tokenizer, session_with};
use crate::{Error, Output, QuestionOutput};

/// Rows per `session.run`: the export's row axis is 1..=4096.
const MAX_ROWS: usize = 4096;
/// Padded tokens per `session.run`. Full attention materialises [rows, heads, seq, seq] scores,
/// so decoders take a smaller budget than encoders.
const TOKEN_BUDGET: usize = 8192;
const INPUTS: [&str; 2] = ["input_ids", "slot_pos"];
const OUTPUT: &str = "label_logits";

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
    #[serde(flatten)]
    decider: DeciderLayout,
}

/// Scoring rows for one request.
#[derive(Debug, Clone)]
pub struct DeciderEncoding {
    pub questions: Vec<Scoring>,
    pub context: Context,
}

pub struct DeciderModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    seq_multiple: usize,
    pub layout: DeciderLayout,
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

impl Engine for DeciderModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        DeciderModel::run(self, state, questions)
    }
}

impl DeciderModel {
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
        if config.engine != "onnx" || config.layout != "decider-slots-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/decider-slots-v1",
                config.engine, config.layout
            )));
        }
        let bad = |e: String| Error::Model(format!("{}: {e}", files.decision.display()));
        config.decider.validate().map_err(|e| bad(e.to_string()))?;
        if config.contract.seq_multiple == 0 {
            return Err(bad("contract.seq_multiple must be positive".into()));
        }
        let calibration = match &files.calibration {
            Some(path) => Calibration::from_file(&read_json::<CalibrationFile>(path)?),
            None => Calibration::default(),
        };
        let tokenizer = load_tokenizer(&files.tokenizer)?;

        let session = session_with(&files.graph, device, intra_threads, |b| {
            configure(b, device)
        })?;
        let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        if inputs.len() != INPUTS.len()
            || !INPUTS.iter().all(|n| inputs.contains(n))
            || !session.outputs().iter().any(|o| o.name() == OUTPUT)
        {
            return Err(Error::Model(format!(
                "graph inputs {inputs:?} do not match the decider contract ({INPUTS:?} -> {OUTPUT:?})"
            )));
        }

        Ok(DeciderModel {
            session: Mutex::new(session),
            tokenizer: Tokenizer(tokenizer),
            seq_multiple: config.contract.seq_multiple,
            layout: config.decider,
            calibration,
            device,
        })
    }

    /// Encode every question's rows after the shared context.
    pub fn encode(&self, state: &Value, questions: &Questions) -> Result<DeciderEncoding, Error> {
        let context = self.layout.encode_context(&self.tokenizer, state)?;
        let questions = questions
            .iter()
            .map(|(qid, q)| self.layout.encode(&self.tokenizer, &context.ids, qid, q))
            .collect::<Result<_, _>>()?;
        Ok(DeciderEncoding { questions, context })
    }

    /// Answer every question.
    pub fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let encoding = self.encode(state, questions)?;
        let logits = self.label_logits(&encoding)?;
        Ok(self.output(&encoding, &logits))
    }

    /// Label logits of every row, question by question. Rows run shortest first, in batches
    /// padded to a multiple of `seq_multiple`.
    pub fn label_logits(&self, encoding: &DeciderEncoding) -> Result<Vec<Vec<f32>>, Error> {
        let rows: Vec<&[u32]> = encoding
            .questions
            .iter()
            .flat_map(|s| s.rows.iter().map(Vec::as_slice))
            .collect();
        let width = encoding
            .questions
            .iter()
            .map(|s| s.options)
            .max()
            .unwrap_or(0)
            .max(2);
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].len());
        let padded: Vec<usize> = order
            .iter()
            .map(|&i| rows[i].len().div_ceil(self.seq_multiple) * self.seq_multiple)
            .collect();
        let pad = i64::from(self.layout.special_tokens.pad);

        let mut logits = vec![Vec::new(); rows.len()];
        let mut session = self.session.lock().expect("session mutex poisoned");
        for range in crate::engine::batches(&padded, TOKEN_BUDGET, MAX_ROWS) {
            let batch = &order[range.clone()];
            let seq = padded[range].iter().copied().max().unwrap_or(0);
            let mut input_ids = Array2::<i64>::from_elem((batch.len(), seq), pad);
            let mut slot_pos = Array1::<i64>::zeros(batch.len());
            for (r, &i) in batch.iter().enumerate() {
                for (c, &id) in rows[i].iter().enumerate() {
                    input_ids[[r, c]] = i64::from(id);
                }
                slot_pos[r] = rows[i].len() as i64 - 1;
            }
            let outputs = session.run(ort::inputs![
                "input_ids" => ort::value::Tensor::from_array(input_ids)?,
                "slot_pos" => ort::value::Tensor::from_array(slot_pos)?,
            ])?;
            let out = outputs[OUTPUT]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix2>()
                .map_err(|e| Error::Model(format!("{OUTPUT}: {e}")))?;
            if out.nrows() != batch.len() || out.ncols() < width {
                return Err(Error::Model(format!(
                    "{OUTPUT} has shape {:?} for {} rows of up to {width} options",
                    out.shape(),
                    batch.len()
                )));
            }
            for (&i, row) in batch.iter().zip(out.rows()) {
                logits[i] = row.to_vec();
            }
        }
        Ok(logits)
    }

    /// Option logits per question from the label logits of every row.
    pub fn output(&self, encoding: &DeciderEncoding, logits: &[Vec<f32>]) -> Output {
        let mut next = 0;
        let questions = encoding
            .questions
            .iter()
            .map(|s| {
                let rows = &logits[next..next + s.rows.len()];
                next += s.rows.len();
                QuestionOutput {
                    logits: self.layout.option_logits(
                        s.readout,
                        s.options,
                        rows.iter().map(Vec::as_slice),
                    ),
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
                .map(Vec::len)
                .sum(),
            state_tokens: encoding.context.tokens,
            state_truncated: encoding.context.truncated,
        }
    }
}

/// Session options for the decoder graphs (decider, kev, qwen3guard) on `device`.
///
/// On the CUDA provider, ONNX Runtime's memory planner gives the int64 shape values that the
/// graph's `Reshape`s read on the host to other values while they are still needed, and runs fail
/// ("requested shape {8,16,0,64}"). Python avoids this with `SessionOptions.enable_mem_reuse =
/// False`, which the C API does not expose. In parallel execution mode the planner reuses no
/// buffers of the main graph (neither freed nor in-place ones), and ORT 1.28 allows that mode
/// with the CUDA provider. The plan has one logic stream per device (CPU and CUDA), so two inter-op threads run
/// it. The CPU provider plans these graphs correctly and keeps the defaults.
pub(crate) fn configure(builder: SessionBuilder, device: Device) -> Result<SessionBuilder, Error> {
    match device {
        Device::Cpu | Device::Metal => Ok(builder),
        Device::Cuda(_) => Ok(builder
            .with_parallel_execution(true)?
            .with_inter_threads(2)?),
    }
}
