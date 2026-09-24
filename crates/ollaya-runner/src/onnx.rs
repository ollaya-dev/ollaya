//! ONNX Runtime engine for encoder decision models (layout `laya-markers-v1`).
//!
//! A model directory holds the layers of one manifest:
//!
//! ```text
//! model.onnx [+ model.onnx.data]   encoder + decision head + marker gather, one graph
//! tokenizer.json                   HF tokenizers file
//! decision.json                    layout, lengths, special tokens, tensor names
//! calibration.json                 temperatures
//! ```

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::Array2;
use ollaya_decision::{
    Calibration, CalibrationFile, LayaLayout, Questions, SpecialTokens, TokenEncoder,
};
use ort::session::Session;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::value::Tensor;
use serde::Deserialize;
use serde_json::Value;

use crate::{Error, Output, QuestionOutput};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    /// CUDA device ordinal.
    Cuda(i32),
}

/// The `decision` layer.
#[derive(Debug, Clone, Deserialize)]
pub struct DecisionConfig {
    pub engine: String,
    pub layout: String,
    pub max_len: usize,
    pub head_max_len: usize,
    pub special_tokens: SpecialTokens,
    /// The graph takes at least this many marker slots (extra slots are masked).
    #[serde(default = "one")]
    pub min_markers: usize,
}

fn one() -> usize {
    1
}

/// Encoder input for one request: every question against the shared state.
#[derive(Debug, Clone)]
pub struct Encoding {
    pub questions: Vec<ollaya_decision::Encoded>,
    /// Tokens in the serialized state, before any truncation.
    pub state_tokens: usize,
}

pub struct OnnxModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    layout: LayaLayout,
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

/// An ONNX Runtime session on `device`. Every engine builds its sessions here, so all of them
/// get the same execution-provider settings.
pub fn session(
    graph: &Path,
    device: Device,
    intra_threads: Option<usize>,
) -> Result<Session, Error> {
    session_with(graph, device, intra_threads, Ok)
}

/// [`session`], with `configure` applied to the builder just before the graph loads: session
/// options that one family's graphs need.
pub fn session_with(
    graph: &Path,
    device: Device,
    intra_threads: Option<usize>,
    configure: impl FnOnce(SessionBuilder) -> Result<SessionBuilder, Error>,
) -> Result<Session, Error> {
    let mut builder =
        Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;
    if let Some(n) = intra_threads {
        builder = builder.with_intra_threads(n)?;
    }
    if let Device::Cuda(id) = device {
        builder = with_cuda(builder, id)?;
    }
    Ok(configure(builder)?.commit_from_file(graph)?)
}

#[cfg(feature = "cuda")]
fn with_cuda(
    builder: ort::session::builder::SessionBuilder,
    device_id: i32,
) -> Result<ort::session::builder::SessionBuilder, Error> {
    // TF32 matmuls keep 10 mantissa bits, which moves calibrated probabilities by ~1e-3 and
    // flips close decisions. fp32 graphs run in true fp32; speed comes from fp16 graphs.
    Ok(builder.with_execution_providers([ort::ep::CUDA::default()
        .with_device_id(device_id)
        .with_tf32(false)
        .build()
        .error_on_failure()])?)
}

#[cfg(not(feature = "cuda"))]
fn with_cuda(
    _builder: ort::session::builder::SessionBuilder,
    _device_id: i32,
) -> Result<ort::session::builder::SessionBuilder, Error> {
    Err(Error::Model(
        "this build of ollaya has no CUDA support".into(),
    ))
}

/// A model's `tokenizer.json`, with any truncation or padding it ships switched off: layouts
/// place every token themselves, and Python's tokenizer calls (the references) ignore those
/// settings too. Some upstream files bake in truncation (e.g. 512) that would silently cut states.
pub fn load_tokenizer(path: &Path) -> Result<tokenizers::Tokenizer, Error> {
    let mut tok = tokenizers::Tokenizer::from_file(path)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    tok.with_truncation(None)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    tok.with_padding(None);
    Ok(tok)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::Model(format!("{}: {e}", path.display())))
}

/// The files one model loads from. In the blob store they are `sha256-<hex>` blobs; in a
/// development export they are the named files of one directory.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub graph: PathBuf,
    pub tokenizer: PathBuf,
    pub decision: PathBuf,
    pub calibration: Option<PathBuf>,
}

impl ModelFiles {
    /// `model.onnx`, `tokenizer.json`, `decision.json`, `calibration.json` in one directory.
    pub fn dir(dir: &Path) -> Self {
        ModelFiles {
            graph: dir.join("model.onnx"),
            tokenizer: dir.join("tokenizer.json"),
            decision: dir.join("decision.json"),
            calibration: Some(dir.join("calibration.json")),
        }
    }
}

impl OnnxModel {
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
        if config.engine != "onnx" || config.layout != "laya-markers-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this runner serves onnx/laya-markers-v1",
                config.engine, config.layout
            )));
        }
        let calibration = match &files.calibration {
            Some(path) => Calibration::from_file(&read_json::<CalibrationFile>(path)?),
            None => Calibration::default(),
        };
        let tokenizer = load_tokenizer(&files.tokenizer)?;

        let session = session(&files.graph, device, intra_threads)?;

        Ok(OnnxModel {
            session: Mutex::new(session),
            tokenizer: Tokenizer(tokenizer),
            layout: LayaLayout {
                max_len: config.max_len,
                head_max_len: config.head_max_len,
                special: config.special_tokens,
            },
            min_markers: config.min_markers,
            calibration,
            device,
        })
    }

    /// Encode every question against the shared state (token ids and marker positions).
    pub fn encode(&self, state: &Value, questions: &Questions) -> Result<Encoding, Error> {
        let state_ids = self
            .layout
            .encode_state(&self.tokenizer, &ollaya_decision::serialize_state(state))?;
        let questions = questions
            .iter()
            .map(|(qid, q)| {
                self.layout
                    .encode(&self.tokenizer, &state_ids, q)
                    .map_err(|e| Error::Decision(e.for_question(qid)))
            })
            .collect::<Result<_, _>>()?;
        Ok(Encoding {
            questions,
            state_tokens: state_ids.len(),
        })
    }

    /// Answer every question in one forward pass.
    pub fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let encoded = self.encode(state, questions)?;
        self.run_encoded(&encoded, questions)
    }

    pub fn run_encoded(&self, encoding: &Encoding, questions: &Questions) -> Result<Output, Error> {
        let encoded = &encoding.questions;
        let qtypes: Vec<i64> = questions.values().map(|q| q.qtype.index() as i64).collect();
        let lens: Vec<usize> = encoded.iter().map(|e| e.ids.len()).collect();
        let mut outputs = Vec::with_capacity(encoded.len());
        for range in crate::engine::batches(&lens, crate::engine::TOKEN_BUDGET, usize::MAX) {
            outputs.extend(self.run_batch(&encoded[range.clone()], &qtypes[range])?);
        }
        Ok(Output {
            questions: outputs,
            input_tokens: lens.iter().sum(),
            state_tokens: encoding.state_tokens,
            state_truncated: encoded.iter().any(|e| e.state_truncated),
        })
    }

    /// One forward pass over `encoded` (one row per question), padded to its longest row.
    fn run_batch(
        &self,
        encoded: &[ollaya_decision::Encoded],
        qtypes: &[i64],
    ) -> Result<Vec<QuestionOutput>, Error> {
        let n = encoded.len();
        let seq = encoded.iter().map(|e| e.ids.len()).max().unwrap_or(0);
        let k = encoded
            .iter()
            .map(|e| e.markers.len())
            .max()
            .unwrap_or(0)
            .max(self.min_markers);

        let pad = i64::from(self.layout.special.pad);
        let mut input_ids = Array2::<i64>::from_elem((n, seq), pad);
        let mut attention = Array2::<i64>::zeros((n, seq));
        let mut marker_pos = Array2::<i64>::zeros((n, k));
        let mut marker_mask = Array2::<bool>::from_elem((n, k), false);
        for (r, e) in encoded.iter().enumerate() {
            for (c, &id) in e.ids.iter().enumerate() {
                input_ids[[r, c]] = i64::from(id);
                attention[[r, c]] = 1;
            }
            for (c, &m) in e.markers.iter().enumerate() {
                marker_pos[[r, c]] = m as i64;
                marker_mask[[r, c]] = true;
            }
        }

        let mut session = self.session.lock().expect("session mutex poisoned");
        let outputs = session.run(ort::inputs![
            "input_ids" => Tensor::from_array(input_ids)?,
            "attention_mask" => Tensor::from_array(attention)?,
            "marker_pos" => Tensor::from_array(marker_pos)?,
            "marker_mask" => Tensor::from_array(marker_mask)?,
            "qtype" => Tensor::from_array(([n], qtypes.to_vec()))?,
        ])?;
        let logits = outputs["logits"].try_extract_array::<f32>()?;
        let act = outputs["act_logits"].try_extract_array::<f32>()?;
        Ok(encoded
            .iter()
            .enumerate()
            .map(|(r, e)| QuestionOutput {
                logits: (0..e.markers.len()).map(|c| logits[[r, c]]).collect(),
                act_logits: Some(act.slice(ndarray::s![r, ..]).to_vec()),
            })
            .collect())
    }
}
