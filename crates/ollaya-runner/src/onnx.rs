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

use std::path::Path;
use std::sync::Mutex;

use ndarray::Array2;
use ollaya_decision::{
    Calibration, CalibrationFile, LayaLayout, Questions, SpecialTokens, TokenEncoder,
};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
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

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::Model(format!("{}: {e}", path.display())))
}

impl OnnxModel {
    pub fn load(dir: &Path, device: Device, intra_threads: Option<usize>) -> Result<Self, Error> {
        let config: DecisionConfig = read_json(&dir.join("decision.json"))?;
        if config.engine != "onnx" || config.layout != "laya-markers-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this runner serves onnx/laya-markers-v1",
                config.engine, config.layout
            )));
        }
        let calibration = Calibration::from_file(&read_json::<CalibrationFile>(
            &dir.join("calibration.json"),
        )?);
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| Error::Model(format!("tokenizer.json: {e}")))?;

        let mut builder =
            Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;
        if let Some(n) = intra_threads {
            builder = builder.with_intra_threads(n)?;
        }
        if let Device::Cuda(id) = device {
            // TF32 matmuls keep 10 mantissa bits, which moves calibrated probabilities by ~1e-3
            // and flips close decisions. fp32 graphs run in true fp32; speed comes from fp16 graphs.
            builder = builder.with_execution_providers([ort::ep::CUDA::default()
                .with_device_id(id)
                .with_tf32(false)
                .build()
                .error_on_failure()])?;
        }
        let session = builder.commit_from_file(dir.join("model.onnx"))?;

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
    pub fn encode(
        &self,
        state: &Value,
        questions: &Questions,
    ) -> Result<Vec<ollaya_decision::Encoded>, Error> {
        let state_ids = self
            .layout
            .encode_state(&self.tokenizer, &ollaya_decision::serialize_state(state))?;
        questions
            .values()
            .map(|q| Ok(self.layout.encode(&self.tokenizer, &state_ids, q)?))
            .collect()
    }

    /// Answer every question in one forward pass.
    pub fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let encoded = self.encode(state, questions)?;
        self.run_encoded(&encoded, questions)
    }

    pub fn run_encoded(
        &self,
        encoded: &[ollaya_decision::Encoded],
        questions: &Questions,
    ) -> Result<Output, Error> {
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
        let mut qtype = Vec::with_capacity(n);
        for (r, (e, q)) in encoded.iter().zip(questions.values()).enumerate() {
            for (c, &id) in e.ids.iter().enumerate() {
                input_ids[[r, c]] = i64::from(id);
                attention[[r, c]] = 1;
            }
            for (c, &m) in e.markers.iter().enumerate() {
                marker_pos[[r, c]] = m as i64;
                marker_mask[[r, c]] = true;
            }
            qtype.push(q.qtype.index() as i64);
        }
        let input_tokens = encoded.iter().map(|e| e.ids.len()).sum();

        let mut session = self.session.lock().expect("session mutex poisoned");
        let outputs = session.run(ort::inputs![
            "input_ids" => Tensor::from_array(input_ids)?,
            "attention_mask" => Tensor::from_array(attention)?,
            "marker_pos" => Tensor::from_array(marker_pos)?,
            "marker_mask" => Tensor::from_array(marker_mask)?,
            "qtype" => Tensor::from_array(([n], qtype))?,
        ])?;
        let logits = outputs["logits"].try_extract_array::<f32>()?;
        let act = outputs["act_logits"].try_extract_array::<f32>()?;

        let questions = encoded
            .iter()
            .enumerate()
            .map(|(r, e)| QuestionOutput {
                logits: (0..e.markers.len()).map(|c| logits[[r, c]]).collect(),
                act_logits: Some(act.slice(ndarray::s![r, ..]).to_vec()),
            })
            .collect();
        Ok(Output {
            questions,
            input_tokens,
        })
    }
}
