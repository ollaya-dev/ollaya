//! Encoder decision models of layout `laya-markers-v1` in the browser.
//!
//! The graph runs in onnxruntime-web; everything around it runs here, compiled to WebAssembly from
//! the same code the native runner uses (`crates/ollaya-runner/src/onnx.rs`): tokenization with the
//! `tokenizers` crate, the sequence layout and calibration of `ollaya-decision`, and the answers of
//! `/api/decide`.
//!
//! ```text
//! const model = new Model(tokenizerJson, decisionJson, calibrationJson);
//! const batch = model.encode(JSON.stringify({model, state, questions}));  // graph inputs
//! const out = await session.run({input_ids: ..., ...});              // onnxruntime-web
//! const response = JSON.parse(model.answer(batch, out.logits.data, out.act_logits.data));
//! ```

use indexmap::IndexMap;
use ollaya_api::decide::engine_questions;
use ollaya_api::error::body_loc;
use ollaya_api::{
    Answer as ApiAnswer, DecideAnswer, DoneReason, ErrorBody, ErrorCode, Extra, LayaExtra,
    MAX_STATE_TOKENS, Usage, ValidationIssue, validate,
};
use ollaya_decision::{
    Answer, Calibration, CalibrationFile, Encoded, LayaLayout, Questions, SpecialTokens,
    TokenEncoder, parse_questions, serialize_state,
};
use serde::Deserialize;
use serde_json::{Value, json};
use wasm_bindgen::prelude::*;

/// The `decision` layer, as `ollaya_runner::onnx::DecisionConfig` reads it.
#[derive(Deserialize)]
struct DecisionConfig {
    engine: String,
    layout: String,
    max_len: usize,
    head_max_len: usize,
    special_tokens: SpecialTokens,
    #[serde(default = "one")]
    min_markers: usize,
}

fn one() -> usize {
    1
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

fn err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// A daemon error body, with its HTTP status, as the error's message.
fn fail(body: ErrorBody) -> JsError {
    let mut v = serde_json::to_value(&body).unwrap_or_default();
    v["status"] = json!(body.status());
    JsError::new(&v.to_string())
}

/// One model: its tokenizer, layout and calibration.
#[wasm_bindgen]
pub struct Model {
    tokenizer: Tokenizer,
    layout: LayaLayout,
    min_markers: usize,
    calibration: Calibration,
}

/// The graph inputs for one request (one row per question, padded to the longest), and what
/// [`Model::answer`] needs to read the outputs back.
#[wasm_bindgen]
pub struct Batch {
    model: String,
    questions: Questions,
    encoded: Vec<Encoded>,
    state_tokens: usize,
    seq: usize,
    markers: usize,
    pad: i64,
    laya: bool,
}

#[wasm_bindgen]
impl Model {
    /// `tokenizer.json`, the `decision` layer and the `calibration` layer (empty: no calibration).
    #[wasm_bindgen(constructor)]
    pub fn new(
        tokenizer_json: &str,
        decision_json: &str,
        calibration_json: &str,
    ) -> Result<Model, JsError> {
        let config: DecisionConfig = serde_json::from_str(decision_json).map_err(err)?;
        if config.engine != "onnx" || config.layout != "laya-markers-v1" {
            return Err(err(format!(
                "unsupported engine/layout {}/{}; this runtime serves onnx/laya-markers-v1",
                config.engine, config.layout
            )));
        }
        let calibration = if calibration_json.is_empty() {
            Calibration::default()
        } else {
            Calibration::from_file(
                &serde_json::from_str::<CalibrationFile>(calibration_json).map_err(err)?,
            )
        };
        // As `ollaya_runner::onnx::load_tokenizer`: the layout places every token itself.
        let mut tok: tokenizers::Tokenizer = tokenizer_json.parse().map_err(err)?;
        tok.with_truncation(None).map_err(err)?;
        tok.with_padding(None);
        Ok(Model {
            tokenizer: Tokenizer(tok),
            layout: LayaLayout {
                max_len: config.max_len,
                head_max_len: config.head_max_len,
                special: config.special_tokens,
            },
            min_markers: config.min_markers,
            calibration,
        })
    }

    /// Encode an `/api/decide` request body, validated as the daemon validates it
    /// (`ollaya_api::validate::decide_request`). Errors carry the daemon's error body, as JSON
    /// `{status, error, code, detail?}`.
    ///
    /// shortcut: every question goes in one batch; the native runner splits at 32768 padded tokens
    /// (`engine::TOKEN_BUDGET`). Split here too if requests with dozens of long questions matter.
    pub fn encode(&self, request_json: &str) -> Result<Batch, JsError> {
        let req =
            validate::body(request_json.as_bytes(), validate::decide_request).map_err(fail)?;
        if req.preset.is_some() || !req.images.is_empty() {
            return Err(fail(ErrorBody::new(
                ErrorCode::InvalidRequest,
                "the web runtime takes questions, not presets or images",
            )));
        }
        let Some(questions) = &req.questions else {
            return Err(fail(ErrorBody::invalid_request(vec![
                ValidationIssue::missing(body_loc(&["questions"])),
            ])));
        };
        // The daemon's engine form: a question without instructions reads its id in their place.
        let questions = parse_questions(&engine_questions(questions))
            .map_err(|e| fail(ErrorBody::new(ErrorCode::InvalidRequest, e.to_string())))?;
        let state = req.state.clone().unwrap_or(Value::Null);
        let state_ids = self
            .layout
            .encode_state(&self.tokenizer, &serialize_state(&state))
            .map_err(err)?;
        if state_ids.len() > MAX_STATE_TOKENS {
            return Err(fail(ErrorBody::input_too_long(state_ids.len())));
        }
        let encoded = questions
            .iter()
            .map(|(qid, q)| {
                self.layout
                    .encode(&self.tokenizer, &state_ids, q)
                    .map_err(|e| match e {
                        ollaya_decision::Error::TooManyOptions { options, .. } => fail(
                            ErrorBody::too_many_options(qid, q.qtype.name(), options, &req.model),
                        ),
                        e => err(e),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Batch {
            seq: encoded.iter().map(|e| e.ids.len()).max().unwrap_or(0),
            markers: encoded
                .iter()
                .map(|e| e.markers.len())
                .max()
                .unwrap_or(0)
                .max(self.min_markers),
            laya: req.wants(Extra::Laya),
            model: req.model,
            questions,
            encoded,
            state_tokens: state_ids.len(),
            pad: i64::from(self.layout.special.pad),
        })
    }

    /// The `/api/decide` response (`model`, `answers`, `usage`, `routing`, `state_truncated`,
    /// `done_reason`; the caller adds timings) from the graph's `logits` [rows, markers] and
    /// `act_logits` [rows, width] (empty when the graph has none).
    pub fn answer(
        &self,
        batch: &Batch,
        logits: &[f32],
        act_logits: &[f32],
    ) -> Result<String, JsError> {
        let rows = batch.encoded.len();
        if logits.len() != rows * batch.markers {
            return Err(err(format!(
                "logits hold {} values, expected {rows} x {}",
                logits.len(),
                batch.markers
            )));
        }
        let act_width = act_logits.len() / rows;
        let mut answers = IndexMap::new();
        for (r, ((qid, q), e)) in batch.questions.iter().zip(&batch.encoded).enumerate() {
            let row = &logits[r * batch.markers..][..e.markers.len()];
            let act = (act_width > 0).then(|| &act_logits[r * act_width..][..act_width]);
            let decided = Answer::new(q, &self.calibration, row, act, batch.state_tokens);
            let answer = DecideAnswer {
                answer: ApiAnswer::from_decision(&decided, q).map_err(err)?,
                laya: batch
                    .laya
                    .then(|| LayaExtra::from_decision(&decided, q))
                    .transpose()
                    .map_err(err)?,
            };
            answers.insert(qid.clone(), answer);
        }
        let usage = Usage {
            input_tokens: batch.encoded.iter().map(|e| e.ids.len() as u64).sum(),
            output_tokens: 0,
        };
        Ok(json!({
            "model": batch.model,
            "answers": answers,
            "usage": usage,
            "routing": null,
            "state_truncated": batch.encoded.iter().any(|e| e.state_truncated),
            "done_reason": DoneReason::Decide,
        })
        .to_string())
    }
}

#[wasm_bindgen]
impl Batch {
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.encoded.len()
    }

    #[wasm_bindgen(getter)]
    pub fn seq(&self) -> usize {
        self.seq
    }

    /// Marker slots per row (at least the model's `min_markers`; extra slots are masked).
    #[wasm_bindgen(getter)]
    pub fn markers(&self) -> usize {
        self.markers
    }

    /// Tokens in the serialized state, before truncation.
    #[wasm_bindgen(getter)]
    pub fn state_tokens(&self) -> usize {
        self.state_tokens
    }

    /// `input_ids` [rows, seq] int64, padded with the pad id.
    pub fn input_ids(&self) -> Vec<i64> {
        self.grid(self.seq, self.pad, |e| {
            e.ids.iter().map(|&i| i64::from(i)).collect()
        })
    }

    /// `attention_mask` [rows, seq] int64.
    pub fn attention_mask(&self) -> Vec<i64> {
        self.grid(self.seq, 0, |e| vec![1; e.ids.len()])
    }

    /// `marker_pos` [rows, markers] int64.
    pub fn marker_pos(&self) -> Vec<i64> {
        self.grid(self.markers, 0, |e| {
            e.markers.iter().map(|&m| m as i64).collect()
        })
    }

    /// `marker_mask` [rows, markers] bool, as bytes.
    pub fn marker_mask(&self) -> Vec<u8> {
        self.grid(self.markers, 0, |e| vec![1; e.markers.len()])
    }

    /// `qtype` [rows] int64.
    pub fn qtype(&self) -> Vec<i64> {
        self.questions
            .values()
            .map(|q| q.qtype.index() as i64)
            .collect()
    }

    /// Each row's token ids and marker positions, as JSON (`[{ids, markers}]`), for parity checks.
    pub fn encodings(&self) -> String {
        Value::from(
            self.encoded
                .iter()
                .map(|e| json!({"ids": e.ids, "markers": e.markers}))
                .collect::<Vec<_>>(),
        )
        .to_string()
    }
}

impl Batch {
    /// A [rows, width] grid, each row `row(encoded)` followed by `fill`.
    fn grid<T: Copy>(&self, width: usize, fill: T, row: impl Fn(&Encoded) -> Vec<T>) -> Vec<T> {
        let mut out = vec![fill; self.encoded.len() * width];
        for (r, e) in self.encoded.iter().enumerate() {
            let v = row(e);
            out[r * width..][..v.len()].copy_from_slice(&v);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENIZER: &str = r#"{"version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [], "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "unk_token": "[UNK]", "vocab": {"[UNK]": 0, "[SEP]": 1,
            "[CLS]": 2, "[PAD]": 3, "<mask>": 4, "yes": 5, "no": 6}}}"#;
    const DECISION: &str = r#"{"engine": "onnx", "layout": "laya-markers-v1", "max_len": 64,
        "head_max_len": 32, "min_markers": 3,
        "special_tokens": {"cls": 2, "sep": 1, "mask": 4, "pad": 3, "mask_text": "<mask>"}}"#;

    #[test]
    fn graph_inputs_are_padded_and_logits_read_back_per_row() {
        let model = Model::new(TOKENIZER, DECISION, "").unwrap();
        let request = r#"{"model": "t", "state": "yes no yes", "extras": ["laya"], "questions": {
            "a": {"type": "choice", "instructions": "yes", "criteria": ["yes", "no"]},
            "b": {"type": "noul", "instructions": "no no no no"}}}"#;
        let batch = model.encode(request).unwrap();
        let (rows, seq, k) = (batch.rows(), batch.seq(), batch.markers());
        assert_eq!((rows, k), (2, 3), "min_markers widens the marker axis");
        let lens: Vec<usize> = batch.encoded.iter().map(|e| e.ids.len()).collect();
        assert_eq!(seq, *lens.iter().max().unwrap());
        assert!(lens[0] < lens[1], "row 0 is padded");
        let ids = batch.input_ids();
        assert_eq!(ids.len(), rows * seq);
        assert_eq!(ids[lens[0]..seq], vec![3; seq - lens[0]][..]);
        assert_eq!(
            batch.attention_mask()[..seq].iter().sum::<i64>(),
            lens[0] as i64
        );
        assert_eq!(batch.marker_mask(), [1, 1, 0, 1, 1, 0]);
        assert_eq!(batch.qtype(), [0, 2]);

        // Row 0 prefers "no"; row 1 prefers true. The masked slot's logit is never read.
        let logits = [0.0, 4.0, 99.0, -3.0, 3.0, 99.0];
        let act = [1.0, 0.0, 0.0, 1.0];
        let response: Value =
            serde_json::from_str(&model.answer(&batch, &logits, &act).unwrap()).unwrap();
        assert_eq!(response["answers"]["a"]["choice"], "no");
        assert!(response["answers"]["b"]["noul"].as_f64().unwrap() > 0.99);
        assert!(
            response["answers"]["a"]["laya"]["act_probability"]
                .as_f64()
                .unwrap()
                > 0.5
        );
        assert_eq!(
            response["usage"]["input_tokens"],
            lens.iter().sum::<usize>()
        );
        assert_eq!(response["model"], "t");
    }
}
