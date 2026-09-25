//! Engine-agnostic decision logic for Ollaya.
//!
//! Everything between the HTTP request and the network lives here: the typed question schema,
//! sequence layouts, temperature calibration and answer rendering. Engines (ONNX Runtime today,
//! llama.cpp later) only turn encoded inputs into option logits, so every model family speaks the
//! same API through this crate.

pub mod answer;
pub mod calibration;
pub mod decider;
pub mod gliclass;
pub mod kev;
pub mod layout;
pub mod nli;
mod printable;
pub mod pyjson;
pub mod pyrepr;
pub mod question;
pub mod qwen3guard;

pub use answer::Answer;
pub use calibration::{Calibration, CalibrationFile, TemperatureMap};
pub use layout::{Encoded, LayaLayout, SpecialTokens, TokenEncoder, serialize_state};
pub use question::{Criteria, QType, Question, Questions, parse_questions};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The request is malformed; the message names the question and what to fix.
    #[error("{0}")]
    Invalid(String),
    #[error(
        "question {question:?}: {options} options exceed the model's option budget (head_max_len={head_max_len})"
    )]
    TooManyOptions {
        /// Filled in by the caller that knows the question id (see [`Error::for_question`]).
        question: String,
        options: usize,
        head_max_len: usize,
    },
    #[error("tokenizer: {0}")]
    Tokenizer(String),
}

impl Error {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::Invalid(msg.into())
    }

    /// Attach the question id to an error raised while encoding that question.
    pub fn for_question(self, qid: &str) -> Self {
        match self {
            Error::TooManyOptions {
                options,
                head_max_len,
                ..
            } => Error::TooManyOptions {
                question: qid.to_owned(),
                options,
                head_max_len,
            },
            e => e,
        }
    }
}
