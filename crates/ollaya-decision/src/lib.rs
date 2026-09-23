//! Engine-agnostic decision logic for Ollaya.
//!
//! Everything between the HTTP request and the network lives here: the typed question schema,
//! sequence layouts, temperature calibration and answer rendering. Engines (ONNX Runtime today,
//! llama.cpp later) only turn encoded inputs into option logits, so every model family speaks the
//! same API through this crate.

pub mod answer;
pub mod calibration;
pub mod layout;
pub mod pyjson;
pub mod question;

pub use answer::Answer;
pub use calibration::{Calibration, CalibrationFile};
pub use layout::{Encoded, LayaLayout, SpecialTokens, TokenEncoder, serialize_state};
pub use question::{Criteria, QType, Question, Questions, parse_questions};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The request is malformed; the message names the question and what to fix.
    #[error("{0}")]
    Invalid(String),
    #[error(
        "question options exceed the model's option budget ({options} options, head_max_len={head_max_len})"
    )]
    TooManyOptions { options: usize, head_max_len: usize },
    #[error("tokenizer: {0}")]
    Tokenizer(String),
}

impl Error {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::Invalid(msg.into())
    }
}
