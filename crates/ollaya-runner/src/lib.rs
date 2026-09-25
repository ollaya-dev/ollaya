//! Inference engines that run inside Ollaya's per-model runner processes.
//!
//! An engine turns a (state, questions) request into raw option logits. Calibration and answer
//! rendering happen in `ollaya-decision`, so engines stay small and interchangeable.

pub mod decider;
pub mod engine;
pub mod gliclass;
pub mod kev;
#[cfg(feature = "mlx")]
pub mod mlx;
pub mod net;
pub mod nli;
pub mod onnx;
pub mod qwen3guard;
pub mod server;
pub mod von;

pub use engine::Engine;
pub use onnx::{Device, Encoding, ModelFiles, OnnxModel};

/// Process-wide settings the engines rely on, set before the process starts any thread. With the
/// `mlx` feature: `MLX_ENABLE_TF32=0`, which MLX reads once (see `ollaya_mlx::disable_tf32`);
/// runner processes inherit it. Otherwise it does nothing.
///
/// # Safety
///
/// Changes the process environment: call it first thing in `main`, while only one thread runs.
pub unsafe fn prepare_process() {
    #[cfg(feature = "mlx")]
    // SAFETY: the caller guarantees the process is single-threaded.
    unsafe {
        ollaya_mlx::disable_tf32()
    };
}

/// Raw network output for one question.
#[derive(Debug, Clone)]
pub struct QuestionOutput {
    /// One logit per option, in option order.
    pub logits: Vec<f32>,
    /// Act/escalate logits, when the model has an act head.
    pub act_logits: Option<Vec<f32>>,
}

/// Raw network output for one request.
#[derive(Debug, Clone)]
pub struct Output {
    pub questions: Vec<QuestionOutput>,
    /// Encoder tokens processed, summed over questions (TypeSafe's `usage.input_tokens`).
    pub input_tokens: usize,
    /// Tokens in the serialized state, before truncation.
    pub state_tokens: usize,
    /// The state was cut to fit at least one question's sequence.
    pub state_truncated: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Decision(#[from] ollaya_decision::Error),
    #[error("onnx runtime: {0}")]
    Ort(#[from] ort::Error),
    #[error("model files: {0}")]
    Model(String),
}

// Session builder errors carry the builder for recovery; we only need the message.
impl From<ort::Error<ort::session::builder::SessionBuilder>> for Error {
    fn from(e: ort::Error<ort::session::builder::SessionBuilder>) -> Self {
        Error::Ort(e.into())
    }
}
