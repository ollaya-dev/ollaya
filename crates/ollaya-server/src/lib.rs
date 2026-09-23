//! The Ollaya daemon: HTTP API, model resolution and the runner scheduler.

pub mod config;
pub mod http;
pub mod launch;
pub mod models;
pub mod scheduler;
pub mod service;
mod views;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("model {0:?} not found, try pulling it first")]
    ModelNotFound(String),
    /// A router's target is missing locally.
    #[error("model {target:?} not found, try pulling it first (routed from {router:?})")]
    RoutedModelNotFound { target: String, router: String },
    /// The request brought no questions and the model has none built in.
    #[error("{0} has no built-in questions; send 'questions' with the request")]
    NoQuestions(String),
    #[error("{0}")]
    InvalidRequest(String),
    /// A question's options do not fit the answering model's option budget.
    #[error("question {question:?}: {options} options do not fit the option budget of {model}")]
    TooManyOptions {
        question: String,
        options: usize,
        model: String,
    },
    /// The state is longer than TypeSafe's token limit.
    #[error("state is {0} tokens long; the limit is {limit}", limit = ollaya_api::MAX_STATE_TOKENS)]
    InputTooLong(usize),
    /// The runner could not load the model (bad files, out of memory, load timeout).
    #[error("{0}")]
    LoadFailed(String),
    /// The runner failed while answering.
    #[error("runner: {0}")]
    Runner(String),
    /// This build cannot run the model's format.
    #[error("{0}")]
    Unsupported(String),
    /// A pull or create is writing this model name.
    #[error("model {0:?} is being pulled or created; retry when it finishes")]
    Busy(String),
    /// The client went away before the decision ran.
    #[error("request cancelled")]
    Cancelled,
    #[error("corrupt model: {0}")]
    Corrupt(String),
    #[error(transparent)]
    Registry(#[from] ollaya_registry::Error),
}
