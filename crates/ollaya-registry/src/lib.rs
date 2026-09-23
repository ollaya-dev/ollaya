//! Model names, manifests, the local blob store, and pulling from a registry.

pub mod manifest;
pub mod name;
pub mod pull;
pub mod store;

pub use manifest::{Descriptor, Manifest, ModelConfig, Router, media};
pub use name::ModelName;
pub use pull::{Progress, Puller};
pub use store::{Entry, Store};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid model name {0:?}; expected [host/][namespace/]model[:tag]")]
    InvalidName(String),
    #[error("invalid digest {0:?}")]
    InvalidDigest(String),
    #[error("model {0} not found")]
    NotFound(String),
    #[error("digest mismatch: expected {expected}, got {got}")]
    DigestMismatch { expected: String, got: String },
    #[error("download stalled: {0}")]
    Stalled(String),
    #[error("corrupt data: {0}")]
    Corrupt(String),
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Error {
    /// Network hiccups worth retrying; a 4xx or a bad digest is not.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::Stalled(_) | Error::Io(_) => true,
            Error::Http(e) => {
                e.is_timeout()
                    || e.is_connect()
                    || e.is_body()
                    || e.status().is_some_and(|s| s.is_server_error())
            }
            _ => false,
        }
    }
}
