//! Model names, manifests, the local blob store, and pulling from a registry.

pub mod adapter;
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
    #[error("invalid model name {0:?}; expected [host/][namespace/]model[:tag][@sha256:<hex>]")]
    InvalidName(String),
    #[error("invalid digest {0:?}")]
    InvalidDigest(String),
    #[error("model {0} not found")]
    NotFound(String),
    /// A pull by digest (`name@sha256:<hex>`) that the registry has no manifest for.
    #[error("model {0} not found: the registry has no manifest with that digest")]
    DigestNotFound(String),
    /// A manifest that is not the one a name's digest pins: what the registry served for
    /// `name@sha256:<hex>`, or what was about to be stored under it. Nothing is stored.
    #[error("the manifest for {name} is {got}, not the one its digest names; nothing was stored")]
    ManifestMismatch { name: String, got: String },
    #[error("digest mismatch: expected {expected}, got {got}")]
    DigestMismatch { expected: String, got: String },
    #[error("download stalled: {0}")]
    Stalled(String),
    #[error("corrupt data: {0}")]
    Corrupt(String),
    /// The server answered a byte-range request with `200` instead of `206`, so it cannot serve
    /// parallel ranges (some mirrors, e.g. Artifactory's `/resolve/`, ignore `Range` and return
    /// the whole body). Large blobs fall back to a single stream.
    #[error("{0}: server does not support byte ranges")]
    NoRanges(String),
    /// This client cannot run the model; found from its config, before any layer downloads.
    #[error("{0}")]
    Unsupported(String),
    /// No HTTPS client: the system has no CA certificates (a minimal container without
    /// `ca-certificates`). Only pulls need one.
    #[error(
        "cannot download over HTTPS: {0}. Install the system CA certificates (for example \
         `apt-get install ca-certificates`)"
    )]
    NoHttps(String),
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
                    || e.status() == Some(reqwest::StatusCode::REQUEST_TIMEOUT)
                    || e.status().is_some_and(|s| s.is_server_error())
            }
            _ => false,
        }
    }
}
