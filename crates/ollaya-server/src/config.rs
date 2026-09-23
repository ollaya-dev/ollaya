//! Daemon settings from the environment (`docs/api.md` §15).

use std::path::PathBuf;
use std::time::Duration;

use ollaya_api::KeepAlive;
use ollaya_api::host::{Host, InvalidHost};

/// The version `/api/version` reports: the release build's, else the crate's.
pub const VERSION: &str = match option_env!("OLLAYA_BUILD_VERSION") {
    Some(v) => v,
    None => env!("CARGO_PKG_VERSION"),
};

#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// `OLLAYA_HOST`: the address to bind.
    pub host: Host,
    /// `OLLAYA_MODELS`: the model store.
    pub models: PathBuf,
    /// `OLLAYA_KEEP_ALIVE`.
    pub keep_alive: KeepAlive,
    /// `OLLAYA_MAX_LOADED_MODELS`.
    pub max_loaded: usize,
    /// `OLLAYA_MAX_QUEUE`: decision requests in flight before `503 QUEUE_FULL`.
    pub max_queue: usize,
    /// `OLLAYA_LOAD_TIMEOUT`.
    pub load_timeout: Duration,
    /// `OLLAYA_DEVICE`: `auto`, `cpu`, `cuda` or `cuda:<n>`, passed to runners.
    pub device: String,
    /// `OLLAYA_API_KEY`: when set, requests need `Authorization: Bearer <key>`.
    pub api_key: Option<String>,
    /// `OLLAYA_ORIGINS`: browser origins allowed on top of the local defaults.
    pub origins: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error(transparent)]
    Host(#[from] InvalidHost),
    #[error("{var}={value:?}: {reason}")]
    Invalid {
        var: &'static str,
        value: String,
        reason: String,
    },
}

pub const DEFAULT_MAX_LOADED: usize = 3;
pub const DEFAULT_LOAD_TIMEOUT: Duration = Duration::from_secs(300);

/// Browser origins always allowed, as Ollama allows them.
pub fn default_origins() -> Vec<String> {
    let mut out = Vec::new();
    for host in ["localhost", "127.0.0.1", "0.0.0.0", "[::1]"] {
        for scheme in ["http", "https"] {
            out.push(format!("{scheme}://{host}"));
            out.push(format!("{scheme}://{host}:*"));
        }
    }
    for scheme in ["app", "file", "tauri", "vscode-webview", "vscode-file"] {
        out.push(format!("{scheme}://*"));
    }
    out
}

impl ServerConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        ServerConfig::from_lookup(|k| std::env::var(k).ok())
    }

    /// Build from any variable lookup (tests pass a map).
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let var = |k: &str| {
            get(k)
                .map(|v| v.trim().to_owned())
                .filter(|v| !v.is_empty())
        };
        let invalid = |var: &'static str, value: &str, reason: String| ConfigError::Invalid {
            var,
            value: value.to_owned(),
            reason,
        };
        let count = |name: &'static str, default: usize| -> Result<usize, ConfigError> {
            match var(name) {
                None => Ok(default),
                Some(v) => v
                    .parse::<usize>()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| invalid(name, &v, "expected a positive integer".into())),
            }
        };
        let host = Host::parse(&var("OLLAYA_HOST").unwrap_or_default())?;
        let models = var("OLLAYA_MODELS")
            .map(PathBuf::from)
            .unwrap_or_else(ollaya_registry::Store::default_root);
        let keep_alive = match var("OLLAYA_KEEP_ALIVE") {
            None => KeepAlive::DEFAULT,
            Some(v) => {
                KeepAlive::parse(&v).map_err(|e| invalid("OLLAYA_KEEP_ALIVE", &v, e.to_string()))?
            }
        };
        let load_timeout = match var("OLLAYA_LOAD_TIMEOUT") {
            None => DEFAULT_LOAD_TIMEOUT,
            Some(v) => match KeepAlive::parse(&v) {
                Ok(KeepAlive::For(d)) if !d.is_zero() => d,
                _ => {
                    return Err(invalid(
                        "OLLAYA_LOAD_TIMEOUT",
                        &v,
                        "expected a positive duration such as \"5m\"".into(),
                    ));
                }
            },
        };
        let device = var("OLLAYA_DEVICE").unwrap_or_else(|| "auto".into());
        let known = matches!(device.as_str(), "auto" | "cpu" | "cuda")
            || device
                .strip_prefix("cuda:")
                .is_some_and(|n| n.parse::<u32>().is_ok());
        if !known {
            return Err(invalid(
                "OLLAYA_DEVICE",
                &device,
                "use auto, cpu, cuda or cuda:<n>".into(),
            ));
        }
        Ok(ServerConfig {
            host,
            models,
            keep_alive,
            max_loaded: count("OLLAYA_MAX_LOADED_MODELS", DEFAULT_MAX_LOADED)?,
            max_queue: count("OLLAYA_MAX_QUEUE", ollaya_api::DEFAULT_MAX_QUEUE)?,
            load_timeout,
            device,
            api_key: var("OLLAYA_API_KEY"),
            origins: var("OLLAYA_ORIGINS")
                .map(|v| {
                    v.split(',')
                        .map(|o| o.trim().to_owned())
                        .filter(|o| !o.is_empty())
                        .collect()
                })
                .unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn with(vars: &[(&str, &str)]) -> Result<ServerConfig, ConfigError> {
        let map: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        ServerConfig::from_lookup(|k| map.get(k).cloned())
    }

    #[test]
    fn defaults() {
        let c = with(&[]).unwrap();
        assert_eq!(c.host.bind_addr(), "127.0.0.1:11435");
        assert_eq!(c.keep_alive, KeepAlive::DEFAULT);
        assert_eq!((c.max_loaded, c.max_queue), (3, 512));
        assert_eq!(c.load_timeout, Duration::from_secs(300));
        assert_eq!(c.device, "auto");
        assert!(c.api_key.is_none() && c.origins.is_empty());
    }

    #[test]
    fn reads_every_variable() {
        let c = with(&[
            ("OLLAYA_HOST", "0.0.0.0:9000"),
            ("OLLAYA_MODELS", "/tmp/m"),
            ("OLLAYA_KEEP_ALIVE", "-1"),
            ("OLLAYA_MAX_LOADED_MODELS", "1"),
            ("OLLAYA_MAX_QUEUE", "8"),
            ("OLLAYA_LOAD_TIMEOUT", "30s"),
            ("OLLAYA_DEVICE", "cuda:1"),
            ("OLLAYA_API_KEY", " secret "),
            (
                "OLLAYA_ORIGINS",
                "https://*.example.com, chrome-extension://*",
            ),
        ])
        .unwrap();
        assert_eq!(c.host.bind_addr(), "0.0.0.0:9000");
        assert_eq!(c.models, PathBuf::from("/tmp/m"));
        assert_eq!(c.keep_alive, KeepAlive::Forever);
        assert_eq!((c.max_loaded, c.max_queue), (1, 8));
        assert_eq!(c.load_timeout, Duration::from_secs(30));
        assert_eq!(c.device, "cuda:1");
        assert_eq!(c.api_key.as_deref(), Some("secret"));
        assert_eq!(c.origins, ["https://*.example.com", "chrome-extension://*"]);
    }

    #[test]
    fn rejects_bad_values() {
        for (k, v) in [
            ("OLLAYA_HOST", "ftp://x"),
            ("OLLAYA_KEEP_ALIVE", "soon"),
            ("OLLAYA_MAX_QUEUE", "0"),
            ("OLLAYA_MAX_LOADED_MODELS", "many"),
            ("OLLAYA_LOAD_TIMEOUT", "0"),
            ("OLLAYA_DEVICE", "tpu"),
        ] {
            assert!(with(&[(k, v)]).is_err(), "{k}={v}");
        }
    }
}
