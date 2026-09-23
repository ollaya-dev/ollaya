//! Model names: `[host/][namespace/]model[:tag]`, as Ollama spells them.
//!
//! `laya` means `<default registry>/library/laya:latest`. Names compare case-insensitively and
//! display in their shortest unambiguous form.

use std::fmt;

use crate::Error;

pub const DEFAULT_NAMESPACE: &str = "library";
pub const DEFAULT_TAG: &str = "latest";
/// The registry that serves the public model library. Overridable with `OLLAYA_REGISTRY`.
pub const DEFAULT_REGISTRY: &str = "ollaya.cobanov.dev";

pub fn default_registry() -> String {
    std::env::var("OLLAYA_REGISTRY")
        .ok()
        .map(|s| s.trim().trim_end_matches('/').to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_REGISTRY.to_owned())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelName {
    /// Registry host, possibly with a scheme (`http://localhost:8080`) for development registries.
    pub host: String,
    pub namespace: String,
    pub model: String,
    pub tag: String,
}

fn valid_part(s: &str, allow_dot_dash: bool) -> bool {
    !s.is_empty()
        && s.len() <= 80
        && s.chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
        && s.chars().all(|c| {
            c.is_ascii_alphanumeric() || c == '_' || (allow_dot_dash && (c == '-' || c == '.'))
        })
}

impl ModelName {
    pub fn parse(s: &str) -> Result<Self, Error> {
        let s = s.trim();
        let bad = || Error::InvalidName(s.to_owned());
        // A scheme only appears on development registries; keep it with the host.
        let (scheme, rest) = match s.split_once("://") {
            Some((scheme, rest)) if scheme == "http" || scheme == "https" => (Some(scheme), rest),
            _ => (None, s),
        };
        let (path, tag) = match rest.rsplit_once(':') {
            // A colon inside the host part (a port) is not a tag separator.
            Some((path, tag)) if !tag.contains('/') => (path, tag.to_owned()),
            _ => (rest, DEFAULT_TAG.to_owned()),
        };
        let parts: Vec<&str> = path.split('/').collect();
        let (host, namespace, model) = match parts.as_slice() {
            [model] => (default_registry(), DEFAULT_NAMESPACE.to_owned(), *model),
            [namespace, model] => (default_registry(), (*namespace).to_owned(), *model),
            [host, namespace, model] => ((*host).to_owned(), (*namespace).to_owned(), *model),
            _ => return Err(bad()),
        };
        let host = match scheme {
            Some(scheme) => format!("{scheme}://{host}"),
            None => host,
        };
        if !valid_part(&namespace, true) || !valid_part(model, true) || !valid_part(&tag, true) {
            return Err(bad());
        }
        Ok(ModelName {
            host: host.to_lowercase(),
            namespace: namespace.to_lowercase(),
            model: model.to_lowercase(),
            tag: tag.to_lowercase(),
        })
    }

    /// Parse a name that appears inside another model (a router's targets): a name without a
    /// host lives in `base`'s registry, and one without a namespace in `base`'s namespace, so a
    /// router pulled from any registry routes within that registry.
    pub fn parse_relative(s: &str, base: &ModelName) -> Result<Self, Error> {
        let mut name = ModelName::parse(s)?;
        let path = s
            .trim()
            .rsplit_once(':')
            .filter(|(_, t)| !t.contains('/'))
            .map_or(s.trim(), |(p, _)| p);
        match path.matches('/').count() {
            0 => {
                name.host = base.host.clone();
                name.namespace = base.namespace.clone();
            }
            1 if !path.contains("://") => name.host = base.host.clone(),
            _ => {}
        }
        Ok(name)
    }

    /// Host without scheme, as used in the on-disk manifest path.
    pub fn host_dir(&self) -> String {
        self.host
            .split_once("://")
            .map_or(self.host.clone(), |(_, h)| h.to_owned())
            .replace(':', "_")
    }

    /// Base URL of the registry (`https://host` unless a scheme was given).
    pub fn registry_url(&self) -> String {
        if self.host.contains("://") {
            self.host.clone()
        } else {
            format!("https://{}", self.host)
        }
    }

    pub fn manifest_url(&self) -> String {
        format!(
            "{}/v2/{}/{}/manifests/{}",
            self.registry_url(),
            self.namespace,
            self.model,
            self.tag
        )
    }

    pub fn blob_url(&self, digest: &str) -> String {
        format!(
            "{}/v2/{}/{}/blobs/{}",
            self.registry_url(),
            self.namespace,
            self.model,
            digest
        )
    }

    /// The same model with another tag.
    pub fn with_tag(&self, tag: &str) -> Self {
        ModelName {
            tag: tag.to_lowercase(),
            ..self.clone()
        }
    }
}

impl fmt::Display for ModelName {
    /// Shortest form: default registry and namespace are omitted, the tag is always shown.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.host != default_registry() {
            write!(f, "{}/{}/", self.host, self.namespace)?;
        } else if self.namespace != DEFAULT_NAMESPACE {
            write!(f, "{}/", self.namespace)?;
        }
        write!(f, "{}:{}", self.model, self.tag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_displays() {
        let n = ModelName::parse("laya").unwrap();
        assert_eq!(
            (n.namespace.as_str(), n.model.as_str(), n.tag.as_str()),
            ("library", "laya", "latest")
        );
        assert_eq!(n.to_string(), "laya:latest");
        assert_eq!(ModelName::parse("Laya:EN").unwrap().to_string(), "laya:en");
        assert_eq!(
            ModelName::parse("acme/triage:v2").unwrap().to_string(),
            "acme/triage:v2"
        );
        let dev = ModelName::parse("http://localhost:8080/library/laya:en-fp32").unwrap();
        assert_eq!(dev.registry_url(), "http://localhost:8080");
        assert_eq!(dev.host_dir(), "localhost_8080");
        assert_eq!(
            dev.manifest_url(),
            "http://localhost:8080/v2/library/laya/manifests/en-fp32"
        );
        let base = ModelName::parse("http://localhost:8080/acme/laya").unwrap();
        let t = ModelName::parse_relative("laya:en", &base).unwrap();
        assert_eq!(
            (t.host.as_str(), t.namespace.as_str(), t.tag.as_str()),
            ("http://localhost:8080", "acme", "en")
        );
        let t = ModelName::parse_relative("other/m", &base).unwrap();
        assert_eq!(
            (t.host.as_str(), t.namespace.as_str()),
            ("http://localhost:8080", "other")
        );
        let t = ModelName::parse_relative("reg.example/ns/m", &base).unwrap();
        assert_eq!(t.host, "reg.example");
        for bad in ["", "a/b/c/d", "-x", "x:", "x:-y", "laya:a/b"] {
            assert!(ModelName::parse(bad).is_err(), "{bad:?}");
        }
    }
}
