//! Model names: `[host/][namespace/]model[:tag][@sha256:<hex>]`, as Ollama spells them.
//!
//! `laya` means `<default registry>/library/laya:latest`. Names compare case-insensitively and
//! display in their shortest unambiguous form.
//!
//! A digest pins a name to one manifest, the sha256 of its bytes. `laya:en@sha256:<hex>` is
//! `laya:en` only while that tag holds that manifest: a pull fetches the manifest by digest and
//! stores it under `en`, and a lookup that finds another manifest there finds nothing. Without a
//! tag, `laya@sha256:<hex>` is stored under the tag `sha256-<hex>`, so `laya:sha256-<hex>` is the
//! same name.

use std::fmt;

use crate::Error;
use crate::store::digest_hex;

pub const DEFAULT_NAMESPACE: &str = "library";
pub const DEFAULT_TAG: &str = "latest";
/// The registry that serves the public model library. Overridable with `OLLAYA_REGISTRY`.
pub const DEFAULT_REGISTRY: &str = "ollaya.dev";
/// Hosts that served the public library before [`DEFAULT_REGISTRY`]. They still serve it, and
/// models pulled from them are moved under the current host (see `Store::migrate_host`).
pub const LEGACY_REGISTRIES: &[&str] = &["ollaya.cobanov.dev"];

/// Whether `OLLAYA_REGISTRY` leaves the default registry in place.
pub fn is_default_registry() -> bool {
    default_registry() == DEFAULT_REGISTRY
}

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
    /// Where the manifest is stored; with only a digest, [`digest_tag`] of it.
    pub tag: String,
    /// `sha256:<hex>`, lowercase: the one manifest this name stands for (`@sha256:<hex>`).
    pub digest: Option<String>,
}

/// The tag a manifest pulled by digest alone is stored under, and served under by the registry:
/// `sha256-<hex>`. Colons are not allowed in file names on Windows, so the digest's own spelling
/// can't be one.
pub fn digest_tag(digest: &str) -> String {
    digest.replacen(':', "-", 1)
}

/// The digest a [`digest_tag`] names: `sha256-<64 hex digits>` -> `sha256:<hex>`.
pub fn tag_digest(tag: &str) -> Option<String> {
    let digest = format!("sha256:{}", tag.strip_prefix("sha256-")?);
    digest_hex(&digest).ok()?;
    Some(digest.to_lowercase())
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

/// `name@sha256:<hex>` -> (`name`, the digest). `Err` for an `@` with anything else after it.
fn split_digest(s: &str) -> Result<(&str, Option<String>), ()> {
    match s.rsplit_once('@') {
        None => Ok((s, None)),
        Some((rest, digest)) => {
            let digest = digest.to_lowercase();
            digest_hex(&digest).map_err(|_| ())?;
            Ok((rest, Some(digest)))
        }
    }
}

impl ModelName {
    pub fn parse(s: &str) -> Result<Self, Error> {
        let s = s.trim();
        let bad = || Error::InvalidName(s.to_owned());
        let (rest, digest) = split_digest(s).map_err(|_| bad())?;
        // A scheme only appears on development registries; keep it with the host.
        let (scheme, rest) = match rest.split_once("://") {
            Some((scheme, rest)) if scheme == "http" || scheme == "https" => (Some(scheme), rest),
            _ => (None, rest),
        };
        let (path, tag) = match rest.rsplit_once(':') {
            // A colon inside the host part (a port) is not a tag separator.
            Some((path, tag)) if !tag.contains('/') => (path, Some(tag)),
            _ => (rest, None),
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
        if !valid_part(&namespace, true)
            || !valid_part(model, true)
            || tag.is_some_and(|t| !valid_part(t, true))
        {
            return Err(bad());
        }
        let (tag, digest) = match (tag.map(str::to_lowercase), digest) {
            (None, None) => (DEFAULT_TAG.to_owned(), None),
            (None, Some(d)) => (digest_tag(&d), Some(d)),
            // `laya:sha256-<hex>` is the name a digest-only pull stores: the same model.
            (Some(t), d) => match (tag_digest(&t), d) {
                (Some(stored), Some(d)) if stored != d => return Err(bad()),
                (Some(stored), _) => (t, Some(stored)),
                (None, d) => (t, d),
            },
        };
        Ok(ModelName {
            host: host.to_lowercase(),
            namespace: namespace.to_lowercase(),
            model: model.to_lowercase(),
            tag,
            digest,
        })
    }

    /// Parse a name that appears inside another model (a router's targets): a name without a
    /// host lives in `base`'s registry, and one without a namespace in `base`'s namespace, so a
    /// router pulled from any registry routes within that registry.
    pub fn parse_relative(s: &str, base: &ModelName) -> Result<Self, Error> {
        let mut name = ModelName::parse(s)?;
        let path = s.trim();
        let path = path.rsplit_once('@').map_or(path, |(p, _)| p);
        let path = path
            .rsplit_once(':')
            .filter(|(_, t)| !t.contains('/'))
            .map_or(path, |(p, _)| p);
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

    /// The manifest's URL: by digest when the name has one (as OCI registries resolve
    /// `/manifests/sha256:<hex>`), else by tag.
    pub fn manifest_url(&self) -> String {
        format!(
            "{}/v2/{}/{}/manifests/{}",
            self.registry_url(),
            self.namespace,
            self.model,
            self.digest.as_deref().unwrap_or(&self.tag)
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

    /// The same model with another tag (and no digest).
    pub fn with_tag(&self, tag: &str) -> Self {
        ModelName {
            tag: tag.to_lowercase(),
            digest: None,
            ..self.clone()
        }
    }

    /// The name of the manifest file this name reads and writes, whatever digest it pins: what
    /// two operations on the same model must not write at once.
    pub fn stored(&self) -> String {
        format!(
            "{}/{}/{}:{}",
            self.host_dir(),
            self.namespace,
            self.model,
            self.tag
        )
    }
}

impl fmt::Display for ModelName {
    /// Shortest form: default registry and namespace are omitted, the tag is always shown, and
    /// a digest follows it (`laya:en@sha256:<hex>`; `laya@sha256:<hex>` when it is the only part).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.host != default_registry() {
            write!(f, "{}/{}/", self.host, self.namespace)?;
        } else if self.namespace != DEFAULT_NAMESPACE {
            write!(f, "{}/", self.namespace)?;
        }
        match &self.digest {
            Some(d) if self.tag == digest_tag(d) => write!(f, "{}@{d}", self.model),
            Some(d) => write!(f, "{}:{}@{d}", self.model, self.tag),
            None => write!(f, "{}:{}", self.model, self.tag),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEX: &str = "4f2a0c6d1e8b3a5f7c9d0e1f2a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d";

    #[test]
    fn parses_and_displays() {
        let n = ModelName::parse("laya").unwrap();
        assert_eq!(
            (n.namespace.as_str(), n.model.as_str(), n.tag.as_str()),
            ("library", "laya", "latest")
        );
        assert_eq!(n.digest, None);
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

    #[test]
    fn a_digest_pins_the_manifest_and_the_tag_says_where_it_is_stored() {
        let digest = format!("sha256:{HEX}");
        let n = ModelName::parse(&format!("laya:en@{digest}")).unwrap();
        assert_eq!(
            (n.model.as_str(), n.tag.as_str(), n.digest.as_deref()),
            ("laya", "en", Some(digest.as_str()))
        );
        assert_eq!(n.to_string(), format!("laya:en@{digest}"));
        assert_eq!(
            n.manifest_url(),
            format!("https://ollaya.dev/v2/library/laya/manifests/{digest}")
        );
        assert_eq!(n.stored(), "ollaya.dev/library/laya:en");
        assert_eq!(n.with_tag("fr").to_string(), "laya:fr");

        // Without a tag: stored under `sha256-<hex>`, which names the same model.
        let n = ModelName::parse(&format!("laya@{digest}")).unwrap();
        assert_eq!(n.tag, format!("sha256-{HEX}"));
        assert_eq!(n.digest.as_deref(), Some(digest.as_str()));
        assert_eq!(n.to_string(), format!("laya@{digest}"));
        assert_eq!(ModelName::parse(&format!("laya:sha256-{HEX}")).unwrap(), n);
        assert_eq!(
            ModelName::parse(&format!("LAYA:SHA256-{}", HEX.to_uppercase())).unwrap(),
            n
        );
        assert_eq!(
            ModelName::parse(&format!("laya:sha256-{HEX}@{digest}")).unwrap(),
            n
        );

        // Host and namespace as usual; the digest's case does not matter.
        let n = ModelName::parse(&format!(
            "http://localhost:8080/acme/laya:en@sha256:{}",
            HEX.to_uppercase()
        ))
        .unwrap();
        assert_eq!(
            (n.host.as_str(), n.namespace.as_str(), n.tag.as_str()),
            ("http://localhost:8080", "acme", "en")
        );
        assert_eq!(n.digest.as_deref(), Some(digest.as_str()));
        assert_eq!(
            n.manifest_url(),
            format!("http://localhost:8080/v2/acme/laya/manifests/{digest}")
        );
        assert_eq!(
            n.to_string(),
            format!("http://localhost:8080/acme/laya:en@{digest}")
        );
        assert_eq!(ModelName::parse(&n.to_string()).unwrap(), n);

        // A router's target may pin a digest too; it still lives in the router's registry.
        let base = ModelName::parse("http://localhost:8080/acme/laya").unwrap();
        let t = ModelName::parse_relative(&format!("laya:en@{digest}"), &base).unwrap();
        assert_eq!(
            (t.host.as_str(), t.namespace.as_str(), t.tag.as_str()),
            ("http://localhost:8080", "acme", "en")
        );
        assert_eq!(t.digest.as_deref(), Some(digest.as_str()));

        let short = &HEX[..63];
        for bad in [
            format!("laya@{HEX}"),
            format!("laya@sha256:{short}"),
            format!("laya@sha256:{short}g"),
            format!("laya@sha512:{HEX}"),
            format!("laya:@{digest}"),
            format!("@{digest}"),
            "laya@".to_owned(),
            format!("laya:en@{digest}@{digest}"),
            // Two different digests.
            format!("laya:sha256-{}@{digest}", "0".repeat(64)),
        ] {
            assert!(ModelName::parse(&bad).is_err(), "{bad:?}");
        }
        // `sha256-` with anything but 64 hex digits is an ordinary tag.
        let n = ModelName::parse("laya:sha256-abc").unwrap();
        assert_eq!((n.tag.as_str(), n.digest), ("sha256-abc", None));
    }
}
