//! The local model store, laid out like Ollama's:
//!
//! ```text
//! <root>/manifests/<host>/<namespace>/<model>/<tag>   manifest JSON
//! <root>/blobs/sha256-<hex>                           content-addressed blobs
//! ```
//!
//! Blobs are shared between models and removed only when no manifest references them.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use sha2::{Digest, Sha256};

use crate::Error;
use crate::manifest::{Descriptor, Manifest};
use crate::name::{DEFAULT_NAMESPACE, ModelName};

#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

/// A locally available model.
#[derive(Debug, Clone)]
pub struct Entry {
    pub name: ModelName,
    pub manifest: Manifest,
    /// sha256 of the manifest bytes: the model's identity.
    pub digest: String,
    pub modified: SystemTime,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// `sha256:<hex>` -> `<hex>`, rejecting anything that is not a sha256 digest.
pub fn digest_hex(digest: &str) -> Result<&str, Error> {
    match digest.strip_prefix("sha256:") {
        Some(h) if h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()) => Ok(h),
        _ => Err(Error::InvalidDigest(digest.to_owned())),
    }
}

impl Store {
    /// `$OLLAYA_MODELS`, else `~/.ollaya/models`.
    pub fn default_root() -> PathBuf {
        match std::env::var_os("OLLAYA_MODELS") {
            Some(p) if !p.is_empty() => PathBuf::from(p),
            _ => dirs_home().join(".ollaya").join("models"),
        }
    }

    pub fn open(root: impl Into<PathBuf>) -> Result<Self, Error> {
        let root = root.into();
        std::fs::create_dir_all(root.join("manifests"))?;
        std::fs::create_dir_all(root.join("blobs"))?;
        Ok(Store { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs")
    }

    /// File name of a blob inside `blobs/` (`sha256-<hex>`). Graphs reference their weights by it.
    pub fn blob_file_name(digest: &str) -> Result<String, Error> {
        Ok(format!("sha256-{}", digest_hex(digest)?))
    }

    pub fn blob_path(&self, digest: &str) -> Result<PathBuf, Error> {
        Ok(self.blobs_dir().join(Self::blob_file_name(digest)?))
    }

    /// Present with the expected size (content was verified when it was written).
    pub fn has_blob(&self, d: &Descriptor) -> bool {
        self.blob_path(&d.digest)
            .ok()
            .and_then(|p| std::fs::metadata(p).ok())
            .is_some_and(|m| m.len() == d.size)
    }

    pub fn read_blob(&self, d: &Descriptor) -> Result<Vec<u8>, Error> {
        Ok(std::fs::read(self.blob_path(&d.digest)?)?)
    }

    pub fn read_blob_json<T: serde::de::DeserializeOwned>(
        &self,
        d: &Descriptor,
    ) -> Result<T, Error> {
        serde_json::from_slice(&self.read_blob(d)?)
            .map_err(|e| Error::Corrupt(format!("{}: {e}", d.digest)))
    }

    /// Store bytes as a blob and return their digest.
    pub fn write_blob(&self, bytes: &[u8]) -> Result<String, Error> {
        let digest = format!("sha256:{}", sha256_hex(bytes));
        let path = self.blob_path(&digest)?;
        if !path.exists() {
            write_atomic(&path, bytes)?;
        }
        Ok(digest)
    }

    pub fn manifest_path(&self, name: &ModelName) -> PathBuf {
        self.root
            .join("manifests")
            .join(name.host_dir())
            .join(&name.namespace)
            .join(&name.model)
            .join(&name.tag)
    }

    pub fn read_manifest(&self, name: &ModelName) -> Result<Option<Entry>, Error> {
        let path = self.manifest_path(name);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let manifest = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Corrupt(format!("{}: {e}", path.display())))?;
        Ok(Some(Entry {
            name: name.clone(),
            manifest,
            digest: format!("sha256:{}", sha256_hex(&bytes)),
            modified: std::fs::metadata(&path)?.modified()?,
        }))
    }

    pub fn write_manifest(&self, name: &ModelName, bytes: &[u8]) -> Result<(), Error> {
        let path = self.manifest_path(name);
        std::fs::create_dir_all(path.parent().expect("manifest path has a parent"))?;
        write_atomic(&path, bytes)
    }

    /// Every local model, sorted by name.
    pub fn list(&self) -> Result<Vec<Entry>, Error> {
        let mut out = Vec::new();
        let base = self.root.join("manifests");
        for host in read_dirs(&base)? {
            for ns in read_dirs(&host)? {
                for model in read_dirs(&ns)? {
                    for tag in std::fs::read_dir(&model)? {
                        let tag = tag?.path();
                        if !tag.is_file() {
                            continue;
                        }
                        let name = ModelName {
                            host: file_name(&host).replace('_', ":"),
                            namespace: file_name(&ns),
                            model: file_name(&model),
                            tag: file_name(&tag),
                        };
                        let name = normalise_host(name);
                        if let Some(entry) = self.read_manifest(&name)? {
                            out.push(entry);
                        }
                    }
                }
            }
        }
        out.sort_by_key(|e| e.name.to_string());
        Ok(out)
    }

    /// Remove a model's manifest, then any blobs nothing references anymore.
    pub fn remove(&self, name: &ModelName) -> Result<bool, Error> {
        let path = self.manifest_path(name);
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e.into()),
        }
        // Drop now-empty directories so `list` stays clean.
        let mut dir = path.parent().map(Path::to_path_buf);
        while let Some(d) = dir {
            if d == self.root.join("manifests") || std::fs::remove_dir(&d).is_err() {
                break;
            }
            dir = d.parent().map(Path::to_path_buf);
        }
        self.prune()?;
        Ok(true)
    }

    /// Moves the models pulled from registry host `from` to host `to`, so that models pulled
    /// before the public library moved keep their short names. A model that already exists under
    /// `to` is kept and its old copy is dropped. Returns how many manifests moved.
    pub fn migrate_host(&self, from: &str, to: &str) -> Result<usize, Error> {
        let manifests = self.root.join("manifests");
        let (from, to) = (manifests.join(from), manifests.join(to));
        let mut moved = 0;
        for ns in read_dirs(&from)? {
            for model in read_dirs(&ns)? {
                for tag in std::fs::read_dir(&model)? {
                    let tag = tag?.path();
                    if !tag.is_file() {
                        continue;
                    }
                    let dest = to
                        .join(file_name(&ns))
                        .join(file_name(&model))
                        .join(file_name(&tag));
                    if dest.exists() {
                        std::fs::remove_file(&tag)?;
                    } else {
                        std::fs::create_dir_all(dest.parent().expect("dest has a parent"))?;
                        std::fs::rename(&tag, &dest)?;
                        moved += 1;
                    }
                }
                let _ = std::fs::remove_dir(&model);
            }
            let _ = std::fs::remove_dir(&ns);
        }
        let _ = std::fs::remove_dir(&from);
        Ok(moved)
    }

    pub fn copy(&self, src: &ModelName, dst: &ModelName) -> Result<(), Error> {
        let bytes =
            std::fs::read(self.manifest_path(src)).map_err(|_| Error::NotFound(src.to_string()))?;
        self.write_manifest(dst, &bytes)
    }

    /// Delete blobs no manifest references. Partial downloads are kept so pulls can resume.
    pub fn prune(&self) -> Result<u64, Error> {
        let mut keep = HashSet::new();
        for entry in self.list()? {
            for d in entry.manifest.blobs() {
                keep.insert(Self::blob_file_name(&d.digest)?);
            }
        }
        let mut freed = 0;
        for f in std::fs::read_dir(self.blobs_dir())? {
            let f = f?;
            let name = f.file_name().to_string_lossy().into_owned();
            if name.starts_with("sha256-") && !name.contains("-partial") && !keep.contains(&name) {
                freed += f.metadata()?.len();
                std::fs::remove_file(f.path())?;
            }
        }
        Ok(freed)
    }
}

/// A `localhost_8080`-style directory came back as `localhost:8080`; the default registry has no port.
fn normalise_host(mut name: ModelName) -> ModelName {
    if name.namespace.is_empty() {
        name.namespace = DEFAULT_NAMESPACE.to_owned();
    }
    name
}

fn read_dirs(dir: &Path) -> Result<Vec<PathBuf>, Error> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd {
            let p = e?.path();
            if p.is_dir() {
                out.push(p);
            }
        }
    }
    Ok(out)
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// `$HOME` when set (as on every Unix, and in tests), else the platform's home directory
/// (`%USERPROFILE%` on Windows).
fn dirs_home() -> PathBuf {
    std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .map(PathBuf::from)
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Write to a temporary sibling, then rename: readers never see a half-written file.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{MANIFEST_V2, media};

    fn manifest(store: &Store, payload: &[u8]) -> Vec<u8> {
        let config = store.write_blob(b"{}").unwrap();
        let layer = store.write_blob(payload).unwrap();
        let m = Manifest {
            schema_version: 2,
            media_type: MANIFEST_V2.into(),
            config: Descriptor {
                media_type: media::CONFIG.into(),
                digest: config,
                size: 2,
                urls: vec![],
                annotations: Default::default(),
            },
            layers: vec![Descriptor {
                media_type: media::LICENSE.into(),
                digest: layer,
                size: payload.len() as u64,
                urls: vec![],
                annotations: Default::default(),
            }],
        };
        serde_json::to_vec(&m).unwrap()
    }

    #[test]
    fn migrate_host_keeps_short_names() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let old = |m: &str| ModelName::parse(&format!("old.example/library/{m}")).unwrap();
        let new = |m: &str| ModelName::parse(&format!("new.example/library/{m}")).unwrap();
        store
            .write_manifest(&old("laya:en"), &manifest(&store, b"en"))
            .unwrap();
        store
            .write_manifest(&old("laya:latest"), &manifest(&store, b"old"))
            .unwrap();
        store
            .write_manifest(&new("laya:latest"), &manifest(&store, b"newer"))
            .unwrap();

        assert_eq!(store.migrate_host("old.example", "new.example").unwrap(), 1);
        let names: Vec<_> = store.list().unwrap().into_iter().map(|e| e.name).collect();
        assert_eq!(names, vec![new("laya:en"), new("laya:latest")]);
        // The model that already existed under the new host is the one kept.
        let kept = store.read_manifest(&new("laya:latest")).unwrap().unwrap();
        assert_eq!(kept.manifest.layers[0].size, 5);
        assert!(!dir.path().join("manifests/old.example").exists());
        // Nothing left to move.
        assert_eq!(store.migrate_host("old.example", "new.example").unwrap(), 0);
    }

    #[test]
    fn write_list_copy_remove_prune() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let a = ModelName::parse("laya:en").unwrap();
        let b = ModelName::parse("acme/triage").unwrap();
        store
            .write_manifest(&a, &manifest(&store, b"shared"))
            .unwrap();
        store.copy(&a, &b).unwrap();
        let names: Vec<_> = store
            .list()
            .unwrap()
            .into_iter()
            .map(|e| e.name.to_string())
            .collect();
        assert_eq!(names, ["acme/triage:latest", "laya:en"]);

        // The layer is shared: removing one model keeps it for the other.
        assert!(store.remove(&a).unwrap());
        let entry = store.read_manifest(&b).unwrap().unwrap();
        assert!(store.has_blob(&entry.manifest.layers[0]));
        assert!(store.remove(&b).unwrap());
        assert_eq!(std::fs::read_dir(store.blobs_dir()).unwrap().count(), 0);
        assert!(!store.remove(&b).unwrap());
    }
}
