//! Pulling models: manifest from the registry, blobs from wherever their descriptors point.
//!
//! Large blobs download as parallel HTTP range requests into a preallocated
//! `sha256-<hex>-partial` file. Per-part progress is saved next to it, so an interrupted pull
//! resumes where it stopped. Every blob is verified against its sha256 before it is renamed into
//! place, and the manifest is written last: a model is either fully present or not listed.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use crate::Error;
use crate::manifest::{Descriptor, Manifest, ModelConfig, media};
use crate::name::ModelName;
use crate::store::{Store, digest_hex, sha256_hex};

/// Blobs up to this size download as one stream.
const SINGLE_STREAM_MAX: u64 = 32 << 20;
/// Target size of one range part; the part count is capped by `MAX_PARTS`.
const PART_SIZE: u64 = 64 << 20;
const MAX_PARTS: u64 = 16;
/// Parts downloading at the same time.
const CONCURRENCY: usize = 8;
const MAX_RETRIES: u32 = 6;

/// One pull status update, in Ollama's NDJSON progress shape.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Progress {
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}

impl Progress {
    pub fn status(s: impl Into<String>) -> Self {
        Progress {
            status: s.into(),
            ..Default::default()
        }
    }
}

/// Whether this client can run a model, judged from its config blob: `Err` says why not. A pull
/// runs it before downloading any layer, so nothing large is fetched for a model that cannot run.
pub type RunCheck = Arc<dyn Fn(&ModelName, &ModelConfig) -> Result<(), String> + Send + Sync>;

#[derive(Clone)]
pub struct Puller {
    http: reqwest::Client,
    store: Store,
    check: Option<RunCheck>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PartState {
    offset: u64,
    size: u64,
    completed: u64,
}

impl Puller {
    pub fn new(store: Store) -> Result<Self, Error> {
        let http = reqwest::Client::builder()
            .user_agent(concat!("ollaya/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(20))
            .read_timeout(Duration::from_secs(60))
            .build()?;
        Ok(Puller {
            http,
            store,
            check: None,
        })
    }

    /// Refuse, before downloading its layers, any model `check` rejects.
    pub fn with_check(mut self, check: RunCheck) -> Self {
        self.check = Some(check);
        self
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Fetch and parse a manifest without storing anything.
    pub async fn fetch_manifest(&self, name: &ModelName) -> Result<(Manifest, Vec<u8>), Error> {
        let resp = self.http.get(name.manifest_url()).send().await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(Error::NotFound(name.to_string()));
        }
        let bytes = resp.error_for_status()?.bytes().await?.to_vec();
        let manifest: Manifest = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Corrupt(format!("manifest for {name}: {e}")))?;
        for d in manifest.blobs() {
            digest_hex(&d.digest)?; // reject malformed digests before any download
        }
        Ok((manifest, bytes))
    }

    /// Pull a model and, for routers, every model it routes to. Reports `success` once, at the end.
    pub async fn pull(
        &self,
        name: &ModelName,
        progress: &(dyn Fn(Progress) + Send + Sync),
    ) -> Result<Manifest, Error> {
        let manifest = self.pull_one(name, progress).await?;
        progress(Progress::status("success"));
        Ok(manifest)
    }

    async fn pull_one(
        &self,
        name: &ModelName,
        progress: &(dyn Fn(Progress) + Send + Sync),
    ) -> Result<Manifest, Error> {
        progress(Progress::status("pulling manifest"));
        let (manifest, bytes) = self.fetch_manifest(name).await?;
        if let Some(check) = &self.check {
            // The config is small: fetch it first and ask whether the model can run here.
            let c = &manifest.config;
            if !self.store.has_blob(c) {
                self.download(name, c, progress).await?;
            }
            let config: ModelConfig = self.store.read_blob_json(c)?;
            check(name, &config).map_err(Error::Unsupported)?;
        }
        for d in manifest.blobs() {
            if self.store.has_blob(d) {
                progress(Progress {
                    status: format!("pulling {}", short(&d.digest)),
                    digest: Some(d.digest.clone()),
                    total: Some(d.size),
                    completed: Some(d.size),
                });
                continue;
            }
            self.download(name, d, progress).await?;
        }
        progress(Progress::status("verifying sha256 digest"));
        if let Some(router) = manifest.layer(media::ROUTER) {
            let router: crate::manifest::Router = self.store.read_blob_json(router)?;
            for target in router.routes.values() {
                let target = ModelName::parse_relative(target, name)?;
                Box::pin(self.pull_one(&target, progress)).await?;
            }
        }
        // Written last: a model is listed only once every blob it needs is present.
        progress(Progress::status("writing manifest"));
        self.store.write_manifest(name, &bytes)?;
        Ok(manifest)
    }

    fn blob_source(&self, name: &ModelName, d: &Descriptor) -> String {
        d.urls
            .first()
            .cloned()
            .unwrap_or_else(|| name.blob_url(&d.digest))
    }

    async fn download(
        &self,
        name: &ModelName,
        d: &Descriptor,
        progress: &(dyn Fn(Progress) + Send + Sync),
    ) -> Result<(), Error> {
        let url = self.blob_source(name, d);
        let final_path = self.store.blob_path(&d.digest)?;
        let partial = PathBuf::from(format!("{}-partial", final_path.display()));
        let state_path = PathBuf::from(format!("{}-partial.json", final_path.display()));
        let report = |completed: u64| {
            progress(Progress {
                status: format!("pulling {}", short(&d.digest)),
                digest: Some(d.digest.clone()),
                total: Some(d.size),
                completed: Some(completed),
            })
        };
        report(0);

        let parts = if d.size <= SINGLE_STREAM_MAX {
            self.fetch_whole(&url, &partial, d.size, &report).await?;
            None
        } else {
            Some(
                self.fetch_ranges(&url, &partial, &state_path, d.size, &report)
                    .await?,
            )
        };

        let got = hash_file(&partial).await?;
        if got != digest_hex(&d.digest)? {
            let _ = tokio::fs::remove_file(&partial).await;
            let _ = tokio::fs::remove_file(&state_path).await;
            return Err(Error::DigestMismatch {
                expected: d.digest.clone(),
                got: format!("sha256:{got}"),
            });
        }
        tokio::fs::rename(&partial, &final_path).await?;
        if parts.is_some() {
            let _ = tokio::fs::remove_file(&state_path).await;
        }
        report(d.size);
        Ok(())
    }

    async fn fetch_whole(
        &self,
        url: &str,
        path: &Path,
        size: u64,
        report: &(dyn Fn(u64) + Send + Sync),
    ) -> Result<(), Error> {
        let mut attempt = 0;
        loop {
            match self.try_fetch_whole(url, path, report).await {
                Ok(n) if n == size => return Ok(()),
                Ok(n) => {
                    return Err(Error::Corrupt(format!(
                        "{url}: got {n} bytes, manifest says {size}"
                    )));
                }
                Err(e) if attempt < MAX_RETRIES && e.is_retryable() => {
                    attempt += 1;
                    tokio::time::sleep(backoff(attempt)).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    async fn try_fetch_whole(
        &self,
        url: &str,
        path: &Path,
        report: &(dyn Fn(u64) + Send + Sync),
    ) -> Result<u64, Error> {
        let resp = self.http.get(url).send().await?.error_for_status()?;
        let mut file = tokio::fs::File::create(path).await?;
        let mut stream = resp.bytes_stream();
        let mut n = 0u64;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            n += chunk.len() as u64;
            report(n);
        }
        file.flush().await?;
        Ok(n)
    }

    async fn fetch_ranges(
        &self,
        url: &str,
        path: &Path,
        state_path: &Path,
        size: u64,
        report: &(dyn Fn(u64) + Send + Sync),
    ) -> Result<Vec<PartState>, Error> {
        // Resume from a previous attempt when its plan still matches the blob.
        let mut parts: Vec<PartState> = match tokio::fs::read(state_path).await {
            Ok(b) if path.exists() => serde_json::from_slice(&b).unwrap_or_default(),
            _ => Vec::new(),
        };
        if parts.iter().map(|p| p.size).sum::<u64>() != size {
            let n = size.div_ceil(PART_SIZE).clamp(1, MAX_PARTS);
            let part = size.div_ceil(n);
            parts = (0..n)
                .map(|i| {
                    let offset = i * part;
                    PartState {
                        offset,
                        size: part.min(size - offset),
                        completed: 0,
                    }
                })
                .collect();
            let file = tokio::fs::File::create(path).await?;
            file.set_len(size).await?;
        }

        let counters: Arc<Vec<AtomicU64>> =
            Arc::new(parts.iter().map(|p| AtomicU64::new(p.completed)).collect());
        let total = || {
            counters
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .sum::<u64>()
        };
        report(total());

        let pending: Vec<(usize, u64, u64)> = parts
            .iter()
            .enumerate()
            .filter(|(_, p)| p.completed < p.size)
            .map(|(i, p)| (i, p.offset, p.size))
            .collect();
        let jobs = pending.into_iter().map(|(i, offset, psize)| {
            let counters = counters.clone();
            let http = self.http.clone();
            let (url, path) = (url.to_owned(), path.to_owned());
            async move { fetch_part(&http, &url, &path, offset, psize, &counters[i]).await }
        });
        let mut running = futures_util::stream::iter(jobs).buffer_unordered(CONCURRENCY);

        let mut ticker = tokio::time::interval(Duration::from_millis(250));
        let result = loop {
            tokio::select! {
                next = running.next() => match next {
                    Some(Ok(())) => continue,
                    Some(Err(e)) => break Err(e),
                    None => break Ok(()),
                },
                _ = ticker.tick() => {
                    report(total());
                    save_state(state_path, &parts, &counters).await;
                }
            }
        };
        drop(running);
        save_state(state_path, &parts, &counters).await;
        result?;
        for (p, c) in parts.iter_mut().zip(counters.iter()) {
            p.completed = c.load(Ordering::Relaxed);
        }
        Ok(parts)
    }
}

async fn save_state(path: &Path, parts: &[PartState], counters: &[AtomicU64]) {
    let snapshot: Vec<PartState> = parts
        .iter()
        .zip(counters)
        .map(|(p, c)| PartState {
            offset: p.offset,
            size: p.size,
            completed: c.load(Ordering::Relaxed),
        })
        .collect();
    if let Ok(bytes) = serde_json::to_vec(&snapshot) {
        let _ = tokio::fs::write(path, bytes).await;
    }
}

/// Download one byte range, resuming from `done` and retrying with backoff.
async fn fetch_part(
    http: &reqwest::Client,
    url: &str,
    path: &Path,
    offset: u64,
    size: u64,
    done: &AtomicU64,
) -> Result<(), Error> {
    let mut attempt = 0;
    loop {
        let start = offset + done.load(Ordering::Relaxed);
        let end = offset + size - 1;
        if start > end {
            return Ok(());
        }
        let result: Result<(), Error> = async {
            let resp = http
                .get(url)
                .header(reqwest::header::RANGE, format!("bytes={start}-{end}"))
                .send()
                .await?
                .error_for_status()?;
            if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
                return Err(Error::Corrupt(format!(
                    "{url}: server ignored the range request"
                )));
            }
            let mut file = tokio::fs::OpenOptions::new().write(true).open(path).await?;
            file.seek(std::io::SeekFrom::Start(start)).await?;
            let mut stream = resp.bytes_stream();
            // A stalled part is restarted rather than waited on forever.
            while let Some(chunk) = tokio::time::timeout(Duration::from_secs(30), stream.next())
                .await
                .map_err(|_| Error::Stalled(url.to_owned()))?
            {
                let chunk = chunk?;
                let room = (end + 1 - (offset + done.load(Ordering::Relaxed))) as usize;
                let chunk = &chunk[..chunk.len().min(room)];
                file.write_all(chunk).await?;
                done.fetch_add(chunk.len() as u64, Ordering::Relaxed);
            }
            file.flush().await?;
            Ok(())
        }
        .await;
        match result {
            Ok(()) if done.load(Ordering::Relaxed) >= size => return Ok(()),
            Ok(()) => {} // connection closed early: resume from where it stopped
            Err(e) if attempt < MAX_RETRIES && e.is_retryable() => {}
            Err(e) => return Err(e),
        }
        attempt += 1;
        if attempt > MAX_RETRIES {
            return Err(Error::Stalled(url.to_owned()));
        }
        tokio::time::sleep(backoff(attempt)).await;
    }
}

fn backoff(attempt: u32) -> Duration {
    Duration::from_millis(500 * 2u64.pow(attempt.min(6)))
}

async fn hash_file(path: &Path) -> Result<String, Error> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 4 << 20];
    loop {
        let n = file.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// First 12 hex characters, as Ollama shows digests.
pub fn short(digest: &str) -> String {
    digest
        .trim_start_matches("sha256:")
        .chars()
        .take(12)
        .collect()
}

/// Verify a local file against a digest (used by `create` and tests).
pub fn verify_bytes(bytes: &[u8], digest: &str) -> Result<(), Error> {
    if sha256_hex(bytes) == digest_hex(digest)? {
        Ok(())
    } else {
        Err(Error::DigestMismatch {
            expected: digest.to_owned(),
            got: format!("sha256:{}", sha256_hex(bytes)),
        })
    }
}
