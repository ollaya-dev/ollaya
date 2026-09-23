//! Pull against a real HTTP server: a static registry plus a range-capable blob host.

use std::sync::Mutex;

use ollaya_registry::manifest::{Descriptor, MANIFEST_V2, Manifest, media};
use ollaya_registry::store::sha256_hex;
use ollaya_registry::{Error, ModelName, Progress, Puller, Store};

/// A static registry served from a temp dir, the way the website serves `/v2/...`.
async fn serve(dir: &std::path::Path) -> String {
    let app = axum::Router::new().fallback_service(tower_http::services::ServeDir::new(dir));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

fn descriptor(media_type: &str, bytes: &[u8], url: String) -> Descriptor {
    Descriptor {
        media_type: media_type.into(),
        digest: format!("sha256:{}", sha256_hex(bytes)),
        size: bytes.len() as u64,
        urls: vec![url],
        annotations: Default::default(),
    }
}

/// Publish `weights` + a config under `library/test:<tag>`; weights are hosted "elsewhere" (a
/// different path, like a Hugging Face repo), referenced through `urls`.
fn publish(
    root: &std::path::Path,
    base: &str,
    tag: &str,
    weights: &[u8],
    advertised: Option<&[u8]>,
) {
    std::fs::create_dir_all(root.join("hf")).unwrap();
    std::fs::write(root.join("hf/model.safetensors"), weights).unwrap();
    let config = br#"{"model_format":"onnx","family":"test"}"#;
    std::fs::write(root.join("hf/config.json"), config).unwrap();
    let manifest = Manifest {
        schema_version: 2,
        media_type: MANIFEST_V2.into(),
        config: descriptor(media::CONFIG, config, format!("{base}/hf/config.json")),
        layers: vec![descriptor(
            media::WEIGHTS,
            advertised.unwrap_or(weights),
            format!("{base}/hf/model.safetensors"),
        )],
    };
    let dir = root.join("v2/library/test/manifests");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(tag), serde_json::to_vec(&manifest).unwrap()).unwrap();
}

fn pseudo_random(n: usize) -> Vec<u8> {
    let mut x: u64 = 0x9e3779b97f4a7c15;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

#[tokio::test]
async fn pulls_ranged_blob_and_resumes() {
    let remote = tempfile::tempdir().unwrap();
    let base = serve(remote.path()).await;
    let weights = pseudo_random(40 << 20); // above the single-stream limit: parallel ranges
    publish(remote.path(), &base, "latest", &weights, None);

    let local = tempfile::tempdir().unwrap();
    let store = Store::open(local.path()).unwrap();
    let puller = Puller::new(store.clone()).unwrap();
    let name = ModelName::parse(&format!("{base}/library/test")).unwrap();

    // Simulate an interrupted earlier pull: half of the first part already on disk.
    let digest = format!("sha256:{}", sha256_hex(&weights));
    let blob = store.blob_path(&digest).unwrap();
    let partial = format!("{}-partial", blob.display());
    let mut pre = vec![0u8; weights.len()];
    pre[..1 << 20].copy_from_slice(&weights[..1 << 20]);
    std::fs::write(&partial, &pre).unwrap();
    let part = (weights.len() as u64).div_ceil(1);
    std::fs::write(
        format!("{partial}.json"),
        format!(r#"[{{"offset":0,"size":{part},"completed":{}}}]"#, 1 << 20),
    )
    .unwrap();

    let seen = Mutex::new(Vec::<Progress>::new());
    let manifest = puller
        .pull(&name, &|p| seen.lock().unwrap().push(p))
        .await
        .unwrap();

    assert_eq!(std::fs::read(&blob).unwrap(), weights);
    assert!(!std::path::Path::new(&partial).exists());
    let entry = store.read_manifest(&name).unwrap().unwrap();
    assert_eq!(entry.manifest, manifest);
    let statuses: Vec<String> = seen
        .lock()
        .unwrap()
        .iter()
        .map(|p| p.status.clone())
        .collect();
    assert_eq!(
        statuses.first().map(String::as_str),
        Some("pulling manifest")
    );
    assert_eq!(statuses.last().map(String::as_str), Some("success"));
    let last_blob = seen
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|p| p.digest.as_deref() == Some(&digest))
        .cloned()
        .unwrap();
    assert_eq!(last_blob.completed, Some(weights.len() as u64));

    // Pulling again downloads nothing.
    puller.pull(&name, &|_| {}).await.unwrap();
}

#[tokio::test]
async fn rejects_digest_mismatch_and_writes_no_manifest() {
    let remote = tempfile::tempdir().unwrap();
    let base = serve(remote.path()).await;
    let weights = pseudo_random(1 << 20);
    let mut tampered = weights.clone();
    tampered[0] ^= 1;
    publish(remote.path(), &base, "bad", &tampered, Some(&weights));

    let local = tempfile::tempdir().unwrap();
    let store = Store::open(local.path()).unwrap();
    let puller = Puller::new(store.clone()).unwrap();
    let name = ModelName::parse(&format!("{base}/library/test:bad")).unwrap();
    let err = puller.pull(&name, &|_| {}).await.unwrap_err();
    assert!(matches!(err, Error::DigestMismatch { .. }), "{err}");
    assert!(store.read_manifest(&name).unwrap().is_none());
    assert!(store.list().unwrap().is_empty());
}

#[tokio::test]
async fn unknown_model_is_not_found() {
    let remote = tempfile::tempdir().unwrap();
    let base = serve(remote.path()).await;
    let local = tempfile::tempdir().unwrap();
    let puller = Puller::new(Store::open(local.path()).unwrap()).unwrap();
    let err = puller
        .pull(
            &ModelName::parse(&format!("{base}/library/nope")).unwrap(),
            &|_| {},
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotFound(_)), "{err}");
}
