//! Parity of an adapter model (decision 0011) against the same adapter merged into the base.
//!
//!     cargo run --release -p ollaya-runner --features mlx --example parity_adapter -- \
//!         <model-dir> <adapter-dir> <goldens.jsonl> [--smoke]
//!
//! `<model-dir>` is a development export of a Laya model (arch.json, its weights file under the
//! upstream name, tokenizer.json, decision.json). `<adapter-dir>` is a peft directory
//! (`adapter_config.json`, `adapter_model.safetensors`). `<goldens.jsonl>` supplies requests only;
//! the reference is computed, not read: the engine's own merged path (`arch.json`'s `lora`, which
//! adds `W + scaling * B A` in float32, as peft's `merge_and_unload` does) over the same adapter.
//!
//! Checks, on every question of every case (Metal only; CI has no Metal):
//! * the adapter model's decision equals the merged model's, and every option logit is within 1e-3
//!   of it (the project's fp32 gate; the tolerance is not loosened);
//! * the adapter model differs from the plain base on the adapted questions (the adapter acts);
//! * bit for bit: the base's answers, and adapter A's, are identical before and after two more
//!   adapters (B, C: the same tensors at other scalings) are loaded and used between requests.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ollaya_decision::parse_questions;
use ollaya_runner::engine::{AdapterLoad, Engine};
use ollaya_runner::{Device, OnnxModel};
use serde_json::{Value, json};

mod smoke;

const TOLERANCE: f32 = 1e-3;

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| path.display().to_string())?,
    )
    .with_context(|| path.display().to_string())
}

/// The adapter's tensor names, from its safetensors header.
fn tensor_names(path: &Path) -> Result<Vec<String>> {
    use std::io::Read as _;
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut header)?;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&header)?;
    Ok(header
        .keys()
        .filter(|k| *k != "__metadata__")
        .cloned()
        .collect())
}

fn logits(engine: &dyn Engine, rec: &Value) -> Result<Vec<Vec<f32>>> {
    let qs = parse_questions(&rec["questions"])?;
    let out = engine.run(&rec["state"], &qs)?;
    Ok(out.questions.into_iter().map(|q| q.logits).collect())
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(
            (0, f32::NEG_INFINITY),
            |b, (i, &x)| if x > b.1 { (i, x) } else { b },
        )
        .0
}

fn main() -> Result<()> {
    // SAFETY: first thing in main, before any thread starts (MLX reads its settings once).
    unsafe { ollaya_runner::prepare_process() };
    let (args, smoke) = smoke::args();
    if args.len() < 4 {
        bail!("usage: parity_adapter <model-dir> <adapter-dir> <goldens.jsonl> [--smoke]");
    }
    let (model_dir, adapter_dir) = (PathBuf::from(&args[1]), PathBuf::from(&args[2]));
    let config = read_json(&adapter_dir.join("adapter_config.json"))?;
    let weights = adapter_dir.join("adapter_model.safetensors");
    let scaling =
        config["lora_alpha"].as_f64().context("lora_alpha")? / config["r"].as_f64().context("r")?;
    let arch = read_json(&model_dir.join("arch.json"))?;
    let backbone_prefix = arch["backbone"]["prefix"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let prefix = "base_model.model.";
    let targets: Vec<String> = tensor_names(&weights)?
        .iter()
        .filter_map(|n| n.strip_prefix(prefix)?.strip_suffix(".lora_A.weight"))
        .map(|m| format!("{backbone_prefix}{m}"))
        .collect();
    if targets.is_empty() {
        bail!("no lora_A tensors under {prefix} in {}", weights.display());
    }
    println!(
        "adapter: r={} scaling={scaling} on {} modules",
        config["r"],
        targets.len()
    );

    let scratch = std::env::temp_dir().join(format!("parity-adapter-{}", std::process::id()));
    std::fs::create_dir_all(&scratch)?;
    let result = run(
        &args[3], smoke, &model_dir, &weights, &arch, scaling, prefix, &targets, &scratch,
    );
    let _ = std::fs::remove_dir_all(&scratch);
    result
}

#[allow(clippy::too_many_arguments)]
fn run(
    goldens: &str,
    smoke: bool,
    model_dir: &Path,
    weights: &Path,
    arch: &Value,
    scaling: f64,
    prefix: &str,
    targets: &[String],
    scratch: &Path,
) -> Result<()> {
    // The merged reference: the model dir, plus the adapter as a second weights file and `lora`.
    let merged_dir = scratch.join("merged");
    std::fs::create_dir_all(&merged_dir)?;
    for e in std::fs::read_dir(model_dir)? {
        let e = e?;
        if e.file_name() != "arch.json" && e.path().is_file() {
            std::os::unix::fs::symlink(e.path().canonicalize()?, merged_dir.join(e.file_name()))?;
        }
    }
    std::os::unix::fs::symlink(
        weights.canonicalize()?,
        merged_dir.join("adapter_model.safetensors"),
    )?;
    let mut merged_arch = arch.clone();
    let more = json!({"format": "safetensors", "file": "adapter_model.safetensors"});
    match merged_arch["weights"]["more"].as_array_mut() {
        Some(list) => list.push(more),
        None => merged_arch["weights"]["more"] = json!([more]),
    }
    merged_arch["lora"] = json!({"prefix": prefix, "scaling": scaling});
    std::fs::write(
        merged_dir.join("arch.json"),
        serde_json::to_vec(&merged_arch)?,
    )?;
    let merged = OnnxModel::load(&merged_dir, Device::Metal, None)?;
    let base = OnnxModel::load(model_dir, Device::Metal, None)?;

    let spec = |scale: f64| -> Result<PathBuf> {
        let path = scratch.join(format!("spec-{scale}.json"));
        std::fs::write(
            &path,
            serde_json::to_vec(
                &json!({"schema": 1, "format": "peft-lora", "rank": 0, "alpha": 0.0,
                "scaling": scale, "dtype": "F32", "prefix": prefix, "targets": targets}),
            )?,
        )?;
        Ok(path)
    };
    // The rank is read from the tensors; the spec's own is checked against them, so it is set here.
    let rank = {
        let a = read_json(&weights.with_file_name("adapter_config.json"))?;
        a["r"].as_u64().context("r")? as usize
    };
    let load = |digest: &str, scale: f64| -> Result<Box<dyn Engine>> {
        let mut s: Value = read_json(&spec(scale)?)?;
        s["rank"] = json!(rank);
        let path = scratch.join(format!("spec-{digest}.json"));
        std::fs::write(&path, serde_json::to_vec(&s)?)?;
        let (engine, bytes) = base.load_adapter(&AdapterLoad {
            digest: digest.into(),
            spec: path,
            weights: weights.to_path_buf(),
            decision: model_dir.join("decision.json"),
            tokenizer: model_dir.join("tokenizer.json"),
        })?;
        println!("loaded {digest} (scaling {scale}): {bytes} bytes");
        Ok(engine)
    };

    let records: Vec<Value> = smoke::lines(goldens, smoke)?
        .iter()
        .map(|l| serde_json::from_str(l))
        .collect::<Result<_, _>>()?;

    // Base answers before any adapter is loaded.
    let base_before: Vec<_> = records
        .iter()
        .map(|r| logits(&base, r))
        .collect::<Result<_>>()?;
    let a = load("sha256:A", scaling)?;

    let (mut questions, mut disagree, mut worst, mut acts) = (0, 0, 0f32, 0);
    let mut a_first: HashMap<usize, Vec<Vec<f32>>> = HashMap::new();
    for (i, rec) in records.iter().enumerate() {
        let want = logits(&merged, rec)?;
        let got = logits(a.as_ref(), rec)?;
        for (w, g) in want.iter().zip(&got) {
            questions += 1;
            let diff = w
                .iter()
                .zip(g)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max);
            worst = worst.max(diff);
            if argmax(w) != argmax(g) {
                disagree += 1;
            }
            if diff > TOLERANCE {
                println!(
                    "LOGITS {}: max diff {diff:.2e} over {TOLERANCE:.0e}",
                    rec["id"]
                );
            }
        }
        if got.iter().zip(&base_before[i]).any(|(g, b)| g != b) {
            acts += 1;
        }
        a_first.insert(i, got);
    }

    // Other adapters in use between requests change nothing, bit for bit.
    let b = load("sha256:B", scaling * 0.5)?;
    let c = load("sha256:C", scaling * 2.0)?;
    let (mut a_moved, mut base_moved) = (0, 0);
    for (i, rec) in records.iter().enumerate() {
        logits(b.as_ref(), rec)?;
        logits(c.as_ref(), rec)?;
        if logits(a.as_ref(), rec)? != a_first[&i] {
            a_moved += 1;
        }
        if logits(&base, rec)? != base_before[i] {
            base_moved += 1;
        }
    }
    // After the others are dropped again, A is still the same.
    base.unload_adapter("sha256:B")?;
    base.unload_adapter("sha256:C")?;
    let mut a_after_drop = 0;
    for (i, rec) in records.iter().enumerate() {
        if logits(a.as_ref(), rec)? != a_first[&i] {
            a_after_drop += 1;
        }
    }

    println!(
        "{} cases, {questions} questions | decisions differ from merged: {disagree} | logit diff max {worst:.2e} (limit {TOLERANCE:.0e}) \
         | cases the adapter changes against the base: {acts} | A moved by B and C: {a_moved} | base moved by adapters: {base_moved} \
         | A moved by dropping B and C: {a_after_drop}",
        records.len()
    );
    if disagree > 0
        || worst > TOLERANCE
        || a_moved > 0
        || base_moved > 0
        || a_after_drop > 0
        || acts == 0
    {
        bail!("adapter parity failed");
    }
    Ok(())
}
