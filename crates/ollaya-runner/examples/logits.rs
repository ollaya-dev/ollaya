//! Raw option logits for a file of requests, through any engine this build runs: the input of the
//! typed-decisions quality report (`ollaya_convert.families.llm_common.quality_runtime`).
//!
//!     cargo run --release -p ollaya-runner --example logits -- <model-dir> <requests.jsonl> <out.jsonl> [cpu|cuda|metal]
//!
//! A directory with a `model.gguf` runs on llama.cpp, loaded from `$OLLAYA_LIBRARY_PATH/llama` with
//! the CUDA backend from `$OLLAYA_LIBRARY_PATH/cuda_v13` (or `cuda_v12`), as `parity_llama` does.
//!
//! Each input line is `{"id", "state", "questions"}`. Each output line is `{"id", "logits": {qid: [...]}}`
//! with the logits in the order answers use (a choice's criteria order, noul `[false, true]`, score
//! levels), or `{"id", "error"}` for a request the model rejects.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_runner::engine::Engine;
use ollaya_runner::llama::{Libraries, LlamaModel, Target};
use ollaya_runner::{Device, ModelFiles};
use serde_json::{Map, Value, json};

fn main() -> Result<()> {
    // SAFETY: first thing in main, before any thread starts (MLX reads its settings once).
    unsafe { ollaya_runner::prepare_process() };
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("usage: logits <model-dir> <requests.jsonl> <out.jsonl> [cpu|cuda|metal]");
    }
    let device = match args.get(4).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        Some("metal") => Device::Metal,
        _ => Device::Cpu,
    };
    let t = Instant::now();
    let dir = PathBuf::from(&args[1]);
    let model: Box<dyn Engine> = if dir.join("model.gguf").is_file() {
        let lib = std::env::var_os("OLLAYA_LIBRARY_PATH")
            .map(PathBuf::from)
            .context("set OLLAYA_LIBRARY_PATH to an install's lib/ollaya")?;
        let cuda = ["cuda_v13", "cuda_v12"]
            .map(|pack| lib.join(pack).join(ollaya_runner::llama::CUDA_BACKEND))
            .into_iter()
            .find(|p| p.is_file());
        let rocm =
            Some(lib.join("rocm").join(ollaya_runner::llama::ROCM_BACKEND)).filter(|p| p.is_file());
        let libs = Libraries {
            dir: lib.join("llama"),
            cuda,
            rocm,
        };
        let target = match device {
            Device::Cuda(n) => Target::Device(format!("CUDA{n}")),
            Device::Metal => Target::Device("MTL0".into()),
            _ => Target::Cpu,
        };
        Box::new(LlamaModel::load(
            &dir.join("model.gguf"),
            &dir.join("decision.json"),
            &libs,
            &target,
            None,
        )?)
    } else {
        ollaya_runner::engine::load(&ModelFiles::dir(&dir), device, None)?
    };
    println!("load {:.1}s on {device:?}", t.elapsed().as_secs_f64());
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[3])?);
    let (mut n, mut rejected) = (0, 0);
    let t = Instant::now();
    for line in std::io::BufReader::new(std::fs::File::open(&args[2])?).lines() {
        let rec: Value = serde_json::from_str(&line?)?;
        let qids: Vec<String> = rec["questions"]
            .as_object()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        let t1 = Instant::now();
        let result = model.run_json(&rec["state"], &rec["questions"]);
        println!(
            "{} {:.0} ms",
            rec["id"].as_str().unwrap_or("?"),
            t1.elapsed().as_secs_f64() * 1000.0
        );
        let row = match result {
            Ok(o) => {
                let logits: Map<String, Value> = qids
                    .into_iter()
                    .zip(o.questions)
                    .map(|(qid, q)| (qid, json!(q.logits)))
                    .collect();
                json!({"id": rec["id"], "logits": logits})
            }
            Err(e) => {
                rejected += 1;
                json!({"id": rec["id"], "error": e.to_string()})
            }
        };
        writeln!(out, "{row}")?;
        n += 1;
    }
    println!(
        "{n} requests ({rejected} rejected) in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    Ok(())
}
