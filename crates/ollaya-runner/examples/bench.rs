//! Latency of full requests (tokenize + encode + forward), on real request shapes.
//!
//!     cargo run --release -p ollaya-runner --example bench -- <model-dir> <goldens.jsonl> [cpu|cuda] [threads]
//!
//! Workloads come from the fixture file: every case as sent (1-7 questions, mostly 5), and the
//! same cases cut down to their first question.

use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use ollaya_decision::{Questions, parse_questions};
use ollaya_runner::{Device, OnnxModel};
use serde_json::Value;

fn percentile(sorted: &[f64], p: usize) -> f64 {
    sorted[(sorted.len() * p / 100).min(sorted.len() - 1)]
}

fn measure(model: &OnnxModel, requests: &[(Value, Questions)], label: &str) -> Result<()> {
    for (s, q) in requests.iter().take(20) {
        model.run(s, q)?; // warm-up: kernel selection, arena growth
    }
    let mut ms = Vec::with_capacity(requests.len() * 2);
    for _ in 0..2 {
        for (s, q) in requests {
            let t = Instant::now();
            model.run(s, q)?;
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
    }
    ms.sort_by(f64::total_cmp);
    let qn: usize = requests.iter().map(|(_, q)| q.len()).sum();
    println!(
        "{label:<22} n={:<4} avg {:.1} q/request | p50 {:>7.1} ms  p95 {:>7.1} ms  p99 {:>7.1} ms",
        ms.len(),
        qn as f64 / requests.len() as f64,
        percentile(&ms, 50),
        percentile(&ms, 95),
        percentile(&ms, 99)
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: bench <model-dir> <goldens.jsonl> [cpu|cuda] [threads]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let threads = args.get(4).and_then(|t| t.parse().ok());
    let t = Instant::now();
    let model = OnnxModel::load(&PathBuf::from(&args[1]), device, threads)?;
    println!("load {:.2} s", t.elapsed().as_secs_f64());

    let mut full = Vec::new();
    for line in std::io::BufReader::new(std::fs::File::open(&args[2])?).lines() {
        let rec: Value = serde_json::from_str(&line?)?;
        full.push((rec["state"].clone(), parse_questions(&rec["questions"])?));
    }
    let single: Vec<_> = full
        .iter()
        .map(|(s, q)| {
            (
                s.clone(),
                q.iter()
                    .take(1)
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            )
        })
        .collect();

    measure(&model, &single, "1 question")?;
    measure(&model, &full, "full request")?;
    Ok(())
}
