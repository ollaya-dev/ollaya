//! Latency and memory of full requests (tokenize + encode + forward), on real request shapes.
//!
//!     cargo run --release -p ollaya-runner --example bench -- <model-dir> <fixtures.jsonl> [cpu|cuda|metal] [threads]
//!         [--ids PREFIX] [--limit N] [--warmup N] [--full-only]
//!
//! Any layout this build runs (the model directory's `decision.json` names it). Workloads come
//! from the fixture file (goldens): every case as sent (1-7 questions, mostly 5), and the same
//! cases cut down to their first question. Records upstream rejects are skipped. `--ids` keeps the
//! records whose id starts with PREFIX (e.g. `td/`, the typed-decisions rows), `--limit` the first N, `--warmup` sets the untimed runs (default 20), `--full-only` skips the
//! one-question pass.
//!
//! Memory: the process's resident set after loading and after the timed runs, and its peak
//! (Linux: VmRSS / VmHWM from /proc; macOS: `ps` for the current RSS, peak via `/usr/bin/time -l`).

use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use ollaya_decision::{Questions, parse_questions};
use ollaya_runner::{Device, Engine, ModelFiles};
use serde_json::Value;

fn percentile(sorted: &[f64], p: usize) -> f64 {
    sorted[(sorted.len() * p / 100).min(sorted.len() - 1)]
}

/// Resident set size (and its peak, where the OS reports it), in GiB.
fn rss() -> String {
    let gib = |kb: f64| kb / (1024.0 * 1024.0);
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        let field = |name: &str| {
            status
                .lines()
                .find(|l| l.starts_with(name))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0)
        };
        return format!(
            "rss {:.2} GiB, peak {:.2} GiB",
            gib(field("VmRSS:")),
            gib(field("VmHWM:"))
        );
    }
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output();
    match out
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
    {
        Some(kb) => format!("rss {:.2} GiB", gib(kb)),
        None => "rss unknown".into(),
    }
}

fn measure(
    model: &dyn Engine,
    requests: &[(Value, Questions)],
    warmup: usize,
    label: &str,
) -> Result<()> {
    for (s, q) in requests.iter().cycle().take(warmup) {
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
        "{label:<22} n={:<4} avg {:.1} q/request | p50 {:>7.1} ms  p95 {:>7.1} ms  p99 {:>7.1} ms | {}",
        ms.len(),
        qn as f64 / requests.len() as f64,
        percentile(&ms, 50),
        percentile(&ms, 95),
        percentile(&ms, 99),
        rss()
    );
    Ok(())
}

fn main() -> Result<()> {
    // SAFETY: first thing in main, before any thread starts (MLX reads its settings once).
    unsafe { ollaya_runner::prepare_process() };
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!(
            "usage: bench <model-dir> <fixtures.jsonl> [cpu|cuda|metal] [threads] [--ids PREFIX] [--limit N] [--warmup N] [--full-only]"
        );
    }
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse::<usize>().ok())
    };
    let limit = flag("--limit").unwrap_or(usize::MAX);
    let warmup = flag("--warmup").unwrap_or(20);
    let full_only = args.iter().any(|a| a == "--full-only");
    let ids = args
        .iter()
        .position(|a| a == "--ids")
        .and_then(|i| args.get(i + 1))
        .map_or("", String::as_str);
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        Some("metal") => Device::Metal,
        _ => Device::Cpu,
    };
    let threads = args.get(4).and_then(|t| t.parse().ok());
    let t = Instant::now();
    let model =
        ollaya_runner::engine::load(&ModelFiles::dir(&PathBuf::from(&args[1])), device, threads)?;
    println!(
        "load {:.2} s on {device:?} | {}",
        t.elapsed().as_secs_f64(),
        rss()
    );

    let mut full = Vec::new();
    for line in std::io::BufReader::new(std::fs::File::open(&args[2])?).lines() {
        let rec: Value = serde_json::from_str(&line?)?;
        if rec.get("error").is_some_and(|e| !e.is_null())
            || !rec["id"].as_str().unwrap_or("").starts_with(ids)
        {
            continue; // upstream rejects the request, or another id prefix
        }
        let Ok(questions) = parse_questions(&rec["questions"]) else {
            continue; // not TypeSafe wire (e.g. a score legend object)
        };
        full.push((rec["state"].clone(), questions));
        if full.len() == limit {
            break;
        }
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

    if !full_only {
        measure(model.as_ref(), &single, warmup, "1 question")?;
    }
    measure(model.as_ref(), &full, warmup, "full request")?;
    Ok(())
}
