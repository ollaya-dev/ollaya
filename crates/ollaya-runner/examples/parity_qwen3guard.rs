//! Compare the `qwen3guard-gen-v1` runtime against golden fixtures from
//! `ollaya_convert.families.qwen3guard.parity` (transformers fp32, TF32 off).
//!
//!     cargo run --release -p ollaya-runner --example parity_qwen3guard -- <model-dir> <goldens.jsonl> [cpu|cuda] [--latency]
//!
//! Every golden state is encoded with the model's preset first:
//! * both rows' token ids (the chat template rendered upstream) and last positions must match;
//! * a request for part of the preset builds the safety row, and the category row only when it
//!   asks `category`; any other question is rejected.
//!
//! Then every state runs through the graph:
//! * all 13 candidate logits of both rows, and every preset question's option logits, must be
//!   within `LOGIT_TOL`;
//! * probabilities (raw, temperature 1) must pick the reference's option (max / p99 reported).
//!
//! `--latency` then times full requests of the whole preset (`run`: tokenize, encode, forward).

use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::{Questions, parse_questions};
use ollaya_runner::Device;
use ollaya_runner::qwen3guard::GuardModel;
use serde_json::{Value, json};

/// Largest raw-logit difference accepted (ONNX Runtime vs PyTorch fp32, TF32 off).
const LOGIT_TOL: f64 = 1e-3;

fn argmax(p: &[f64]) -> usize {
    p.iter()
        .enumerate()
        .fold(
            (0, f64::NEG_INFINITY),
            |b, (i, &v)| if v > b.1 { (i, v) } else { b },
        )
        .0
}

fn max_diff(a: &[f32], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - y).abs())
        .fold(0.0, f64::max)
}

fn percentile(sorted: &[f64], p: usize) -> f64 {
    sorted
        .get((sorted.len() * p / 100).min(sorted.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0)
}

/// Part of the preset reads only the rows it needs; anything else is rejected.
fn check_preset_rules(model: &GuardModel, preset: &Questions) -> Result<()> {
    let state = json!("Is this safe?");
    let full = model.encode(&state, preset)?;
    for qid in preset.keys() {
        let one: Questions = [(qid.clone(), preset[qid].clone())].into_iter().collect();
        let rows = model.encode(&state, &one)?;
        let want = if qid == "category" { 2 } else { 1 };
        if rows.rows != full.rows[..want] {
            bail!("PRESET {qid} alone: {} rows", rows.rows.len());
        }
    }
    for custom in [
        json!({"spam": {"type": "noul", "instructions": "Is this spam?"}}),
        json!({"unsafe": {"type": "noul", "instructions": "Is it harmful?"}}),
    ] {
        let q = parse_questions(&custom)?;
        match model.encode(&state, &q) {
            Err(e) if e.to_string().contains("built-in questions") => {}
            other => bail!("PRESET {custom} is not rejected: {other:?}"),
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: parity_qwen3guard <model-dir> <goldens.jsonl> [cpu|cuda] [--latency]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let latency = args.iter().any(|a| a == "--latency");
    let t = Instant::now();
    let model = GuardModel::load(&PathBuf::from(&args[1]), device, None)?;
    println!("load {:.1}s on {device:?}", t.elapsed().as_secs_f64());
    let preset = model.layout.preset.clone();
    check_preset_rules(&model, &preset)?;

    let file = std::fs::File::open(&args[2]).with_context(|| args[2].clone())?;
    let mut records = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        records.push(serde_json::from_str::<Value>(&line?)?);
    }

    // Encoding: both rows of every state.
    let mut encoded = Vec::new();
    let mut enc_bad = 0;
    for rec in &records {
        let id = rec["id"].as_str().unwrap_or("?");
        let rows = model.encode(&rec["state"], &preset)?;
        let want: Vec<Vec<u32>> = serde_json::from_value(rec["rows"].clone())?;
        let last: Vec<usize> = serde_json::from_value(rec["last_pos"].clone())?;
        let got_last: Vec<usize> = rows.rows.iter().map(|r| r.len() - 1).collect();
        if rows.rows != want || got_last != last {
            enc_bad += 1;
            if enc_bad <= 5 {
                let first = want
                    .iter()
                    .zip(&rows.rows)
                    .find_map(|(a, b)| a.iter().zip(b).position(|(x, y)| x != y));
                println!(
                    "ENCODING {id}: lengths {:?} vs {:?}, first differing token {first:?}",
                    got_last, last
                );
            }
        }
        encoded.push(rows);
    }
    let tokens: usize = encoded.iter().flat_map(|r| &r.rows).map(Vec::len).sum();
    println!(
        "encoding: {} states, {} rows, {tokens} tokens | encoding mismatches: {enc_bad}",
        records.len(),
        2 * records.len()
    );
    if enc_bad > 0 {
        bail!("encoding parity failed");
    }

    // Numbers: candidate logits, option logits, probabilities.
    let (mut cand_max, mut logit_max, mut disagree) = (0f64, 0f64, 0);
    let mut prob_diffs = Vec::new();
    let mut forward = 0f64;
    for (rec, rows) in records.iter().zip(&encoded) {
        let id = rec["id"].as_str().unwrap_or("?");
        let t = Instant::now();
        let cand = model.cand_logits(rows)?;
        forward += t.elapsed().as_secs_f64();
        let want: Vec<Vec<f64>> = serde_json::from_value(rec["cand_logits"].clone())?;
        for (got, want) in cand.iter().zip(&want) {
            if got.len() != want.len() {
                bail!("CAND {id}: {} candidates vs {}", got.len(), want.len());
            }
            cand_max = cand_max.max(max_diff(got, want));
        }
        let out = model.output(rows, &cand);
        for ((qid, q), got) in preset.iter().zip(&out.questions) {
            let want: Vec<f64> = serde_json::from_value(rec["option_logits"][qid].clone())?;
            if got.logits.len() != want.len() {
                bail!(
                    "LOGITS {id} {qid}: {} options vs {}",
                    got.logits.len(),
                    want.len()
                );
            }
            logit_max = logit_max.max(max_diff(&got.logits, &want));
            let want: Vec<f64> = serde_json::from_value(rec["probabilities"][qid].clone())?;
            let probs = model
                .calibration
                .probabilities(q.qtype, &got.logits, out.state_tokens);
            prob_diffs.push(
                want.iter()
                    .zip(&probs)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max),
            );
            if argmax(&want) != argmax(&probs) {
                disagree += 1;
                if disagree <= 5 {
                    println!("DECISION {id} {qid}: {probs:.4?} vs reference {want:.4?}");
                }
            }
        }
    }
    let nq = records.len() * preset.len();
    prob_diffs.sort_by(f64::total_cmp);
    println!(
        "numbers: candidate logit diff max {cand_max:.1e}, option logit diff max {logit_max:.1e} \
         | decisions agree: {:.2}% ({disagree} of {nq} differ) | prob diff max {:.1e} p99 {:.1e} \
         | forward {forward:.1}s ({:.0} ms/state)",
        100.0 * (nq - disagree) as f64 / nq.max(1) as f64,
        prob_diffs.last().copied().unwrap_or(0.0),
        percentile(&prob_diffs, 99),
        1000.0 * forward / records.len().max(1) as f64,
    );
    if cand_max.max(logit_max) > LOGIT_TOL {
        bail!("logits differ by more than {LOGIT_TOL:e}");
    }
    if disagree > 0 {
        bail!("decisions differ from the reference");
    }

    if latency {
        let mut ms = Vec::new();
        for _ in 0..2 {
            for rec in &records {
                let t = Instant::now();
                model.run(&rec["state"], &preset)?;
                ms.push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        ms.sort_by(f64::total_cmp);
        println!(
            "latency, the whole preset (n={}, {:.0} tokens/request): p50 {:.1} ms  p95 {:.1} ms  max {:.1} ms",
            ms.len(),
            tokens as f64 / records.len().max(1) as f64,
            percentile(&ms, 50),
            percentile(&ms, 95),
            ms.last().copied().unwrap_or(0.0)
        );
    }
    Ok(())
}
