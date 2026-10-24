//! Compare the `gliclass-kv-v1` engine against goldens from the Python reference
//! (`ollaya_convert.families.gliclass_kv.goldens`, the `gliclass` package in fp32).
//!
//!     cargo run --release -p ollaya-runner --example parity_gliclass_kv -- <model-dir> <goldens.jsonl> [cpu|cuda] [--smoke]
//!
//! Per question:
//! * the upstream call (labels, prompt, mode) must match exactly;
//! * the decoder row, the `<<LABEL>>` positions and the label section must match exactly;
//! * the calibrated label logits must be within `LOGIT_TOL` of the fp32 reference;
//! * the probabilities must pick the same option (max / p99 absolute difference reported).
//!
//! It prints one `record:` JSON line for `ollaya_convert.parity_run` before its checks.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::gliclass::Call;
use ollaya_decision::parse_questions;
use ollaya_runner::Device;
use ollaya_runner::gliclass_kv::GliclassKvModel;
use serde_json::Value;

mod smoke;

/// Max absolute label-logit difference against the fp32 reference.
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

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn percentile(sorted: &[f64], p: usize) -> f64 {
    sorted
        .get((sorted.len() * p / 100).min(sorted.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0)
}

fn main() -> Result<()> {
    // SAFETY: first thing in main, before any thread starts (MLX reads its settings once).
    unsafe { ollaya_runner::prepare_process() };
    let (args, smoke) = smoke::args();
    if args.len() < 3 {
        bail!("usage: parity_gliclass_kv <model-dir> <goldens.jsonl> [cpu|cuda] [--smoke]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        Some("metal") => Device::Metal,
        _ => Device::Cpu,
    };
    let t = Instant::now();
    let model = GliclassKvModel::load(&PathBuf::from(&args[1]), device, None)?;
    println!("load {:.1}s on {device:?}", t.elapsed().as_secs_f64());

    let (mut cases, mut nq, mut call_bad, mut enc_bad, mut disagree) = (0, 0, 0, 0, 0);
    let mut logit_max = 0f64;
    let mut prob_diffs = Vec::new();
    let mut forward = 0f64;
    for line in smoke::lines(&args[2], smoke)? {
        let rec: Value = serde_json::from_str(&line)?;
        let id = rec["id"].as_str().unwrap_or("?").to_owned();
        let qs = parse_questions(&rec["questions"]).with_context(|| id.clone())?;
        cases += 1;
        let state_text = model.layout.state_text(&rec["state"]);
        if state_text != rec["state_text"].as_str().unwrap_or_default() {
            call_bad += 1;
            println!("STATE {id}: state text differs");
        }
        let encoding = model
            .encode(&rec["state"], &qs)
            .with_context(|| format!("{id}: the runtime rejects a request upstream accepts"))?;
        let items = rec["items"].as_array().context("items")?;
        if items.len() != qs.len() {
            bail!(
                "{id}: {} golden items for {} questions",
                items.len(),
                qs.len()
            );
        }
        let mut same = true;
        for ((qid, q), (row, gold)) in qs.iter().zip(encoding.rows.iter().zip(items)) {
            let call = Call::new(q);
            let labels: Vec<String> = call
                .labels
                .iter()
                .map(|l| model.layout.sanitize(l))
                .collect();
            let want_labels: Vec<String> = serde_json::from_value(gold["labels"].clone())?;
            if labels != want_labels
                || model.layout.sanitize(&call.prompt)
                    != gold["prompt"].as_str().unwrap_or_default()
                || call.mode.name() != gold["mode"].as_str().unwrap_or_default()
            {
                call_bad += 1;
                if call_bad <= 5 {
                    println!("CALL {id} {qid}: {call:?}");
                }
            }
            let ids: Vec<u32> = serde_json::from_value(gold["ids"].clone())?;
            let markers: Vec<usize> = serde_json::from_value(gold["markers"].clone())?;
            if row.ids != ids
                || row.markers != markers
                || Some(row.section_start as u64) != gold["section_start"].as_u64()
                || Some(row.sep as u64) != gold["sep"].as_u64()
            {
                same = false;
                if enc_bad < 5 {
                    let first = row.ids.iter().zip(&ids).position(|(a, b)| a != b);
                    println!(
                        "ENCODING {id} {qid}: len {} vs {}, markers {:?} vs {markers:?}, section {} / {} vs {} / {}, first diff at {first:?}",
                        row.ids.len(),
                        ids.len(),
                        row.markers,
                        row.section_start,
                        row.sep,
                        gold["section_start"],
                        gold["sep"]
                    );
                }
            }
        }
        if !same {
            enc_bad += 1;
            continue;
        }

        let t = Instant::now();
        let labels = model.label_logits(&encoding)?;
        forward += t.elapsed().as_secs_f64();
        for ((qid, q), (row, (got, gold))) in qs
            .iter()
            .zip(encoding.rows.iter().zip(labels.iter().zip(items)))
        {
            nq += 1;
            let want: Vec<f64> = serde_json::from_value(gold["logits"].clone())?;
            let got64: Vec<f64> = got.iter().map(|&x| f64::from(x)).collect();
            let d = max_diff(&got64, &want);
            if d > LOGIT_TOL {
                println!(
                    "LOGITS {id} {qid}: {got64:.5?} vs reference {want:.5?} ({} steps upstream)",
                    gold["steps"]
                );
            }
            logit_max = logit_max.max(d);
            let gold_p: Vec<f64> = serde_json::from_value(gold["probabilities"].clone())?;
            let option = row.mode.option_logits(got);
            let probs = model
                .calibration
                .probabilities(q.qtype, &option, encoding.state_tokens);
            prob_diffs.push(max_diff(&probs, &gold_p));
            if argmax(&gold_p) != argmax(&probs) {
                disagree += 1;
                if disagree <= 5 {
                    println!("DECISION {id} {qid}: {probs:.4?} vs reference {gold_p:.4?}");
                }
            }
        }
    }
    prob_diffs.sort_by(f64::total_cmp);
    println!(
        "encoding: {cases} cases, {nq} questions | call mismatches: {call_bad} | encoding mismatches: {enc_bad}"
    );
    println!(
        "numbers: label logit diff max {logit_max:.1e} | decisions agree: {:.2}% ({disagree} differ) | prob diff max \
         {:.1e} p99 {:.1e} | forward {forward:.1}s ({:.0} ms/request)",
        100.0 * (nq - disagree) as f64 / nq.max(1) as f64,
        prob_diffs.last().copied().unwrap_or(0.0),
        percentile(&prob_diffs, 99),
        1000.0 * forward / cases.max(1) as f64,
    );
    // One JSON line for `ollaya_convert.parity_run`, which records it in results/runs/.
    println!(
        "record: {}",
        serde_json::json!({
            "questions": nq,
            "decisions_same": nq - disagree,
            "max_logit_diff": logit_max,
            "max_prob_diff": prob_diffs.last().copied().unwrap_or(0.0),
            "p99_prob_diff": percentile(&prob_diffs, 99),
        })
    );
    if call_bad + enc_bad > 0 {
        bail!("encoding parity failed");
    }
    if logit_max > LOGIT_TOL {
        bail!("logits differ by more than {LOGIT_TOL:e}");
    }
    if disagree > 0 {
        bail!("{disagree} decisions differ from the reference");
    }
    Ok(())
}
