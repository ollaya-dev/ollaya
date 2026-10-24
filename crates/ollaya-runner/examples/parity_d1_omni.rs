//! Compare the `d1-omni-markers-v1` runtime against goldens from
//! `ollaya_convert.families.d1_omni.goldens` (the model repo's own code in fp32, TF32 off).
//!
//!     cargo run --release -p ollaya-runner --example parity_d1_omni -- <model-dir> <goldens.jsonl> [cpu|cuda] [--smoke]
//!
//! Every case is encoded first: a request upstream rejects must be rejected here; otherwise every
//! question's token ids and marker positions must match exactly. Then every case runs through the
//! graph, its rows batched as the runtime batches them:
//! * each question's marker logits must be within `LOGIT_TOL` of the reference;
//! * the calibration must apply the reference's temperature, and the calibrated probabilities must
//!   pick the reference's option (max / p99 difference reported).
//!
//! It prints one `record:` JSON line for `ollaya_convert.parity_run` before its checks.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::parse_questions;
use ollaya_runner::Device;
use ollaya_runner::d1_omni::D1OmniModel;
use serde_json::Value;

mod smoke;

/// Largest raw-logit difference accepted (ONNX Runtime fp32 vs PyTorch fp32).
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
        bail!("usage: parity_d1_omni <model-dir> <goldens.jsonl> [cpu|cuda] [--smoke]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        Some("metal") => Device::Metal,
        _ => Device::Cpu,
    };
    let t = Instant::now();
    let model = D1OmniModel::load(&PathBuf::from(&args[1]), device, None)?;
    println!("load {:.1}s on {device:?}", t.elapsed().as_secs_f64());

    let (mut cases, mut nq, mut enc_bad, mut reject_bad, mut rejected) = (0, 0, 0, 0, 0);
    let (mut logit_max, mut temp_bad, mut disagree) = (0f64, 0, 0);
    let mut prob_diffs = Vec::new();
    let mut forward = 0f64;
    for line in smoke::lines(&args[2], smoke)? {
        let rec: Value = serde_json::from_str(&line)?;
        let id = rec["id"].as_str().unwrap_or("?").to_owned();
        let questions = parse_questions(&rec["questions"]);
        let encoding = questions
            .as_ref()
            .ok()
            .map(|qs| model.encode(&rec["state"], qs));
        if !rec["error"].is_null() {
            rejected += 1;
            if matches!(encoding, Some(Ok(_))) {
                reject_bad += 1;
                println!(
                    "REJECT {id}: upstream rejects it ({}), the runtime does not",
                    rec["error"]
                );
            }
            continue;
        }
        let questions = questions.with_context(|| id.clone())?;
        let encoding = encoding
            .unwrap()
            .with_context(|| format!("{id}: the runtime rejects a request upstream accepts"))?;
        cases += 1;
        let items = rec["items"].as_array().context("items")?;
        if items.len() != questions.len() {
            bail!(
                "{id}: {} golden items for {} questions",
                items.len(),
                questions.len()
            );
        }
        let mut same = true;
        for (row, item) in encoding.questions.iter().zip(items) {
            let ids: Vec<u32> = serde_json::from_value(item["ids"].clone())?;
            let markers: Vec<usize> = serde_json::from_value(item["markers"].clone())?;
            if row.ids != ids || row.markers != markers {
                same = false;
                if enc_bad < 5 {
                    let first = row.ids.iter().zip(&ids).position(|(a, b)| a != b);
                    println!(
                        "ENCODING {id} {}: len {} vs {}, markers {:?} vs {markers:?}, first diff at {first:?}",
                        item["qid"],
                        row.ids.len(),
                        ids.len(),
                        row.markers
                    );
                }
            }
        }
        if !same {
            enc_bad += 1;
            continue;
        }

        let t = Instant::now();
        let logits = model.row_logits(&encoding)?;
        forward += t.elapsed().as_secs_f64();
        for ((qid, q), (got, item)) in questions.iter().zip(logits.iter().zip(items)) {
            nq += 1;
            let want: Vec<f64> = serde_json::from_value(item["logits"].clone())?;
            let got64: Vec<f64> = got.iter().map(|&x| f64::from(x)).collect();
            let d = max_diff(&got64, &want);
            if d > LOGIT_TOL {
                println!("LOGITS {id} {qid}: {got64:.5?} vs reference {want:.5?}");
            }
            logit_max = logit_max.max(d);
            let gold_t = item["temperature"].as_f64().context("temperature")?;
            let t_got = model.calibration.temperature(q.qtype, got.len());
            if (t_got - gold_t).abs() > 1e-9 {
                temp_bad += 1;
                println!("TEMPERATURE {id} {qid}: {t_got} vs reference {gold_t}");
            }
            let gold_p: Vec<f64> = serde_json::from_value(item["probabilities"].clone())?;
            let probs = model
                .calibration
                .probabilities(q.qtype, got, encoding.state_tokens);
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
        "encoding: {cases} cases, {nq} questions, {rejected} requests rejected upstream | encoding mismatches: \
         {enc_bad} | rejection mismatches: {reject_bad}"
    );
    println!(
        "numbers: logit diff max {logit_max:.1e} | decisions agree: {:.2}% ({disagree} differ) | prob diff max \
         {:.1e} p99 {:.1e} | temperatures off: {temp_bad} | forward {forward:.1}s ({:.0} ms/request)",
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
    if enc_bad + reject_bad > 0 {
        bail!("encoding parity failed");
    }
    if logit_max > LOGIT_TOL {
        bail!("logits differ by more than {LOGIT_TOL:e}");
    }
    if temp_bad > 0 || disagree > 0 {
        bail!("{temp_bad} temperatures and {disagree} decisions differ from the reference");
    }
    Ok(())
}
