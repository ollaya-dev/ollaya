//! Compare the `nli-pairs-v1` runtime against golden fixtures from
//! `ollaya_convert.families.nli.goldens` (the fp32 PyTorch reference, one unpadded pair at a time).
//!
//!     cargo run --release -p ollaya-runner --example parity_nli -- <model-dir> <goldens.jsonl> [cpu|cuda]
//!
//! Per case:
//! * the premise, and which questions are rejected (400), must match;
//! * per question: mode, hypotheses and every row's token ids must match exactly;
//! * row scores and option logits must be within `LOGIT_TOL` of the reference;
//! * calibrated probabilities must pick the reference's option (max / p99 difference reported).

use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::{Questions, parse_questions, serialize_state};
use ollaya_runner::Device;
use ollaya_runner::nli::NliModel;
use serde_json::Value;

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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: parity_nli <model-dir> <goldens.jsonl> [cpu|cuda]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let model = NliModel::load(&PathBuf::from(&args[1]), device, None)?;
    let file = std::fs::File::open(&args[2]).with_context(|| args[2].clone())?;

    let (mut cases, mut questions, mut rows, mut rej_bad, mut enc_bad, mut disagree) =
        (0, 0, 0, 0, 0, 0);
    let (mut score_max, mut logit_max) = (0f64, 0f64);
    let mut prob_diffs = Vec::new();
    let mut forward = 0f64;
    for line in std::io::BufReader::new(file).lines() {
        let rec: Value = serde_json::from_str(&line?)?;
        let id = rec["id"].as_str().unwrap_or("?").to_owned();
        let state = &rec["state"];
        cases += 1;
        if serialize_state(state) != rec["premise"].as_str().context("premise")? {
            bail!("PREMISE {id}: serialize_state differs from the reference");
        }

        // The reference drops rejected questions and answers the rest.
        let all = parse_questions(&rec["questions"])?;
        let rejected = rec["rejected"].as_object().context("rejected")?;
        for (qid, q) in &all {
            let one: Questions = [(qid.clone(), q.clone())].into_iter().collect();
            if model.encode(state, &one).is_err() != rejected.contains_key(qid) {
                rej_bad += 1;
                println!(
                    "REJECTION {id} {qid}: reference rejected={}",
                    rejected.contains_key(qid)
                );
            }
        }
        let qs: Questions = all
            .into_iter()
            .filter(|(qid, _)| !rejected.contains_key(qid))
            .collect();
        let items = rec["items"].as_array().context("items")?;
        if items.len() != qs.len() {
            bail!(
                "ITEMS {id}: {} answered questions vs {}",
                qs.len(),
                items.len()
            );
        }
        if qs.is_empty() {
            continue;
        }
        let encoding = model.encode(state, &qs)?;
        let t = Instant::now();
        let scores = model.scores(&encoding)?;
        forward += t.elapsed().as_secs_f64();
        let out = model.output(&encoding, &scores);

        let mut next = 0;
        for (r, ((qid, q), gold)) in qs.iter().zip(items).enumerate() {
            questions += 1;
            let pairs = &encoding.questions[r];
            let (hyps, mode) = model.layout.hypotheses(q)?;
            let gold_rows = gold["rows"].as_array().context("rows")?;
            let gold_ids = gold_rows
                .iter()
                .map(|g| serde_json::from_value::<Vec<u32>>(g["ids"].clone()))
                .collect::<Result<Vec<_>, _>>()?;
            let gold_hyps: Vec<&str> = gold_rows
                .iter()
                .map(|g| g["hypothesis"].as_str().unwrap_or_default())
                .collect();
            let row_scores = &scores[next..next + pairs.rows.len()];
            next += pairs.rows.len();
            rows += pairs.rows.len();
            if gold["qid"] != qid.as_str()
                || gold["mode"] != mode.name()
                || hyps != gold_hyps
                || pairs.rows != gold_ids
            {
                enc_bad += 1;
                if enc_bad <= 5 {
                    let first = pairs.rows.iter().zip(&gold_ids).position(|(a, b)| a != b);
                    println!(
                        "ENCODING {id} {qid}: mode {} vs {}, {} vs {} rows, first differing row {first:?}, hypotheses {hyps:?} vs {gold_hyps:?}",
                        mode.name(),
                        gold["mode"],
                        pairs.rows.len(),
                        gold_ids.len()
                    );
                }
                continue;
            }

            for (got, g) in row_scores.iter().zip(gold_rows) {
                let want: Vec<f64> = serde_json::from_value(g["scores"].clone())?;
                score_max = score_max.max(max_diff(got, &want));
            }
            let logits = &out.questions[r].logits;
            let want_logits: Vec<f64> = serde_json::from_value(gold["option_logits"].clone())?;
            if logits.len() != want_logits.len() {
                bail!(
                    "LOGITS {id} {qid}: {} options vs {}",
                    logits.len(),
                    want_logits.len()
                );
            }
            logit_max = logit_max.max(max_diff(logits, &want_logits));

            let want: Vec<f64> = serde_json::from_value(gold["probabilities"].clone())?;
            let got = model.calibration.probabilities(q.qtype, logits);
            prob_diffs.push(
                want.iter()
                    .zip(&got)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max),
            );
            if argmax(&want) != argmax(&got) {
                disagree += 1;
                if disagree <= 5 {
                    println!("DECISION {id} {qid}: {got:.4?} vs reference {want:.4?}");
                }
            }
        }
    }
    prob_diffs.sort_by(f64::total_cmp);
    let p99 = prob_diffs
        .get(prob_diffs.len() * 99 / 100)
        .copied()
        .unwrap_or(0.0);
    println!(
        "{cases} cases, {questions} questions, {rows} rows | rejection mismatches: {rej_bad} \
         | encoding mismatches: {enc_bad} \
         | score diff max {score_max:.1e}, option logit diff max {logit_max:.1e} \
         | decisions agree: {:.2}% ({disagree} differ) | prob diff max {:.1e} p99 {p99:.1e} \
         | forward {forward:.1}s",
        100.0 * (questions - enc_bad - disagree) as f64 / (questions - enc_bad).max(1) as f64,
        prob_diffs.last().copied().unwrap_or(0.0),
    );
    if rej_bad + enc_bad > 0 {
        bail!("encoding parity failed");
    }
    if score_max.max(logit_max) > LOGIT_TOL {
        bail!("logits differ by more than {LOGIT_TOL:e}");
    }
    if disagree > 0 {
        bail!("decisions differ from the reference");
    }
    Ok(())
}
