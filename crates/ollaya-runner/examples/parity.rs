//! Compare the Rust runtime against golden fixtures produced by the `laya` Python package.
//!
//!     cargo run --release -p ollaya-runner --example parity -- <model-dir> <goldens.jsonl> [cpu|cuda]
//!
//! Per question:
//! * encoder token ids and marker positions must match exactly;
//! * the decision (most probable option) is compared against the fp32 reference;
//! * calibrated probabilities are compared (max / p99 absolute difference);
//! * when the fixture carries laya's own answers, the final answer in laya's shape must match
//!   (strings exactly, numbers within rounding).

use std::io::BufRead;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use ollaya_decision::{Answer, parse_questions};
use ollaya_runner::{Device, OnnxModel};
use serde_json::Value;

fn argmax(p: &[f64]) -> usize {
    p.iter()
        .enumerate()
        .fold(
            (0, f64::NEG_INFINITY),
            |b, (i, &v)| if v > b.1 { (i, v) } else { b },
        )
        .0
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: parity <model-dir> <goldens.jsonl> [cpu|cuda]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let model = OnnxModel::load(&PathBuf::from(&args[1]), device, None)?;
    let file = std::fs::File::open(&args[2]).with_context(|| args[2].clone())?;

    let (mut cases, mut questions, mut enc_bad, mut disagree, mut ans_checked, mut ans_bad) =
        (0, 0, 0, 0, 0, 0);
    let mut prob_diffs = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        let rec: Value = serde_json::from_str(&line?)?;
        let id = rec["id"].as_str().unwrap_or("?").to_owned();
        let qs = parse_questions(&rec["questions"])?;
        let encoded = model.encode(&rec["state"], &qs)?;
        let out = model.run_encoded(&encoded, &qs)?;
        cases += 1;

        for (r, ((qid, q), gold)) in qs.iter().zip(rec["items"].as_array().unwrap()).enumerate() {
            questions += 1;
            let ids: Vec<u32> = serde_json::from_value(gold["ids"].clone())?;
            let markers: Vec<usize> = serde_json::from_value(gold["markers"].clone())?;
            if encoded.questions[r].ids != ids || encoded.questions[r].markers != markers {
                enc_bad += 1;
                if enc_bad <= 5 {
                    let first = encoded.questions[r]
                        .ids
                        .iter()
                        .zip(&ids)
                        .position(|(a, b)| a != b);
                    println!(
                        "ENCODING {id} {qid}: len {} vs {}, markers {:?} vs {:?}, first diff at {first:?}",
                        encoded.questions[r].ids.len(),
                        ids.len(),
                        encoded.questions[r].markers,
                        markers
                    );
                }
                continue;
            }
            let gold_logits: Vec<f32> = serde_json::from_value(gold["logits"].clone())?;
            let want = model.calibration.probabilities(q.qtype, &gold_logits);
            let got = model
                .calibration
                .probabilities(q.qtype, &out.questions[r].logits);
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

            if let Some(want) = rec["answers"].get(qid) {
                ans_checked += 1;
                let answer = Answer::new(
                    q,
                    &model.calibration,
                    &out.questions[r].logits,
                    out.questions[r].act_logits.as_deref(),
                );
                if let Some(diff) = compare(&answer.to_laya(q), want, 2e-4, "") {
                    ans_bad += 1;
                    if ans_bad <= 5 {
                        println!("ANSWER {id} {qid}: {diff}");
                    }
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
        "{cases} cases, {questions} questions | encoding mismatches: {enc_bad} | decisions agree: {:.2}% ({disagree} differ) \
         | prob diff max {:.1e} p99 {p99:.1e} | laya answers outside rounding: {ans_bad}/{ans_checked}",
        100.0 * (questions - enc_bad - disagree) as f64 / (questions - enc_bad).max(1) as f64,
        prob_diffs.last().copied().unwrap_or(0.0),
    );
    if enc_bad > 0 {
        bail!("encoding parity failed");
    }
    Ok(())
}

/// First difference between two JSON values: strings/bools exact, numbers within `tol`.
fn compare(got: &Value, want: &Value, tol: f64, path: &str) -> Option<String> {
    match (got, want) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64()?, b.as_f64()?);
            ((a - b).abs() > tol).then(|| format!("{path}: {a} vs {b}"))
        }
        (Value::Object(a), Value::Object(b)) => {
            if a.len() != b.len() || a.keys().ne(b.keys()) {
                return Some(format!(
                    "{path}: keys {:?} vs {:?}",
                    a.keys().collect::<Vec<_>>(),
                    b.keys().collect::<Vec<_>>()
                ));
            }
            a.iter()
                .find_map(|(k, v)| compare(v, &b[k], tol, &format!("{path}.{k}")))
        }
        (a, b) => (a != b).then(|| format!("{path}: {a} vs {b}")),
    }
}
