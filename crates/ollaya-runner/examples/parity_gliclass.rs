//! Compare the `gliclass-uni-v1` engine against goldens from the Python reference
//! (`ollaya_convert.families.gliclass.goldens`).
//!
//!     cargo run --release -p ollaya-runner --example parity_gliclass -- <model-dir> <goldens.jsonl> [cpu|cuda]
//!
//! Per question:
//! * the upstream call (labels, prompt, mode) and input text must match exactly;
//! * encoder token ids and `<<LABEL>>` positions must match exactly;
//! * option logits must be within `LOGIT_TOL` of the fp32 reference;
//! * calibrated probabilities must pick the same option (max / p99 absolute difference reported).
//!
//! A question the reference rejects must fail the whole request with `TooManyOptions` naming it;
//! the request's other questions are then checked without it.

use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::gliclass::{Call, GliclassLayout, GliclassTokens};
use ollaya_decision::parse_questions;
use ollaya_runner::gliclass::GliclassModel;
use ollaya_runner::{Device, Error};
use serde_json::Value;

/// Max absolute option-logit difference against the fp32 reference.
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

fn p99(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() * 99 / 100).copied().unwrap_or(0.0)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: parity_gliclass <model-dir> <goldens.jsonl> [cpu|cuda]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let dir = PathBuf::from(&args[1]);
    let model = GliclassModel::load(&dir, device, None)?;
    let decision: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("decision.json"))?)?;
    let layout = GliclassLayout {
        max_len: serde_json::from_value(decision["max_len"].clone())?,
        special: serde_json::from_value::<GliclassTokens>(decision["special_tokens"].clone())?,
        sanitize: serde_json::from_value(decision["sanitize"].clone())?,
    };
    let file = std::fs::File::open(&args[2]).with_context(|| args[2].clone())?;

    let (mut cases, mut questions, mut call_bad, mut enc_bad, mut logit_bad, mut disagree) =
        (0, 0, 0, 0, 0, 0);
    let (mut rejected, mut reject_bad) = (0, 0);
    let (mut logit_diffs, mut prob_diffs) = (Vec::new(), Vec::new());
    let mut run_time = std::time::Duration::ZERO;
    for line in std::io::BufReader::new(file).lines() {
        let rec: Value = serde_json::from_str(&line?)?;
        let id = rec["id"].as_str().unwrap_or("?").to_owned();
        let mut qs = parse_questions(&rec["questions"])?;
        cases += 1;

        let gold_rejected = rec["rejected"].as_object().cloned().unwrap_or_default();
        if !gold_rejected.is_empty() {
            rejected += gold_rejected.len();
            let first = qs.keys().find(|q| gold_rejected.contains_key(*q)).cloned();
            match model.encode(&rec["state"], &qs) {
                Err(Error::Decision(ollaya_decision::Error::TooManyOptions {
                    question, ..
                })) if Some(&question) == first.as_ref() => {}
                other => {
                    reject_bad += 1;
                    println!("REJECT {id}: expected TooManyOptions for {first:?}, got {other:?}");
                }
            }
            qs.retain(|q, _| !gold_rejected.contains_key(q));
            if qs.is_empty() {
                continue;
            }
        }

        let state_text = layout.state_text(&rec["state"]);
        if state_text != rec["state_text"].as_str().unwrap_or_default() {
            println!("STATE {id}: state text differs");
            call_bad += 1;
        }
        let encoded = model.encode(&rec["state"], &qs)?;
        let t0 = Instant::now();
        let out = model.run_encoded(&encoded, &qs)?;
        run_time += t0.elapsed();

        let items = rec["items"].as_array().context("items")?;
        if items.len() != qs.len() {
            bail!(
                "{id}: {} golden items for {} questions",
                items.len(),
                qs.len()
            );
        }
        for (r, ((qid, q), gold)) in qs.iter().zip(items).enumerate() {
            questions += 1;
            if gold["qid"] != qid.as_str() {
                bail!("{id}: golden item {r} is {} not {qid}", gold["qid"]);
            }
            let call = Call::new(q);
            let labels: Vec<String> = call.labels.iter().map(|l| layout.sanitize(l)).collect();
            let want_labels: Vec<String> = serde_json::from_value(gold["labels"].clone())?;
            if labels != want_labels
                || layout.sanitize(&call.prompt) != gold["prompt"].as_str().unwrap_or_default()
                || call.mode.name() != gold["mode"].as_str().unwrap_or_default()
                || layout.text(&state_text, &call) != gold["text"].as_str().unwrap_or_default()
            {
                call_bad += 1;
                if call_bad <= 5 {
                    println!("CALL {id} {qid}: {call:?}");
                }
            }

            let row = &encoded.rows[r];
            let ids: Vec<u32> = serde_json::from_value(gold["ids"].clone())?;
            let markers: Vec<usize> = serde_json::from_value(gold["markers"].clone())?;
            if row.ids != ids || row.markers != markers {
                enc_bad += 1;
                if enc_bad <= 5 {
                    let first = row.ids.iter().zip(&ids).position(|(a, b)| a != b);
                    println!(
                        "ENCODING {id} {qid}: len {} vs {}, markers {:?} vs {:?}, first diff at {first:?}",
                        row.ids.len(),
                        ids.len(),
                        row.markers,
                        markers
                    );
                }
                continue;
            }

            let got: Vec<f64> = out.questions[r]
                .logits
                .iter()
                .map(|&z| f64::from(z))
                .collect();
            let want: Vec<f64> = serde_json::from_value(gold["option_logits"].clone())?;
            let d = max_diff(&got, &want);
            logit_diffs.push(d);
            if d > LOGIT_TOL {
                logit_bad += 1;
                if logit_bad <= 5 {
                    println!("LOGITS {id} {qid}: {got:.5?} vs reference {want:.5?}");
                }
            }

            let p = model.calibration.probabilities(
                q.qtype,
                &out.questions[r].logits,
                out.state_tokens,
            );
            let want_p: Vec<f64> = serde_json::from_value(gold["probabilities"].clone())?;
            prob_diffs.push(max_diff(&p, &want_p));
            if argmax(&p) != argmax(&want_p) {
                disagree += 1;
                if disagree <= 5 {
                    println!("DECISION {id} {qid}: {p:.4?} vs reference {want_p:.4?}");
                }
            }
        }
    }

    let compared = questions - enc_bad;
    println!(
        "{cases} cases, {questions} questions (+{rejected} rejected, {reject_bad} wrong) | call/text mismatches: {call_bad} \
         | encoding mismatches: {enc_bad} | logit diff max {:.1e} p99 {:.1e} ({logit_bad} > {LOGIT_TOL:.0e}) \
         | decisions agree: {:.2}% ({disagree} differ) | prob diff max {:.1e} p99 {:.1e} | run {:.2}s",
        logit_diffs.iter().copied().fold(0.0, f64::max),
        p99(&mut logit_diffs),
        100.0 * (compared - disagree) as f64 / compared.max(1) as f64,
        prob_diffs.iter().copied().fold(0.0, f64::max),
        p99(&mut prob_diffs),
        run_time.as_secs_f64(),
    );
    if call_bad + enc_bad + reject_bad + logit_bad + disagree > 0 {
        bail!("gliclass parity failed");
    }
    Ok(())
}
