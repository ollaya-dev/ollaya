//! Compare the `decider-slots-v1` runtime against golden fixtures from
//! `ollaya_convert.families.decider.goldens` (upstream decider in fp32, TF32 off).
//!
//!     cargo run --release -p ollaya-runner --example parity_decider -- <model-dir> <goldens.jsonl> [cpu|cuda] [--latency]
//!
//! Every case is encoded first, before anything runs through the graph:
//! * a request upstream rejects must be rejected, and each of its questions must be rejected on
//!   its own exactly when upstream rejects it (its `#valid` record holds the accepted ones);
//! * per question: type, readout, option count, and every row's token ids and slot position must
//!   match exactly.
//!
//! Then every case runs through the graph:
//! * each row's label logits and each question's option logits must be within `LOGIT_TOL`;
//! * calibrated probabilities must pick the reference's option (max / p99 difference reported).
//!
//! A score question whose criteria are an object (a legend) is skipped and counted: upstream
//! accepts it, but it is not TypeSafe wire, so the API rejects it before it reaches a runner.
//!
//! `--latency` then times full requests (`run`: tokenize, encode, forward) of 5 questions.

use std::collections::HashMap;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::{Question, Questions};
use ollaya_runner::Device;
use ollaya_runner::decider::{DeciderEncoding, DeciderModel};
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

fn percentile(sorted: &[f64], p: usize) -> f64 {
    sorted
        .get((sorted.len() * p / 100).min(sorted.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0)
}

/// Upstream accepts a score legend object; TypeSafe wire (and so Ollaya's parser) does not.
fn is_legend(def: &Value) -> bool {
    def["type"] == "score" && def["criteria"].is_object()
}

/// The questions Ollaya parses, minus legend objects (counted in `skipped`). `None` if a
/// question fails to parse for any other reason.
fn parse(questions: &Value, skipped: &mut usize) -> Option<Questions> {
    let mut out = Questions::new();
    for (qid, def) in questions.as_object()? {
        match Question::parse(qid, def) {
            Ok(q) => {
                out.insert(qid.clone(), q);
            }
            Err(_) if is_legend(def) => *skipped += 1,
            Err(_) => return None,
        }
    }
    Some(out)
}

/// Does the runtime reject this request (parsing or encoding)?
fn rejects(model: &DeciderModel, state: &Value, questions: &Value) -> bool {
    match ollaya_decision::parse_questions(questions) {
        Ok(qs) => model.encode(state, &qs).is_err(),
        Err(_) => true,
    }
}

struct Case {
    id: String,
    rec: Value,
    questions: Questions,
    encoding: DeciderEncoding,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: parity_decider <model-dir> <goldens.jsonl> [cpu|cuda] [--latency]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let latency = args.iter().any(|a| a == "--latency");
    let t = Instant::now();
    let model = DeciderModel::load(&PathBuf::from(&args[1]), device, None)?;
    println!("load {:.1}s on {device:?}", t.elapsed().as_secs_f64());

    let file = std::fs::File::open(&args[2]).with_context(|| args[2].clone())?;
    let mut records = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        records.push(serde_json::from_str::<Value>(&line?)?);
    }
    let by_id: HashMap<&str, &Value> = records
        .iter()
        .map(|r| (r["id"].as_str().unwrap_or("?"), r))
        .collect();

    // Encoding: rejections, token rows and slots.
    let (mut rej_bad, mut enc_bad, mut skipped, mut rejected) = (0, 0, 0, 0);
    let mut cases = Vec::new();
    for rec in &records {
        let id = rec["id"].as_str().unwrap_or("?").to_owned();
        let state = &rec["state"];
        if !rec["error"].is_null() {
            rejected += 1;
            if !rejects(&model, state, &rec["questions"]) {
                rej_bad += 1;
                println!(
                    "REJECTION {id}: upstream rejects the request ({})",
                    rec["error"]
                );
            }
            let valid = by_id
                .get(format!("{id}#valid").as_str())
                .and_then(|r| r["questions"].as_object());
            for (qid, def) in rec["questions"].as_object().context("questions")? {
                if is_legend(def) {
                    skipped += 1;
                    continue;
                }
                let upstream = !valid.is_some_and(|v| v.contains_key(qid));
                let one = Value::Object([(qid.clone(), def.clone())].into_iter().collect());
                if rejects(&model, state, &one) != upstream {
                    rej_bad += 1;
                    println!("REJECTION {id} {qid}: upstream rejects={upstream}");
                }
            }
            continue;
        }
        let Some(questions) = parse(&rec["questions"], &mut skipped) else {
            rej_bad += 1;
            println!("REJECTION {id}: a question upstream accepts does not parse");
            continue;
        };
        if questions.is_empty() {
            continue;
        }
        let encoding = match model.encode(state, &questions) {
            Ok(e) => e,
            Err(e) => {
                rej_bad += 1;
                println!("REJECTION {id}: upstream accepts, the runtime rejects: {e}");
                continue;
            }
        };
        let gold_rows = rec["rows"].as_array().context("rows")?;
        let plan = rec["plan"].as_array().context("plan")?;
        for ((qid, q), scoring) in questions.iter().zip(&encoding.questions) {
            let p = plan
                .iter()
                .find(|p| p["qid"] == qid.as_str())
                .with_context(|| format!("{id}: no plan for {qid}"))?;
            let idx: Vec<usize> = serde_json::from_value(p["rows"].clone())?;
            let want = idx
                .iter()
                .map(|&r| {
                    let ids: Vec<u32> = serde_json::from_value(gold_rows[r]["ids"].clone())?;
                    let slot = gold_rows[r]["slot"].as_u64().context("slot")? as usize;
                    Ok((ids, slot))
                })
                .collect::<Result<Vec<_>>>()?;
            let same_rows = want.len() == scoring.rows.len()
                && want
                    .iter()
                    .zip(&scoring.rows)
                    .all(|((ids, slot), row)| ids == row && *slot == row.len() - 1);
            if p["type"] != q.qtype.name()
                || p["kind"] != scoring.readout.name()
                || p["k"].as_u64() != Some(scoring.options as u64)
                || !same_rows
            {
                enc_bad += 1;
                if enc_bad <= 5 {
                    let first = want
                        .iter()
                        .zip(&scoring.rows)
                        .find_map(|((ids, _), row)| ids.iter().zip(row).position(|(a, b)| a != b));
                    println!(
                        "ENCODING {id} {qid}: {} {} k={} vs {} {} k={}, {} vs {} rows, first differing token {first:?}",
                        q.qtype.name(),
                        scoring.readout.name(),
                        scoring.options,
                        p["type"],
                        p["kind"],
                        p["k"],
                        scoring.rows.len(),
                        want.len()
                    );
                }
            }
        }
        cases.push(Case {
            id,
            rec: rec.clone(),
            questions,
            encoding,
        });
    }
    let rows: usize = cases
        .iter()
        .flat_map(|c| &c.encoding.questions)
        .map(|s| s.rows.len())
        .sum();
    let nq: usize = cases.iter().map(|c| c.questions.len()).sum();
    println!(
        "encoding: {} cases ({rejected} rejected upstream), {nq} questions, {rows} rows | \
         rejection mismatches: {rej_bad} | encoding mismatches: {enc_bad} | skipped legend objects: {skipped}",
        records.len()
    );
    if rej_bad + enc_bad > 0 {
        bail!("encoding parity failed");
    }

    // Numbers: label logits, option logits, calibrated probabilities.
    let (mut label_max, mut logit_max, mut disagree) = (0f64, 0f64, 0);
    let mut prob_diffs = Vec::new();
    let mut forward = 0f64;
    for case in &cases {
        let t = Instant::now();
        let logits = model.label_logits(&case.encoding)?;
        forward += t.elapsed().as_secs_f64();
        let out = model.output(&case.encoding, &logits);
        let plan = case.rec["plan"].as_array().context("plan")?;
        let mut next = 0;
        for (r, (qid, q)) in case.questions.iter().enumerate() {
            let scoring = &case.encoding.questions[r];
            let row_logits = &logits[next..next + scoring.rows.len()];
            next += scoring.rows.len();
            let p = plan
                .iter()
                .find(|p| p["qid"] == qid.as_str())
                .context("plan")?;
            let want_rows: Vec<Vec<f64>> = serde_json::from_value(p["label_logits"].clone())?;
            for (got, want) in row_logits.iter().zip(&want_rows) {
                label_max = label_max.max(max_diff(&got[..want.len()], want));
            }
            let got = &out.questions[r].logits;
            let want: Vec<f64> = serde_json::from_value(p["option_logits"].clone())?;
            if got.len() != want.len() {
                bail!(
                    "LOGITS {} {qid}: {} options vs {}",
                    case.id,
                    got.len(),
                    want.len()
                );
            }
            logit_max = logit_max.max(max_diff(got, &want));

            let want: Vec<f64> = serde_json::from_value(p["probabilities"].clone())?;
            let probs = model.calibration.probabilities(q.qtype, got);
            prob_diffs.push(
                want.iter()
                    .zip(&probs)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max),
            );
            if argmax(&want) != argmax(&probs) {
                disagree += 1;
                if disagree <= 5 {
                    println!(
                        "DECISION {} {qid}: {probs:.4?} vs reference {want:.4?}",
                        case.id
                    );
                }
            }
        }
    }
    prob_diffs.sort_by(f64::total_cmp);
    println!(
        "numbers: label logit diff max {label_max:.1e}, option logit diff max {logit_max:.1e} \
         | decisions agree: {:.2}% ({disagree} differ) | prob diff max {:.1e} p99 {:.1e} \
         | forward {forward:.1}s ({:.0} ms/request)",
        100.0 * (nq - disagree) as f64 / nq.max(1) as f64,
        prob_diffs.last().copied().unwrap_or(0.0),
        percentile(&prob_diffs, 99),
        1000.0 * forward / cases.len().max(1) as f64,
    );
    if label_max.max(logit_max) > LOGIT_TOL {
        bail!("logits differ by more than {LOGIT_TOL:e}");
    }
    if disagree > 0 {
        bail!("decisions differ from the reference");
    }

    if latency {
        let five: Vec<&Case> = cases.iter().filter(|c| c.questions.len() == 5).collect();
        let mut ms = Vec::new();
        for _ in 0..2 {
            for c in &five {
                let t = Instant::now();
                model.run(&c.rec["state"], &c.questions)?;
                ms.push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        ms.sort_by(f64::total_cmp);
        let rows: usize = five
            .iter()
            .flat_map(|c| &c.encoding.questions)
            .map(|s| s.rows.len())
            .sum();
        let tokens: usize = five
            .iter()
            .flat_map(|c| &c.encoding.questions)
            .flat_map(|s| &s.rows)
            .map(Vec::len)
            .sum();
        println!(
            "latency, 5-question requests (n={}, {:.1} rows and {:.0} tokens/request): p50 {:.1} ms  p95 {:.1} ms  max {:.1} ms",
            ms.len(),
            rows as f64 / five.len().max(1) as f64,
            tokens as f64 / five.len().max(1) as f64,
            percentile(&ms, 50),
            percentile(&ms, 95),
            ms.last().copied().unwrap_or(0.0)
        );
    }
    Ok(())
}
