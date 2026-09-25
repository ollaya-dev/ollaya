//! Compare the `decision-endpoint-v1` runtime against golden fixtures from
//! `ollaya_convert.families.decision.goldens` (the author's code in fp32, TF32 off, exact SDPA).
//!
//!     cargo run --release -p ollaya-runner --example parity_decision -- <model-dir> <goldens.jsonl> [cpu|cuda] [--latency]
//!
//! Every case is encoded first, before anything runs through the graph:
//! * a request the author rejects must be rejected, and each of its questions must be rejected on
//!   its own exactly when the author rejects it (its `#valid` record holds the accepted ones);
//! * per question: type, option count, token ids, query position and candidate positions must
//!   match exactly.
//!
//! Then every case runs through the graph:
//! * each question's option logits (raw candidate logits) must be within `LOGIT_TOL`;
//! * calibrated probabilities must pick the reference's option (max / p99 difference reported);
//! * the TypeSafe answers must match the author's `typed_answer` (choice, noul, expected score,
//!   confidence, legend and probabilities). Ollaya rounds wire numbers to 4 decimals and the author
//!   does not, so they may differ by half a step.
//!
//! `--latency` then times full requests (`run`: tokenize, encode, forward) of 5 questions.

use std::collections::HashMap;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::{Answer, Questions};
use ollaya_runner::Device;
use ollaya_runner::decision::{DecisionEncoding, DecisionModel};
use serde_json::Value;

/// Largest raw-logit difference accepted (ONNX Runtime vs PyTorch fp32, TF32 off).
const LOGIT_TOL: f64 = 1e-3;
/// Ollaya rounds wire numbers to 4 decimals, the author does not: half a step plus float noise.
const WIRE_TOL: f64 = 1e-4 + 1e-9;

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

/// Does the runtime reject this request (parsing or encoding)?
fn rejects(model: &DecisionModel, state: &Value, questions: &Value) -> bool {
    match ollaya_decision::parse_questions(questions) {
        Ok(qs) => model.encode(state, &qs).is_err(),
        Err(_) => true,
    }
}

/// Largest difference between two TypeSafe answers' numbers, or `None` if their decisions,
/// shapes or legends differ.
fn wire_diff(ours: &Value, upstream: &Value) -> Option<f64> {
    let num = |v: &Value, k: &str| v[k].as_f64();
    if ours["type"] != upstream["type"] {
        return None;
    }
    let mut d = 0f64;
    match ours["type"].as_str()? {
        "noul" => return Some((num(ours, "noul")? - num(upstream, "noul")?).abs()),
        "choice" => {
            if ours["choice"] != upstream["choice"] {
                return None;
            }
        }
        "score" => {
            if ours["legend"] != upstream["legend"] {
                return None;
            }
            d = d.max((num(ours, "score")? - num(upstream, "score")?).abs());
        }
        _ => return None,
    }
    d = d.max((num(ours, "confidence")? - num(upstream, "confidence")?).abs());
    let (a, b) = (
        ours["probabilities"].as_object()?,
        upstream["probabilities"].as_object()?,
    );
    if !a.keys().eq(b.keys()) {
        return None;
    }
    for (x, y) in a.values().zip(b.values()) {
        d = d.max((x.as_f64()? - y.as_f64()?).abs());
    }
    Some(d)
}

struct Case {
    id: String,
    rec: Value,
    questions: Questions,
    encoding: DecisionEncoding,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: parity_decision <model-dir> <goldens.jsonl> [cpu|cuda] [--latency]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let latency = args.iter().any(|a| a == "--latency");
    let t = Instant::now();
    let model = DecisionModel::load(&PathBuf::from(&args[1]), device, None)?;
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

    // Encoding: rejections, token rows, query and candidate positions.
    let (mut rej_bad, mut enc_bad, mut rejected) = (0, 0, 0);
    let mut cases = Vec::new();
    for rec in &records {
        let id = rec["id"].as_str().unwrap_or("?").to_owned();
        let state = &rec["state"];
        if !rec["error"].is_null() {
            rejected += 1;
            if !rejects(&model, state, &rec["questions"]) {
                rej_bad += 1;
                println!(
                    "REJECTION {id}: the author rejects the request ({})",
                    rec["error"]
                );
            }
            let valid = by_id
                .get(format!("{id}#valid").as_str())
                .and_then(|r| r["questions"].as_object());
            for (qid, def) in rec["questions"].as_object().context("questions")? {
                let upstream = !valid.is_some_and(|v| v.contains_key(qid));
                let one = Value::Object([(qid.clone(), def.clone())].into_iter().collect());
                if rejects(&model, state, &one) != upstream {
                    rej_bad += 1;
                    println!("REJECTION {id} {qid}: the author rejects={upstream}");
                }
            }
            continue;
        }
        let questions = match ollaya_decision::parse_questions(&rec["questions"]) {
            Ok(q) => q,
            Err(e) => {
                rej_bad += 1;
                println!("REJECTION {id}: the author accepts, the runtime rejects: {e}");
                continue;
            }
        };
        let encoding = match model.encode(state, &questions) {
            Ok(e) => e,
            Err(e) => {
                rej_bad += 1;
                println!("REJECTION {id}: the author accepts, the runtime rejects: {e}");
                continue;
            }
        };
        let gold_rows = rec["rows"].as_array().context("rows")?;
        let plan = rec["plan"].as_array().context("plan")?;
        if gold_rows.len() != encoding.rows.len() || plan.len() != questions.len() {
            enc_bad += 1;
            println!(
                "ENCODING {id}: {} rows vs {}",
                encoding.rows.len(),
                gold_rows.len()
            );
            continue;
        }
        for (((qid, q), row), (gold, p)) in questions
            .iter()
            .zip(&encoding.rows)
            .zip(gold_rows.iter().zip(plan))
        {
            let ids: Vec<u32> = serde_json::from_value(gold["ids"].clone())?;
            let cands: Vec<usize> = serde_json::from_value(gold["cands"].clone())?;
            let query = gold["query"].as_u64().context("query")? as usize;
            if p["qid"] != qid.as_str()
                || p["type"] != q.qtype.name()
                || p["k"].as_u64() != Some(row.cands.len() as u64)
                || ids != row.ids
                || cands != row.cands
                || query != row.query
            {
                enc_bad += 1;
                if enc_bad <= 5 {
                    let first = ids.iter().zip(&row.ids).position(|(a, b)| a != b);
                    println!(
                        "ENCODING {id} {qid}: {} tokens vs {}, first differing token {first:?}, \
                         query {} vs {query}, cands {:?} vs {cands:?}",
                        row.ids.len(),
                        ids.len(),
                        row.query,
                        row.cands
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
    let nq: usize = cases.iter().map(|c| c.questions.len()).sum();
    let tokens: usize = cases
        .iter()
        .flat_map(|c| &c.encoding.rows)
        .map(|r| r.ids.len())
        .sum();
    println!(
        "encoding: {} cases ({rejected} rejected by the author), {nq} questions / rows, {tokens} tokens \
         | rejection mismatches: {rej_bad} | encoding mismatches: {enc_bad}",
        records.len()
    );
    if rej_bad + enc_bad > 0 {
        bail!("encoding parity failed");
    }

    // Numbers: option logits, calibrated probabilities, wire answers.
    let (mut logit_max, mut wire_max, mut disagree, mut wire_bad) = (0f64, 0f64, 0, 0);
    let mut prob_diffs = Vec::new();
    let mut forward = 0f64;
    for case in &cases {
        let t = Instant::now();
        let logits = model.logits(&case.encoding)?;
        forward += t.elapsed().as_secs_f64();
        let plan = case.rec["plan"].as_array().context("plan")?;
        for (((qid, q), got), p) in case.questions.iter().zip(&logits).zip(plan) {
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
            let answer = Answer::new(q, &model.calibration, got, None, case.encoding.state_tokens);
            let probs = &answer.probabilities;
            prob_diffs.push(
                want.iter()
                    .zip(probs)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max),
            );
            if argmax(&want) != argmax(probs) {
                disagree += 1;
                if disagree <= 5 {
                    println!(
                        "DECISION {} {qid}: {probs:.4?} vs reference {want:.4?}",
                        case.id
                    );
                }
            }
            let upstream = &case.rec["answers"][qid.as_str()];
            match wire_diff(&answer.to_typesafe(q), upstream) {
                Some(d) => wire_max = wire_max.max(d),
                None => {
                    wire_bad += 1;
                    if wire_bad <= 5 {
                        println!(
                            "ANSWER {} {qid}: {} vs the author's {upstream}",
                            case.id,
                            answer.to_typesafe(q)
                        );
                    }
                }
            }
        }
    }
    prob_diffs.sort_by(f64::total_cmp);
    println!(
        "numbers: option logit diff max {logit_max:.1e} | decisions agree: {:.2}% ({disagree} differ) \
         | prob diff max {:.1e} p99 {:.1e} | wire answers vs the author: max diff {wire_max:.1e}, \
         {wire_bad} differ | forward {forward:.1}s ({:.0} ms/request)",
        100.0 * (nq - disagree) as f64 / nq.max(1) as f64,
        prob_diffs.last().copied().unwrap_or(0.0),
        percentile(&prob_diffs, 99),
        1000.0 * forward / cases.len().max(1) as f64,
    );
    if logit_max > LOGIT_TOL {
        bail!("logits differ by more than {LOGIT_TOL:e}");
    }
    if disagree > 0 {
        bail!("decisions differ from the reference");
    }
    if wire_bad > 0 || wire_max > WIRE_TOL {
        bail!("answers differ from the author's typed_answer");
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
        let tokens: usize = five
            .iter()
            .flat_map(|c| &c.encoding.rows)
            .map(|r| r.ids.len())
            .sum();
        println!(
            "latency, 5-question requests (n={}, {:.0} tokens/request): p50 {:.1} ms  p95 {:.1} ms  max {:.1} ms",
            ms.len(),
            tokens as f64 / five.len().max(1) as f64,
            percentile(&ms, 50),
            percentile(&ms, 95),
            ms.last().copied().unwrap_or(0.0)
        );
    }
    Ok(())
}
