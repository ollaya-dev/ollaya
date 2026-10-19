//! Compare the `clef-joint-v1` runtime against golden fixtures from
//! `ollaya_convert.families.clef.goldens` (the authors' `joint_schema_model.py`: `encode_record` and
//! their model with the joint schema head, fp32).
//!
//!     cargo run --release -p ollaya-runner --example parity_clef -- <model-dir> <goldens.jsonl> [cpu|cuda] [--latency] [--fp16]
//!
//! Every case is checked for its sequence first:
//! * a request upstream rejects must be rejected, and each of its questions exactly when upstream
//!   rejects it on its own. Ollaya's shared question rules (such as required `instructions`) are
//!   counted separately;
//! * the token ids, every question span, option span and option id must match exactly.
//!
//! Then every case runs through the graph:
//! * each question's option logits (upstream's option order) must be within `LOGIT_TOL`;
//! * the answer probabilities (the engine's output, in the caller's option order) must pick the
//!   reference's option (max / p99 difference reported).
//!
//! `--latency` then times every accepted request again (p50, p95, and by question count).

use std::collections::HashMap;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::{Answer, Criteria, Question};
use ollaya_runner::Device;
use ollaya_runner::clef::ClefModel;
use ollaya_runner::engine::Engine;
use serde_json::Value;

/// Largest logit difference accepted (ONNX Runtime vs PyTorch, both fp32 on the BF16 weights).
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

fn percentile(sorted: &[f64], p: usize) -> f64 {
    sorted
        .get((sorted.len() * p / 100).min(sorted.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0)
}

/// Why the runtime rejects a request, or `None`: `Some(true)` for Ollaya's shared question rules,
/// `Some(false)` for the layout's.
fn rejects(model: &ClefModel, state: &Value, questions: &Value) -> Option<bool> {
    let parsed = match ollaya_decision::parse_questions(questions) {
        Ok(q) => q,
        Err(_) => return Some(true),
    };
    let defs: Vec<(&str, &Value)> = parsed
        .iter()
        .map(|(k, q)| (k.as_str(), &q.definition))
        .collect();
    model.row(state, &defs).err().map(|_| false)
}

/// The reference probabilities in the order answers use (the caller's labels, noul [false, true]).
fn in_answer_order(q: &Question, option_ids: &[String], probs: &[f64]) -> Vec<f64> {
    let labels: Vec<String> = match &q.criteria {
        Criteria::Choice(m) => m.keys().cloned().collect(),
        Criteria::Noul { .. } => vec!["false".into(), "true".into()],
        Criteria::Score(levels) => (0..levels.len()).map(|i| i.to_string()).collect(),
    };
    labels
        .iter()
        .map(|l| {
            option_ids
                .iter()
                .position(|o| o == l)
                .map_or(f64::NAN, |i| probs[i])
        })
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: parity_clef <model-dir> <goldens.jsonl> [cpu|cuda] [--latency] [--fp16]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let latency = args.iter().any(|a| a == "--latency");
    // An fp16 graph's gate (docs/decisions/0008-fp16-decoder-tags.md): the encoding stays exact, at least 99.5%
    // of decisions match the fp32 reference, and the numbers are reported rather than bounded.
    let fp16 = args.iter().any(|a| a == "--fp16");
    let t = Instant::now();
    let model = ClefModel::load(&PathBuf::from(&args[1]), device, None)?;
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

    // Sequences: rejections, token ids, spans.
    let (mut rej_bad, mut row_bad, mut rejected, mut ollaya_rule) = (0, 0, 0, 0);
    let mut cases = Vec::new();
    for rec in &records {
        let id = rec["id"].as_str().unwrap_or("?").to_owned();
        let state = &rec["state"];
        if !rec["error"].is_null() {
            rejected += 1;
            if rejects(&model, state, &rec["questions"]).is_none() {
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
                let upstream = !valid.is_some_and(|v| v.contains_key(qid));
                let one = Value::Object([(qid.clone(), def.clone())].into_iter().collect());
                match (rejects(&model, state, &one), upstream) {
                    (Some(_), true) | (None, false) => {}
                    (Some(true), false) => ollaya_rule += 1,
                    (r, _) => {
                        rej_bad += 1;
                        println!(
                            "REJECTION {id} {qid}: upstream rejects={upstream}, runtime {r:?}"
                        );
                    }
                }
            }
            continue;
        }
        match rejects(&model, state, &rec["questions"]) {
            None => {}
            Some(true) => {
                ollaya_rule += 1;
                continue;
            }
            Some(false) => {
                rej_bad += 1;
                println!("REJECTION {id}: upstream accepts, the layout rejects");
                continue;
            }
        }
        let questions = ollaya_decision::parse_questions(&rec["questions"])?;
        let defs: Vec<(&str, &Value)> = questions
            .iter()
            .map(|(k, q)| (k.as_str(), &q.definition))
            .collect();
        let row = model.row(state, &defs)?;
        let gold = &rec["row"];
        let ids: Vec<u32> = serde_json::from_value(gold["ids"].clone())?;
        let mut same = ids == row.ids;
        let gq = gold["questions"].as_array().context("row.questions")?;
        same &= gq.len() == row.questions.len();
        for (g, q) in gq.iter().zip(&row.questions) {
            let span: (usize, usize) = serde_json::from_value(g["span"].clone())?;
            let options: Vec<(usize, usize)> = serde_json::from_value(g["options"].clone())?;
            let option_ids: Vec<String> = serde_json::from_value(g["option_ids"].clone())?;
            same &= g["qid"] == q.qid.as_str()
                && span == q.span
                && options == q.options
                && option_ids == q.option_ids;
        }
        if !same {
            row_bad += 1;
            if row_bad <= 5 {
                let at = ids.iter().zip(&row.ids).take_while(|(a, b)| a == b).count();
                println!(
                    "SEQUENCE {id}: {} vs {} ids, first difference at {at}",
                    row.ids.len(),
                    ids.len()
                );
            }
        }
        cases.push((id, rec, questions, row));
    }
    let nq: usize = cases.iter().map(|c| c.2.len()).sum();
    println!(
        "sequences: {} cases ({rejected} rejected upstream), {nq} questions | rejection mismatches: {rej_bad} \
         | sequence mismatches: {row_bad} | refused by Ollaya's shared question rules: {ollaya_rule}",
        records.len()
    );
    if rej_bad + row_bad > 0 {
        bail!("sequence parity failed");
    }

    // Numbers.
    let (mut logit_max, mut disagree) = (0f64, 0);
    let mut prob_diffs = Vec::new();
    let mut forward = 0f64;
    for (id, rec, questions, row) in &cases {
        let t = Instant::now();
        let raw = model.logits(row)?;
        forward += t.elapsed().as_secs_f64();
        let out = model.run(&rec["state"], questions)?;
        let plan = rec["plan"].as_array().context("plan")?;
        for ((((qid, q), z), got), p) in questions.iter().zip(&raw).zip(&out.questions).zip(plan) {
            let want: Vec<f64> = serde_json::from_value(p["option_logits"].clone())?;
            if z.len() != want.len() {
                bail!("LOGITS {id} {qid}: {} options vs {}", z.len(), want.len());
            }
            let d = z
                .iter()
                .zip(&want)
                .map(|(&x, &y)| (f64::from(x) - y).abs())
                .fold(0.0, f64::max);
            logit_max = logit_max.max(d);
            let option_ids: Vec<String> = serde_json::from_value(p["option_ids"].clone())?;
            let gold: Vec<f64> = serde_json::from_value(p["probabilities"].clone())?;
            let want = in_answer_order(q, &option_ids, &gold);
            let answer = Answer::new(q, &model.calibration, &got.logits, None, 0);
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
                    println!("DECISION {id} {qid}: {probs:.4?} vs reference {want:.4?}");
                }
            }
        }
    }
    prob_diffs.sort_by(f64::total_cmp);
    println!(
        "numbers: option logit diff max {logit_max:.1e} | decisions agree: {:.2}% ({disagree} differ) \
         | prob diff max {:.1e} p99 {:.1e} \
         | forward {forward:.1}s ({:.0} ms/request)",
        100.0 * (nq - disagree) as f64 / nq.max(1) as f64,
        prob_diffs.last().copied().unwrap_or(0.0),
        percentile(&prob_diffs, 99),
        1000.0 * forward / cases.len().max(1) as f64,
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
    if !fp16 && logit_max > LOGIT_TOL || prob_diffs.last().is_some_and(|d| d.is_nan()) {
        bail!("logits differ by more than {LOGIT_TOL:e}");
    }
    if disagree > if fp16 { nq / 200 } else { 0 } {
        bail!("decisions differ from the reference");
    }

    if latency {
        let mut ms = Vec::new();
        for (_, rec, questions, _) in &cases {
            model.run(&rec["state"], questions)?;
            let t = Instant::now();
            model.run(&rec["state"], questions)?;
            ms.push((questions.len(), t.elapsed().as_secs_f64() * 1000.0));
        }
        let mut all: Vec<f64> = ms.iter().map(|m| m.1).collect();
        all.sort_by(f64::total_cmp);
        println!(
            "latency, every request (n={}): p50 {:.1} ms  p95 {:.1} ms",
            all.len(),
            percentile(&all, 50),
            percentile(&all, 95)
        );
        let mut five: Vec<f64> = ms.iter().filter(|m| m.0 == 5).map(|m| m.1).collect();
        five.sort_by(f64::total_cmp);
        if !five.is_empty() {
            println!(
                "latency, 5-question requests (n={}): p50 {:.1} ms  p95 {:.1} ms",
                five.len(),
                percentile(&five, 50),
                percentile(&five, 95)
            );
        }
    }
    Ok(())
}
