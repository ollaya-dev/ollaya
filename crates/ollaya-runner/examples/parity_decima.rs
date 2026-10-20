//! Parity of the `decima-late-interaction-v1` and `-v2` runtime against the goldens of
//! `ollaya_convert.families.decima.goldens` (the author's `decima/model.py` and `systemone.py`,
//! fp32).
//!
//!     cargo run --release -p ollaya-runner --example parity_decima -- <model-dir> <goldens.jsonl> [cpu|cuda] [--latency]
//!
//! Every case is checked for its inputs first:
//! * a request upstream rejects for an invalid question must be rejected, and each of its
//!   questions exactly when upstream rejects it on its own;
//! * every question's state row and option rows must be identical, and so must the state's
//!   truncation (upstream's server answers `422 STATE_TRUNCATED`, which `/v1/systemone` reports
//!   from this flag).
//!
//! Then every accepted case runs through the graph:
//! * each option's raw score and ordinal projections must be within `SCORE_TOL`;
//! * the answer probabilities (calibrated as the daemon calibrates them, in the caller's option
//!   order) must pick the reference's option, and the `/v1/systemone` answers must name the same
//!   choice as upstream's `system_one` (max / p99 differences reported).
//!
//! `--latency` then times every accepted request again (p50, p95, and the five-question requests).

use std::collections::HashMap;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::{Answer, QType, Question};
use ollaya_runner::Device;
use ollaya_runner::decima::DecimaModel;
use ollaya_runner::engine::Engine;
use serde_json::Value;

/// Largest difference accepted on the graph's outputs (ONNX Runtime vs PyTorch, both fp32).
const SCORE_TOL: f64 = 1e-3;

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

fn max_diff(a: &[f32], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - y).abs())
        .fold(0.0, f64::max)
}

/// Why the runtime rejects a request, or `None`: `Some(true)` for Ollaya's shared question rules,
/// `Some(false)` for the layout's.
fn rejects(model: &DecimaModel, state: &Value, questions: &Value) -> Option<bool> {
    match ollaya_decision::parse_questions(questions) {
        Err(_) => Some(true),
        Ok(q) => model.encode(state, &q).err().map(|_| false),
    }
}

/// Upstream's probabilities (a choice's labels, score levels, noul [yes, no]) in the order answers
/// use (noul [false, true]).
fn in_answer_order(q: &Question, probs: &[f64]) -> Vec<f64> {
    match q.qtype {
        QType::Noul => probs.iter().rev().copied().collect(),
        _ => probs.to_vec(),
    }
}

/// The largest difference between two `/v1/systemone` answers' numbers, and whether a choice
/// answer names a different label.
fn answer_diff(ours: &Value, theirs: &Value) -> (f64, bool) {
    let mut d = 0f64;
    for key in ["confidence", "score", "noul"] {
        if let (Some(a), Some(b)) = (ours[key].as_f64(), theirs[key].as_f64()) {
            d = d.max((a - b).abs());
        }
    }
    if let (Some(a), Some(b)) = (
        ours["probabilities"].as_object(),
        theirs["probabilities"].as_object(),
    ) {
        for (k, v) in a {
            let other = b.get(k).and_then(Value::as_f64).unwrap_or(f64::NAN);
            d = d.max((v.as_f64().unwrap_or(f64::NAN) - other).abs());
        }
    }
    (d, ours["choice"] != theirs["choice"])
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: parity_decima <model-dir> <goldens.jsonl> [cpu|cuda] [--latency]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let latency = args.iter().any(|a| a == "--latency");
    let t = Instant::now();
    let model = DecimaModel::load(&PathBuf::from(&args[1]), device, None)?;
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

    // Inputs: rejections, rows, truncation.
    let (mut rej_bad, mut row_bad, mut rejected, mut ollaya_rule, mut truncated) = (0, 0, 0, 0, 0);
    let mut cases = Vec::new();
    for rec in &records {
        let id = rec["id"].as_str().unwrap_or("?").to_owned();
        let state = &rec["state"];
        let code = rec["error"]["code"].as_str();
        if rec.get("rows").is_none() {
            rejected += 1;
            if rejects(&model, state, &rec["questions"]).is_none() {
                rej_bad += 1;
                println!("REJECTION {id}: upstream rejects the request ({code:?})");
            }
            let valid = by_id
                .get(format!("{id}#valid").as_str())
                .and_then(|r| r["questions"].as_object());
            for (qid, def) in rec["questions"].as_object().context("questions")? {
                let one = Value::Object([(qid.clone(), def.clone())].into_iter().collect());
                let upstream = !valid.is_some_and(|v| v.contains_key(qid));
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
        let request = model.encode(state, &questions)?;
        let rows = rec["rows"].as_array().context("rows")?;
        let mut same = rows.len() == request.questions.len();
        for (g, q) in rows.iter().zip(&request.questions) {
            let state_ids: Vec<u32> = serde_json::from_value(g["state_ids"].clone())?;
            let option_ids: Vec<Vec<u32>> = serde_json::from_value(g["option_ids"].clone())?;
            same &= state_ids == q.state_ids
                && option_ids == q.option_ids
                && g["state_truncated"].as_bool() == Some(q.state_truncated);
        }
        let flagged = request.questions.iter().any(|q| q.state_truncated);
        same &= flagged == (code == Some("STATE_TRUNCATED"));
        truncated += usize::from(flagged);
        if !same {
            row_bad += 1;
            println!("ROWS {id}: the rows or the truncation differ");
        }
        cases.push((id, rec, questions, request));
    }
    let nq: usize = cases.iter().map(|c| c.2.len()).sum();
    let rows: usize = cases
        .iter()
        .flat_map(|c| &c.3.questions)
        .map(|q| 1 + q.option_ids.len())
        .sum();
    println!(
        "inputs: {} records ({rejected} rejected upstream), {} accepted ({truncated} with a truncated state), \
         {nq} questions, {rows} rows | rejection mismatches: {rej_bad} | row mismatches: {row_bad} \
         | refused by Ollaya's shared question rules: {ollaya_rule}",
        records.len(),
        cases.len()
    );
    if rej_bad + row_bad > 0 {
        bail!("input parity failed");
    }

    // Numbers.
    let (mut score_max, mut proj_max, mut disagree, mut wire_choice) = (0f64, 0f64, 0, 0);
    let mut prob_diffs = Vec::new();
    let mut wire_max = 0f64;
    let mut forward = 0f64;
    for (id, rec, questions, request) in &cases {
        let t = Instant::now();
        let raw = model.scores(&request.questions)?;
        forward += t.elapsed().as_secs_f64();
        let out = model.run(&rec["state"], questions)?;
        let plan = rec["plan"].as_array().context("plan")?;
        for ((((qid, q), s), got), p) in questions.iter().zip(&raw).zip(&out.questions).zip(plan) {
            let want = |k: &str| serde_json::from_value::<Vec<f64>>(p[k].clone());
            let (scores, g, gap) = (want("scores")?, want("ordinal_g")?, want("ordinal_gap")?);
            if s.scores.len() != scores.len() {
                bail!(
                    "SCORES {id} {qid}: {} options vs {}",
                    s.scores.len(),
                    scores.len()
                );
            }
            score_max = score_max.max(max_diff(&s.scores, &scores));
            proj_max = proj_max
                .max(max_diff(&s.ordinal_g, &g))
                .max(max_diff(&s.ordinal_gap, &gap));
            let gold = in_answer_order(q, &want("probabilities")?);
            let answer = Answer::new(q, &model.calibration, &got.logits, None, out.state_tokens);
            let probs = &answer.probabilities;
            prob_diffs.push(
                gold.iter()
                    .zip(probs)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max),
            );
            if argmax(&gold) != argmax(probs) {
                disagree += 1;
                if disagree <= 5 {
                    println!("DECISION {id} {qid}: {probs:.4?} vs reference {gold:.4?}");
                }
            }
            let (d, other) = answer_diff(&answer.to_typesafe(q), &rec["answers"][qid]);
            wire_max = wire_max.max(d);
            if other {
                wire_choice += 1;
                println!("ANSWER {id} {qid}: another choice than upstream's system_one");
            }
        }
    }
    prob_diffs.sort_by(f64::total_cmp);
    println!(
        "numbers: score diff max {score_max:.1e} | ordinal projections max {proj_max:.1e} \
         | decisions agree: {:.2}% ({disagree} differ) | prob diff max {:.1e} p99 {:.1e} \
         | /v1/systemone answers vs upstream's (4 decimals): max {wire_max:.1e}, {wire_choice} other choices \
         | forward {forward:.2}s ({:.1} ms/request)",
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
            "max_logit_diff": score_max,
            "max_ordinal_diff": proj_max,
            "max_prob_diff": prob_diffs.last().copied().unwrap_or(0.0),
            "p99_prob_diff": percentile(&prob_diffs, 99),
            "max_answer_diff": wire_max,
            "answers_other_choice": wire_choice,
        })
    );
    if score_max.max(proj_max) > SCORE_TOL || prob_diffs.last().is_some_and(|d| d.is_nan()) {
        bail!("scores differ by more than {SCORE_TOL:e}");
    }
    if disagree + wire_choice > 0 {
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
