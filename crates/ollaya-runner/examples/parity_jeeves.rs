//! Compare the `jeeves-markers-v1` runtime against golden fixtures from
//! `ollaya_convert.families.jeeves.goldens` (the authors' Encoder and their own fp32 model with the
//! pointer head, no thinking).
//!
//!     cargo run --release -p ollaya-runner --example parity_jeeves -- <model-dir> <goldens.jsonl> [cpu|cuda] [--latency] [--smoke]
//!
//! Every case is checked for rows first:
//! * a request upstream rejects must be rejected, and each of its questions exactly when upstream
//!   rejects it on its own. Ollaya's shared question rules (such as required `instructions`) are
//!   counted separately;
//! * per question: the token ids, the decide position and the option positions must match exactly.
//!
//! Then every case runs through the graph:
//! * each question's option logits must be within `LOGIT_TOL`;
//! * the calibrated probabilities must pick the reference's option (max / p99 difference reported);
//!
//! `--latency` then times full requests of 5 questions.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::Answer;
use ollaya_runner::Device;
use ollaya_runner::engine::Engine;
use ollaya_runner::jeeves::JeevesModel;
use serde_json::Value;

mod smoke;

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
fn rejects(model: &JeevesModel, state: &Value, questions: &Value) -> Option<bool> {
    let parsed = match ollaya_decision::parse_questions(questions) {
        Ok(q) => q,
        Err(_) => return Some(true),
    };
    let defs: Vec<(&str, &Value)> = parsed
        .iter()
        .map(|(k, q)| (k.as_str(), &q.definition))
        .collect();
    model.rows(state, &defs).err().map(|_| false)
}

fn main() -> Result<()> {
    let (args, smoke) = smoke::args();
    if args.len() < 3 {
        bail!("usage: parity_jeeves <model-dir> <goldens.jsonl> [cpu|cuda] [--latency] [--smoke]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let latency = args.iter().any(|a| a == "--latency");
    let t = Instant::now();
    let model = JeevesModel::load(&PathBuf::from(&args[1]), device, None)?;
    println!("load {:.1}s on {device:?}", t.elapsed().as_secs_f64());

    let mut records = Vec::new();
    for line in smoke::lines(&args[2], smoke)? {
        records.push(serde_json::from_str::<Value>(&line)?);
    }
    let by_id: HashMap<&str, &Value> = records
        .iter()
        .map(|r| (r["id"].as_str().unwrap_or("?"), r))
        .collect();

    // Rows: rejections, token ids, candidates.
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
        let rows = model.rows(state, &defs)?;
        let gold = rec["rows"].as_array().context("rows")?;
        if gold.len() != rows.len() {
            row_bad += 1;
            println!("ROWS {id}: {} rows vs {}", rows.len(), gold.len());
            continue;
        }
        for ((row, g), qid) in rows.iter().zip(gold).zip(questions.keys()) {
            let ids: Vec<u32> = serde_json::from_value(g["ids"].clone())?;
            let opts: Vec<usize> = serde_json::from_value(g["opts"].clone())?;
            let decide = g["decide"].as_u64().unwrap_or(0) as usize;
            if ids != row.ids || opts != row.opts || decide != row.decide {
                row_bad += 1;
                if row_bad <= 5 {
                    let at = ids.iter().zip(&row.ids).take_while(|(a, b)| a == b).count();
                    println!(
                        "ROWS {id} {qid}: {} vs {} ids, first difference at {at}",
                        row.ids.len(),
                        ids.len()
                    );
                }
            }
        }
        cases.push((id, rec, questions));
    }
    let nq: usize = cases.iter().map(|c| c.2.len()).sum();
    println!(
        "rows: {} cases ({rejected} rejected upstream), {nq} questions | rejection mismatches: {rej_bad} \
         | row mismatches: {row_bad} | refused by Ollaya's shared question rules: {ollaya_rule}",
        records.len()
    );
    if rej_bad + row_bad > 0 {
        bail!("row parity failed");
    }

    // Numbers.
    let (mut logit_max, mut disagree) = (0f64, 0);
    let mut prob_diffs = Vec::new();
    let mut forward = 0f64;
    for (id, rec, questions) in &cases {
        let t = Instant::now();
        let out = model.run(&rec["state"], questions)?;
        forward += t.elapsed().as_secs_f64();
        let plan = rec["plan"].as_array().context("plan")?;
        for (((qid, q), got), p) in questions.iter().zip(&out.questions).zip(plan) {
            let want: Vec<f64> = serde_json::from_value(p["option_logits"].clone())?;
            if got.logits.len() != want.len() {
                bail!(
                    "LOGITS {id} {qid}: {} options vs {}",
                    got.logits.len(),
                    want.len()
                );
            }
            let d = got
                .logits
                .iter()
                .zip(&want)
                .map(|(&x, &y)| (f64::from(x) - y).abs())
                .fold(0.0, f64::max);
            logit_max = logit_max.max(d);
            let want: Vec<f64> = serde_json::from_value(p["probabilities"].clone())?;
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
    if logit_max > LOGIT_TOL {
        bail!("logits differ by more than {LOGIT_TOL:e}");
    }
    if disagree > 0 {
        bail!("decisions differ from the reference");
    }

    if latency {
        let five: Vec<_> = cases.iter().filter(|c| c.2.len() == 5).collect();
        let mut ms = Vec::new();
        for (_, rec, questions) in &five {
            model.run(&rec["state"], questions)?;
            let t = Instant::now();
            model.run(&rec["state"], questions)?;
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        ms.sort_by(f64::total_cmp);
        println!(
            "latency, 5-question requests (n={}): p50 {:.1} ms  p95 {:.1} ms",
            ms.len(),
            percentile(&ms, 50),
            percentile(&ms, 95)
        );
    }
    Ok(())
}
