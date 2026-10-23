//! Compare the `clm-v1` runtime against golden fixtures from `ollaya_convert.families.clm.goldens`
//! (upstream `clm.schema` and heads, the Qwen3-8B encoder in fp32 on its BF16 weights).
//!
//!     cargo run --release -p ollaya-runner --example parity_clm -- <model-dir> <goldens.jsonl> [cpu|cuda] [--latency] [--smoke]
//!
//! Every case is checked for texts first:
//! * a request upstream rejects must be rejected, and each of its questions exactly when
//!   upstream rejects it on its own. Ollaya requires `instructions` on every question, for every
//!   model; upstream CLM does not, so a question without them is counted separately;
//! * per question: the state text, the option labels and texts, and every text's token ids must
//!   match exactly.
//!
//! Then every case runs through the graph (the projection cache fills as it goes, as in production):
//! * each question's option logits (`scale * cosine`) must be within `LOGIT_TOL`;
//! * the probabilities must pick the reference's option (max / p99 difference reported);
//! * the TypeSafe answers must match upstream's `answer_from_logits` (4 decimals on the wire).
//!
//! `--latency` then times full requests of 5 questions, with nothing cached and with every text
//! cached (a repeated request; repeated questions and options hit the cache the same way).

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::Answer;
use ollaya_runner::Device;
use ollaya_runner::clm::ClmModel;
use ollaya_runner::engine::Engine;
use serde_json::Value;

mod smoke;

/// Largest logit difference accepted (ONNX Runtime vs PyTorch, both fp32 on the BF16 weights).
const LOGIT_TOL: f64 = 1e-3;
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

fn percentile(sorted: &[f64], p: usize) -> f64 {
    sorted
        .get((sorted.len() * p / 100).min(sorted.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0)
}

/// Why the runtime rejects a request, or `None`: `Some(true)` for Ollaya's own rule (missing
/// instructions and the other typed-question checks every model shares), `Some(false)` for the
/// layout's.
fn rejects(model: &ClmModel, state: &Value, questions: &Value) -> Option<bool> {
    let parsed = match ollaya_decision::parse_questions(questions) {
        Ok(q) => q,
        Err(_) => return Some(true),
    };
    let defs: Vec<(&str, &Value)> = parsed
        .iter()
        .map(|(k, q)| (k.as_str(), &q.definition))
        .collect();
    let Ok(texts) = model.questions(state, &defs) else {
        return Some(false);
    };
    for q in &texts {
        if model.row(&q.state, false).is_err()
            || q.options.iter().any(|o| model.row(o, true).is_err())
        {
            return Some(false);
        }
    }
    None
}

fn wire_diff(ours: &Value, upstream: &Value) -> Option<f64> {
    let num = |v: &Value, k: &str| v[k].as_f64();
    if ours["type"] != upstream["type"] {
        return None;
    }
    let mut d = 0f64;
    match ours["type"].as_str()? {
        "noul" => d = d.max((num(ours, "noul")? - num(upstream, "noul")?).abs()),
        "choice" => {
            if ours["choice"] != upstream["choice"] {
                return None;
            }
            d = d.max((num(ours, "confidence")? - num(upstream, "confidence")?).abs());
        }
        "score" => {
            d = d.max((num(ours, "score")? - num(upstream, "score")?).abs());
            d = d.max((num(ours, "confidence")? - num(upstream, "confidence")?).abs());
        }
        _ => return None,
    }
    if ours["type"] != "noul" {
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
    }
    Some(d)
}

fn main() -> Result<()> {
    let (args, smoke) = smoke::args();
    if args.len() < 3 {
        bail!("usage: parity_clm <model-dir> <goldens.jsonl> [cpu|cuda] [--latency] [--smoke]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let latency = args.iter().any(|a| a == "--latency");
    let t = Instant::now();
    let model = ClmModel::load(&PathBuf::from(&args[1]), device, None)?;
    println!("load {:.1}s on {device:?}", t.elapsed().as_secs_f64());

    let mut records = Vec::new();
    for line in smoke::lines(&args[2], smoke)? {
        records.push(serde_json::from_str::<Value>(&line)?);
    }
    let by_id: HashMap<&str, &Value> = records
        .iter()
        .map(|r| (r["id"].as_str().unwrap_or("?"), r))
        .collect();

    // Texts: rejections, state and option texts, token ids.
    let (mut rej_bad, mut text_bad, mut rejected, mut ollaya_rule) = (0, 0, 0, 0);
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
        let texts = model.questions(state, &defs)?;
        let gold = rec["texts"].as_array().context("texts")?;
        if gold.len() != texts.len() {
            text_bad += 1;
            println!("TEXTS {id}: {} questions vs {}", texts.len(), gold.len());
            continue;
        }
        for (q, g) in texts.iter().zip(gold) {
            let options: Vec<String> = serde_json::from_value(g["options"].clone())?;
            let keys: Vec<String> = serde_json::from_value(g["keys"].clone())?;
            let state_ids: Vec<u32> = serde_json::from_value(g["state_ids"].clone())?;
            let option_ids: Vec<Vec<u32>> = serde_json::from_value(g["option_ids"].clone())?;
            let ours_ids = model.row(&q.state, false)?.ids;
            let ours_opts: Vec<Vec<u32>> = q
                .options
                .iter()
                .map(|o| model.row(o, true).map(|r| r.ids))
                .collect::<Result<_, _>>()?;
            if g["state"] != q.state.as_str()
                || keys != q.keys
                || options != q.options
                || state_ids != ours_ids
                || option_ids != ours_opts
            {
                text_bad += 1;
                if text_bad <= 5 {
                    println!(
                        "TEXTS {id} {}: state {:?} vs {:?}, options {:?} vs {options:?}",
                        g["qid"], q.state, g["state"], q.options
                    );
                }
            }
        }
        cases.push((id, rec, questions));
    }
    let nq: usize = cases.iter().map(|c| c.2.len()).sum();
    println!(
        "texts: {} cases ({rejected} rejected upstream), {nq} questions | rejection mismatches: {rej_bad} \
         | text mismatches: {text_bad} | refused by Ollaya's shared question rules: {ollaya_rule}",
        records.len()
    );
    if rej_bad + text_bad > 0 {
        bail!("text parity failed");
    }

    // Numbers.
    let (mut logit_max, mut wire_max, mut disagree, mut wire_bad) = (0f64, 0f64, 0, 0);
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
            let upstream = &rec["answers"][qid.as_str()];
            match wire_diff(&answer.to_typesafe(q), upstream) {
                Some(d) => wire_max = wire_max.max(d),
                None => {
                    wire_bad += 1;
                    if wire_bad <= 5 {
                        println!(
                            "ANSWER {id} {qid}: {} vs upstream {upstream}",
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
         | prob diff max {:.1e} p99 {:.1e} | wire answers vs upstream: max diff {wire_max:.1e}, {wire_bad} differ \
         | forward {forward:.1}s ({:.0} ms/request)",
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
        bail!("answers differ from upstream answer_from_logits");
    }

    if latency {
        let five: Vec<_> = cases.iter().filter(|c| c.2.len() == 5).collect();
        let (mut cold, mut warm) = (Vec::new(), Vec::new());
        for (_, rec, questions) in &five {
            model.clear_cache();
            for times in [&mut cold, &mut warm] {
                let t = Instant::now();
                model.run(&rec["state"], questions)?;
                times.push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        for (pass, ms) in [
            ("cold (nothing cached)", &mut cold),
            ("warm (the same request again)", &mut warm),
        ] {
            ms.sort_by(f64::total_cmp);
            println!(
                "latency, 5-question requests, {pass} (n={}): p50 {:.1} ms  p95 {:.1} ms",
                ms.len(),
                percentile(ms, 50),
                percentile(ms, 95)
            );
        }
    }
    Ok(())
}
