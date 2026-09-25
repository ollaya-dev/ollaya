//! Compare the `von-option-marker-v1` runtime against goldens from
//! `ollaya_convert.families.von.goldens` (upstream von 1.1 run in float64, one unpadded row at a
//! time; see "Why the goldens are fp64" in `docs/families/von.md`).
//!
//!     cargo run --release -p ollaya-runner --example parity_von -- <model-dir> <goldens.jsonl> [cpu|cuda|metal] [--latency]
//!
//! Every case is encoded first, before anything runs through the graph:
//! * the rendered state text, and its token count (the temperature map's input), must match;
//! * per question: type, zero-shot flag, and every row's packed text, token ids and marker
//!   positions must match exactly.
//!
//! Then every case runs through the graph, all rows of a request batched as the runtime runs them:
//! * each row's marker logits and each question's option logits must be within `LOGIT_TOL`;
//! * calibrated probabilities (input-conditioned temperature) must pick the reference's option
//!   (max / p99 difference reported);
//! * the calibration alone, applied to the reference's option logits, must reproduce the
//!   reference's temperature and probabilities within `CALIBRATION_TOL`.
//!
//! Where upstream accepts the whole request, the runtime's probabilities are also compared with
//! von's own `evaluate()` answers, which von rounds to 4 decimals (reported only).
//!
//! `--latency` then times full requests (`run`: render, tokenize, forward) of 5 questions.

use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::question::Criteria;
use ollaya_decision::von::state_text;
use ollaya_decision::{Questions, parse_questions};
use ollaya_runner::Device;
use ollaya_runner::von::{VonEncoding, VonModel};
use serde_json::Value;

/// Largest raw-logit difference accepted (ONNX Runtime in fp32 vs the network in float64).
const LOGIT_TOL: f64 = 1e-3;
/// Largest temperature / probability difference of the calibration on the reference's own logits.
const CALIBRATION_TOL: f64 = 1e-6;

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
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - y).abs())
        .fold(0.0, f64::max)
}

fn max_diff64(a: &[f64], b: &[f64]) -> f64 {
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

/// von's own answer as probabilities in Ollaya option order (`[false, true]` for noul).
fn upstream_probabilities(answer: &Value) -> Option<Vec<f64>> {
    match answer["type"].as_str()? {
        "noul" => {
            let p = answer["noul"].as_f64()?;
            Some(vec![1.0 - p, p])
        }
        _ => answer["probabilities"]
            .as_object()?
            .values()
            .map(Value::as_f64)
            .collect(),
    }
}

struct Case {
    id: String,
    rec: Value,
    questions: Questions,
    encoding: VonEncoding,
}

fn main() -> Result<()> {
    // SAFETY: first thing in main, before any thread starts (MLX reads its settings once).
    unsafe { ollaya_runner::prepare_process() };
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: parity_von <model-dir> <goldens.jsonl> [cpu|cuda|metal] [--latency]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        Some("metal") => Device::Metal,
        _ => Device::Cpu,
    };
    let latency = args.iter().any(|a| a == "--latency");
    let t = Instant::now();
    let model = VonModel::load(&PathBuf::from(&args[1]), device, None)?;
    println!("load {:.1}s on {device:?}", t.elapsed().as_secs_f64());
    if model.calibration.map().is_none() {
        bail!("calibration.json has no usable temperature_map");
    }

    let file = std::fs::File::open(&args[2]).with_context(|| args[2].clone())?;
    let mut records = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        records.push(serde_json::from_str::<Value>(&line?)?);
    }

    // Encoding: state text, rows, markers.
    let (mut state_bad, mut enc_bad, mut rows_total) = (0, 0, 0);
    let mut cases = Vec::new();
    for rec in &records {
        let id = rec["id"].as_str().unwrap_or("?").to_owned();
        let state = &rec["state"];
        let questions = parse_questions(&rec["questions"]).with_context(|| id.clone())?;
        let encoding = model
            .encode(state, &questions)
            .with_context(|| format!("{id}: the runtime rejects a request upstream accepts"))?;
        // The goldens hold the temperature map's input, `max(tokens, 1)`.
        if state_text(state) != rec["state_text"].as_str().unwrap_or_default()
            || Some(encoding.state_tokens.max(1) as u64) != rec["state_tokens"].as_u64()
        {
            state_bad += 1;
            println!(
                "STATE {id}: {} tokens vs {}, text equal: {}",
                encoding.state_tokens,
                rec["state_tokens"],
                state_text(state) == rec["state_text"].as_str().unwrap_or_default()
            );
        }
        let items = rec["items"].as_array().context("items")?;
        if items.len() != questions.len() {
            bail!(
                "{id}: {} golden items for {} questions",
                items.len(),
                questions.len()
            );
        }
        for ((qid, q), (scoring, item)) in
            questions.iter().zip(encoding.questions.iter().zip(items))
        {
            let gold_rows = item["rows"].as_array().context("rows")?;
            rows_total += gold_rows.len();
            let mut same = item["qid"] == qid.as_str()
                && item["qtype"] == q.qtype.name()
                && item["zero_shot"].as_bool() == Some(scoring.zero_shot)
                && gold_rows.len() == scoring.rows.len();
            let mut first = None;
            for (row, gold) in scoring.rows.iter().zip(gold_rows) {
                let ids: Vec<u32> = serde_json::from_value(gold["ids"].clone())?;
                let markers: Vec<usize> = serde_json::from_value(gold["markers"].clone())?;
                if row.text != gold["text"].as_str().unwrap_or_default()
                    || row.ids != ids
                    || row.markers != markers
                {
                    same = false;
                    first = first.or_else(|| {
                        Some((
                            row.text
                                .chars()
                                .zip(gold["text"].as_str()?.chars())
                                .position(|(a, b)| a != b),
                            row.ids.iter().zip(&ids).position(|(a, b)| a != b),
                            row.ids.len(),
                            ids.len(),
                        ))
                    });
                }
            }
            if !same {
                enc_bad += 1;
                if enc_bad <= 5 {
                    println!(
                        "ENCODING {id} {qid}: {} rows vs {}, zero-shot {} vs {}, \
                         first difference (text char, token, len, gold len) {first:?}",
                        scoring.rows.len(),
                        gold_rows.len(),
                        scoring.zero_shot,
                        item["zero_shot"]
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
    let truncated = cases
        .iter()
        .filter(|c| c.encoding.questions.iter().any(|s| s.truncated))
        .count();
    println!(
        "encoding: {} cases, {nq} questions, {rows_total} rows ({truncated} cases cut to max_len) \
         | state mismatches: {state_bad} | encoding mismatches: {enc_bad}",
        cases.len()
    );
    if state_bad + enc_bad > 0 {
        bail!("encoding parity failed");
    }

    // Numbers: row logits, option logits, temperatures, probabilities.
    let (mut row_max, mut logit_max, mut disagree) = (0f64, 0f64, 0);
    let (mut row_diffs, mut row_worst) = (Vec::new(), String::new());
    let (mut cal_t_max, mut cal_p_max) = (0f64, 0f64);
    let (mut up_n, mut up_disagree, mut up_max) = (0, 0, 0f64);
    let mut prob_diffs = Vec::new();
    let mut forward = 0f64;
    for case in &cases {
        let t = Instant::now();
        let logits = model.row_logits(&case.encoding)?;
        forward += t.elapsed().as_secs_f64();
        let out = model.output(&case.encoding, &logits);
        let items = case.rec["items"].as_array().context("items")?;
        let mut next = 0;
        for (r, ((qid, q), item)) in case.questions.iter().zip(items).enumerate() {
            let scoring = &case.encoding.questions[r];
            let gold_rows = item["rows"].as_array().context("rows")?;
            for (got, gold) in logits[next..next + scoring.rows.len()]
                .iter()
                .zip(gold_rows)
            {
                let want: Vec<f64> = serde_json::from_value(gold["logits"].clone())?;
                let d = max_diff(got, &want);
                let tokens = gold["ids"].as_array().map_or(0, Vec::len);
                if d > LOGIT_TOL {
                    println!(
                        "ROW {} {qid}: a {tokens}-token row, logits {got:.5?} vs reference {want:.5?}",
                        case.id
                    );
                }
                if d > row_max {
                    row_worst = format!("{} {qid}, a {tokens}-token row", case.id);
                }
                row_max = row_max.max(d);
                row_diffs.push(d);
            }
            next += scoring.rows.len();

            let got = &out.questions[r].logits;
            let want: Vec<f64> = serde_json::from_value(item["option_logits"].clone())?;
            logit_max = logit_max.max(max_diff(got, &want));

            // The calibration on the reference's own logits reproduces its temperature and probabilities.
            let gold_t = item["temperature"].as_f64().context("temperature")?;
            let gold_p: Vec<f64> = serde_json::from_value(item["probabilities"].clone())?;
            let want32: Vec<f32> = want.iter().map(|&x| x as f32).collect();
            let t_ref = model
                .calibration
                .temperature_for(q.qtype, &want32, out.state_tokens);
            cal_t_max = cal_t_max.max((t_ref - gold_t).abs());
            let p_ref = model
                .calibration
                .probabilities(q.qtype, &want32, out.state_tokens);
            cal_p_max = cal_p_max.max(max_diff64(&p_ref, &gold_p));

            let probs = model
                .calibration
                .probabilities(q.qtype, got, out.state_tokens);
            prob_diffs.push(max_diff64(&probs, &gold_p));
            if argmax(&gold_p) != argmax(&probs) {
                disagree += 1;
                if disagree <= 5 {
                    println!(
                        "DECISION {} {qid}: {probs:.4?} vs reference {gold_p:.4?}",
                        case.id
                    );
                }
            }

            if let Some(up) = case.rec["answers"]
                .get(qid)
                .and_then(upstream_probabilities)
            {
                up_n += 1;
                up_max = up_max.max(max_diff64(&probs, &up));
                let same = match (&q.criteria, case.rec["answers"][qid]["choice"].as_str()) {
                    (Criteria::Choice(m), Some(label)) => {
                        m.get_index(argmax(&probs)).map(|(k, _)| k.as_str()) == Some(label)
                    }
                    _ => argmax(&probs) == argmax(&up) || max_diff64(&probs, &up) < 1e-4,
                };
                up_disagree += usize::from(!same);
            }
        }
    }
    prob_diffs.sort_by(f64::total_cmp);
    row_diffs.sort_by(f64::total_cmp);
    println!(
        "numbers: row logit diff max {row_max:.1e} p99 {:.1e} (worst: {row_worst}), \
         option logit diff max {logit_max:.1e} | decisions agree: {:.2}% ({disagree} differ) \
         | prob diff max {:.1e} p99 {:.1e} | forward {forward:.1}s ({:.0} ms/request)",
        percentile(&row_diffs, 99),
        100.0 * (nq - disagree) as f64 / nq.max(1) as f64,
        prob_diffs.last().copied().unwrap_or(0.0),
        percentile(&prob_diffs, 99),
        1000.0 * forward / cases.len().max(1) as f64,
    );
    println!(
        "calibration on the reference logits: temperature diff max {cal_t_max:.1e}, prob diff max {cal_p_max:.1e}"
    );
    println!(
        "vs von evaluate() (rounded to 4 dp): {up_n} questions, {up_disagree} decisions differ, prob diff max {up_max:.1e}"
    );
    if row_max.max(logit_max) > LOGIT_TOL {
        bail!("logits differ by more than {LOGIT_TOL:e}");
    }
    if cal_t_max.max(cal_p_max) > CALIBRATION_TOL {
        bail!("the calibration differs from the reference by more than {CALIBRATION_TOL:e}");
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
        let rows = five
            .iter()
            .flat_map(|c| &c.encoding.questions)
            .map(|s| s.rows.len())
            .sum::<usize>();
        let tokens: usize = five
            .iter()
            .flat_map(|c| &c.encoding.questions)
            .flat_map(|s| &s.rows)
            .map(|r| r.ids.len())
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
