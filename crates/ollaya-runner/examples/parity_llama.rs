//! Compare the llama.cpp engine (`winnow-v1`, `llm-logits-v1`) against golden fixtures from
//! `ollaya_convert.families.llm_common.export_llama` (or `replay`, for another device): the
//! family's Python reference prompt, run through the same pinned llama.cpp build's llama-server
//! with the same fixed evaluation plan.
//!
//!     cargo run --release -p ollaya-runner --example parity_llama -- \
//!         <model-dir> <goldens.jsonl> [auto|cpu|cuda|cuda:<n>] [--latency]
//!
//! `<model-dir>` holds `decision.json`, `calibration.json` and `model.gguf` (the author's GGUF, or
//! a link to it). llama.cpp is loaded from `$OLLAYA_LIBRARY_PATH/llama`, and its CUDA backend from
//! `$OLLAYA_LIBRARY_PATH/cuda_v13/libggml-cuda.so` when that exists: the install layout.
//!
//! Every case is encoded first:
//! * a request the reference rejects must be rejected, with the same error class;
//! * per question: the prompt's token ids, its split point and the label candidates must be
//!   identical, and so must the state's token count and truncation.
//!
//! Then every question is evaluated:
//! * option logits within `LOGIT_TOL` of the reference, compared as log-probabilities over the
//!   question's options: the reference reads them through llama-server's sampler (softmax over
//!   the candidates), the engine reads the logits themselves, and the two differ by a constant
//!   per question that calibration and softmax ignore;
//! * calibrated probabilities (`calibration.json`) must pick the reference's option; their
//!   difference is reported as max and p99.
//!
//! `--latency` then times full requests (encode and evaluate) of 5 questions.

use std::path::PathBuf;
use std::time::Instant;

use ollaya_decision::{Calibration, CalibrationFile, QType};
use ollaya_runner::Error;
use ollaya_runner::llama::{Libraries, LlamaModel, Target};
use serde_json::Value;

/// Largest option-logit difference accepted: the family tolerance of every Ollaya parity check.
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

/// `z - logsumexp(z)`: logits up to their additive constant.
fn log_softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lse = m + z.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
    z.iter().map(|x| x - lse).collect()
}

fn qtype(def: &Value) -> QType {
    match def["type"].as_str() {
        Some("choice") => QType::Choice,
        Some("score") => QType::Score,
        _ => QType::Noul,
    }
}

fn error_class(e: &Error) -> &'static str {
    match e {
        Error::Decision(ollaya_decision::Error::TooManyOptions { .. }) => "too_many_options",
        _ => "invalid",
    }
}

fn ids(v: &Value) -> Vec<i32> {
    serde_json::from_value(v.clone()).unwrap_or_default()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let latency = args.iter().any(|a| a == "--latency");
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if pos.len() < 2 {
        return Err(
            "usage: parity_llama <model-dir> <goldens.jsonl> [auto|cpu|cuda|cuda:<n>] [--latency]"
                .into(),
        );
    }
    let dir = PathBuf::from(pos[0]);
    let goldens = PathBuf::from(pos[1]);
    let target = match pos.get(2).map(|s| s.as_str()).unwrap_or("auto") {
        "auto" => Target::Auto,
        "cpu" => Target::Cpu,
        "cuda" => Target::Device("CUDA0".into()),
        d => Target::Device(
            d.strip_prefix("cuda:")
                .map(|n| format!("CUDA{n}"))
                .unwrap_or_else(|| d.to_owned()),
        ),
    };
    let lib = std::env::var_os("OLLAYA_LIBRARY_PATH")
        .map(PathBuf::from)
        .ok_or("set OLLAYA_LIBRARY_PATH to an install's lib/ollaya")?;
    let cuda = lib.join("cuda_v13").join("libggml-cuda.so");
    let libs = Libraries {
        dir: lib.join("llama"),
        cuda: cuda.is_file().then_some(cuda),
    };
    let calibration: CalibrationFile =
        serde_json::from_str(&std::fs::read_to_string(dir.join("calibration.json"))?)?;
    let calibration = Calibration::from_file(&calibration);

    let t0 = Instant::now();
    let model = LlamaModel::load(
        &dir.join("model.gguf"),
        &dir.join("decision.json"),
        &libs,
        &target,
        None,
    )?;
    let dev = model.device.clone();
    println!(
        "loaded {} on {dev} in {:.1}s",
        model.precision,
        t0.elapsed().as_secs_f64()
    );

    let text = std::fs::read_to_string(&goldens)?;
    let (mut cases, mut rejected, mut questions) = (0usize, 0usize, 0usize);
    let mut failures: Vec<String> = Vec::new();
    let mut logit_diffs: Vec<f64> = Vec::new();
    let mut prob_diffs: Vec<f64> = Vec::new();
    let mut agree = 0usize;
    let t_eval = Instant::now();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let case: Value = serde_json::from_str(line)?;
        let id = case["id"].as_str().unwrap_or("?").to_owned();
        cases += 1;
        let enc = model.encode(&case["state"], &case["questions"]);
        if let Some(want) = case.get("error") {
            rejected += 1;
            let want = want.as_str().unwrap_or("invalid");
            match enc {
                Err(e) if error_class(&e) == want => {}
                Err(e) => failures.push(format!(
                    "{id}: rejected as {} ({e}), the reference: {want}",
                    error_class(&e)
                )),
                Ok(_) => {
                    failures.push(format!("{id}: accepted; the reference rejects it ({want})"))
                }
            }
            continue;
        }
        let enc = match enc {
            Ok(e) => e,
            Err(e) => {
                failures.push(format!("{id}: rejected ({e}); the reference accepts it"));
                continue;
            }
        };
        if enc.state_tokens as u64 != case["state_tokens"].as_u64().unwrap_or(u64::MAX)
            || enc.state_truncated != case["state_truncated"].as_bool().unwrap_or(false)
        {
            failures.push(format!(
                "{id}: state {} tokens (truncated {}), the reference {} ({})",
                enc.state_tokens,
                enc.state_truncated,
                case["state_tokens"],
                case["state_truncated"]
            ));
        }
        let rows = case["rows"].as_array().cloned().unwrap_or_default();
        if rows.len() != enc.rows.len() {
            failures.push(format!(
                "{id}: {} questions, the reference {}",
                enc.rows.len(),
                rows.len()
            ));
            continue;
        }
        let mut same = true;
        for ((qid, row), want) in enc.rows.iter().zip(&rows) {
            let mut bad = Vec::new();
            if qid != want["qid"].as_str().unwrap_or("") {
                bad.push(format!("order ({})", want["qid"]));
            }
            let w = ids(&want["ids"]);
            if row.ids != w {
                let at = row.ids.iter().zip(&w).take_while(|(a, b)| a == b).count();
                bad.push(format!(
                    "token ids differ at {at} ({} vs {} tokens)",
                    row.ids.len(),
                    w.len()
                ));
            }
            if row.p as u64 != want["P"].as_u64().unwrap_or(u64::MAX) {
                bad.push(format!("split {} vs {}", row.p, want["P"]));
            }
            if row.candidates != ids(&want["candidates"]) {
                bad.push("candidates".into());
            }
            if !bad.is_empty() {
                same = false;
                failures.push(format!("{id}/{qid}: {}", bad.join(", ")));
            }
        }
        if !same {
            continue;
        }
        let logits = model.evaluate(&enc)?;
        for (((qid, _), got), want) in enc.rows.iter().zip(&logits).zip(&rows) {
            questions += 1;
            let want: Vec<f64> = serde_json::from_value(want["option_logits"].clone())?;
            let got64: Vec<f64> = got.iter().map(|&x| f64::from(x)).collect();
            if got64.len() != want.len() {
                failures.push(format!(
                    "{id}/{qid}: {} options, the reference {}",
                    got64.len(),
                    want.len()
                ));
                continue;
            }
            let diff = log_softmax(&got64)
                .iter()
                .zip(log_softmax(&want))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            logit_diffs.push(diff);
            if diff > LOGIT_TOL {
                failures.push(format!("{id}/{qid}: option logits differ by {diff:.3e}"));
            }
            let t = qtype(&case["questions"][qid]);
            let p_got = calibration.probabilities(t, got, enc.state_tokens);
            let want32: Vec<f32> = want.iter().map(|&x| x as f32).collect();
            let p_want = calibration.probabilities(t, &want32, enc.state_tokens);
            let pd = p_got
                .iter()
                .zip(&p_want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            prob_diffs.push(pd);
            if argmax(&p_got) == argmax(&p_want) {
                agree += 1;
            } else {
                failures.push(format!(
                    "{id}/{qid}: the decision differs (p diff {pd:.3e})"
                ));
            }
        }
    }
    let eval_s = t_eval.elapsed().as_secs_f64();
    logit_diffs.sort_by(f64::total_cmp);
    prob_diffs.sort_by(f64::total_cmp);
    println!(
        "{cases} cases ({rejected} rejected), {questions} questions on {dev} in {eval_s:.1}s: \
         decisions {agree}/{questions} ({:.2}%), option logits max {:.3e} p99 {:.3e}, \
         probabilities max {:.3e} p99 {:.3e}",
        100.0 * agree as f64 / questions.max(1) as f64,
        logit_diffs.last().copied().unwrap_or(0.0),
        percentile(&logit_diffs, 99),
        prob_diffs.last().copied().unwrap_or(0.0),
        percentile(&prob_diffs, 99),
    );

    if latency {
        // A different state every request, so no request reuses the previous one's prefix.
        let state = |i: usize| {
            Value::String(format!(
                "Hi, I was charged twice for invoice #{i}. If it is not refunded today we will \
                 cancel and move to a competitor."
            ))
        };
        let qs: Value = serde_json::from_str(
            r#"{"refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
                "team": {"type": "choice", "instructions": "Which team should handle this?",
                         "criteria": {"billing": "payments and refunds", "technical": "bugs and outages",
                                      "sales": "plans and pricing"}},
                "urgency": {"type": "score", "instructions": "How urgent is it?",
                            "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"]},
                "churn": {"type": "noul", "instructions": "Is the customer threatening to leave?"},
                "sentiment": {"type": "choice", "instructions": "Sentiment?",
                              "criteria": ["positive", "neutral", "negative"]}}"#,
        )?;
        for i in 0..3 {
            model.run_json(&state(i), &qs)?;
        }
        let mut times: Vec<f64> = Vec::new();
        for i in 0..20 {
            let t = Instant::now();
            model.run_json(&state(100 + i), &qs)?;
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(f64::total_cmp);
        println!(
            "latency on {dev}, 5 questions, 20 requests: p50 {:.1} ms, p90 {:.1} ms, min {:.1} ms",
            percentile(&times, 50),
            percentile(&times, 90),
            times[0]
        );
    }

    if !failures.is_empty() {
        for f in failures.iter().take(40) {
            eprintln!("FAIL {f}");
        }
        return Err(format!("{} failures", failures.len()).into());
    }
    println!("PASS");
    Ok(())
}
