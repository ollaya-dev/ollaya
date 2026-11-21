//! Compare the `valen-decision-1.5-v1` runtime against golden fixtures from
//! `ollaya_convert.families.valen.goldens` (the authors' Valen-Decision-1.5 code in fp32).
//!
//!     cargo run --release -p ollaya-runner --example parity_valen -- <model-dir> <goldens.jsonl> [cpu|cuda] [--smoke]
//!
//! Every case is encoded first, before anything runs through the graphs:
//! * a request upstream rejects must be rejected;
//! * an image must decode and resize to the reference's pixels exactly, with the same merged grid;
//! * the sequence must match upstream's exactly: token ids, the prefix length, every token's
//!   attention block, the M-RoPE positions, and each question's keys, instruction span,
//!   candidate spans and decision position.
//!
//! Then every case runs through both graphs:
//! * each question's candidate logits must be within `LOGIT_TOL` of the reference's;
//! * the probabilities (softmax, temperature 1 as upstream) must pick the same candidate.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::Questions;
use ollaya_runner::Device;
use ollaya_runner::valen::{Valen15Encoding, Valen15Model};
use ollaya_runner::vision::{self, Rgb};
use serde_json::Value;

mod smoke;

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

fn softmax(z: &[f32]) -> Vec<f64> {
    let m = z
        .iter()
        .fold(f64::NEG_INFINITY, |a, &v| a.max(f64::from(v)));
    let e: Vec<f64> = z.iter().map(|&v| (f64::from(v) - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

fn percentile(sorted: &[f64], p: usize) -> f64 {
    sorted
        .get((sorted.len() * p / 100).min(sorted.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0)
}

fn b64(v: &Value) -> Result<Vec<u8>> {
    vision::from_base64(v.as_str().context("base64 string")?).map_err(Into::into)
}

fn image_of(rec: &Value) -> Result<Option<Rgb>, ollaya_runner::Error> {
    match rec["image"].as_str() {
        Some(s) => Ok(Some(vision::decode(&vision::from_base64(s)?)?)),
        None => Ok(None),
    }
}

fn defs(questions: &Questions) -> Vec<(&str, &Value)> {
    questions
        .iter()
        .map(|(qid, q)| (qid.as_str(), &q.definition))
        .collect()
}

/// Does the runtime reject this request (image, parsing or encoding)?
fn rejects(model: &Valen15Model, rec: &Value) -> bool {
    let Ok(image) = image_of(rec) else {
        return true;
    };
    match ollaya_decision::parse_questions(&rec["questions"]) {
        Ok(qs) => model
            .encode(&rec["state"], &defs(&qs), image.as_ref())
            .is_err(),
        Err(_) => true,
    }
}

/// The runtime's sequence in the goldens' shape (`row`).
fn row_json(e: &Valen15Encoding) -> Value {
    let r = &e.row;
    serde_json::json!({
        "ids": r.ids,
        "prefix": r.prefix,
        "blocks": r.blocks,
        "positions": r.positions,
        "questions": r.questions.iter().map(|q| serde_json::json!({
            "qid": q.qid,
            "type": q.qtype.name(),
            "keys": q.keys,
            "instruction": [q.instruction.0, q.instruction.1],
            "candidates": q.candidates.iter().map(|s| [s.0, s.1]).collect::<Vec<_>>(),
            "decision": q.decision,
        })).collect::<Vec<_>>(),
    })
}

/// The first field of the row that differs, for the report.
fn first_difference(got: &Value, want: &Value) -> String {
    for key in ["ids", "prefix", "blocks", "positions", "questions"] {
        if got[key] != want[key] {
            if let (Some(a), Some(b)) = (got[key].as_array(), want[key].as_array()) {
                let at = a.iter().zip(b).position(|(x, y)| x != y);
                return format!(
                    "{key} ({} vs {} entries, first difference at {at:?})",
                    a.len(),
                    b.len()
                );
            }
            return format!("{key} ({} vs {})", got[key], want[key]);
        }
    }
    "nothing".into()
}

struct Case {
    id: String,
    rec: Value,
    questions: Questions,
    encoding: Valen15Encoding,
}

fn main() -> Result<()> {
    let (args, smoke) = smoke::args();
    if args.len() < 3 {
        bail!("usage: parity_valen <model-dir> <goldens.jsonl> [cpu|cuda] [--smoke]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let t = Instant::now();
    let model = Valen15Model::load(&PathBuf::from(&args[1]), device, None)?;
    println!("load {:.1}s on {device:?}", t.elapsed().as_secs_f64());

    let mut records = Vec::new();
    for line in smoke::lines(&args[2], smoke)? {
        records.push(serde_json::from_str::<Value>(&line)?);
    }

    // Encoding: rejections, pixels, the sequence.
    let (mut rej_bad, mut pix_bad, mut enc_bad, mut rejected) = (0, 0, 0, 0);
    let mut tokens = 0usize;
    let mut cases = Vec::new();
    for rec in &records {
        let id = rec["id"].as_str().unwrap_or("?").to_owned();
        if !rec["error"].is_null() {
            rejected += 1;
            if !rejects(&model, rec) {
                rej_bad += 1;
                println!(
                    "REJECTION {id}: upstream rejects the request ({})",
                    rec["error"]
                );
            }
            continue;
        }
        let questions = match ollaya_decision::parse_questions(&rec["questions"]) {
            Ok(q) => q,
            Err(e) => {
                rej_bad += 1;
                println!("REJECTION {id}: questions upstream accepts do not parse: {e}");
                continue;
            }
        };
        let image = image_of(rec)?;
        let encoding = match model.encode(&rec["state"], &defs(&questions), image.as_ref()) {
            Ok(e) => e,
            Err(e) => {
                rej_bad += 1;
                println!("REJECTION {id}: upstream accepts, the runtime rejects: {e}");
                continue;
            }
        };
        if let Some(img) = &encoding.image {
            let want = &rec["resized"];
            let rgb = b64(&want["rgb"])?;
            let grid: Vec<usize> = serde_json::from_value(rec["grid"].clone())?;
            let same = want["width"].as_u64() == Some(img.resized.width as u64)
                && want["height"].as_u64() == Some(img.resized.height as u64)
                && rgb == img.resized.data
                && grid == [img.grid.0, img.grid.1];
            if !same {
                pix_bad += 1;
                println!(
                    "PIXELS {id}: {}x{} grid {:?} vs {}x{} grid {grid:?}",
                    img.resized.width, img.resized.height, img.grid, want["width"], want["height"],
                );
            }
        } else if !rec["grid"].is_null() {
            pix_bad += 1;
            println!("PIXELS {id}: the reference read an image, the runtime none");
        }
        let got = row_json(&encoding);
        if got != rec["row"] {
            enc_bad += 1;
            println!(
                "ENCODING {id}: differs in {}",
                first_difference(&got, &rec["row"])
            );
        }
        tokens += encoding.row.ids.len();
        cases.push(Case {
            id,
            rec: rec.clone(),
            questions,
            encoding,
        });
    }
    let nq: usize = cases.iter().map(|c| c.questions.len()).sum();
    let images = cases.iter().filter(|c| c.encoding.image.is_some()).count();
    println!(
        "encoding: {} records ({rejected} rejected upstream), {} run ({images} with an image), {nq} questions, \
         {tokens} tokens | rejection mismatches: {rej_bad} | pixel mismatches: {pix_bad} | sequence mismatches: {enc_bad}",
        records.len(),
        cases.len(),
    );
    if rej_bad + pix_bad + enc_bad > 0 {
        bail!("encoding parity failed");
    }

    // Numbers: candidate logits and the decisions.
    let (mut logit_max, mut image_max, mut disagree) = (0f64, 0f64, 0);
    let mut prob_diffs = Vec::new();
    let mut forward = 0f64;
    for case in &cases {
        let t = Instant::now();
        let logits = model.logits(&case.encoding)?;
        forward += t.elapsed().as_secs_f64();
        // The wire order must also come out (noul [false, true]).
        model.output(&case.encoding, &logits)?;
        let plan = case.rec["plan"].as_array().context("plan")?;
        if plan.len() != logits.len() {
            bail!(
                "{}: {} questions vs {} in the plan",
                case.id,
                logits.len(),
                plan.len()
            );
        }
        for (p, (q, got)) in plan
            .iter()
            .zip(case.encoding.row.questions.iter().zip(&logits))
        {
            if p["qid"] != q.qid.as_str() {
                bail!("{}: plan question {} is not {}", case.id, p["qid"], q.qid);
            }
            let want: Vec<f64> = serde_json::from_value(p["option_logits"].clone())?;
            if got.len() != want.len() {
                bail!(
                    "LOGITS {} {}: {} candidates vs {}",
                    case.id,
                    q.qid,
                    got.len(),
                    want.len()
                );
            }
            let d = got
                .iter()
                .zip(&want)
                .map(|(&x, &y)| (f64::from(x) - y).abs())
                .fold(0.0, f64::max);
            logit_max = logit_max.max(d);
            if case.encoding.image.is_some() {
                image_max = image_max.max(d);
            }
            let want: Vec<f64> = serde_json::from_value(p["probabilities"].clone())?;
            let probs = softmax(got);
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
                        "DECISION {} {}: {probs:.4?} vs reference {want:.4?}",
                        case.id, q.qid
                    );
                }
            }
        }
    }
    prob_diffs.sort_by(f64::total_cmp);
    println!(
        "numbers: candidate logit diff max {logit_max:.1e} (with an image {image_max:.1e}) \
         | decisions agree: {:.2}% ({disagree} differ) | prob diff max {:.1e} p99 {:.1e} \
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
    Ok(())
}
