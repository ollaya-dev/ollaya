//! Compare the `decider-vision-v1` runtime against golden fixtures from
//! `ollaya_convert.families.decider_vision.goldens` (upstream `VisionDecisionModel` in fp32).
//!
//!     cargo run --release -p ollaya-runner --example parity_decider_vision -- <model-dir> <goldens.jsonl> [cpu|cuda] [--latency]
//!
//! Every case is encoded first, before anything runs through the graphs:
//! * a request upstream rejects must be rejected;
//! * an image must decode and resize to upstream's pixels exactly (PIL's bicubic, byte for byte),
//!   with the same patch grid;
//! * every row's token ids, slot position and (t, h, w) positions must match exactly.
//!
//! Then every case runs through both graphs:
//! * each question's option logits must be within `LOGIT_TOL`;
//! * calibrated probabilities must pick the reference's option (max / p99 difference reported).
//!
//! A score question whose criteria are an object (a legend) is skipped and counted: upstream
//! accepts it, but it is not TypeSafe wire, so the API rejects it before it reaches a runner.
//!
//! `--latency` then times full requests (decode, resize, both graphs) with an image.

use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::{Question, Questions};
use ollaya_runner::Device;
use ollaya_runner::decider_vision::{VisionDeciderModel, VisionEncoding};
use ollaya_runner::vision::{self, Rgb};
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

fn b64(v: &Value) -> Result<Vec<u8>> {
    vision::from_base64(v.as_str().context("base64 string")?).map_err(Into::into)
}

/// Upstream accepts a score legend object; TypeSafe wire (and so Ollaya's parser) does not.
fn is_legend(def: &Value) -> bool {
    def["type"] == "score" && def["criteria"].is_object()
}

/// The questions Ollaya parses, minus legend objects (counted in `skipped`), and the indices of
/// the kept ones in the request.
fn parse(questions: &Value, skipped: &mut usize) -> Result<(Questions, Vec<usize>)> {
    let (mut out, mut kept) = (Questions::new(), Vec::new());
    for (i, (qid, def)) in questions
        .as_object()
        .context("questions")?
        .iter()
        .enumerate()
    {
        match Question::parse(qid, def) {
            Ok(q) => {
                out.insert(qid.clone(), q);
                kept.push(i);
            }
            Err(_) if is_legend(def) => *skipped += 1,
            Err(e) => return Err(e.into()),
        }
    }
    Ok((out, kept))
}

/// Does the runtime reject this request (image, parsing or encoding)?
fn rejects(model: &VisionDeciderModel, rec: &Value) -> bool {
    let image = match rec["image"].as_str() {
        Some(s) => match vision::from_base64(s).and_then(|b| vision::decode(&b)) {
            Ok(i) => Some(i),
            Err(_) => return true,
        },
        None => None,
    };
    match ollaya_decision::parse_questions(&rec["questions"]) {
        Ok(qs) => model.encode(&rec["state"], &qs, image.as_ref()).is_err(),
        Err(_) => true,
    }
}

struct Case {
    id: String,
    rec: Value,
    questions: Questions,
    /// The kept questions' indices in the request (and so in the golden rows and plan).
    kept: Vec<usize>,
    image: Option<Rgb>,
    encoding: VisionEncoding,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: parity_decider_vision <model-dir> <goldens.jsonl> [cpu|cuda] [--latency]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let latency = args.iter().any(|a| a == "--latency");
    let t = Instant::now();
    let model = VisionDeciderModel::load(&PathBuf::from(&args[1]), device, None)?;
    println!("load {:.1}s on {device:?}", t.elapsed().as_secs_f64());

    let file = std::fs::File::open(&args[2]).with_context(|| args[2].clone())?;
    let mut records = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        records.push(serde_json::from_str::<Value>(&line?)?);
    }

    // Encoding: rejections, pixels, token rows, slots and positions.
    let (mut rej_bad, mut pix_bad, mut enc_bad, mut rejected, mut skipped) = (0, 0, 0, 0, 0);
    let mut pix_max = 0u8;
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
        let (questions, kept) = match parse(&rec["questions"], &mut skipped) {
            Ok(q) => q,
            Err(e) => {
                rej_bad += 1;
                println!("REJECTION {id}: a question upstream accepts does not parse: {e}");
                continue;
            }
        };
        let image = if rec["image"].is_null() {
            None
        } else {
            Some(vision::decode(&b64(&rec["image"])?)?)
        };
        let encoding = match model.encode(&rec["state"], &questions, image.as_ref()) {
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
            let same_size = want["width"].as_u64() == Some(img.resized.width as u64)
                && want["height"].as_u64() == Some(img.resized.height as u64);
            if same_size {
                let d = rgb
                    .iter()
                    .zip(&img.resized.data)
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap_or(0);
                pix_max = pix_max.max(d);
            }
            if !same_size
                || rgb != img.resized.data
                || grid != [1, img.patches.grid_h, img.patches.grid_w]
            {
                pix_bad += 1;
                println!(
                    "PIXELS {id}: {}x{} grid {}x{} vs {}x{} grid {grid:?}",
                    img.resized.width,
                    img.resized.height,
                    img.patches.grid_h,
                    img.patches.grid_w,
                    want["width"],
                    want["height"],
                );
            }
        }
        if questions.is_empty() {
            continue;
        }
        let gold_rows: Vec<&Value> = kept.iter().map(|&i| &rec["rows"][i]).collect();
        let gold_pos: Vec<&Value> = kept.iter().map(|&i| &rec["positions"][i]).collect();
        let plan: Vec<&Value> = kept.iter().map(|&i| &rec["plan"][i]).collect();
        let rows: Vec<&Vec<u32>> = encoding.questions.iter().flat_map(|s| &s.rows).collect();
        if rows.len() != gold_rows.len() {
            enc_bad += 1;
            println!("ENCODING {id}: {} rows vs {}", rows.len(), gold_rows.len());
        }
        for (i, (row, want)) in rows.iter().zip(gold_rows).enumerate() {
            let ids: Vec<u32> = serde_json::from_value(want["ids"].clone())?;
            let slot = want["slot"].as_u64().context("slot")? as usize;
            let pos: Vec<Vec<i64>> = serde_json::from_value(gold_pos[i].clone())?;
            let same_pos =
                pos.len() == 3 && pos.iter().zip(&encoding.positions[i]).all(|(a, b)| a == b);
            if ids != **row || slot != row.len() - 1 || !same_pos {
                enc_bad += 1;
                if enc_bad <= 5 {
                    let first = ids.iter().zip(row.iter()).position(|(a, b)| a != b);
                    println!(
                        "ENCODING {id} row {i}: {} vs {} tokens, slot {} vs {slot}, first differing token {first:?}, positions equal: {same_pos}",
                        row.len(),
                        ids.len(),
                        row.len() - 1
                    );
                }
            }
        }
        for (s, p) in encoding.questions.iter().zip(&plan) {
            if p["k"].as_u64() != Some(s.options as u64) {
                enc_bad += 1;
                println!("ENCODING {id}: k {} vs {}", s.options, p["k"]);
            }
        }
        cases.push(Case {
            id,
            rec: rec.clone(),
            questions,
            kept,
            image,
            encoding,
        });
    }
    let rows: usize = cases
        .iter()
        .flat_map(|c| &c.encoding.questions)
        .map(|s| s.rows.len())
        .sum();
    let nq: usize = cases.iter().map(|c| c.questions.len()).sum();
    let images = cases.iter().filter(|c| c.image.is_some()).count();
    println!(
        "encoding: {} cases ({rejected} rejected upstream, {images} with an image), {nq} questions, {rows} rows \
         | rejection mismatches: {rej_bad} | pixel mismatches: {pix_bad} (max diff {pix_max}) | encoding mismatches: {enc_bad} \
         | skipped legend objects: {skipped}",
        records.len()
    );
    if rej_bad + pix_bad + enc_bad > 0 {
        bail!("encoding parity failed");
    }

    // Numbers: option logits, calibrated probabilities.
    let (mut logit_max, mut image_max, mut disagree) = (0f64, 0f64, 0);
    let mut prob_diffs = Vec::new();
    let mut forward = 0f64;
    for case in &cases {
        let t = Instant::now();
        let logits = model.letter_logits(&case.encoding)?;
        forward += t.elapsed().as_secs_f64();
        let out = model.output(&case.encoding, &logits);
        for (r, (qid, q)) in case.questions.iter().enumerate() {
            let p = &case.rec["plan"][case.kept[r]];
            if p["qid"] != qid.as_str() {
                bail!("{}: plan {r} is {} not {qid}", case.id, p["qid"]);
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
            let d = max_diff(got, &want);
            logit_max = logit_max.max(d);
            if case.image.is_some() {
                image_max = image_max.max(d);
            }
            let want: Vec<f64> = serde_json::from_value(p["probabilities"].clone())?;
            let probs = model
                .calibration
                .probabilities(q.qtype, got, out.state_tokens);
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
        "numbers: option logit diff max {logit_max:.1e} (with an image {image_max:.1e}) \
         | decisions agree: {:.2}% ({disagree} differ) | prob diff max {:.1e} p99 {:.1e} \
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

    if latency {
        let with_image: Vec<&Case> = cases.iter().filter(|c| c.image.is_some()).collect();
        for c in &with_image {
            let mut ms = Vec::new();
            for _ in 0..3 {
                let t = Instant::now();
                model.run(&c.rec["state"], &c.questions, c.image.as_ref())?;
                ms.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            ms.sort_by(f64::total_cmp);
            let img = c.encoding.image.as_ref().context("image")?;
            println!(
                "latency {}: {} questions, {} visual tokens: median {:.0} ms",
                c.id,
                c.questions.len(),
                img.tokens,
                ms[1]
            );
        }
    }
    Ok(())
}
