//! `ollaya calibrate`: fit a model's temperatures to labelled examples, on the model as installed.
//!
//! Every example goes through the daemon with `"extras": ["logits"]`, so the fit sees the logits
//! of this model file on this device (a quantized GGUF does not give its checkpoint's logits).
//! One temperature is fitted per media (`text`, `image`, `audio`, `image+audio`) and question
//! type, by temperature scaling on the cross-entropy against the labels, and written as the
//! `media` entries of a calibration layer that keeps the model's other temperatures. A Modelfile's
//! `CALIBRATION` then bakes it into a new model.
//!
//! An example is one JSON line:
//!
//! ```json
//! {"state": {...}, "questions": {...}, "audio": ["clip.wav"], "images": ["photo.png"],
//!  "labels": {"species": "Turdus merula", "calls": true, "level": 2, "mood": [0.1, 0.9]}}
//! ```
//!
//! Media are file paths, relative to the JSON Lines file, or base64 as `/api/decide` takes them.
//! A label is the option key (choice), the level index (score), `true`/`false` (noul), a
//! probability of yes (noul), or a probability per option in option order (any type: soft labels).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use futures_util::{StreamExt, stream};
use ollaya_api::{Client, DecideRequest, Extra, KeepAlive, Questions};
use ollaya_decision::fit::{Sample, accuracy, cross_entropy, ece, fit_temperature};
use ollaya_decision::{
    Calibration, CalibrationFile, Criteria, QType, calibration::bucket, media_key,
};
use serde_json::{Map, Value, json};

/// The search range for a fitted temperature. Wider than any clamp a model ships: the fit's own
/// result decides the entry's range.
const SEARCH: [f64; 2] = [0.01, 10_000.0];

#[derive(Debug, Clone, clap::Args)]
pub struct CalibrateArgs {
    /// The model to calibrate
    pub model: String,
    /// Labelled examples to fit on, as JSON Lines (see `docs/api.md`, "Calibrating a model")
    pub data: PathBuf,
    /// Where to write the calibration, for a Modelfile's CALIBRATION
    #[arg(short, long, default_value = "calibration.json")]
    pub out: PathBuf,
    /// Labelled examples to report on before and after, never fitted on (a test split)
    #[arg(long)]
    pub eval: Option<PathBuf>,
    /// Also fit a temperature per option-count bucket (`temperature_by_options`)
    #[arg(long)]
    pub by_options: bool,
    /// Fewest questions a temperature is fitted from; smaller groups keep their temperature
    #[arg(long, default_value_t = 30)]
    pub min_samples: usize,
    /// Equal-width confidence bins for the expected calibration error
    #[arg(long, default_value_t = 10)]
    pub bins: usize,
    /// Requests in flight
    #[arg(long, default_value_t = 2)]
    pub parallel: usize,
}

/// One answered question: which group it calibrates, and its logits and target.
#[derive(Debug, Clone)]
struct Scored {
    media: &'static str,
    qtype: QType,
    sample: Sample,
}

pub async fn calibrate(client: &Client, args: &CalibrateArgs) -> Result<()> {
    let show = client.show(&args.model).await?;
    if show.router.is_some() {
        bail!(
            "{} is a router: calibrate the model it routes to",
            args.model
        );
    }
    let base: Value = show.calibration.clone().unwrap_or_else(|| json!({}));
    let fit_rows = read_examples(&args.data)?;
    eprintln!("{}: {} examples", args.data.display(), fit_rows.len());
    let fitted = score(client, args, &fit_rows).await?;
    let layer = fit(&base, &fitted, args)?;
    std::fs::write(&args.out, serde_json::to_string_pretty(&layer)? + "\n")
        .with_context(|| format!("writing {}", args.out.display()))?;

    let before = load(&base)?;
    let after = load(&layer)?;
    println!(
        "Fitted on {} ({} questions)",
        args.data.display(),
        fitted.len()
    );
    report(&fitted, &before, &after, args.bins);
    if let Some(eval) = &args.eval {
        let rows = read_examples(eval)?;
        let scored = score(client, args, &rows).await?;
        println!(
            "\nHeld out: {} ({} questions)",
            eval.display(),
            scored.len()
        );
        report(&scored, &before, &after, args.bins);
    }
    println!(
        "\nWrote {}. To make a model with it, write a Modelfile:\n  FROM {}\n  CALIBRATION {}\nand run `ollaya create NAME -f Modelfile`.",
        args.out.display(),
        args.model,
        args.out.display()
    );
    Ok(())
}

/// A labelled example: the request, and the label of each question that has one.
#[derive(Debug, Clone)]
struct Example {
    line: usize,
    request: DecideRequest,
    labels: Map<String, Value>,
}

fn read_examples(path: &Path) -> Result<Vec<Example>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line_no = i + 1;
        if line.trim().is_empty() {
            continue;
        }
        let at = || format!("{}:{line_no}", path.display());
        let row: Value = serde_json::from_str(line).with_context(at)?;
        let questions: Questions = serde_json::from_value(row["questions"].clone())
            .with_context(|| format!("{}: questions", at()))?;
        let labels = match &row["labels"] {
            Value::Object(m) => m.clone(),
            _ => bail!("{}: no labels object", at()),
        };
        let mut request = DecideRequest::new(
            String::new(),
            row.get("state").cloned().unwrap_or(json!({})),
            Some(questions),
        );
        request.images = media(&row["images"], dir).with_context(|| format!("{}: images", at()))?;
        request.audio = media(&row["audio"], dir).with_context(|| format!("{}: audio", at()))?;
        request.extras = vec![Extra::Logits];
        out.push(Example {
            line: line_no,
            request,
            labels,
        });
    }
    Ok(out)
}

/// Paths (read and encoded) or base64 strings, as `/api/decide` takes them.
fn media(value: &Value, dir: &Path) -> Result<Vec<String>> {
    use base64::Engine as _;
    let items = match value {
        Value::Null => return Ok(Vec::new()),
        Value::Array(items) => items,
        _ => bail!("expected a list of paths or base64 strings"),
    };
    items
        .iter()
        .map(|item| {
            let s = item.as_str().context("expected a string")?;
            let path = dir.join(s);
            if s.len() < 4096 && path.is_file() {
                let bytes =
                    std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
                Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                Ok(s.to_owned())
            }
        })
        .collect()
}

/// Every labelled question's raw logits, from the installed model.
async fn score(client: &Client, args: &CalibrateArgs, rows: &[Example]) -> Result<Vec<Scored>> {
    let total = rows.len();
    let mut done = 0;
    let mut out = Vec::new();
    let mut answers = stream::iter(rows.iter().map(|row| async move {
        let mut req = row.request.clone();
        req.model = args.model.clone();
        req.keep_alive = Some(KeepAlive::For(std::time::Duration::from_secs(600)));
        let resp = client.decide(&req).await;
        (row, resp)
    }))
    .buffer_unordered(args.parallel.max(1));
    while let Some((row, resp)) = answers.next().await {
        let resp = resp.with_context(|| format!("example on line {}", row.line))?;
        let questions = row.request.questions.as_ref().expect("read with questions");
        let engine =
            ollaya_decision::parse_questions(&ollaya_api::decide::engine_questions(questions))
                .with_context(|| format!("line {}: questions", row.line))?;
        let media = media_key(row.request.images.len(), row.request.audio.len());
        for (qid, label) in &row.labels {
            let q = engine.get(qid).with_context(|| {
                format!("line {}: label for unknown question {qid:?}", row.line)
            })?;
            let answer = resp
                .answers
                .get(qid)
                .with_context(|| format!("line {}: no answer for {qid:?}", row.line))?;
            let logits = answer
                .logits
                .clone()
                .context("the server sent no logits; update ollaya")?;
            let target = target(q, label)
                .with_context(|| format!("line {}: label for {qid:?}", row.line))?;
            if target.len() != logits.len() {
                bail!(
                    "line {}: {qid:?} has {} options, the label {}",
                    row.line,
                    logits.len(),
                    target.len()
                );
            }
            out.push(Scored {
                media,
                qtype: q.head(),
                sample: Sample { logits, target },
            });
        }
        done += 1;
        if done % 50 == 0 || done == total {
            eprintln!("{done}/{total} examples");
        }
    }
    Ok(out)
}

/// The target distribution over a question's options, in option order.
fn target(q: &ollaya_decision::Question, label: &Value) -> Result<Vec<f64>> {
    let k = q.num_options();
    let one_hot = |i: usize| -> Result<Vec<f64>> {
        if i >= k {
            bail!("option {i} of {k}");
        }
        let mut t = vec![0.0; k];
        t[i] = 1.0;
        Ok(t)
    };
    match (label, &q.criteria) {
        (Value::Array(p), _) => {
            let t: Vec<f64> = p
                .iter()
                .map(|v| v.as_f64().filter(|x| x.is_finite() && *x >= 0.0))
                .collect::<Option<_>>()
                .context("soft labels are non-negative numbers")?;
            let sum: f64 = t.iter().sum();
            if sum <= 0.0 {
                bail!("soft labels sum to 0");
            }
            Ok(t.into_iter().map(|x| x / sum).collect())
        }
        (Value::Bool(yes), _) if q.qtype == QType::Noul => one_hot(usize::from(*yes)),
        (Value::Number(p), _) if q.qtype == QType::Noul => {
            let p = p
                .as_f64()
                .filter(|p| (0.0..=1.0).contains(p))
                .context("a probability")?;
            Ok(vec![1.0 - p, p])
        }
        (Value::String(key), Criteria::Choice(options)) => {
            let i = options
                .keys()
                .position(|k| k == key)
                .with_context(|| format!("{key:?} is not an option"))?;
            one_hot(i)
        }
        (Value::Number(i), Criteria::Score(_)) => {
            one_hot(i.as_u64().context("a level index")? as usize)
        }
        _ => bail!(
            "expected the option key (choice), level index (score), true/false (noul) or soft labels"
        ),
    }
}

/// The calibration layer: the model's own, with a fitted `media` entry per media in the data.
fn fit(base: &Value, scored: &[Scored], args: &CalibrateArgs) -> Result<Value> {
    let file: CalibrationFile = serde_json::from_value(base.clone())?;
    let mut groups: BTreeMap<(&str, usize), Vec<Sample>> = BTreeMap::new();
    for s in scored {
        groups
            .entry((s.media, s.qtype as usize))
            .or_default()
            .push(s.sample.clone());
    }
    let mut layer = base.as_object().cloned().unwrap_or_default();
    let mut media = layer
        .remove("media")
        .and_then(|m| m.as_object().cloned())
        .unwrap_or_default();
    let base_cal = Calibration::from_file(&file);
    let types = [QType::Choice, QType::Score, QType::Noul];
    let keys: Vec<&str> = {
        let mut k: Vec<&str> = scored.iter().map(|s| s.media).collect();
        k.sort();
        k.dedup();
        k
    };
    for key in keys {
        // Start from the temperatures this media uses now, so that unfitted types keep theirs.
        let current = base_cal.for_media(key);
        let mut temps: Vec<f64> = types.iter().map(|&t| current.type_temperature(t)).collect();
        let mut by_options = Map::new();
        let mut fitted_any = false;
        for (i, &qtype) in types.iter().enumerate() {
            let Some(samples) = groups.get(&(key, i)) else {
                continue;
            };
            if samples.len() < args.min_samples {
                eprintln!(
                    "{key}/{}: {} questions, under --min-samples {}; kept T = {}",
                    qtype.name(),
                    samples.len(),
                    args.min_samples,
                    temps[i]
                );
                continue;
            }
            temps[i] = round6(fit_temperature(samples, SEARCH));
            fitted_any = true;
            if args.by_options {
                let mut buckets: BTreeMap<String, Vec<Sample>> = BTreeMap::new();
                for s in samples {
                    buckets
                        .entry(bucket(qtype, s.logits.len()))
                        .or_default()
                        .push(s.clone());
                }
                for (b, bs) in buckets {
                    if bs.len() >= args.min_samples {
                        by_options.insert(b, json!(round6(fit_temperature(&bs, SEARCH))));
                    }
                }
            }
        }
        if !fitted_any {
            continue;
        }
        let mut entry = Map::new();
        entry.insert("temperature".into(), json!(temps));
        if !by_options.is_empty() {
            entry.insert(
                "temperature_by_options".into(),
                Value::Object(by_options.clone()),
            );
        }
        // A range that holds the fit, so that loading does not clamp it.
        let all = temps
            .iter()
            .copied()
            .chain(by_options.values().filter_map(Value::as_f64));
        let (lo, hi) = all.fold((f64::INFINITY, 0.0f64), |(lo, hi), t| {
            (lo.min(t), hi.max(t))
        });
        let [flo, fhi] = file.range();
        if lo < flo || hi > fhi {
            entry.insert(
                "temperature_range".into(),
                json!([lo.min(flo), hi.max(fhi)]),
            );
        }
        media.insert(key.to_owned(), Value::Object(entry));
    }
    layer.insert("media".into(), Value::Object(media));
    Ok(Value::Object(layer))
}

fn round6(t: f64) -> f64 {
    format!("{t:.6}").parse().unwrap_or(t)
}

fn load(layer: &Value) -> Result<Calibration> {
    Ok(Calibration::from_file(&serde_json::from_value(
        layer.clone(),
    )?))
}

/// Per media and question type: n, temperature, cross-entropy, ECE before and after, accuracy.
fn report(scored: &[Scored], before: &Calibration, after: &Calibration, bins: usize) {
    let mut groups: BTreeMap<(&str, usize), Vec<&Scored>> = BTreeMap::new();
    for s in scored {
        groups
            .entry((s.media, s.qtype as usize))
            .or_default()
            .push(s);
    }
    println!(
        "{:<12} {:<7} {:>6} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8} {:>7}",
        "MEDIA",
        "TYPE",
        "N",
        "T BEFORE",
        "T AFTER",
        "NLL BEF",
        "NLL AFT",
        "ECE BEF",
        "ECE AFT",
        "ACC"
    );
    for ((media, _), group) in groups {
        let qtype = group[0].qtype;
        let samples: Vec<Sample> = group.iter().map(|s| s.sample.clone()).collect();
        // A group's temperature depends on the option count only with buckets; report the
        // temperatures of the most common count.
        let k = mode(samples.iter().map(|s| s.logits.len()));
        let tb = before.for_media(media).temperature(qtype, k);
        let ta = after.for_media(media).temperature(qtype, k);
        let (nb, na) = (
            tempered_nll(&samples, before.for_media(media), qtype),
            tempered_nll(&samples, after.for_media(media), qtype),
        );
        let (eb, ea) = (
            tempered_ece(&samples, before.for_media(media), qtype, bins),
            tempered_ece(&samples, after.for_media(media), qtype, bins),
        );
        println!(
            "{:<12} {:<7} {:>6} {:>10.4} {:>10.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>6.1}%",
            media,
            qtype.name(),
            samples.len(),
            tb,
            ta,
            nb,
            na,
            eb,
            ea,
            100.0 * accuracy(&samples)
        );
    }
}

fn mode(ks: impl Iterator<Item = usize>) -> usize {
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for k in ks {
        *counts.entry(k).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, n)| n)
        .map_or(2, |(k, _)| k)
}

/// Metrics under a calibration's own temperature per question (buckets included): each sample
/// is scaled to temperature 1 first.
fn rescaled(samples: &[Sample], cal: &Calibration, qtype: QType) -> Vec<Sample> {
    samples
        .iter()
        .map(|s| {
            let t = cal.temperature(qtype, s.logits.len());
            Sample {
                logits: s
                    .logits
                    .iter()
                    .map(|&z| (f64::from(z) / t) as f32)
                    .collect(),
                target: s.target.clone(),
            }
        })
        .collect()
}

fn tempered_nll(samples: &[Sample], cal: &Calibration, qtype: QType) -> f64 {
    cross_entropy(&rescaled(samples, cal, qtype), 1.0)
}

fn tempered_ece(samples: &[Sample], cal: &Calibration, qtype: QType, bins: usize) -> f64 {
    ece(&rescaled(samples, cal, qtype), 1.0, bins)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn question(v: Value) -> ollaya_decision::Question {
        ollaya_decision::Question::parse("q", &v).unwrap()
    }

    #[test]
    fn labels_become_targets_in_option_order() {
        let choice = question(
            json!({"type": "choice", "instructions": "Which?", "criteria": {"a": null, "b": null, "c": null}}),
        );
        assert_eq!(target(&choice, &json!("b")).unwrap(), [0.0, 1.0, 0.0]);
        assert_eq!(
            target(&choice, &json!([1, 1, 2])).unwrap(),
            [0.25, 0.25, 0.5]
        );
        assert!(target(&choice, &json!("d")).is_err());
        assert!(target(&choice, &json!([0, 0, 0])).is_err());
        let score = question(
            json!({"type": "score", "instructions": "How much?", "criteria": ["low", "mid", "high"]}),
        );
        assert_eq!(target(&score, &json!(2)).unwrap(), [0.0, 0.0, 1.0]);
        assert!(target(&score, &json!(3)).is_err());
        let noul = question(json!({"type": "noul", "instructions": "Yes?"}));
        assert_eq!(target(&noul, &json!(true)).unwrap(), [0.0, 1.0]);
        assert_eq!(target(&noul, &json!(false)).unwrap(), [1.0, 0.0]);
        assert_eq!(target(&noul, &json!(0.25)).unwrap(), [0.75, 0.25]);
        assert!(target(&noul, &json!("yes")).is_err());
    }

    fn args(min_samples: usize) -> CalibrateArgs {
        CalibrateArgs {
            model: "m".into(),
            data: "d.jsonl".into(),
            out: "c.json".into(),
            eval: None,
            by_options: false,
            min_samples,
            bins: 10,
            parallel: 1,
        }
    }

    #[test]
    fn the_fit_adds_media_entries_and_keeps_the_rest() {
        // Audio choice logits drawn so that T = 90 is calibrated: z = 90 * ln p.
        let scored: Vec<Scored> = (0..60)
            .map(|i| {
                let p = 0.2 + 0.6 * (i as f64 / 60.0);
                let z = [0.0, (90.0 * (p / (1.0 - p)).ln()) as f32];
                Scored {
                    media: "audio",
                    qtype: QType::Choice,
                    sample: Sample {
                        logits: z.to_vec(),
                        target: vec![1.0 - p, p],
                    },
                }
            })
            .collect();
        let base = json!({"temperature": [68.58, 15.0, 101.09], "temperature_range": [0.5, 200.0],
                          "source": "kept", "media": {"text": {"temperature": [2.0, 2.0, 2.0]}}});
        let layer = fit(&base, &scored, &args(30)).unwrap();
        assert_eq!(layer["source"], "kept");
        assert_eq!(layer["temperature"], base["temperature"]);
        assert_eq!(layer["media"]["text"], base["media"]["text"]);
        let audio = &layer["media"]["audio"];
        let t = audio["temperature"][0].as_f64().unwrap();
        assert!((t - 90.0).abs() < 1e-3, "{t}");
        // Unfitted types keep the temperatures audio requests used before; the range holds.
        assert_eq!(audio["temperature"][1], 15.0);
        assert_eq!(audio["temperature"][2], 101.09);
        assert!(audio.get("temperature_range").is_none());
        let cal = load(&layer).unwrap();
        assert!((cal.for_media("audio").temperature(QType::Choice, 2) - t).abs() < 1e-9);

        // Too few questions: nothing fitted, nothing added.
        let layer = fit(&base, &scored[..10], &args(30)).unwrap();
        assert!(layer["media"].get("audio").is_none());

        // A fit outside the clamp range widens the entry's own range.
        let layer = fit(&json!({}), &scored, &args(30)).unwrap();
        assert_eq!(
            layer["media"]["audio"]["temperature_range"][1]
                .as_f64()
                .map(|h| h > 89.0),
            Some(true)
        );
    }
}
