//! Terminal rendering of answers: one row per question, a bar for the deciding probability.

use std::io::IsTerminal;

use serde_json::Value;

const BAR: usize = 16;

struct Style {
    on: bool,
}

impl Style {
    fn dim(&self, s: &str) -> String {
        if self.on {
            format!("\x1b[2m{s}\x1b[0m")
        } else {
            s.to_owned()
        }
    }
    fn bold(&self, s: &str) -> String {
        if self.on {
            format!("\x1b[1m{s}\x1b[0m")
        } else {
            s.to_owned()
        }
    }
}

fn bar(p: f64) -> String {
    let filled = ((p.clamp(0.0, 1.0) * BAR as f64).round() as usize).min(BAR);
    format!("{}{}", "█".repeat(filled), "░".repeat(BAR - filled))
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_owned()
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    }
}

fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}

/// Render `answers` (TypeSafe answer objects keyed by question id) as a table.
/// `verbose` adds every option's probability under each row.
pub fn answers(answers: &Value, verbose: bool) -> String {
    let style = Style {
        on: std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none(),
    };
    let Some(map) = answers.as_object() else {
        return String::new();
    };
    let width = map
        .keys()
        .map(|k| k.chars().count())
        .max()
        .unwrap_or(8)
        .clamp(8, 28);
    let mut out = String::new();
    for (qid, a) in map {
        let (answer, p) = match a["type"].as_str() {
            Some("choice") => {
                let choice = text(&a["choice"]);
                let p = a["probabilities"][choice.as_str()].as_f64().unwrap_or(0.0);
                (choice, p)
            }
            Some("score") => {
                let score = a["score"].as_f64().unwrap_or(0.0);
                let levels = a["legend"].as_object().map_or(0, |l| l.len());
                let nearest = a["legend"][score.round().to_string()].clone();
                let label = if nearest.is_null() {
                    String::new()
                } else {
                    format!("  {}", truncate(&text(&nearest), 28))
                };
                (
                    format!("{score:.2} / {}{label}", levels.saturating_sub(1)),
                    a["confidence"].as_f64().unwrap_or(0.0),
                )
            }
            Some("noul") => {
                let p = a["noul"].as_f64().unwrap_or(0.0);
                (
                    (if p >= 0.5 { "yes" } else { "no" }).to_owned(),
                    if p >= 0.5 { p } else { 1.0 - p },
                )
            }
            _ => (a.to_string(), 0.0),
        };
        out.push_str(&format!(
            "{:<width$}  {:<36}  {} {:.2}\n",
            style.dim(&truncate(qid, width)),
            style.bold(&truncate(&answer, 36)),
            bar(p),
            p,
            width = width + if style.on { 8 } else { 0 },
        ));
        if verbose && let Some(probs) = a["probabilities"].as_object() {
            for (label, p) in probs {
                let p = p.as_f64().unwrap_or(0.0);
                out.push_str(&style.dim(&format!(
                    "{:width$}    {:<32}  {} {:.2}\n",
                    "",
                    truncate(label, 32),
                    bar(p),
                    p,
                    width = width
                )));
            }
        }
    }
    out
}

/// Left-aligned columns separated by three spaces, like Ollama's tables.
pub fn table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut widths: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
    for row in rows {
        for (w, cell) in widths.iter_mut().zip(row) {
            *w = (*w).max(cell.chars().count());
        }
    }
    let line = |cells: Vec<&str>| {
        let n = cells.len();
        let mut s = String::new();
        for (i, (cell, w)) in cells.into_iter().zip(&widths).enumerate() {
            if i + 1 == n {
                s.push_str(cell);
            } else {
                s.push_str(&format!("{cell:<w$}   "));
            }
        }
        s.trim_end().to_owned() + "\n"
    };
    let mut out = line(headers.to_vec());
    for row in rows {
        out.push_str(&line(row.iter().map(String::as_str).collect()));
    }
    out
}

/// Decimal units, as Ollama prints sizes: `846 MB`, `11 KB`, `1.2 GB`.
pub fn bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut unit = 0;
    while v >= 1000.0 && unit + 1 < UNITS.len() {
        v /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} B")
    } else if v < 10.0 {
        format!("{v:.1} {}", UNITS[unit])
    } else {
        format!("{v:.0} {}", UNITS[unit])
    }
}

fn span(secs: i64) -> String {
    let unit = |n: i64, what: &str| {
        if n == 1 {
            format!("1 {what}")
        } else {
            format!("{n} {what}s")
        }
    };
    match secs {
        s if s < 60 => unit(s.max(1), "second"),
        s if s < 3600 => unit(s / 60, "minute"),
        s if s < 86_400 => unit(s / 3600, "hour"),
        s if s < 7 * 86_400 => unit(s / 86_400, "day"),
        s if s < 30 * 86_400 => unit(s / (7 * 86_400), "week"),
        s if s < 365 * 86_400 => unit(s / (30 * 86_400), "month"),
        s => unit(s / (365 * 86_400), "year"),
    }
}

/// `2 hours ago`.
pub fn ago(t: chrono::DateTime<chrono::Utc>) -> String {
    let secs = (chrono::Utc::now() - t).num_seconds();
    if secs < 1 {
        "Less than a second ago".into()
    } else {
        format!("{} ago", span(secs))
    }
}

/// `4 minutes from now`; `Forever` for a model kept loaded.
pub fn until(t: Option<chrono::DateTime<chrono::Utc>>) -> String {
    match t {
        None => "Forever".into(),
        Some(t) => {
            let secs = (t - chrono::Utc::now()).num_seconds();
            if secs < 1 {
                "Now".into()
            } else {
                format!("{} from now", span(secs))
            }
        }
    }
}

/// `18.7ms`, `2.2s`.
pub fn nanos(ns: u64) -> String {
    let d = std::time::Duration::from_nanos(ns);
    if d.as_secs() >= 1 {
        format!("{:.2}s", d.as_secs_f64())
    } else {
        format!("{:.1}ms", d.as_secs_f64() * 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    fn humanizes() {
        assert_eq!(super::bytes(845_897_529), "846 MB");
        assert_eq!(super::bytes(11_862), "12 KB");
        assert_eq!(super::bytes(1_203_765_248), "1.2 GB");
        assert_eq!(super::bytes(318), "318 B");
        let now = chrono::Utc::now();
        assert_eq!(super::ago(now - chrono::Duration::hours(2)), "2 hours ago");
        assert_eq!(
            super::until(Some(now + chrono::Duration::seconds(245))),
            "4 minutes from now"
        );
        assert_eq!(super::until(None), "Forever");
        assert_eq!(super::nanos(18_734_512), "18.7ms");
        let t = super::table(
            &["NAME", "SIZE"],
            &[vec!["laya:en".into(), "846 MB".into()]],
        );
        assert_eq!(t, "NAME      SIZE\nlaya:en   846 MB\n");
    }

    #[test]
    fn renders_each_type() {
        let a = json!({
            "team": {"type": "choice", "choice": "billing", "confidence": 0.8, "probabilities": {"billing": 0.9, "tech": 0.1}},
            "urgency": {"type": "score", "score": 1.6, "confidence": 0.4, "legend": {"0": "low", "1": "normal", "2": "high"}, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
            "angry": {"type": "noul", "noul": 0.12}
        });
        let out = super::answers(&a, false);
        assert!(out.contains("billing") && out.contains("0.90"), "{out}");
        assert!(out.contains("1.60 / 2  high"), "{out}");
        assert!(out.contains("no") && out.contains("0.88"), "{out}");
    }
}
