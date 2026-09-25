//! Model management subcommands: `pull`, `list`, `ps`, `show`, `rm`, `cp`, `stop`, `create`.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ollaya_api::models::short_digest;
use ollaya_api::{
    CalibrationSpec, Client, CreateParameters, CreateRequest, License, ProgressResponse,
    PullRequest, Question, ShowResponse,
};

use crate::{modelfile, render};

/// Pull with Ollama-style progress: a bar per layer, a line per step.
pub async fn pull(client: &Client, model: &str, insecure: bool) -> Result<()> {
    let bars = MultiProgress::new();
    let style = ProgressStyle::with_template(
        "{msg} {percent:>3}% ▕{bar:20}▏ {binary_bytes:>9}/{binary_total_bytes:<9} {binary_bytes_per_sec:>11}",
    )
    .expect("valid template")
    .progress_chars("██ ");
    let mut layers: HashMap<String, ProgressBar> = HashMap::new();
    let mut step: Option<ProgressBar> = None;
    let req = PullRequest {
        insecure,
        ..PullRequest::new(model)
    };
    let result = client
        .pull(&req, |p: &ProgressResponse| match (&p.digest, p.total) {
            (Some(digest), Some(total)) => {
                let bar = layers.entry(digest.clone()).or_insert_with(|| {
                    if let Some(s) = step.take() {
                        s.finish();
                    }
                    let b = bars.add(ProgressBar::new(total));
                    b.set_style(style.clone());
                    b.set_message(format!("pulling {}:", short_digest(digest)));
                    b
                });
                bar.set_length(total);
                bar.set_position(p.completed.unwrap_or(0));
                if p.completed == Some(total) && !bar.is_finished() {
                    bar.finish();
                }
            }
            _ => {
                if let Some(s) = step.take() {
                    s.finish();
                }
                let s = bars.add(ProgressBar::new_spinner());
                s.set_message(p.status.clone());
                s.enable_steady_tick(Duration::from_millis(100));
                step = Some(s);
            }
        })
        .await;
    if let Some(s) = step.take() {
        s.finish();
    }
    for b in layers.values() {
        if !b.is_finished() {
            b.abandon();
        }
    }
    result.map_err(Into::into)
}

pub async fn list(client: &Client) -> Result<()> {
    let tags = client.tags().await?;
    let rows: Vec<Vec<String>> = tags
        .models
        .iter()
        .map(|m| {
            vec![
                m.name.clone(),
                m.digest.chars().take(12).collect(),
                render::bytes(m.size),
                render::ago(m.modified_at),
            ]
        })
        .collect();
    print!(
        "{}",
        render::table(&["NAME", "ID", "SIZE", "MODIFIED"], &rows)
    );
    Ok(())
}

pub async fn ps(client: &Client) -> Result<()> {
    let ps = client.ps().await?;
    let rows: Vec<Vec<String>> = ps
        .models
        .iter()
        .map(|m| {
            vec![
                m.name.clone(),
                m.digest.chars().take(12).collect(),
                render::bytes(m.size),
                m.device.clone(),
                m.details.quantization_level.clone(),
                render::until(m.expires_at),
            ]
        })
        .collect();
    print!(
        "{}",
        render::table(
            &["NAME", "ID", "SIZE", "DEVICE", "PRECISION", "UNTIL"],
            &rows
        )
    );
    Ok(())
}

/// What `ollaya show` prints besides the default summary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShowOnly {
    pub questions: bool,
    pub license: bool,
    pub modelfile: bool,
    pub parameters: bool,
}

pub async fn show(client: &Client, model: &str, only: ShowOnly) -> Result<()> {
    let s = client.show(model).await?;
    if only.license {
        println!("{}", s.license.trim_end());
    }
    if only.modelfile {
        print!("{}", s.modelfile);
    }
    if only.parameters {
        println!("{}", s.parameters);
    }
    if only.questions {
        match &s.questions {
            Some(q) => println!("{}", serde_json::to_string_pretty(q)?),
            None => println!("{model} has no built-in questions"),
        }
    }
    if only != ShowOnly::default() {
        return Ok(());
    }
    print!("{}", summary(&s));
    Ok(())
}

fn question_kind(q: &Question) -> String {
    match q {
        Question::Choice(_) => format!("choice ({} options)", q.num_options()),
        Question::Score(_) => format!("score ({} levels)", q.num_options()),
        Question::Noul(_) => "noul".into(),
    }
}

/// Ollama-style `show` output: indented sections of key/value pairs.
pub fn summary(s: &ShowResponse) -> String {
    let mut out = String::new();
    let mut section = |title: &str, rows: Vec<(String, String)>| {
        if rows.is_empty() {
            return;
        }
        let width = rows
            .iter()
            .map(|(k, _)| k.chars().count())
            .max()
            .unwrap_or(0)
            .max(16);
        out.push_str(&format!("  {title}\n"));
        for (k, v) in rows {
            out.push_str(format!("    {k:<width$}    {v}").trim_end());
            out.push('\n');
        }
        out.push('\n');
    };
    let info = |key: &str| {
        s.model_info
            .iter()
            .find(|(k, _)| k.ends_with(key))
            .map(|(_, v)| match v {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Array(a) => a
                    .iter()
                    .map(|x| x.as_str().map_or_else(|| x.to_string(), str::to_owned))
                    .collect::<Vec<_>>()
                    .join(", "),
                other => other.to_string(),
            })
    };
    let d = &s.details;
    let mut model = vec![("architecture".to_owned(), d.family.clone())];
    for (label, value) in [
        ("parameters", Some(d.parameter_size.clone())),
        ("context length", info(".context_length")),
        ("precision", Some(d.quantization_level.clone())),
        ("format", Some(d.format.clone())),
        ("engine", info("general.engine")),
        ("languages", info("general.languages")),
        ("derived from", Some(d.parent_model.clone())),
    ] {
        if let Some(v) = value.filter(|v| !v.is_empty()) {
            model.push((label.to_owned(), v));
        }
    }
    section("Model", model);
    if let Some(r) = &s.router {
        let mut rows = vec![
            ("strategy".to_owned(), r.strategy.clone()),
            ("default".to_owned(), r.default.clone()),
        ];
        rows.extend(r.routes.iter().map(|(k, v)| (format!("→ {k}"), v.clone())));
        section("Router", rows);
    }
    section(
        "Capabilities",
        s.capabilities
            .iter()
            .map(|c| (c.clone(), String::new()))
            .collect(),
    );
    if let Some(q) = &s.questions {
        section(
            "Questions",
            q.iter()
                .map(|(id, q)| (id.clone(), question_kind(q)))
                .collect(),
        );
    }
    if !s.parameters.is_empty() {
        section(
            "Parameters",
            s.parameters
                .lines()
                .map(|l| {
                    let (k, v) = l.split_once(' ').unwrap_or((l, ""));
                    (k.to_owned(), v.to_owned())
                })
                .collect(),
        );
    }
    let license: Vec<&str> = s.license.lines().filter(|l| !l.trim().is_empty()).collect();
    if !license.is_empty() {
        let mut rows: Vec<(String, String)> = license
            .iter()
            .take(2)
            .map(|l| (l.trim().to_owned(), String::new()))
            .collect();
        if license.len() > 2 {
            rows.push(("...".into(), String::new()));
        }
        section("License", rows);
    }
    out
}

pub async fn rm(client: &Client, models: &[String]) -> Result<()> {
    for m in models {
        client.delete(m).await?;
        println!("deleted '{m}'");
    }
    Ok(())
}

pub async fn cp(client: &Client, source: &str, destination: &str) -> Result<()> {
    client.copy(source, destination).await?;
    println!("copied '{source}' to '{destination}'");
    Ok(())
}

pub async fn stop(client: &Client, model: &str) -> Result<()> {
    client.unload(model).await?;
    Ok(())
}

/// A create request from a Modelfile.
pub fn create_request(name: &str, mf: modelfile::Modelfile) -> Result<CreateRequest> {
    let questions = mf
        .questions
        .map(serde_json::from_value)
        .transpose()
        .context("QUESTIONS: not a question schema")?;
    let calibration: Option<CalibrationSpec> = mf
        .calibration
        .map(serde_json::from_value)
        .transpose()
        .context(
            "CALIBRATION: expected {\"temperature\": [...], \"temperature_by_options\": {...}}",
        )?;
    Ok(CreateRequest {
        model: name.to_owned(),
        from: mf.from,
        questions,
        calibration,
        parameters: mf
            .precision
            .map(|p| CreateParameters { precision: Some(p) }),
        license: mf.license.map(License::One),
        description: mf.description,
        stream: None,
    })
}

/// `ollaya create NAME -f Modelfile`: pulls `FROM` first when it is not local (the API never
/// pulls implicitly).
pub async fn create(client: &Client, name: &str, file: &Path) -> Result<()> {
    let mf = modelfile::read(file)?;
    let req = create_request(name, mf)?;
    match client.show(&req.from).await {
        Ok(_) => {}
        Err(e) if e.is_model_not_found() => pull(client, &req.from, false).await?,
        Err(e) => return Err(e.into()),
    }
    client.create(&req, |p| println!("{}", p.status)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modelfile_to_request() {
        let mf = modelfile::parse(
            "FROM laya:en\nQUESTIONS {\"urgent\": {\"type\": \"noul\"}}\nPARAMETER precision fp32\nDESCRIPTION Triage\nLICENSE \"\"\"MIT\"\"\"",
            Path::new("."),
        )
        .unwrap();
        let req = create_request("triage", mf).unwrap();
        assert_eq!(req.from, "laya:en");
        assert!(req.questions.unwrap().contains_key("urgent"));
        assert_eq!(req.parameters.unwrap().precision.as_deref(), Some("fp32"));
        assert_eq!(req.license, Some(License::One("MIT".into())));
        assert_eq!(req.description.as_deref(), Some("Triage"));
    }
}
