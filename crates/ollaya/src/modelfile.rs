//! Modelfiles: derive a model from another, with questions and calibration baked in.
//!
//! ```text
//! # support ticket triage
//! FROM laya
//! QUESTIONS ./triage.json
//! CALIBRATION ./calibration.json
//! PARAMETER precision fp32
//! LICENSE ./LICENSE
//! DESCRIPTION Support ticket triage
//! ```
//!
//! Directives are case-insensitive. A value may be a path (relative to the Modelfile), inline
//! JSON, or a `"""`-quoted block spanning lines. `#` starts a comment line.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::Value;

#[derive(Debug, Default, PartialEq)]
pub struct Modelfile {
    pub from: String,
    pub questions: Option<Value>,
    pub calibration: Option<Value>,
    pub precision: Option<String>,
    pub license: Option<String>,
    pub description: Option<String>,
}

pub fn read(path: &Path) -> Result<Modelfile> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse(&text, path.parent().unwrap_or(Path::new(".")))
}

/// Parse Modelfile text; relative paths resolve against `base`.
pub fn parse(text: &str, base: &Path) -> Result<Modelfile> {
    let mut mf = Modelfile::default();
    let mut lines = text.lines().enumerate();
    while let Some((i, line)) = lines.next() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (directive, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let mut value = rest.trim().to_owned();
        if let Some(start) = value.strip_prefix("\"\"\"") {
            // Multi-line block: everything up to the closing triple quote.
            let mut block = start.to_owned();
            if let Some(end) = block.find("\"\"\"") {
                block.truncate(end);
            } else {
                loop {
                    let Some((_, next)) = lines.next() else {
                        bail!("line {}: unterminated \"\"\"", i + 1)
                    };
                    if let Some(end) = next.find("\"\"\"") {
                        block.push('\n');
                        block.push_str(&next[..end]);
                        break;
                    }
                    block.push('\n');
                    block.push_str(next);
                }
            }
            value = block.trim().to_owned();
            if value.is_empty() {
                bail!("line {}: empty {directive}", i + 1);
            }
            apply(&mut mf, directive, Source::Inline(value), base, i + 1)?;
        } else {
            if value.is_empty() {
                bail!("line {}: {directive} needs a value", i + 1);
            }
            apply(&mut mf, directive, Source::Word(value), base, i + 1)?;
        }
    }
    if mf.from.is_empty() {
        bail!("a Modelfile needs a FROM line");
    }
    Ok(mf)
}

enum Source {
    /// A `"""` block: always literal content.
    Inline(String),
    /// A one-line value: inline JSON if it looks like JSON, otherwise a path.
    Word(String),
}

fn resolve(base: &Path, p: &str) -> PathBuf {
    let p = match p.strip_prefix("~/") {
        Some(rest) => dirs::home_dir().unwrap_or_default().join(rest),
        None => PathBuf::from(p),
    };
    if p.is_absolute() { p } else { base.join(p) }
}

fn content(src: Source, base: &Path, what: &str) -> Result<String> {
    match src {
        Source::Inline(s) => Ok(s),
        Source::Word(w) if w.starts_with('{') || w.starts_with('[') => Ok(w),
        Source::Word(w) => {
            let path = resolve(base, &w);
            std::fs::read_to_string(&path)
                .with_context(|| format!("{what}: reading {}", path.display()))
        }
    }
}

fn json(src: Source, base: &Path, what: &str) -> Result<Value> {
    let text = content(src, base, what)?;
    serde_json::from_str(&text).with_context(|| format!("{what}: not valid JSON"))
}

fn apply(mf: &mut Modelfile, directive: &str, src: Source, base: &Path, line: usize) -> Result<()> {
    match directive.to_ascii_uppercase().as_str() {
        "FROM" => {
            mf.from = match src {
                Source::Word(w) | Source::Inline(w) => w,
            }
        }
        "QUESTIONS" => mf.questions = Some(json(src, base, "QUESTIONS")?),
        "CALIBRATION" => mf.calibration = Some(json(src, base, "CALIBRATION")?),
        "LICENSE" => mf.license = Some(content(src, base, "LICENSE")?),
        "DESCRIPTION" => {
            mf.description = Some(match src {
                Source::Word(w) | Source::Inline(w) => w,
            })
        }
        "PARAMETER" => {
            let (Source::Word(v) | Source::Inline(v)) = src;
            let (key, val) = v.split_once(char::is_whitespace).unwrap_or((&v, ""));
            match key {
                "precision" => mf.precision = Some(val.trim().to_owned()),
                other => bail!("line {line}: unknown PARAMETER {other:?}; supported: precision"),
            }
        }
        other => bail!("line {line}: unknown directive {other:?}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_paths_inline_json_and_blocks() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("cal.json"),
            r#"{"temperature":[1.2,1.0,1.5]}"#,
        )
        .unwrap();
        let mf = parse(
            r#"
# triage
from laya
QUESTIONS """
{"urgent": {"type": "noul", "instructions": "Is it urgent?"}}
"""
CALIBRATION ./cal.json
PARAMETER precision fp32
DESCRIPTION Support triage
LICENSE """MIT"""
"#,
            dir.path(),
        )
        .unwrap();
        assert_eq!(mf.from, "laya");
        assert_eq!(
            mf.questions,
            Some(json!({"urgent": {"type": "noul", "instructions": "Is it urgent?"}}))
        );
        assert_eq!(
            mf.calibration,
            Some(json!({"temperature": [1.2, 1.0, 1.5]}))
        );
        assert_eq!(mf.precision.as_deref(), Some("fp32"));
        assert_eq!(mf.description.as_deref(), Some("Support triage"));
        assert_eq!(mf.license.as_deref(), Some("MIT"));
    }

    #[test]
    fn rejects_bad_input() {
        let base = Path::new(".");
        assert!(parse("QUESTIONS {}", base).is_err()); // no FROM
        assert!(parse("FROM laya\nTEMPLATE x", base).is_err());
        assert!(parse("FROM laya\nPARAMETER top_k 3", base).is_err());
        assert!(parse("FROM laya\nQUESTIONS \"\"\"\n{", base).is_err());
        assert!(parse("FROM laya\nQUESTIONS {not json", base).is_err());
    }
}
