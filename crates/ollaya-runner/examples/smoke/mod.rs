//! `--smoke` for the parity examples (tier K1 in CLAUDE.md): only the goldens' cases listed in
//! `cases.txt`, checked with the same tolerances as the full set.

use std::path::Path;

use anyhow::{Context, Result, bail};

const CASES: &str = include_str!("cases.txt");

/// The command line without `--smoke`, and whether it was given.
pub fn args() -> (Vec<String>, bool) {
    let (smoke, args): (Vec<String>, Vec<String>) = std::env::args().partition(|a| a == "--smoke");
    (args, !smoke.is_empty())
}

/// The goldens' records, one JSON line each; with `smoke`, only the listed cases.
pub fn lines(path: impl AsRef<Path>, smoke: bool) -> Result<Vec<String>> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    let lines: Vec<String> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && (!smoke || listed(l)))
        .map(str::to_owned)
        .collect();
    if smoke {
        if lines.is_empty() {
            bail!(
                "--smoke: none of the cases in examples/smoke/cases.txt is in {}; list some of its ids there",
                path.display()
            );
        }
        println!("smoke: {} cases of {}", lines.len(), path.display());
    }
    Ok(lines)
}

fn listed(line: &str) -> bool {
    let rec: serde_json::Value = serde_json::from_str(line).unwrap_or_default();
    let id = rec["id"].as_str().unwrap_or_default();
    // parity_decider_vision's goldens carry the shared cases as `text/<id>`.
    let id = id.strip_prefix("text/").unwrap_or(id);
    CASES
        .lines()
        .filter_map(|l| l.split('#').next().map(str::trim))
        .any(|c| !c.is_empty() && c == id)
}
