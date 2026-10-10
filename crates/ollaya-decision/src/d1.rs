//! LiquidAI d1's default JSON-only, no-system, no-reasoning decision protocol.
//! Reference: LiquidAI/d1-3B prompt.py at 051bcc464b01b9f92942b364d9586b0ef5912432.
//! Independently implemented; answer forms are max-pooled, never summed or averaged.
use crate::{
    Error, pyjson,
    question::{Criteria, parse_questions},
};
use serde::Deserialize;
use serde_json::Value;
pub const LAYOUT: &str = "d1-v1";
pub const PRE: &str = "<|startoftext|><|im_start|>user\n";
pub const POST: &str = "<|im_end|>\n<|im_start|>assistant\n";
#[derive(Debug, Clone, Deserialize)]
pub struct D1Config {
    pub layout: String,
}
#[derive(Debug, Clone)]
pub struct Prompt {
    pub suffix: String,
    pub groups: Vec<Vec<u32>>,
}
impl D1Config {
    pub fn validate(&self) -> Result<(), Error> {
        if self.layout != LAYOUT {
            return Err(Error::invalid("invalid d1 layout"));
        }
        Ok(())
    }
}
pub fn state_text(state: &Value) -> String {
    if state.is_null() {
        return String::new();
    }
    let s = state
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| pyjson::dumps_indent(state, false, 2));
    format!("{s}\n\n\nQUESTION:\n")
}
fn text(v: &Value) -> Result<&str, Error> {
    v.as_str()
        .ok_or_else(|| Error::invalid("d1 descriptions must be strings or null"))
}
fn forms<F: FnMut(&str) -> Result<Vec<u32>, Error>>(
    texts: &[String],
    tok: &mut F,
) -> Result<Vec<u32>, Error> {
    let mut out = Vec::new();
    for s in texts {
        let ids = tok(s)?;
        if ids.len() == 1 && !out.contains(&ids[0]) {
            out.push(ids[0]);
        }
    }
    if out.is_empty() {
        return Err(Error::invalid("d1 answer has no single-token form"));
    }
    Ok(out)
}
fn fallback() -> Vec<String> {
    let mut out: Vec<String> = ('A'..='Z').map(String::from).collect();
    out.extend((0..100).map(|i| format!("{i:02}")));
    out.extend(('a'..='z').map(String::from));
    out.extend((0..200).map(|i| format!("#{i}")));
    out.extend(('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}"))));
    out
}
pub fn questions<F: FnMut(&str) -> Result<Vec<u32>, Error>>(
    value: &Value,
    mut tok: F,
) -> Result<Vec<(String, Prompt)>, Error> {
    let mut out = Vec::new();
    for (qid, q) in parse_questions(value)? {
        if qid.contains('\u{1f}') {
            return Err(Error::invalid("reserved question-name character"));
        }
        // Structured descriptions/instructions are not part of the author's typed protocol.
        if !q.definition["instructions"].is_string() {
            return Err(Error::invalid("d1 instructions must be a string"));
        }
        let (block, groups) = match q.criteria {
            Criteria::Choice(m) => {
                if m.len() > 255 {
                    return Err(Error::TooManyOptions {
                        question: qid,
                        options: m.len(),
                        head_max_len: 255,
                    });
                }
                let labels: Vec<_> = m.keys().map(|s| s.trim()).collect();
                let native = labels
                    .iter()
                    .all(|s| s.chars().count() == 1 && s.chars().all(char::is_alphabetic));
                let pool = fallback();
                let mut used = Vec::new();
                let mut groups = Vec::new();
                let mut lines = Vec::new();
                for (i, (label, desc)) in m.iter().enumerate() {
                    let code = if native {
                        labels[i].to_owned()
                    } else if m.len() <= 26 {
                        ((b'A' + i as u8) as char).to_string()
                    } else {
                        format!("{i:02}")
                    };
                    let mut chosen = None;
                    for candidate in std::iter::once(&code).chain(pool.iter()) {
                        let ids = tok(candidate)?;
                        if ids.len() == 1 && !used.contains(&ids[0]) {
                            chosen = Some((candidate.clone(), ids[0]));
                            break;
                        }
                    }
                    let (code, id) = chosen.ok_or_else(|| {
                        Error::invalid("d1 has no distinct single-token alias left")
                    })?;
                    used.push(id);
                    let mut g = vec![id];
                    let ids = tok(&format!(" {code}"))?;
                    if ids.len() == 1 && ids[0] != id {
                        g.push(ids[0]);
                    }
                    groups.push(g);
                    let desc = if desc.is_null() { "" } else { text(desc)? };
                    let desc = if desc.is_empty() {
                        label.replace('_', " ")
                    } else {
                        desc.to_owned()
                    };
                    lines.push(format!("{code} {desc}"));
                }
                (
                    format!(
                        "{}\n\nOptions:\n{}\n\nReply with the option code only.",
                        q.instructions,
                        lines.join("\n")
                    ),
                    groups,
                )
            }
            Criteria::Noul { .. } => {
                let extra = match q
                    .definition
                    .get("criteria")
                    .and_then(Value::as_object)
                    .filter(|m| !m.is_empty())
                {
                    None => String::new(),
                    Some(m) => {
                        let desc = |k| -> Result<String, Error> {
                            match m.get(k) {
                                None | Some(Value::Null) => Ok("None".into()),
                                Some(v) => Ok(text(v)?.into()),
                            }
                        };
                        format!("\nYes: {}\nNo: {}", desc("true")?, desc("false")?)
                    }
                };
                // Wire order is false, true; author's prompt/readout order is yes, no.
                let no = forms(&["no".into(), "No".into(), "NO".into()], &mut tok)?;
                let yes = forms(&["yes".into(), "Yes".into(), "YES".into()], &mut tok)?;
                (
                    format!("{}{extra}\n\nReply with yes or no only.", q.instructions),
                    vec![no, yes],
                )
            }
            Criteria::Score(levels) => {
                if !(2..=10).contains(&levels.len()) {
                    return Err(Error::invalid("d1 scores require 2 to 10 levels"));
                }
                let lines = levels
                    .iter()
                    .enumerate()
                    .map(|(i, v)| Ok(format!("{i} {}", text(v)?)))
                    .collect::<Result<Vec<_>, Error>>()?;
                let groups = (0..levels.len())
                    .map(|i| forms(&[i.to_string()], &mut tok))
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    format!(
                        "{}\n\n{}\n\nReply with a single digit 0-{} only.",
                        q.instructions,
                        lines.join("\n"),
                        levels.len() - 1
                    ),
                    groups,
                )
            }
        };
        if block.contains('\0') {
            return Err(Error::invalid("NUL in d1 prompt"));
        }
        out.push((
            qid,
            Prompt {
                suffix: block + POST,
                groups,
            },
        ));
    }
    Ok(out)
}
/// Max-pool answer forms in wire order. Every group is nonempty by construction.
pub fn pool(logits: &[f32], groups: &[Vec<usize>]) -> Vec<f32> {
    groups
        .iter()
        .map(|g| {
            g.iter()
                .map(|&i| logits[i])
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn pinned_author_protocol() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/d1_protocol.json")).unwrap();
        for case in fixture["fixtures"].as_array().unwrap() {
            let q = serde_json::json!({"q":case["question"]});
            let got = questions(&q, |s| {
                serde_json::from_value(fixture["tokenizations"][s].clone())
                    .map_err(|e| Error::invalid(e.to_string()))
            })
            .unwrap();
            assert_eq!(got[0].1.suffix, case["suffix"].as_str().unwrap());
            assert_eq!(
                serde_json::to_value(&got[0].1.groups).unwrap(),
                case["groups"]
            );
        }
    }
    #[test]
    fn rejects_invalid_rubrics() {
        for q in [
            serde_json::json!({"type":"score","instructions":"x","criteria":["one"]}),
            serde_json::json!({"type":"choice","instructions":"x","criteria":{"a":42}}),
            serde_json::json!({"type":"noul","instructions":"a\u{0}b"}),
        ] {
            assert!(questions(&serde_json::json!({"q":q}), |_| Ok(vec![1])).is_err());
        }
    }
    #[test]
    fn state_and_pool() {
        assert_eq!(state_text(&Value::Null), "");
        assert_eq!(
            state_text(&json!({"x": "é"})),
            "{\n  \"x\": \"é\"\n}\n\n\nQUESTION:\n"
        );
        assert_eq!(pool(&[1., 3., 2.], &[vec![0, 1], vec![2]]), [3., 2.]);
    }
    #[test]
    fn protocol() {
        let q = json!({"n":{"type":"noul","instructions":"Yes?","criteria":{"true":"ok"}},"s":{"type":"score","instructions":"Rate","criteria":["low","high"]}});
        let out = questions(&q, |s| Ok(vec![s.bytes().map(u32::from).sum()])).unwrap();
        assert_eq!(
            out[0].1.suffix,
            format!("Yes?\nYes: ok\nNo: None\n\nReply with yes or no only.{POST}")
        );
        assert_eq!(out[0].1.groups[0], [221, 189, 157]);
        assert_eq!(
            out[1].1.suffix,
            format!("Rate\n\n0 low\n1 high\n\nReply with a single digit 0-1 only.{POST}")
        );
    }
}
