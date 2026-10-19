//! `pplx-joint-v1`: Perplexity's pplx-decider (a post-trained Qwen3.5 with a 255-code readout),
//! whose sequences follow upstream `pplx_decider.protocol` (`ollaya_convert.families.pplx.layout`).
//!
//! A request is one sequence holding the state and every question, each followed by its options
//! and a marker token; the readout is read at each marker.
//!
//! ```text
//! text = pre "State:\n" esc(describe(state))
//!        for question i: "\n\nQuestion {i}:\n" esc(describe(instructions or default)) "\nOptions:\n"
//!                        "{code_1}: " esc(option_1) "\n" ... "{code_k}: " esc(option_k) "\n" marker
//!        post
//! ```
//!
//! `describe` is the string itself or `json.dumps(ensure_ascii=False)`; `esc` writes every special
//! token with full-width brackets. A sequence over `max_length` tokens is split by halving its
//! questions, recursively, every part keeping the state; one question that does not fit is rejected.

use serde::Deserialize;
use serde_json::Value;

use crate::layout::TokenEncoder;
use crate::question::{Criteria, QType, Question, Questions};
use crate::{Error, pyjson};

#[derive(Debug, Clone, Deserialize)]
pub struct NoulDefaults {
    pub r#false: String,
    pub r#true: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Templates {
    /// The chat template before and after the user text (system prompt, generation prompt).
    pub pre: String,
    pub post: String,
    pub default_instructions: String,
    pub noul_defaults: NoulDefaults,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Marker {
    pub token: String,
    pub id: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PplxTokens {
    pub pad: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PplxLayout {
    /// Tokens per sequence.
    pub max_length: usize,
    pub max_options: usize,
    /// Answer codes, in option order.
    pub codes: Vec<String>,
    pub marker: Marker,
    /// Strings neutralized in user text, longest first.
    pub escapes: Vec<String>,
    pub special_tokens: PplxTokens,
    pub templates: Templates,
}

/// One sequence: the request, or part of its questions.
#[derive(Debug, Clone, PartialEq)]
pub struct PplxSequence {
    pub ids: Vec<u32>,
    /// Each question's marker position.
    pub markers: Vec<usize>,
    /// Each question's option count.
    pub counts: Vec<usize>,
    /// Each question's index in the request.
    pub questions: Vec<usize>,
}

fn describe(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => pyjson::dumps(other, false),
    }
}

/// Python truthiness (`x or default`).
fn falsy(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64() == Some(0.0),
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

impl PplxLayout {
    pub fn validate(&self) -> Result<(), Error> {
        if self.max_length == 0 || self.max_options == 0 || self.codes.len() < self.max_options {
            return Err(Error::invalid(format!(
                "max_length={}, max_options={} and {} codes are inconsistent",
                self.max_length,
                self.max_options,
                self.codes.len()
            )));
        }
        Ok(())
    }

    fn esc(&self, text: &str) -> String {
        let mut text = text.to_owned();
        for s in &self.escapes {
            text = text.replace(s, &s.replace('<', "＜").replace('>', "＞"));
        }
        text
    }

    /// Option descriptions in prompt order. Noul criteria are read as sent (case-sensitive keys).
    fn options(&self, q: &Question) -> Vec<String> {
        match &q.criteria {
            Criteria::Choice(m) => m
                .iter()
                .map(|(k, v)| match v {
                    Value::Null => k.clone(),
                    v => format!("{k}: {}", describe(v)),
                })
                .collect(),
            Criteria::Score(levels) => levels.iter().map(describe).collect(),
            Criteria::Noul { .. } => {
                let crit = q.definition.get("criteria").filter(|c| !falsy(c));
                let side = |key: &str, default: &str| match crit.and_then(|c| c.get(key)) {
                    Some(v) if !falsy(v) => describe(v),
                    _ => default.to_owned(),
                };
                let d = &self.templates.noul_defaults;
                vec![side("false", &d.r#false), side("true", &d.r#true)]
            }
        }
    }

    fn check(&self, qid: &str, q: &Question) -> Result<(), Error> {
        let bad = |msg: String| Err(Error::invalid(format!("question {qid:?}: {msg}")));
        if qid.is_empty() {
            return Err(Error::invalid("question ids must be nonempty strings"));
        }
        let n = q.num_options();
        if n == 0 || n > self.max_options {
            return bad(format!(
                "{n} options; 1 to {} are supported",
                self.max_options
            ));
        }
        if q.qtype == QType::Score && n < 2 {
            return bad("a score needs at least two ordered levels".into());
        }
        Ok(())
    }

    fn text(&self, state: &Value, items: &[(&str, &Question)]) -> String {
        let mut parts = vec![format!("State:\n{}", self.esc(&describe(state)))];
        for (i, (_, q)) in items.iter().enumerate() {
            let instructions = match q.definition.get("instructions") {
                Some(v) if !falsy(v) => describe(v),
                _ => self.templates.default_instructions.clone(),
            };
            let lines: Vec<String> = self
                .codes
                .iter()
                .zip(self.options(q))
                .map(|(c, d)| format!("{c}: {}", self.esc(&d)))
                .collect();
            parts.push(format!(
                "Question {i}:\n{}\nOptions:\n{}\n{}",
                self.esc(&instructions),
                lines.join("\n"),
                self.marker.token
            ));
        }
        format!(
            "{}{}{}",
            self.templates.pre,
            parts.join("\n\n"),
            self.templates.post
        )
    }

    fn split<T: TokenEncoder>(
        &self,
        tok: &T,
        state: &Value,
        items: &[(usize, &str, &Question)],
        out: &mut Vec<PplxSequence>,
    ) -> Result<(), Error> {
        let pairs: Vec<(&str, &Question)> = items.iter().map(|&(_, k, q)| (k, q)).collect();
        let ids = tok.encode(&self.text(state, &pairs))?;
        if ids.len() > self.max_length {
            if items.len() == 1 {
                return Err(Error::invalid(format!(
                    "question {:?} needs {} tokens with the state, over the {}-token limit",
                    items[0].1,
                    ids.len(),
                    self.max_length
                )));
            }
            let mid = items.len() / 2;
            self.split(tok, state, &items[..mid], out)?;
            return self.split(tok, state, &items[mid..], out);
        }
        let markers: Vec<usize> = (0..ids.len())
            .filter(|&i| ids[i] == self.marker.id)
            .collect();
        if markers.len() != items.len() {
            return Err(Error::invalid(
                "readout markers do not match questions after tokenization",
            ));
        }
        out.push(PplxSequence {
            ids,
            markers,
            counts: items.iter().map(|(_, _, q)| q.num_options()).collect(),
            questions: items.iter().map(|&(i, _, _)| i).collect(),
        });
        Ok(())
    }

    /// The request's sequences (upstream `Encoder.plan`).
    pub fn encode<T: TokenEncoder>(
        &self,
        tok: &T,
        state: &Value,
        questions: &Questions,
    ) -> Result<Vec<PplxSequence>, Error> {
        for (qid, q) in questions {
            self.check(qid, q)?;
        }
        let items: Vec<(usize, &str, &Question)> = questions
            .iter()
            .enumerate()
            .map(|(i, (k, q))| (i, k.as_str(), q))
            .collect();
        let mut out = Vec::new();
        self.split(tok, state, &items, &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_questions;
    use serde_json::json;

    /// One token per char; `#` is the marker.
    struct Chars;
    impl TokenEncoder for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            Ok(text.chars().map(|c| if c == '#' { 7 } else { 1 }).collect())
        }
    }

    fn layout(max_length: usize) -> PplxLayout {
        serde_json::from_value(json!({
            "max_length": max_length, "max_options": 3, "codes": ["A", "B", "C"],
            "marker": {"token": "#", "id": 7}, "escapes": ["<x>"],
            "special_tokens": {"pad": 0},
            "templates": {"pre": "[", "post": "]", "default_instructions": "Choose.",
                          "noul_defaults": {"false": "No", "true": "Yes"}}
        }))
        .unwrap()
    }

    #[test]
    fn the_text_follows_upstream_and_escapes_markers() {
        let qs = parse_questions(&json!({
            "c": {"type": "choice", "instructions": null, "criteria": {"a": null, "b<x>": 0}},
            "n": {"type": "noul", "instructions": "<x>?", "criteria": {"True": "t", "false": ""}}
        }))
        .unwrap();
        let items: Vec<(&str, &Question)> = qs.iter().map(|(k, q)| (k.as_str(), q)).collect();
        assert_eq!(
            layout(1000).text(&json!({"k": "ü"}), &items),
            "[State:\n{\"k\": \"ü\"}\n\nQuestion 0:\nChoose.\nOptions:\nA: a\nB: b＜x＞: 0\n#\
             \n\nQuestion 1:\n＜x＞?\nOptions:\nA: No\nB: Yes\n#]"
        );
    }

    #[test]
    fn long_requests_split_in_halves_and_one_long_question_is_rejected() {
        let qs = parse_questions(&json!({
            "a": {"type": "noul", "instructions": "x".repeat(40)},
            "b": {"type": "noul", "instructions": "y"},
            "c": {"type": "noul", "instructions": "z"}
        }))
        .unwrap();
        let seqs = layout(90).encode(&Chars, &json!("s"), &qs).unwrap();
        let parts: Vec<Vec<usize>> = seqs.iter().map(|s| s.questions.clone()).collect();
        assert_eq!(parts, vec![vec![0], vec![1, 2]]);
        assert!(layout(40).encode(&Chars, &json!("s"), &qs).is_err());
        let one = parse_questions(
            &json!({"s": {"type": "score", "instructions": "", "criteria": ["x"]}}),
        )
        .unwrap();
        assert!(layout(1000).encode(&Chars, &json!("s"), &one).is_err());
    }
}
