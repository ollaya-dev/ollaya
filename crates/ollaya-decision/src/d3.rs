//! `d3-code-readout-v1`: vllm-sr's d3 (Decision 3.0, a fine-tuned Qwen3.5 with a 255-code
//! readout), whose prompts follow upstream `d3_format` (`ollaya_convert.families.d3.layout`).
//!
//! Every question is its own sequence, read at its last token; it runs on the pplx graph.
//!
//! ```text
//! text = pre "State:\n" (empty if state is null or "" else describe(state))
//!        "\n\nQuestion:\n" (describe(instructions) if truthy else default)
//!        "\n\nOptions:\n" "{code_1}: " option_1 "\n" ... "{code_k}: " option_k
//!        "\n\n" closing post
//! ```
//!
//! `describe` is the string itself or `json.dumps(ensure_ascii=False)`; nothing is escaped, so
//! special tokens in user text are read as special tokens, as upstream reads them. Questions are
//! checked as upstream's `normalize_question` checks them; a question over `max_length` tokens is
//! rejected.

use serde::Deserialize;
use serde_json::Value;

use crate::layout::TokenEncoder;
use crate::pplx::{NoulDefaults, PplxSequence, PplxTokens};
use crate::question::{Criteria, QType, Question, Questions};
use crate::{Error, pyjson};

#[derive(Debug, Clone, Deserialize)]
pub struct D3Templates {
    /// The chat template before and after the user text (system prompt, generation prompt).
    pub pre: String,
    pub post: String,
    pub default_instructions: String,
    pub noul_defaults: NoulDefaults,
    /// What the state line says for a null or empty state.
    pub empty_state: String,
    pub closing_line: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct D3Layout {
    /// Tokens per question.
    pub max_length: usize,
    pub max_options: usize,
    /// The fewest and most levels of a score question.
    pub score_levels: [usize; 2],
    /// Answer codes, in option order.
    pub codes: Vec<String>,
    pub special_tokens: PplxTokens,
    pub templates: D3Templates,
}

fn describe(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => pyjson::dumps(other, false),
    }
}

/// Python truthiness (`x if x else default`).
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

impl D3Layout {
    pub fn validate(&self) -> Result<(), Error> {
        let [low, high] = self.score_levels;
        if self.max_length == 0
            || self.max_options == 0
            || self.codes.len() < self.max_options
            || low < 2
            || high < low
        {
            return Err(Error::invalid(format!(
                "max_length={}, max_options={}, {} codes and score_levels={:?} are inconsistent",
                self.max_length,
                self.max_options,
                self.codes.len(),
                self.score_levels
            )));
        }
        Ok(())
    }

    /// The option texts in prompt order, or the reason upstream's `normalize_question` rejects
    /// the question. Criteria are read as sent (noul keys are case-sensitive).
    fn options(&self, qid: &str, q: &Question) -> Result<Vec<String>, Error> {
        let bad = |msg: String| Err(Error::invalid(format!("question {qid:?}: {msg}")));
        let crit = q.definition.get("criteria").filter(|c| !c.is_null());
        match q.qtype {
            // the parser already turned a list of labels into labels without descriptions
            QType::Choice => self.choice(qid, q),
            QType::Noul => {
                let given = match crit {
                    None => None,
                    Some(Value::Object(m)) => {
                        if m.keys().any(|k| k != "false" && k != "true") {
                            return bad(
                                "noul criteria may only describe \"false\" and \"true\"".into()
                            );
                        }
                        Some(m)
                    }
                    Some(_) => {
                        return bad("noul criteria may only describe \"false\" and \"true\"".into());
                    }
                };
                let side = |key: &str, default: &str| match given.and_then(|m| m.get(key)) {
                    Some(v) if !falsy(v) => describe(v),
                    _ => default.to_owned(),
                };
                let d = &self.templates.noul_defaults;
                Ok(vec![side("false", &d.r#false), side("true", &d.r#true)])
            }
            QType::Score => {
                let [low, high] = self.score_levels;
                let levels: Vec<String> = match crit {
                    Some(Value::Array(levels)) if (low..=high).contains(&levels.len()) => {
                        levels.iter().map(describe).collect()
                    }
                    _ => {
                        return bad(format!(
                            "score criteria must be an ordered list of {low} to {high} levels"
                        ));
                    }
                };
                let distinct: std::collections::HashSet<&String> = levels.iter().collect();
                if levels.iter().any(String::is_empty) || distinct.len() != levels.len() {
                    return bad("score levels must be distinct and nonempty".into());
                }
                Ok(levels)
            }
        }
    }

    fn choice(&self, qid: &str, q: &Question) -> Result<Vec<String>, Error> {
        let Criteria::Choice(m) = &q.criteria else {
            unreachable!("a choice question has choice criteria")
        };
        if m.is_empty() || m.len() > self.max_options {
            return Err(Error::invalid(format!(
                "question {qid:?}: choice criteria must map 1 to {} option keys to descriptions",
                self.max_options
            )));
        }
        if m.keys().any(String::is_empty) {
            return Err(Error::invalid(format!(
                "question {qid:?}: choice option keys must be nonempty strings"
            )));
        }
        Ok(m.iter()
            .map(|(k, v)| match v {
                Value::Null => k.clone(),
                v => format!("{k}: {}", describe(v)),
            })
            .collect())
    }

    fn text(&self, state: &Value, q: &Question, options: &[String]) -> String {
        let t = &self.templates;
        let state = match state {
            Value::Null => t.empty_state.clone(),
            Value::String(s) if s.is_empty() => t.empty_state.clone(),
            v => describe(v),
        };
        let instructions = match q.definition.get("instructions") {
            Some(v) if !falsy(v) => describe(v),
            _ => t.default_instructions.clone(),
        };
        let mut lines = vec![
            "State:".to_owned(),
            state,
            String::new(),
            "Question:".to_owned(),
            instructions,
            String::new(),
            "Options:".to_owned(),
        ];
        lines.extend(
            self.codes
                .iter()
                .zip(options)
                .map(|(c, o)| format!("{c}: {o}")),
        );
        lines.extend([String::new(), t.closing_line.clone()]);
        format!("{}{}{}", t.pre, lines.join("\n"), t.post)
    }

    /// One sequence per question, in request order (upstream `D3.prepare`).
    pub fn encode<T: TokenEncoder>(
        &self,
        tok: &T,
        state: &Value,
        questions: &Questions,
    ) -> Result<Vec<PplxSequence>, Error> {
        let mut out = Vec::with_capacity(questions.len());
        for (i, (qid, q)) in questions.iter().enumerate() {
            if qid.is_empty() {
                return Err(Error::invalid("question ids must be nonempty strings"));
            }
            let options = self.options(qid, q)?;
            let ids = tok.encode(&self.text(state, q, &options))?;
            if ids.len() > self.max_length {
                return Err(Error::invalid(format!(
                    "question {qid:?}: the question prompt has {} tokens, over the {}-token limit",
                    ids.len(),
                    self.max_length
                )));
            }
            out.push(PplxSequence {
                markers: vec![ids.len() - 1],
                counts: vec![options.len()],
                questions: vec![i],
                ids,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_questions;
    use serde_json::json;

    struct Chars;
    impl TokenEncoder for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            Ok(text.chars().map(|_| 1).collect())
        }
    }

    fn layout(max_length: usize) -> D3Layout {
        serde_json::from_value(json!({
            "max_length": max_length, "max_options": 3, "score_levels": [2, 3],
            "codes": ["A", "B", "C"], "special_tokens": {"pad": 0},
            "templates": {"pre": "[", "post": "]", "default_instructions": "Choose.",
                          "noul_defaults": {"false": "No", "true": "Yes"},
                          "empty_state": "(empty)", "closing_line": "Code?"}
        }))
        .unwrap()
    }

    fn texts(state: Value, qs: Value) -> Result<Vec<String>, Error> {
        let lay = layout(1000);
        let qs = parse_questions(&qs).unwrap();
        qs.iter()
            .map(|(k, q)| Ok(lay.text(&state, q, &lay.options(k, q)?)))
            .collect()
    }

    #[test]
    fn the_text_follows_upstream() {
        assert_eq!(
            texts(
                json!({"k": "ü"}),
                json!({
                    "c": {"type": "choice", "instructions": null, "criteria": {"a": null, "<b>": 0}},
                    "n": {"type": "noul", "instructions": ["x"], "criteria": {"true": "t", "false": ""}},
                    "s": {"type": "score", "instructions": "", "criteria": ["lo", {"m": 1}]}
                })
            )
            .unwrap(),
            vec![
                "[State:\n{\"k\": \"ü\"}\n\nQuestion:\nChoose.\n\nOptions:\nA: a\nB: <b>: 0\n\nCode?]",
                "[State:\n{\"k\": \"ü\"}\n\nQuestion:\n[\"x\"]\n\nOptions:\nA: No\nB: t\n\nCode?]",
                "[State:\n{\"k\": \"ü\"}\n\nQuestion:\nChoose.\n\nOptions:\nA: lo\nB: {\"m\": 1}\n\nCode?]",
            ]
        );
        let q = json!({"q": {"type": "noul", "instructions": "?"}});
        for empty in [json!(null), json!("")] {
            assert!(texts(empty, q.clone()).unwrap()[0].starts_with("[State:\n(empty)\n"));
        }
        assert!(texts(json!(0), q).unwrap()[0].starts_with("[State:\n0\n"));
    }

    #[test]
    fn questions_upstream_rejects_are_rejected() {
        for q in [
            json!({"type": "noul", "instructions": "?", "criteria": {"True": "x"}}),
            json!({"type": "score", "instructions": "?", "criteria": ["one"]}),
            json!({"type": "score", "instructions": "?", "criteria": ["a", "b", "c", "d"]}),
            json!({"type": "score", "instructions": "?", "criteria": ["a", "a"]}),
            json!({"type": "score", "instructions": "?", "criteria": ["", "a"]}),
            json!({"type": "choice", "instructions": "?", "criteria": {"a": 1, "b": 2, "c": 3, "d": 4}}),
            json!({"type": "choice", "instructions": "?", "criteria": {"": 1}}),
        ] {
            assert!(texts(json!("s"), json!({"q": q})).is_err(), "{q}");
        }
    }

    #[test]
    fn every_question_is_a_sequence_read_at_its_last_token() {
        let qs = parse_questions(&json!({
            "a": {"type": "noul", "instructions": "x".repeat(40)},
            "b": {"type": "choice", "instructions": "y", "criteria": ["p", "q", "r"]}
        }))
        .unwrap();
        let seqs = layout(1000).encode(&Chars, &json!("s"), &qs).unwrap();
        assert_eq!(seqs.len(), 2);
        for (i, s) in seqs.iter().enumerate() {
            assert_eq!(
                (s.markers.clone(), s.questions.clone()),
                (vec![s.ids.len() - 1], vec![i])
            );
        }
        assert_eq!(seqs[1].counts, vec![3]);
        assert!(layout(60).encode(&Chars, &json!("s"), &qs).is_err());
    }
}
