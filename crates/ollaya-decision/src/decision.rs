//! `decision-endpoint-v1`: llm-semantic-router's Decision 1.0 models (a fine-tuned Qwen3.5 text
//! backbone plus an endpoint head), whose rows follow the author's `question_row` and `encode`
//! (prompt version `structured-segmented-candidate-endpoints-global-query-v2`,
//! `ollaya_convert.families.decision.layout`).
//!
//! Every question becomes one causal row of three kinds of segment, each tokenized on its own so
//! no BPE merge crosses a boundary:
//!
//! ```text
//! enc("Context:\n" + payload(state) + "\n\nTask type: " + type + "\nQuestion:\n"
//!     + payload(instructions) + "\nOptions:")
//! enc("\n<option>\n" + canonical({"key": key, "description": description}) + "\n</option>")   per option
//! enc("\n\nSelect the single option best supported by the context and instructions.\nDecision:")
//! ```
//!
//! `cand_pos[j]` is the last token of option j and `query_pos` the last token of the row. The head
//! scores every option from its endpoint and the query; those logits are the option logits of every
//! question type. `payload` is a string verbatim and anything else [`pyjson::dumps_canonical`].

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::layout::TokenEncoder;
use crate::question::{Criteria, Question};
use crate::{Error, pyjson};

/// The prompt this layout builds; `decision.json` must name the same one.
pub const PROMPT_VERSION: &str = "structured-segmented-candidate-endpoints-global-query-v2";
const OPTION_OPEN: &str = "\n<option>\n";
const OPTION_CLOSE: &str = "\n</option>";
const SUFFIX: &str =
    "\n\nSelect the single option best supported by the context and instructions.\nDecision:";
/// The author's descriptions of a noul question's outcomes when the criteria leave one out.
const NOUL_FALSE: &str = "The answer to the question is no.";
const NOUL_TRUE: &str = "The answer to the question is yes.";

/// How a choice option with a null description is rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NullDescription {
    /// As JSON `null` (Eos and Lux).
    Null,
    /// As the option's key (Nox, release 1.3.2).
    Key,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct DecisionLayout {
    pub prompt_version: String,
    /// A longer row rejects the request; nothing is truncated.
    pub max_row_tokens: usize,
    pub min_options: usize,
    /// Choice options.
    pub max_options: usize,
    pub max_score_levels: usize,
    pub choice_null_description: NullDescription,
    /// Right-padding id.
    pub pad: u32,
}

/// One question's row.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRow {
    pub ids: Vec<u32>,
    /// Position of the row's last token, where the question is read.
    pub query: usize,
    /// Position of each option segment's last token, in option order.
    pub cands: Vec<usize>,
}

impl DecisionLayout {
    /// Reject configurations this layout cannot run.
    pub fn validate(&self) -> Result<(), Error> {
        if self.prompt_version != PROMPT_VERSION {
            return Err(Error::invalid(format!(
                "prompt_version {:?} is not {PROMPT_VERSION:?}",
                self.prompt_version
            )));
        }
        if self.max_row_tokens == 0 {
            return Err(Error::invalid("max_row_tokens must be positive"));
        }
        if self.min_options < 1
            || self.min_options > self.max_options
            || self.min_options > self.max_score_levels
        {
            return Err(Error::invalid(format!(
                "min_options={} does not fit max_options={} and max_score_levels={}",
                self.min_options, self.max_options, self.max_score_levels
            )));
        }
        Ok(())
    }

    /// One question's options as (key, description), validated as the author's `question_row`.
    pub fn options(&self, qid: &str, q: &Question) -> Result<Vec<(String, Value)>, Error> {
        let bad = |msg: String| Error::invalid(format!("question {qid:?}: {msg}"));
        let criteria = q.definition.get("criteria").filter(|c| !c.is_null());
        let range = |n: usize, max: usize, what: &str| {
            if n < self.min_options || n > max {
                Err(bad(format!(
                    "this model takes {}..{max} {what}, got {n}",
                    self.min_options
                )))
            } else {
                Ok(())
            }
        };
        match &q.criteria {
            Criteria::Choice(m) => {
                if !criteria.is_some_and(Value::is_object) {
                    return Err(bad(
                        "this model takes choice criteria as an object of label -> description; \
                         write a list of labels as {\"label\": null, ...}"
                            .into(),
                    ));
                }
                if m.len() > self.max_options {
                    return Err(Error::TooManyOptions {
                        question: qid.to_owned(),
                        options: m.len(),
                        head_max_len: self.max_options,
                    });
                }
                range(m.len(), self.max_options, "choice options")?;
                Ok(m.iter()
                    .map(|(k, v)| {
                        let d = match (v, self.choice_null_description) {
                            (Value::Null, NullDescription::Key) => Value::String(k.clone()),
                            (v, _) => v.clone(),
                        };
                        (k.clone(), d)
                    })
                    .collect())
            }
            Criteria::Score(levels) => {
                range(levels.len(), self.max_score_levels, "score levels")?;
                Ok(levels
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), v.clone()))
                    .collect())
            }
            // The author reads the exact keys `false` and `true` and rejects any other key; a key
            // given as null keeps its null description.
            Criteria::Noul { .. } => {
                let empty = Map::new();
                let c = match criteria {
                    None => &empty,
                    Some(Value::Object(c)) => c,
                    Some(_) => return Err(bad("noul criteria must be an object".into())),
                };
                if let Some(k) = c.keys().find(|k| *k != "true" && *k != "false") {
                    return Err(bad(format!(
                        "noul criteria may contain only \"true\" and \"false\", got {k:?}"
                    )));
                }
                let side = |key: &str, default: &str| {
                    c.get(key)
                        .cloned()
                        .unwrap_or_else(|| Value::String(default.to_owned()))
                };
                Ok(vec![
                    ("false".to_owned(), side("false", NOUL_FALSE)),
                    ("true".to_owned(), side("true", NOUL_TRUE)),
                ])
            }
        }
    }

    /// Encode one question's row.
    pub fn encode(
        &self,
        enc: &dyn TokenEncoder,
        state: &Value,
        qid: &str,
        q: &Question,
    ) -> Result<DecisionRow, Error> {
        let options = self.options(qid, q)?;
        let instructions = q.definition.get("instructions").unwrap_or(&Value::Null);
        let prefix = format!(
            "Context:\n{}\n\nTask type: {}\nQuestion:\n{}\nOptions:",
            payload(state),
            q.qtype.name(),
            payload(instructions)
        );
        let mut ids = enc.encode(&prefix)?;
        let mut cands = Vec::with_capacity(options.len());
        for (key, description) in &options {
            let mut option = Map::new();
            option.insert("key".into(), Value::String(key.clone()));
            option.insert("description".into(), description.clone());
            let text = format!(
                "{OPTION_OPEN}{}{OPTION_CLOSE}",
                pyjson::dumps_canonical(&Value::Object(option))
            );
            ids.extend(enc.encode(&text)?);
            cands.push(ids.len() - 1);
        }
        ids.extend(enc.encode(SUFFIX)?);
        if ids.len() > self.max_row_tokens {
            return Err(Error::invalid(format!(
                "question {qid:?}: the row is {} tokens; the model's limit is {} and nothing is truncated",
                ids.len(),
                self.max_row_tokens
            )));
        }
        Ok(DecisionRow {
            query: ids.len() - 1,
            ids,
            cands,
        })
    }
}

/// The author's `payload`: a string verbatim, anything else canonical JSON.
pub fn payload(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => pyjson::dumps_canonical(v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One id per char, so rows can be read back.
    struct Chars;

    impl TokenEncoder for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            Ok(text.chars().map(u32::from).collect())
        }
    }

    fn text(ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| char::from_u32(i).unwrap_or('#'))
            .collect()
    }

    fn layout(null: NullDescription) -> DecisionLayout {
        DecisionLayout {
            prompt_version: PROMPT_VERSION.into(),
            max_row_tokens: 400,
            min_options: 2,
            max_options: 3,
            max_score_levels: 3,
            choice_null_description: null,
            pad: 0,
        }
    }

    fn question(def: Value) -> Question {
        Question::parse("q", &def).unwrap()
    }

    #[test]
    fn builds_rows_like_the_author() {
        let l = layout(NullDescription::Null);
        let q = question(
            json!({"type": "choice", "instructions": {"task": "route", "a": 1.0},
            "criteria": {"b": null, "a": {"z": 1, "y": [true]}}}),
        );
        let row = l
            .encode(&Chars, &json!({"x": "é", "n": 1e-5}), "q", &q)
            .unwrap();
        let want = "Context:\n{\"n\":1e-05,\"x\":\"é\"}\n\nTask type: choice\nQuestion:\n\
            {\"a\":1.0,\"task\":\"route\"}\nOptions:\
            \n<option>\n{\"description\":null,\"key\":\"b\"}\n</option>\
            \n<option>\n{\"description\":{\"y\":[true],\"z\":1},\"key\":\"a\"}\n</option>\
            \n\nSelect the single option best supported by the context and instructions.\nDecision:";
        assert_eq!(text(&row.ids), want);
        assert_eq!(row.query, row.ids.len() - 1);
        assert_eq!(row.cands.len(), 2);
        assert!(row.cands.iter().all(|&p| row.ids[p] == u32::from('>')));
        assert!(row.cands[0] < row.cands[1] && row.cands[1] < row.query);

        // Nox renders a null choice description as the key.
        let nox = layout(NullDescription::Key);
        assert_eq!(
            nox.options("q", &q).unwrap()[0],
            ("b".to_owned(), json!("b"))
        );
    }

    #[test]
    fn noul_and_score_options() {
        let l = layout(NullDescription::Key);
        let bare = question(json!({"type": "noul", "instructions": "ok?"}));
        assert_eq!(
            l.options("q", &bare).unwrap(),
            vec![
                ("false".to_owned(), json!(NOUL_FALSE)),
                ("true".to_owned(), json!(NOUL_TRUE))
            ]
        );
        // A key given as null stays null; the other side takes the default.
        let half =
            question(json!({"type": "noul", "instructions": "ok?", "criteria": {"false": null}}));
        assert_eq!(
            l.options("q", &half).unwrap(),
            vec![
                ("false".to_owned(), Value::Null),
                ("true".to_owned(), json!(NOUL_TRUE))
            ]
        );
        let score =
            question(json!({"type": "score", "instructions": 7, "criteria": ["low", null]}));
        assert_eq!(
            l.options("q", &score).unwrap(),
            vec![
                ("0".to_owned(), json!("low")),
                ("1".to_owned(), Value::Null)
            ]
        );
        let row = l.encode(&Chars, &json!("s"), "q", &score).unwrap();
        assert!(text(&row.ids).starts_with("Context:\ns\n\nTask type: score\nQuestion:\n7\n"));
    }

    #[test]
    fn rejects_like_the_author() {
        let l = layout(NullDescription::Null);
        for def in [
            json!({"type": "choice", "instructions": "x", "criteria": ["a", "b"]}),
            json!({"type": "choice", "instructions": "x", "criteria": {"a": null}}),
            json!({"type": "score", "instructions": "x", "criteria": ["a"]}),
            json!({"type": "score", "instructions": "x", "criteria": [1, 2, 3, 4]}),
            json!({"type": "noul", "instructions": "x", "criteria": {"True": "yes"}}),
            json!({"type": "noul", "instructions": "x", "criteria": {"true": "y", "maybe": "m"}}),
        ] {
            assert!(
                matches!(
                    l.options("q", &question(def.clone())),
                    Err(Error::Invalid(_))
                ),
                "{def}"
            );
        }
        let many = question(
            json!({"type": "choice", "instructions": "x", "criteria": {"a": 1, "b": 2, "c": 3, "d": 4}}),
        );
        assert!(matches!(
            l.options("q", &many),
            Err(Error::TooManyOptions { options: 4, .. })
        ));
        let long = question(json!({"type": "noul", "instructions": "y".repeat(400)}));
        assert!(
            matches!(l.encode(&Chars, &json!(""), "q", &long), Err(Error::Invalid(m)) if m.contains("limit"))
        );
    }
}
