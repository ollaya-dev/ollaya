//! `nimble-codes-v1`: Bespoke Labs' Nimble (a LoRA on Qwen3.5-9B), whose rows follow the author's
//! `serving_schema.prepare_prompts` on the schema that the author's `/v1/systemone` server builds
//! from TypeSafe questions (`ollaya_convert.families.nimble.ref`).
//!
//! One chat row per question. Every row carries the whole request; only the requested field differs:
//!
//! ```text
//! row    = tok(pre + safe_json({"context": serialize(state), "schema": fields})
//!              + "\n\nRequested field: " + safe_json(qid) + post)
//! fields = [{"name": qid, "description": serialize(instructions),
//!            "choices": [{"code": codes[j], "value": value_j, "description": text_j}, ...]}, ...]
//! noul   values [false, true], texts criteria["false"] or "No", criteria["true"] or "Yes"
//! choice values = labels, texts = the description, or the label when it is null
//! score  values "0".."n-1", texts = the levels
//! ```
//!
//! Nimble v3 ships no TypeSafe server; its mapping is the author's Decision Index engine
//! (`mapping: "decision-index"`, `nimble.evaluation.decision_index_engine`): a missing or null
//! description leaves the choice without one (no label, no `Yes`/`No`), noul criteria may also be
//! keyed `yes`/`no`, and a blank string state is read as its JSON form.
//!
//! `serialize` is the string itself or `json.dumps(ensure_ascii=False)`; `safe_json` also writes `<`
//! and `>` as `\u003c` and `\u003e`, so user text can never form a chat control token and the row is
//! tokenized whole with special tokens parsed. Codes are `A`..`Z`, then two-letter codes; a request
//! whose widest question has more than 26 options uses the "short" template for every row. The graph
//! returns the next-token logits of the 255 codes at each row's last token; a question reads its
//! first k. Rows longer than the model's context are rejected, never truncated.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::layout::TokenEncoder;
use crate::{Error, pyjson};

/// The chat template around the user turn, for one system prompt.
#[derive(Debug, Clone, Deserialize)]
pub struct Template {
    pub pre: String,
    pub post: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Templates {
    /// The training prompt ("one-letter code"), for requests of at most `letter_codes` options.
    pub letter: Template,
    /// The serving extension ("short code"), when a question has more.
    pub short: Template,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Codes {
    pub strings: Vec<String>,
    pub ids: Vec<u32>,
    /// How many codes the training prompt covers (26: `A`..`Z`).
    pub letter_codes: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NoulDefaults {
    pub r#false: String,
    pub r#true: String,
}

/// How TypeSafe questions become schema fields: the author's server for each checkpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mapping {
    /// v2: `nimble/serving/compiler.py` with openjev's defaults.
    #[default]
    Openjev,
    /// v3: `nimble.evaluation.decision_index_engine` (`native_question`, `native_schema`).
    DecisionIndex,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct NimbleLayout {
    pub templates: Templates,
    pub codes: Codes,
    #[serde(default)]
    pub mapping: Mapping,
    /// Noul descriptions when the request gives none (openjev only).
    #[serde(default)]
    pub noul_defaults: Option<NoulDefaults>,
    /// A longer row is rejected (upstream never truncates).
    pub max_prompt_tokens: usize,
}

/// One question's row: its token ids and option count.
#[derive(Debug, Clone, PartialEq)]
pub struct NimbleRow {
    pub ids: Vec<u32>,
    pub options: usize,
}

/// Python's `str.isspace`: Rust's White_Space plus the four information separators (U+001C..U+001F).
fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn blank(s: &str) -> bool {
    s.chars().all(py_isspace)
}

/// `nimble.serving.compiler.serialize`.
fn serialize(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => pyjson::dumps(v, false),
    }
}

/// `safe_json`: `json.dumps(ensure_ascii=False)` with `<` and `>` escaped.
fn safe_json(v: &Value) -> String {
    pyjson::dumps(v, false)
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
}

/// A question as the schema lists it: its description and option values and texts (a choice
/// without a text has no `description` key).
struct Field {
    description: String,
    values: Vec<Value>,
    texts: Vec<Option<String>>,
}

impl NimbleLayout {
    /// Reject configurations this layout cannot run.
    pub fn validate(&self) -> Result<(), Error> {
        let c = &self.codes;
        if c.strings.is_empty()
            || c.strings.len() != c.ids.len()
            || c.letter_codes == 0
            || c.letter_codes > c.strings.len()
            || self.max_prompt_tokens == 0
            || (self.mapping == Mapping::Openjev && self.noul_defaults.is_none())
        {
            return Err(Error::invalid(format!(
                "codes ({} strings, {} ids, {} letter codes), max_prompt_tokens={} and the noul \
                 defaults are inconsistent",
                c.strings.len(),
                c.ids.len(),
                c.letter_codes,
                self.max_prompt_tokens
            )));
        }
        Ok(())
    }

    /// The most options one question may have.
    pub fn max_options(&self) -> usize {
        self.codes.strings.len()
    }

    /// The code token ids the graph returns logits for, in order.
    pub fn code_ids(&self) -> &[u32] {
        &self.codes.ids
    }

    /// One question's field, validated as upstream's compiler and `validate_schema` do. `def` is the
    /// definition as the caller sent it (the typed parse has already checked its shape).
    fn field(&self, qid: &str, def: &Value) -> Result<Field, Error> {
        let bad = |msg: &str| Error::invalid(format!("question {qid:?}: {msg}"));
        if blank(qid) {
            return Err(Error::invalid(
                "question ids must not be blank for this model",
            ));
        }
        let description = serialize(
            def.get("instructions")
                .ok_or_else(|| bad("no 'instructions'; add the text the model should answer"))?,
        );
        if blank(&description) {
            return Err(bad("instructions must not be blank for this model"));
        }
        let crit = def.get("criteria").unwrap_or(&Value::Null);
        let di = self.mapping == Mapping::DecisionIndex;
        let (values, texts): (Vec<Value>, Vec<Option<String>>) = match def
            .get("type")
            .and_then(Value::as_str)
        {
            Some("noul") => {
                let empty = Map::new();
                let c = match crit {
                    Value::Null => &empty,
                    Value::Object(m) => m,
                    _ => return Err(bad("noul criteria must be an object")),
                };
                let true_false = c.keys().all(|k| k == "true" || k == "false");
                // native_question: exactly {yes, no} reads as {true, false}
                let yes_no = di && c.len() == 2 && c.contains_key("yes") && c.contains_key("no");
                if !true_false && !yes_no {
                    return Err(bad(if di {
                        "this model takes noul criteria keyed \"true\"/\"false\" or \"yes\"/\"no\""
                    } else {
                        "this model takes noul criteria with only \"true\" and \"false\" keys"
                    }));
                }
                let (f, t) = if yes_no {
                    ("no", "yes")
                } else {
                    ("false", "true")
                };
                let text = |key: &str, default: Option<&String>| match c.get(key) {
                    None | Some(Value::Null) => default.cloned(),
                    Some(v) => Some(serialize(v)),
                };
                let d = self.noul_defaults.as_ref();
                (
                    vec![Value::Bool(false), Value::Bool(true)],
                    vec![
                        text(f, d.map(|d| &d.r#false)),
                        text(t, d.map(|d| &d.r#true)),
                    ],
                )
            }
            Some("choice") => {
                let pairs: Vec<(String, Option<String>)> = match crit {
                    Value::Object(m) => m
                        .iter()
                        .map(|(k, d)| {
                            let text = match d {
                                Value::Null if di => None,
                                Value::Null => Some(k.clone()),
                                d => Some(serialize(d)),
                            };
                            (k.clone(), text)
                        })
                        .collect(),
                    Value::Array(items) => {
                        // A list of labels: duplicates collapse onto their first position.
                        let mut labels: Vec<String> = Vec::with_capacity(items.len());
                        for item in items {
                            let label = item
                                .as_str()
                                .ok_or_else(|| bad("choice labels must be strings"))?;
                            if !labels.iter().any(|l| l == label) {
                                labels.push(label.to_owned());
                            }
                        }
                        labels
                            .into_iter()
                            .map(|l| {
                                let text = (!di).then(|| l.clone());
                                (l, text)
                            })
                            .collect()
                    }
                    _ => return Err(bad("choice criteria must be an object or a list of labels")),
                };
                if pairs.iter().any(|(label, _)| blank(label)) {
                    return Err(bad("choice labels must not be blank for this model"));
                }
                pairs
                    .into_iter()
                    .map(|(label, text)| (Value::String(label), text))
                    .unzip()
            }
            Some("score") => {
                let levels = crit
                    .as_array()
                    .ok_or_else(|| bad("this model takes score criteria as a list of levels"))?;
                levels
                    .iter()
                    .enumerate()
                    .map(|(i, level)| (Value::String(i.to_string()), Some(serialize(level))))
                    .unzip()
            }
            _ => return Err(bad("unknown type; use one of choice, noul, score")),
        };
        if values.is_empty() || values.len() > self.max_options() {
            return Err(bad(&format!(
                "{} options; this model takes 1 to {}",
                values.len(),
                self.max_options()
            )));
        }
        Ok(Field {
            description,
            values,
            texts,
        })
    }

    /// Every question's prompt text, in request order, and its option count.
    pub fn prompts(
        &self,
        state: &Value,
        defs: &[(&str, &Value)],
    ) -> Result<Vec<(String, usize)>, Error> {
        if defs.is_empty() {
            return Err(Error::invalid(
                "questions must contain at least one question",
            ));
        }
        let fields = defs
            .iter()
            .map(|(qid, def)| self.field(qid, def))
            .collect::<Result<Vec<_>, _>>()?;
        let context = match state {
            // the Decision Index engine passes a blank string state as `json.dumps(state)` (ASCII)
            Value::String(s) if self.mapping == Mapping::DecisionIndex && blank(s) => {
                pyjson::dumps(state, true)
            }
            _ => serialize(state),
        };
        if blank(&context) {
            return Err(Error::invalid("state must not be blank for this model"));
        }
        let schema: Vec<Value> = defs
            .iter()
            .zip(&fields)
            .map(|((qid, _), f)| {
                let choices = self
                    .codes
                    .strings
                    .iter()
                    .zip(f.values.iter().zip(&f.texts))
                    .map(|(code, (value, text))| {
                        let mut c = Map::new();
                        c.insert("code".into(), Value::String(code.clone()));
                        c.insert("value".into(), value.clone());
                        if let Some(text) = text {
                            c.insert("description".into(), Value::String(text.clone()));
                        }
                        Value::Object(c)
                    })
                    .collect();
                let mut m = Map::new();
                m.insert("name".into(), Value::String((*qid).to_owned()));
                m.insert("description".into(), Value::String(f.description.clone()));
                m.insert("choices".into(), Value::Array(choices));
                Value::Object(m)
            })
            .collect();
        let mut request = Map::new();
        request.insert("context".into(), Value::String(context));
        request.insert("schema".into(), Value::Array(schema));
        let widest = fields.iter().map(|f| f.values.len()).max().unwrap_or(0);
        let t = if widest <= self.codes.letter_codes {
            &self.templates.letter
        } else {
            &self.templates.short
        };
        let head = format!(
            "{}{}\n\nRequested field: ",
            t.pre,
            safe_json(&Value::Object(request))
        );
        Ok(defs
            .iter()
            .zip(&fields)
            .map(|((qid, _), f)| {
                let name = safe_json(&Value::String((*qid).to_owned()));
                (format!("{head}{name}{}", t.post), f.values.len())
            })
            .collect())
    }

    /// Every question's row, tokenized whole (special tokens parsed, none added).
    pub fn rows<T: TokenEncoder>(
        &self,
        tok: &T,
        state: &Value,
        defs: &[(&str, &Value)],
    ) -> Result<Vec<NimbleRow>, Error> {
        self.prompts(state, defs)?
            .into_iter()
            .zip(defs)
            .map(|((prompt, options), (qid, _))| {
                let ids = tok.encode(&prompt)?;
                if ids.len() > self.max_prompt_tokens {
                    return Err(Error::invalid(format!(
                        "question {qid:?}: the prompt is {} tokens; this model reads up to {} and never \
                         truncates, so shorten the state or split the questions",
                        ids.len(),
                        self.max_prompt_tokens
                    )));
                }
                Ok(NimbleRow { ids, options })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn layout() -> NimbleLayout {
        let mut strings: Vec<String> = ('A'..='Z').map(String::from).collect();
        strings.extend(["AA", "AB", "AC"].map(String::from));
        let ids = (0..strings.len() as u32).collect();
        NimbleLayout {
            templates: Templates {
                letter: Template {
                    pre: "<L>".into(),
                    post: "</L>".into(),
                },
                short: Template {
                    pre: "<S>".into(),
                    post: "</S>".into(),
                },
            },
            codes: Codes {
                strings,
                ids,
                letter_codes: 26,
            },
            mapping: Mapping::Openjev,
            noul_defaults: Some(NoulDefaults {
                r#false: "No".into(),
                r#true: "Yes".into(),
            }),
            max_prompt_tokens: 8192,
        }
    }

    #[test]
    fn prompts_follow_the_upstream_schema() {
        let q = json!({"type": "noul", "instructions": {"ask": "<b>refund?</b>"},
                       "criteria": {"true": "wants money back"}});
        let c = json!({"type": "choice", "instructions": "Team?", "criteria": {"billing": null, "tech": "bugs"}});
        let s = json!({"type": "score", "instructions": "How bad?", "criteria": ["fine", {"worse": 1.0}]});
        let defs = [("refund", &q), ("team", &c), ("bad", &s)];
        let p = layout().prompts(&json!("Zoë <3"), &defs).unwrap();
        assert_eq!(p.len(), 3);
        let body = concat!(
            r#"<L>{"context": "Zoë \u003c3", "schema": ["#,
            r#"{"name": "refund", "description": "{\"ask\": \"\u003cb\u003erefund?\u003c/b\u003e\"}", "choices": ["#,
            r#"{"code": "A", "value": false, "description": "No"}, {"code": "B", "value": true, "description": "wants money back"}]}, "#,
            r#"{"name": "team", "description": "Team?", "choices": ["#,
            r#"{"code": "A", "value": "billing", "description": "billing"}, {"code": "B", "value": "tech", "description": "bugs"}]}, "#,
            r#"{"name": "bad", "description": "How bad?", "choices": ["#,
            r#"{"code": "A", "value": "0", "description": "fine"}, {"code": "B", "value": "1", "description": "{\"worse\": 1.0}"}]}"#,
            "]}\n\nRequested field: "
        );
        assert_eq!(p[0], (format!("{body}\"refund\"</L>"), 2));
        assert_eq!(p[1], (format!("{body}\"team\"</L>"), 2));
        assert_eq!(p[2].1, 2);
    }

    #[test]
    fn a_wide_question_switches_every_row_to_the_short_template() {
        let labels: Vec<String> = (0..27).map(|i| format!("o{i}")).collect();
        let wide = json!({"type": "choice", "instructions": "Which?", "criteria": labels});
        let yes = json!({"type": "noul", "instructions": "Ok?"});
        let p = layout()
            .prompts(&json!({"a": 1}), &[("w", &wide), ("y", &yes)])
            .unwrap();
        assert!(
            p.iter()
                .all(|(s, _)| s.starts_with("<S>") && s.ends_with("</S>"))
        );
        assert!(
            p[0].0
                .contains(r#"{"code": "AA", "value": "o26", "description": "o26"}"#)
        );
        assert_eq!((p[0].1, p[1].1), (27, 2));
    }

    #[test]
    fn upstream_rejections() {
        let l = layout();
        let ok = json!({"type": "noul", "instructions": "Ok?"});
        let reject = |state: Value, def: Value| l.prompts(&state, &[("q", &def)]).is_err();
        assert!(l.prompts(&json!(" \u{1c}\n"), &[("q", &ok)]).is_err());
        assert!(l.prompts(&json!("x"), &[(" ", &ok)]).is_err());
        assert!(reject(
            json!("x"),
            json!({"type": "noul", "instructions": "  "})
        ));
        assert!(reject(
            json!("x"),
            json!({"type": "noul", "instructions": "q", "criteria": {"True": "y"}})
        ));
        assert!(reject(
            json!("x"),
            json!({"type": "score", "instructions": "q", "criteria": {"0": "a"}})
        ));
        assert!(reject(
            json!("x"),
            json!({"type": "choice", "instructions": "q", "criteria": {" ": null}})
        ));
        let many: Vec<String> = (0..30).map(|i| format!("o{i}")).collect();
        assert!(reject(
            json!("x"),
            json!({"type": "choice", "instructions": "q", "criteria": many})
        ));
        assert!(!reject(
            json!({"n": 0}),
            json!({"type": "noul", "instructions": 5, "criteria": {"true": null}})
        ));
    }

    #[test]
    fn the_decision_index_mapping_leaves_missing_texts_out() {
        let l = NimbleLayout {
            mapping: Mapping::DecisionIndex,
            noul_defaults: None,
            ..layout()
        };
        let n =
            json!({"type": "noul", "instructions": "Ok?", "criteria": {"yes": "fine", "no": null}});
        let c = json!({"type": "choice", "instructions": "Team?", "criteria": {"billing": null, "tech": "bugs"}});
        let l2 = json!({"type": "choice", "instructions": "Which?", "criteria": ["a", "b", "a"]});
        let p = l
            .prompts(&json!(" "), &[("n", &n), ("c", &c), ("l", &l2)])
            .unwrap();
        let body = concat!(
            r#"<L>{"context": "\" \"", "schema": ["#,
            r#"{"name": "n", "description": "Ok?", "choices": ["#,
            r#"{"code": "A", "value": false}, {"code": "B", "value": true, "description": "fine"}]}, "#,
            r#"{"name": "c", "description": "Team?", "choices": ["#,
            r#"{"code": "A", "value": "billing"}, {"code": "B", "value": "tech", "description": "bugs"}]}, "#,
            r#"{"name": "l", "description": "Which?", "choices": ["#,
            r#"{"code": "A", "value": "a"}, {"code": "B", "value": "b"}]}"#,
            "]}\n\nRequested field: "
        );
        assert_eq!(p[0], (format!("{body}\"n\"</L>"), 2));
        let mixed =
            json!({"type": "noul", "instructions": "q", "criteria": {"yes": "y", "false": "n"}});
        assert!(l.prompts(&json!("x"), &[("q", &mixed)]).is_err());
        assert!(layout().prompts(&json!(" "), &[("n", &c)]).is_err());
    }
}
