//! Typed questions: parsing, validation and option rendering.
//!
//! The wire format is TypeSafe's (`type`, `instructions`, `criteria`), which Laya shares. Parsing
//! follows `laya.Agent._check_question` / `_to_internal` so every accepted question renders the
//! same option text the model was trained on.

use indexmap::IndexMap;
use serde_json::Value;

use crate::{Error, pyjson};

/// Question type, numbered as the decision head's type embedding expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "choice" => Some(QType::Choice),
            "score" => Some(QType::Score),
            "noul" => Some(QType::Noul),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Criteria {
    /// Option label -> optional description, in the caller's order.
    Choice(IndexMap<String, Value>),
    /// Level descriptions, level 0 first.
    Score(Vec<Value>),
    /// Optional descriptions of the false and true outcomes.
    Noul {
        r#false: Option<Value>,
        r#true: Option<Value>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub qtype: QType,
    /// Instructions as the model reads them: strings verbatim, anything else as `json.dumps`.
    pub instructions: String,
    pub criteria: Criteria,
}

/// An ordered set of questions, keyed by the caller's question ids.
pub type Questions = IndexMap<String, Question>;

/// Parse a `{"<id>": {type, instructions, criteria}}` object.
pub fn parse_questions(value: &Value) -> Result<Questions, Error> {
    let obj = value.as_object().ok_or_else(|| {
        Error::invalid("questions must be an object of question id -> definition")
    })?;
    if obj.is_empty() {
        return Err(Error::invalid(
            "questions must contain at least one question",
        ));
    }
    obj.iter()
        .map(|(qid, def)| Ok((qid.clone(), Question::parse(qid, def)?)))
        .collect()
}

impl Question {
    pub fn parse(qid: &str, def: &Value) -> Result<Self, Error> {
        let bad = |msg: String| Error::invalid(format!("question {qid:?}: {msg}"));
        let def = def
            .as_object()
            .ok_or_else(|| bad(format!("definition must be an object, got {}", kind(def))))?;
        let qtype = def
            .get("type")
            .and_then(Value::as_str)
            .and_then(QType::parse)
            .ok_or_else(|| {
                bad(format!(
                    "unknown type {}; use one of [\"choice\", \"noul\", \"score\"]",
                    def.get("type").map_or("null".into(), |t| t.to_string())
                ))
            })?;
        let instructions = match def.get("instructions") {
            None => {
                return Err(bad(
                    "no 'instructions'; add the text the model should answer".into(),
                ));
            }
            Some(Value::String(s)) => s.clone(),
            // laya renders structured instructions with plain json.dumps (ensure_ascii on).
            Some(other) => pyjson::dumps(other, true),
        };
        let crit = def.get("criteria").filter(|c| !c.is_null());
        let criteria = match qtype {
            QType::Choice => match crit {
                Some(Value::Object(m)) if !m.is_empty() => {
                    Criteria::Choice(m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                }
                Some(Value::Array(items)) if !items.is_empty() => {
                    // `{c: None for c in criteria}`: duplicates collapse onto their first position.
                    let mut labels = IndexMap::with_capacity(items.len());
                    for item in items {
                        let label = item.as_str().ok_or_else(|| {
                            bad(format!("choice labels must be strings, got {}", kind(item)))
                        })?;
                        labels.entry(label.to_owned()).or_insert(Value::Null);
                    }
                    Criteria::Choice(labels)
                }
                Some(Value::Object(_)) | Some(Value::Array(_)) => {
                    return Err(bad("a choice question needs at least one criterion".into()));
                }
                _ => {
                    return Err(bad(
                        "a choice question takes 'criteria' as a dict of label -> description, or a list of labels"
                            .into(),
                    ));
                }
            },
            QType::Score => match crit {
                Some(Value::Array(levels)) if !levels.is_empty() => Criteria::Score(levels.clone()),
                Some(Value::Array(_)) => {
                    return Err(bad("a score question needs at least one level".into()));
                }
                _ => {
                    return Err(bad(
                        "a score question takes 'criteria' as a list of level descriptions, index 0 first".into(),
                    ));
                }
            },
            QType::Noul => match crit {
                None => Criteria::Noul {
                    r#false: None,
                    r#true: None,
                },
                Some(Value::Object(m)) => {
                    // laya lower-cases keys, so "True"/"FALSE" work; the last spelling wins.
                    let mut t = None;
                    let mut f = None;
                    for (k, v) in m {
                        match k.to_lowercase().as_str() {
                            "true" => t = Some(v.clone()),
                            "false" => f = Some(v.clone()),
                            _ => {}
                        }
                    }
                    Criteria::Noul {
                        r#false: f,
                        r#true: t,
                    }
                }
                Some(_) => {
                    return Err(bad(
                        "a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it"
                            .into(),
                    ));
                }
            },
        };
        Ok(Question {
            qtype,
            instructions,
            criteria,
        })
    }

    /// Number of answer options the model scores.
    pub fn num_options(&self) -> usize {
        match &self.criteria {
            Criteria::Choice(m) => m.len(),
            Criteria::Score(levels) => levels.len(),
            Criteria::Noul { .. } => 2,
        }
    }

    /// Option texts in label-index order, as `laya.common.render_options` writes them.
    pub fn render_options(&self) -> Vec<String> {
        match &self.criteria {
            Criteria::Choice(m) => m
                .iter()
                .map(|(k, v)| match v {
                    Value::Null => k.clone(),
                    Value::String(s) if s.is_empty() => k.clone(),
                    v => format!("{k}: {}", render_criterion(v)),
                })
                .collect(),
            Criteria::Score(levels) => levels
                .iter()
                .enumerate()
                .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
                .collect(),
            Criteria::Noul { r#false, r#true } => vec![
                format!(
                    "false: {}",
                    described(r#false).unwrap_or("no, the statement does not hold".into())
                ),
                format!(
                    "true: {}",
                    described(r#true).unwrap_or("yes, the statement holds".into())
                ),
            ],
        }
    }
}

/// A criterion as text: strings verbatim, anything structured as compact JSON.
pub fn render_criterion(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        v => pyjson::dumps(v, false),
    }
}

fn described(value: &Option<Value>) -> Option<String> {
    match value {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(v) => Some(render_criterion(v)),
    }
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "list",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_like_laya() {
        let q = Question::parse(
            "q",
            &json!({"type": "choice", "instructions": {"task": "route"},
                    "criteria": {"tier1": {"desc": "simple", "sla_h": 24}, "tier2": ["bugs"], "tier3": 0, "none": false, "bare": ""}}),
        )
        .unwrap();
        assert_eq!(q.instructions, "{\"task\": \"route\"}");
        assert_eq!(
            q.render_options(),
            vec![
                "tier1: {\"desc\": \"simple\", \"sla_h\": 24}",
                "tier2: [\"bugs\"]",
                "tier3: 0",
                "none: false",
                "bare"
            ]
        );
        let n = Question::parse("n", &json!({"type": "noul", "instructions": "x", "criteria": {"True": "yes!", "false": ""}}))
            .unwrap();
        assert_eq!(
            n.render_options(),
            vec!["false: no, the statement does not hold", "true: yes!"]
        );
        let l = Question::parse(
            "l",
            &json!({"type": "choice", "instructions": "x", "criteria": ["a", "b", "a"]}),
        )
        .unwrap();
        assert_eq!(l.render_options(), vec!["a", "b"]);
    }

    #[test]
    fn rejects_malformed() {
        for def in [
            json!("nope"),
            json!({"type": "maybe", "instructions": "x"}),
            json!({"type": "noul"}),
            json!({"type": "choice", "instructions": "x"}),
            json!({"type": "choice", "instructions": "x", "criteria": []}),
            json!({"type": "choice", "instructions": "x", "criteria": [1, 2]}),
            json!({"type": "score", "instructions": "x", "criteria": {"a": 1}}),
            json!({"type": "noul", "instructions": "x", "criteria": ["a"]}),
        ] {
            assert!(Question::parse("q", &def).is_err(), "{def}");
        }
    }
}
