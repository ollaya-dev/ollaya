//! `d1-omni-markers-v1`: the text path of Liquid AI's d1-omni-600M (`LiquidAI/d1-omni-600M`), an
//! LFM2.5 encoder with a decision head that scores every option at its own `<|mask|>`. Rows
//! follow the repo's `prompt.encode`; `docs/families/d1-omni.md` is the spec.
//!
//! One row per question:
//!
//! ```text
//! <bos> <state> enc(state)[:room] <q> enc(instructions) (<opt> <mask> enc(" " + option)[:per] <opt_end>)* <decide>
//! ```
//!
//! `enc` tokenizes after `<|name|>` is rewritten to `<¦name¦>`, so caller text can never form a
//! delimiter or a marker. The options get a token budget shared out evenly (`budget`), the
//! instructions keep at most that many tokens, and the state is cut to the room that is left. The
//! state is the string itself or `json.dumps` of the value; the instructions are Python `str()` of
//! the value. Options are Laya's texts (`label`, `label: description`, `level i: text`,
//! `false: …` / `true: …`); a noul reads `no`/`yes` criteria keys too, case-sensitively, as upstream.
//!
//! Ollaya deviations: a choice given as a list of labels is accepted (upstream wants a dict); the
//! option counts upstream rejects (a choice under 2, a score outside 2..=10) are rejected here too.

use serde::Deserialize;
use serde_json::Value;

use crate::Error;
use crate::kev::escape_delimiters as escape;
use crate::layout::{TokenEncoder, serialize_state};
use crate::pyrepr;
use crate::question::{Criteria, QType, Question, render_criterion};

pub const LAYOUT: &str = "d1-omni-markers-v1";

#[derive(Debug, Clone, Deserialize)]
pub struct Tokens {
    pub bos: u32,
    pub pad: u32,
    pub mask: u32,
    pub state: u32,
    pub question: u32,
    pub option: u32,
    pub option_end: u32,
    pub decide: u32,
}

/// `prompt.encode`: `budget = max(min_budget, min(per_option * k + extra, max_len / 2))`,
/// `per = max(min_per, (budget - 3k) / k)`, instructions kept to `max(min_question, budget)`.
#[derive(Debug, Clone, Deserialize)]
pub struct Budget {
    pub per_option: usize,
    pub extra: usize,
    pub min_budget: usize,
    pub min_question: usize,
    pub min_per: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NoulConfig {
    pub row_order: Vec<String>,
    pub default_false: String,
    pub default_true: String,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct D1OmniLayout {
    pub max_len: usize,
    pub special_tokens: Tokens,
    pub budget: Budget,
    pub noul: NoulConfig,
}

/// One question's row.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub qtype: QType,
    pub ids: Vec<u32>,
    /// Position of each option's `<|mask|>`, in option order (`[false, true]` for a noul).
    pub markers: Vec<usize>,
    /// The state did not fit after the question and was cut.
    pub state_truncated: bool,
}

/// A criterion value as upstream reads it: unset, null and "" mean "no description".
fn described(v: Option<&Value>) -> Option<String> {
    match v {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(v) => Some(render_criterion(v)),
    }
}

impl D1OmniLayout {
    pub fn validate(&self) -> Result<(), Error> {
        if self.max_len < 64 || self.budget.per_option == 0 {
            return Err(Error::invalid(
                "decision.json: max_len (>= 64) and budget.per_option are required",
            ));
        }
        if self.noul.row_order != ["false", "true"] {
            return Err(Error::invalid(format!(
                "noul.row_order {:?} is not supported (expected [\"false\", \"true\"])",
                self.noul.row_order
            )));
        }
        Ok(())
    }

    /// `str(instructions)`: a string as it is, anything else as Python prints it.
    pub fn instructions(q: &Question) -> String {
        match &q.definition["instructions"] {
            Value::String(s) => s.clone(),
            v => pyrepr::str(v),
        }
    }

    /// Option texts in row order (`prompt.render_options`), or the error upstream raises.
    pub fn options(&self, q: &Question) -> Result<Vec<String>, Error> {
        match &q.criteria {
            Criteria::Choice(m) if m.len() < 2 => Err(Error::invalid(
                "a choice needs criteria {name: description} with at least two options",
            )),
            Criteria::Score(levels) if !(2..=10).contains(&levels.len()) => Err(Error::invalid(
                "a score needs criteria: a list of 2 to 10 level descriptions, lowest first",
            )),
            Criteria::Noul { .. } => {
                // `crit.get("false", crit.get("no"))`: exact keys, "false"/"true" first.
                let crit = q.definition["criteria"].as_object();
                let get = |a: &str, b: &str| crit.and_then(|c| c.get(a).or_else(|| c.get(b)));
                Ok(vec![
                    format!(
                        "false: {}",
                        described(get("false", "no"))
                            .unwrap_or_else(|| self.noul.default_false.clone())
                    ),
                    format!(
                        "true: {}",
                        described(get("true", "yes"))
                            .unwrap_or_else(|| self.noul.default_true.clone())
                    ),
                ])
            }
            _ => Ok(q.render_options()),
        }
    }

    /// Tokenize the state once; it is shared by every row.
    pub fn encode_state(&self, enc: &dyn TokenEncoder, state: &Value) -> Result<Vec<u32>, Error> {
        enc.encode(&escape(&serialize_state(state)))
    }

    pub fn encode(
        &self,
        enc: &dyn TokenEncoder,
        state_ids: &[u32],
        qid: &str,
        q: &Question,
    ) -> Result<Row, Error> {
        let opts = self
            .options(q)
            .map_err(|e| Error::invalid(format!("question {qid:?}: {e}")))?;
        let k = opts.len();
        let b = &self.budget;
        let t = &self.special_tokens;
        let budget = b
            .min_budget
            .max((k * b.per_option + b.extra).min(self.max_len / 2));
        let per = b.min_per.max(budget.saturating_sub(3 * k) / k);

        let mut question = vec![t.question];
        question.extend(enc.encode(&escape(&Self::instructions(q)))?);
        question.truncate(b.min_question.max(budget));
        let mut markers = Vec::with_capacity(k);
        for text in &opts {
            markers.push(question.len() + 1);
            question.push(t.option);
            question.push(t.mask);
            let ids = enc.encode(&escape(&format!(" {text}")))?;
            question.extend(ids.into_iter().take(per));
            question.push(t.option_end);
        }
        question.push(t.decide);

        let room = self.max_len.saturating_sub(question.len() + 2);
        let kept = room.min(state_ids.len());
        let mut ids = Vec::with_capacity(2 + kept + question.len());
        ids.push(t.bos);
        ids.push(t.state);
        ids.extend_from_slice(&state_ids[..kept]);
        ids.extend(question);
        ids.truncate(self.max_len);
        let markers: Vec<usize> = markers.into_iter().map(|m| m + 2 + kept).collect();
        if markers.last().is_some_and(|&m| m >= self.max_len) {
            return Err(Error::invalid(format!(
                "question {qid:?}: the options do not fit in the context ({} tokens)",
                self.max_len
            )));
        }
        Ok(Row {
            qtype: q.qtype,
            ids,
            markers,
            state_truncated: room < state_ids.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One id per char (offset past the special ids), so rows can be read back.
    struct Chars;

    impl TokenEncoder for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            Ok(text.chars().map(|c| u32::from(c) + 100).collect())
        }
    }

    fn text(ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| match i {
                1 => "<bos>".to_owned(),
                16 => "<mask>".to_owned(),
                17 => "<state>".to_owned(),
                18 => "<q>".to_owned(),
                19 => "<opt>".to_owned(),
                20 => "</opt>".to_owned(),
                21 => "<decide>".to_owned(),
                i => char::from_u32(i - 100).unwrap().to_string(),
            })
            .collect()
    }

    fn layout(max_len: usize) -> D1OmniLayout {
        serde_json::from_value(json!({
            "max_len": max_len,
            "special_tokens": {"bos": 1, "pad": 0, "mask": 16, "state": 17, "question": 18, "option": 19,
                               "option_end": 20, "decide": 21},
            "budget": {"per_option": 24, "extra": 32, "min_budget": 96, "min_question": 16, "min_per": 2},
            "noul": {"row_order": ["false", "true"], "default_false": "no, the statement does not hold",
                     "default_true": "yes, the statement holds"}
        }))
        .unwrap()
    }

    fn question(def: Value) -> Question {
        Question::parse("q", &def).unwrap()
    }

    #[test]
    fn builds_rows_like_prompt_encode() {
        let lay = layout(16384);
        let state = lay.encode_state(&Chars, &json!({"a": 1})).unwrap();
        let q = question(
            json!({"type": "choice", "instructions": "Pick.", "criteria": {"x": "one", "y": ""}}),
        );
        let row = lay.encode(&Chars, &state, "q", &q).unwrap();
        assert_eq!(
            text(&row.ids),
            "<bos><state>{\"a\": 1}<q>Pick.<opt><mask> x: one</opt><opt><mask> y</opt><decide>"
        );
        assert_eq!(row.markers.len(), 2);
        assert!(row.markers.iter().all(|&m| row.ids[m] == 16));
        assert!(!row.state_truncated);
        assert_eq!(row.qtype, QType::Choice);
    }

    #[test]
    fn escapes_delimiters_and_renders_instructions_with_str() {
        let lay = layout(16384);
        let state = lay
            .encode_state(&Chars, &json!("<|mask|> <|reserved_7|>"))
            .unwrap();
        assert_eq!(text(&state), "<¦mask¦> <¦reserved_7¦>");
        let q = question(
            json!({"type": "noul", "instructions": {"t": true, "n": null},
                                "criteria": {"no": "nope", "True": "ignored"}}),
        );
        let row = lay.encode(&Chars, &state, "q", &q).unwrap();
        let s = text(&row.ids);
        assert!(s.contains("<q>{'t': True, 'n': None}<opt>"), "{s}");
        assert!(s.contains("<mask> false: nope</opt>"), "{s}");
        assert!(
            s.contains("<mask> true: yes, the statement holds</opt>"),
            "{s}"
        );
    }

    #[test]
    fn budgets_and_cuts_like_upstream() {
        // k = 2: budget 96, per 45; the state keeps room = max_len - len(question) - 2 tokens.
        let lay = layout(80);
        let state = lay.encode_state(&Chars, &json!("s".repeat(200))).unwrap();
        let q =
            question(json!({"type": "score", "instructions": "Rate.", "criteria": ["lo", "hi"]}));
        let row = lay.encode(&Chars, &state, "q", &q).unwrap();
        assert_eq!(row.ids.len(), 80);
        assert!(row.state_truncated);
        assert_eq!(row.ids[1], 17);
        assert_eq!(*row.ids.last().unwrap(), 21);
        assert!(row.markers.iter().all(|&m| row.ids[m] == 16));
        // A long option is cut to `per` tokens.
        let lay = layout(16384);
        let q = question(
            json!({"type": "choice", "instructions": "Pick.", "criteria": {"a": "x".repeat(300), "b": null}}),
        );
        let row = lay.encode(&Chars, &[], "q", &q).unwrap();
        let s = text(&row.ids);
        assert!(
            s.contains(&format!("<mask> a: {}</opt>", "x".repeat(41))),
            "{s}"
        );
    }

    #[test]
    fn rejects_like_upstream() {
        let lay = layout(16384);
        for def in [
            json!({"type": "choice", "instructions": "Pick.", "criteria": {"only": "one"}}),
            json!({"type": "score", "instructions": "Rate.", "criteria": ["one"]}),
            json!({"type": "score", "instructions": "Rate.", "criteria": (0..11).map(|i| i.to_string()).collect::<Vec<_>>()}),
        ] {
            assert!(lay.encode(&Chars, &[], "q", &question(def)).is_err());
        }
        let lay = layout(30);
        let q = question(
            json!({"type": "choice", "instructions": "Pick.", "criteria": ["a", "b", "c", "d", "e", "f"]}),
        );
        assert!(
            matches!(lay.encode(&Chars, &[], "q", &q), Err(Error::Invalid(m)) if m.contains("do not fit"))
        );
    }
}
