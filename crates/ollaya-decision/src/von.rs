//! `von-option-marker-v1`: Von 1.1 (`wfzyx/von`), a ModernBERT-large encoder that scores every
//! option at its own `[MASK]` in one sequence. Rows follow upstream `OptionMarkerBackend`
//! (`von-sdk` 1.1.1) and `ollaya_convert.families.von.ref`; `docs/families/von.md` is the spec.
//!
//! Every question is one row (`OptionMarkerModel.pack_sequence`, then `[CLS] … [SEP]`):
//!
//! ```text
//! [CLS] <instructions> <state> [SEP] [MASK] <description 0> [MASK] <description 1> ... [SEP]
//! ```
//!
//! * The state is Python `str()` text, one `key: value` line per top-level key of an object.
//! * An option is read from its description alone (the label only when it has none).
//! * A noul question without criteria (zero-shot) adds a second row with an empty state. Its
//!   logits estimate the question's own yes/no bias, which is subtracted from the first row's.
//!
//! Ollaya deviations, each on inputs upstream rejects or mis-scores:
//! 1. A literal mask token in the state, the instructions or an option becomes a space; upstream
//!    reads every mask token as an option marker.
//! 2. A row over `max_len` tokens keeps the longest prefix of the state that fits; upstream has
//!    no limit.
//! 3. Inputs von's schema rejects are rendered, not refused: structured instructions (as the
//!    shared parser dumps them), list choice criteria, non-string criterion values (Python
//!    `str()`), non-string score examples (`str()` too, where Python would raise).

use std::cell::OnceCell;

use serde::Deserialize;
use serde_json::Value;

use crate::Error;
use crate::layout::TokenEncoder;
use crate::pyrepr::{self, strip, truthy};
use crate::question::{Criteria, QType, Question};

/// Row order of a noul question's options; option logits are returned as `[false, true]`.
const NOUL_ROW_ORDER: [&str; 2] = ["true", "false"];

/// A tokenizer that can also say where each token ends, for cutting the state at a token boundary.
pub trait CharOffsetEncoder: TokenEncoder {
    /// For each token of `encode(text)`, the character (code point) offset where it ends.
    fn char_ends(&self, text: &str) -> Result<Vec<usize>, Error>;
}

#[derive(Debug, Clone, Deserialize)]
pub struct VonTokens {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub pad: u32,
    pub mask_text: String,
    pub sep_text: String,
}

/// `correction = a * (n_true - n_false) + b`, from the empty-state row of a zero-shot noul.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct ZeroShotPrior {
    pub a: f64,
    pub b: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NoulConfig {
    pub row_order: Vec<String>,
    pub default_true: String,
    pub default_false: String,
    pub zero_shot_prior: ZeroShotPrior,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct VonLayout {
    /// Longest row in tokens, `[CLS]` and `[SEP]` included.
    pub max_len: usize,
    pub special_tokens: VonTokens,
    pub noul: NoulConfig,
}

/// The request's state, shared by every row.
#[derive(Debug, Clone)]
pub struct StateText {
    /// Tokens in the rendered state before sanitising: the temperature map's `state_tokens`.
    pub tokens: usize,
    /// The rendered state with mask tokens replaced, as rows read it.
    pub text: String,
    /// Where each token of `text` ends (in characters), once a row has had to cut the state.
    ends: OnceCell<Vec<usize>>,
}

/// One encoder row.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// The packed text (`[CLS]`/`[SEP]` around it are added as ids).
    pub text: String,
    pub ids: Vec<u32>,
    /// Position of each option's `[MASK]`, in row order.
    pub markers: Vec<usize>,
}

/// One question's rows: the question itself, then the empty-state row of a zero-shot noul.
#[derive(Debug, Clone, PartialEq)]
pub struct Scoring {
    pub qtype: QType,
    pub rows: Vec<Row>,
    pub zero_shot: bool,
    /// The state was cut to fit `max_len`.
    pub truncated: bool,
}

/// The state as von renders it (`option_marker_backend._format_state`): a string as it is; an
/// object as one `key: str(value)` line per key; anything else as `str(state)`.
pub fn state_text(state: &Value) -> String {
    match state {
        Value::Object(m) => m
            .iter()
            .map(|(k, v)| format!("{k}: {}", pyrepr::str(v)))
            .collect::<Vec<_>>()
            .join("\n"),
        v => pyrepr::str(v),
    }
}

/// A score level's description (`evaluate_score`).
fn level_description(level: &Value) -> String {
    match level {
        Value::Object(m) => {
            let what = m.get("what").map(pyrepr::str).unwrap_or_default();
            let examples = match m.get("examples") {
                Some(e) if truthy(e) => format!(" Examples: {}", join_examples(e)),
                _ => String::new(),
            };
            strip(&format!("{what}{examples}")).to_owned()
        }
        v => strip(&pyrepr::str(v)).to_owned(),
    }
}

/// `", ".join(examples)`. Python joins a string's characters and a dict's keys, and raises on
/// anything but strings; Ollaya renders those with `str()` instead.
fn join_examples(examples: &Value) -> String {
    match examples {
        Value::String(s) => s.chars().map(String::from).collect::<Vec<_>>().join(", "),
        Value::Array(items) => items.iter().map(pyrepr::str).collect::<Vec<_>>().join(", "),
        Value::Object(m) => m.keys().cloned().collect::<Vec<_>>().join(", "),
        other => pyrepr::str(other),
    }
}

/// The first `chars` characters of `s`.
fn char_prefix(s: &str, chars: usize) -> &str {
    s.char_indices().nth(chars).map_or(s, |(i, _)| &s[..i])
}

impl VonLayout {
    /// Reject configurations this layout cannot run.
    pub fn validate(&self) -> Result<(), Error> {
        let bad = |msg: String| Err(Error::invalid(msg));
        if self.special_tokens.mask_text.is_empty() || self.special_tokens.sep_text.is_empty() {
            return bad("special_tokens: mask_text and sep_text must not be empty".into());
        }
        if self.max_len < 8 {
            return bad(format!("max_len={} is too short", self.max_len));
        }
        if self.noul.row_order != NOUL_ROW_ORDER {
            return bad(format!(
                "noul.row_order {:?} is not supported (expected {NOUL_ROW_ORDER:?})",
                self.noul.row_order
            ));
        }
        let p = self.noul.zero_shot_prior;
        if !(p.a.is_finite() && p.b.is_finite()) {
            return bad("noul.zero_shot_prior must be finite".into());
        }
        Ok(())
    }

    /// Deviation 1: no literal mask token may reach the tokenizer outside the option markers.
    fn sanitize(&self, text: &str) -> String {
        text.replace(&self.special_tokens.mask_text, " ")
    }

    /// Render and measure the state once; it is shared by every row.
    pub fn encode_state(&self, enc: &dyn TokenEncoder, state: &Value) -> Result<StateText, Error> {
        let raw = state_text(state);
        Ok(StateText {
            tokens: enc.encode(&raw)?.len(),
            text: self.sanitize(&raw),
            ends: OnceCell::new(),
        })
    }

    /// Option descriptions in row order (`[true, false]` for noul), and whether the question is a
    /// zero-shot noul.
    pub fn descriptions(&self, q: &Question) -> (Vec<String>, bool) {
        match &q.criteria {
            // evaluate_choice: `desc.strip() if desc else opt.strip()`
            Criteria::Choice(m) => {
                let descriptions = m
                    .iter()
                    .map(|(label, v)| {
                        if truthy(v) {
                            strip(&pyrepr::str(v)).to_owned()
                        } else {
                            strip(label).to_owned()
                        }
                    })
                    .collect();
                (descriptions, false)
            }
            Criteria::Score(levels) => (levels.iter().map(level_description).collect(), false),
            // evaluate_noul: defaults when a side has no description; zero-shot when neither has.
            Criteria::Noul { r#false, r#true } => {
                let t = r#true.as_ref().filter(|v| truthy(v));
                let f = r#false.as_ref().filter(|v| truthy(v));
                let zero_shot = t.is_none() && f.is_none();
                let descriptions = vec![
                    t.map_or_else(|| self.noul.default_true.clone(), pyrepr::str),
                    f.map_or_else(|| self.noul.default_false.clone(), pyrepr::str),
                ];
                (descriptions, zero_shot)
            }
        }
    }

    /// `OptionMarkerModel.pack_sequence`.
    fn pack(&self, state: &str, instructions: &str, descriptions: &[String]) -> String {
        let t = &self.special_tokens;
        let prefix = if instructions.is_empty() {
            strip(state).to_owned()
        } else {
            strip(&format!("{instructions} {state}")).to_owned()
        };
        let mut text = format!("{prefix} {} ", t.sep_text);
        for (i, d) in descriptions.iter().enumerate() {
            if i > 0 {
                text.push(' ');
            }
            text.push_str(&t.mask_text);
            text.push(' ');
            text.push_str(strip(d));
        }
        text
    }

    /// `tok(text)`: the tokenizer's `[CLS] $A [SEP]` template.
    fn tokenize(&self, enc: &dyn TokenEncoder, text: &str) -> Result<Vec<u32>, Error> {
        let body = enc.encode(text)?;
        let mut ids = Vec::with_capacity(body.len() + 2);
        ids.push(self.special_tokens.cls);
        ids.extend(body);
        ids.push(self.special_tokens.sep);
        Ok(ids)
    }

    /// One row; if it is longer than `max_len`, the state is cut at a token boundary until it fits
    /// (deviation 2). Returns the row and whether the state was cut.
    fn fit_row(
        &self,
        enc: &dyn CharOffsetEncoder,
        state: &str,
        ends: &OnceCell<Vec<usize>>,
        instructions: &str,
        descriptions: &[String],
        qid: &str,
    ) -> Result<(Row, bool), Error> {
        let mut text = self.pack(state, instructions, descriptions);
        let mut ids = self.tokenize(enc, &text)?;
        let state_chars = state.chars().count();
        let mut kept = state_chars;
        if ids.len() > self.max_len {
            let ends = match ends.get() {
                Some(e) => e,
                None => {
                    let e = enc.char_ends(state)?;
                    ends.get_or_init(|| e)
                }
            };
            let mut n = ends.len() as isize;
            while ids.len() > self.max_len {
                if n <= 0 {
                    return Err(Error::invalid(format!(
                        "question {qid:?}: its instructions and options alone take {} tokens; \
                         the model reads at most {}",
                        ids.len(),
                        self.max_len
                    )));
                }
                n -= (ids.len() - self.max_len) as isize;
                kept = if n > 0 { ends[n as usize - 1] } else { 0 };
                text = self.pack(char_prefix(state, kept), instructions, descriptions);
                ids = self.tokenize(enc, &text)?;
            }
        }
        let mask = self.special_tokens.mask;
        let markers: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|&(_, &id)| id == mask)
            .map(|(i, _)| i)
            .collect();
        if markers.len() != descriptions.len() {
            return Err(Error::invalid(format!(
                "question {qid:?}: the text tokenizes to {} option markers for {} options",
                markers.len(),
                descriptions.len()
            )));
        }
        Ok((Row { text, ids, markers }, kept < state_chars))
    }

    /// Encode one question's rows against the shared state.
    pub fn encode(
        &self,
        enc: &dyn CharOffsetEncoder,
        state: &StateText,
        qid: &str,
        q: &Question,
    ) -> Result<Scoring, Error> {
        let (descriptions, zero_shot) = self.descriptions(q);
        let instructions = self.sanitize(&q.instructions);
        let descriptions: Vec<String> = descriptions.iter().map(|d| self.sanitize(d)).collect();
        let (row, truncated) = self.fit_row(
            enc,
            &state.text,
            &state.ends,
            &instructions,
            &descriptions,
            qid,
        )?;
        let mut rows = vec![row];
        if zero_shot {
            let empty = OnceCell::new();
            let (row, _) = self.fit_row(enc, "", &empty, &instructions, &descriptions, qid)?;
            rows.push(row);
        }
        Ok(Scoring {
            qtype: q.qtype,
            rows,
            zero_shot,
            truncated,
        })
    }

    /// Option logits in Ollaya's option order from the logits of a question's rows (one per
    /// marker): the first row as it is for choice and score; `[false, true]` for noul, with the
    /// empty-state row's bias taken off `true` for a zero-shot question.
    pub fn option_logits(&self, scoring: &Scoring, rows: &[&[f32]]) -> Vec<f32> {
        let Some(first) = rows.first() else {
            return Vec::new();
        };
        if scoring.qtype != QType::Noul {
            return first.to_vec();
        }
        let (l_true, l_false) = (f64::from(first[0]), f64::from(first[1]));
        let correction = match (scoring.zero_shot, rows.get(1)) {
            (true, Some(null)) => {
                let p = self.noul.zero_shot_prior;
                p.a * (f64::from(null[0]) - f64::from(null[1])) + p.b
            }
            _ => 0.0,
        };
        vec![l_false as f32, (l_true - correction) as f32]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One id per char, `[MASK]` / `[SEP]` as single ids (as the tokenizer's added tokens are),
    /// so rows can be read back.
    struct Chars;

    const MASK: u32 = 0x10_0000;
    const SEP: u32 = 0x10_0001;
    const CLS: u32 = 0x10_0002;

    impl TokenEncoder for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            let mut ids = Vec::new();
            let mut rest = text;
            while let Some(c) = rest.chars().next() {
                if let Some(r) = rest.strip_prefix("[MASK]") {
                    // lstrip: the space before a marker is absorbed.
                    if ids.last() == Some(&u32::from(' ')) {
                        ids.pop();
                    }
                    ids.push(MASK);
                    rest = r;
                } else if let Some(r) = rest.strip_prefix("[SEP]") {
                    ids.push(SEP);
                    rest = r;
                } else {
                    ids.push(u32::from(c));
                    rest = &rest[c.len_utf8()..];
                }
            }
            Ok(ids)
        }
    }

    impl CharOffsetEncoder for Chars {
        fn char_ends(&self, text: &str) -> Result<Vec<usize>, Error> {
            Ok((1..=text.chars().count()).collect())
        }
    }

    fn layout(max_len: usize) -> VonLayout {
        serde_json::from_value(json!({
            "max_len": max_len,
            "special_tokens": {"cls": CLS, "sep": SEP, "mask": MASK, "pad": 0,
                               "mask_text": "[MASK]", "sep_text": "[SEP]"},
            "noul": {"row_order": ["true", "false"], "default_true": "Yes, condition holds true.",
                     "default_false": "No, condition is false.", "zero_shot_prior": {"a": 0.7, "b": 0.0}},
        }))
        .unwrap()
    }

    fn question(def: Value) -> Question {
        Question::parse("q", &def).unwrap()
    }

    #[test]
    fn renders_state_like_python_str() {
        let state = json!({"subject": "it's", "n": 1, "p": 19.99, "ok": true, "none": null,
                           "tags": ["a", "b"], "meta": {"k": "v", "b": false}});
        assert_eq!(
            state_text(&state),
            "subject: it's\nn: 1\np: 19.99\nok: True\nnone: None\ntags: ['a', 'b']\nmeta: {'k': 'v', 'b': False}"
        );
        assert_eq!(state_text(&json!("  raw  ")), "  raw  ");
        assert_eq!(
            state_text(&json!([{"role": "user", "content": "It's"}])),
            "[{'role': 'user', 'content': \"It's\"}]"
        );
        assert_eq!(state_text(&json!(null)), "None");
        assert_eq!(state_text(&json!({})), "");
    }

    #[test]
    fn describes_options_like_von() {
        let l = layout(8192);
        let choice = question(json!({"type": "choice", "instructions": "x",
            "criteria": {" a ": null, "b": " desc ", "c": {"k": 1}, "d": 0, "e": [], "f": 2.5}}));
        assert_eq!(
            l.descriptions(&choice),
            (
                vec![
                    "a".into(),
                    "desc".into(),
                    "{'k': 1}".into(),
                    "d".into(),
                    "e".into(),
                    "2.5".into()
                ],
                false
            )
        );
        let score = question(json!({"type": "score", "instructions": "x", "criteria": [
            " low ", {"what": "mid", "examples": ["a", "b"]}, {"what": "hi", "examples": "xy"},
            {"examples": [1, true]}, {"what": null}, {"what": "w", "examples": []}, 3]}));
        assert_eq!(
            l.descriptions(&score).0,
            [
                "low",
                "mid Examples: a, b",
                "hi Examples: x, y",
                "Examples: 1, True",
                "None",
                "w",
                "3"
            ]
        );
        let zero = question(
            json!({"type": "noul", "instructions": "x", "criteria": {"true": "", "false": null}}),
        );
        assert_eq!(
            l.descriptions(&zero),
            (
                vec![
                    "Yes, condition holds true.".into(),
                    "No, condition is false.".into()
                ],
                true
            )
        );
        let one =
            question(json!({"type": "noul", "instructions": "x", "criteria": {"False": "no way"}}));
        assert_eq!(
            l.descriptions(&one),
            (
                vec!["Yes, condition holds true.".into(), "no way".into()],
                false
            )
        );
    }

    #[test]
    fn packs_rows_like_pack_sequence() {
        let l = layout(8192);
        let state = l.encode_state(&Chars, &json!(" s [MASK] t ")).unwrap();
        assert_eq!((state.tokens, state.text.as_str()), (6, " s   t "));
        let q = question(json!({"type": "choice", "instructions": "Pick [MASK]?",
            "criteria": {"a": "first [MASK]", "b": null}}));
        let s = l.encode(&Chars, &state, "q", &q).unwrap();
        assert_eq!(s.rows.len(), 1);
        let row = &s.rows[0];
        assert_eq!(row.text, "Pick  ?  s   t [SEP] [MASK] first [MASK] b");
        assert_eq!(row.ids[0], CLS);
        assert_eq!(*row.ids.last().unwrap(), SEP);
        assert_eq!(row.markers.len(), 2);
        assert!(row.markers.iter().all(|&m| row.ids[m] == MASK));
        assert!(!s.truncated);

        // No instructions: the prefix is the stripped state; a zero-shot noul adds an empty-state row.
        let q = question(json!({"type": "noul", "instructions": ""}));
        let s = l.encode(&Chars, &state, "q", &q).unwrap();
        assert_eq!(
            s.rows[0].text,
            "s   t [SEP] [MASK] Yes, condition holds true. [MASK] No, condition is false."
        );
        assert_eq!(
            s.rows[1].text,
            " [SEP] [MASK] Yes, condition holds true. [MASK] No, condition is false."
        );
        assert!(s.zero_shot);
    }

    #[test]
    fn cuts_the_state_to_fit() {
        let q = question(json!({"type": "choice", "instructions": "Q", "criteria": ["a", "b"]}));
        // Without the state: [CLS] Q ␠ [SEP] [MASK] ␠ a [MASK] ␠ b [SEP] -> 11 tokens.
        let fits = |max_len: usize, state: &str| {
            let l = layout(max_len);
            let st = l.encode_state(&Chars, &json!(state)).unwrap();
            l.encode(&Chars, &st, "q", &q)
        };
        let s = fits(20, "0123456789").unwrap();
        assert_eq!(s.rows[0].text, "Q 01234567 [SEP] [MASK] a [MASK] b");
        assert_eq!(s.rows[0].ids.len(), 20);
        assert!(s.truncated);
        assert!(!fits(22, "0123456789").unwrap().truncated);
        let s = fits(11, "0123456789").unwrap();
        assert_eq!(s.rows[0].text, "Q [SEP] [MASK] a [MASK] b");
        assert!(
            matches!(fits(10, "0123456789"), Err(Error::Invalid(m)) if m.contains("alone take 11 tokens"))
        );
    }

    #[test]
    fn derives_option_logits() {
        let l = layout(8192);
        let rows = |qtype, zero_shot| Scoring {
            qtype,
            rows: vec![],
            zero_shot,
            truncated: false,
        };
        assert_eq!(
            l.option_logits(&rows(QType::Choice, false), &[&[1.0, 2.0, 3.0]]),
            [1.0, 2.0, 3.0]
        );
        assert_eq!(
            l.option_logits(&rows(QType::Noul, false), &[&[1.0, -1.0]]),
            [-1.0, 1.0]
        );
        let z = l.option_logits(&rows(QType::Noul, true), &[&[1.0, -1.0], &[2.0, 0.0]]);
        assert_eq!(z[0], -1.0);
        assert!((z[1] - (1.0 - 0.7 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn validates_the_config() {
        assert!(layout(8192).validate().is_ok());
        let mut l = layout(8192);
        l.noul.row_order.reverse();
        assert!(l.validate().is_err());
        let mut l = layout(8192);
        l.special_tokens.mask_text.clear();
        assert!(l.validate().is_err());
    }
}
