//! `decima-late-interaction-v1` and `-v2`: Decima (`amyrmahdy/decima-small`, `-base`, `-agent`), an
//! encoder with a late-interaction option scorer. Requests follow the author's `decima/systemone.py`,
//! `decima/model.py` and `decima/render.py` at v2.3.0 (`ollaya_convert.families.decima.layout`);
//! `docs/families/decima.md` is the spec.
//!
//! Every question becomes one state row and one row per option, each encoded on its own:
//!
//! ```text
//! state row   [cls] tok(state_prefix + normalize(text + "\n" + state))[..max_state_tokens - 2] [sep]
//! option row  [cls] tok(option_text)[..max_option_tokens - 2] [sep]
//! full        option_prefix + normalize(text + " " + option)
//! fitted      option_prefix + head + " … " + normalize(option)   (option_prefix + normalize(option)
//!             when no word fits), head the longest run of normalize(text)'s leading words that fits
//! ```
//!
//! `option_text` is `render.py`'s `choice_texts`. v2 (`option_fit`, the author's `choice_fit`): an
//! option whose full text fits `max_option_tokens` keeps it, any other is fitted. v1: every option
//! keeps its full text, unless cutting them makes two of the question's rows identical; then the
//! question's options are rendered as in v2.
//!
//! * `text` is the question's instructions (the question id when they are absent or null; JSON
//!   other than a string as `json.dumps(ensure_ascii=False)`). A noul question whose criteria give
//!   a `true` or `false` description appends `"\nTrue if: …\nFalse if: …"`.
//! * Options: a choice's `label: description` (the label alone without a description), each score
//!   level's text, and `yes` / `no` for noul.
//! * The state is the string itself, or `json.dumps(ensure_ascii=False)` of an object or array.
//! * `normalize` is NFC, then Python's `str.strip()`.
//!
//! The graph scores every option against its question's state row. Choice and noul answers are a
//! softmax of the scores over the calibration temperature (the author's); score answers come from
//! the cumulative-link ordinal head ([`ordinal_log_probs`]), which reads the scores over the
//! same temperature.
//!
//! A state row over `max_state_tokens` keeps its first tokens and is flagged, so `/v1/systemone`
//! answers 422 `STATE_TRUNCATED` as upstream's server does (its `model.py` truncates, as
//! `/api/decide` does). An option row over `max_option_tokens` is cut silently, as upstream cuts it.

use std::collections::HashSet;

use serde::Deserialize;
use serde_json::Value;

use crate::layout::TokenEncoder;
use crate::pyrepr::{split, strip};
use crate::question::{Criteria, QType, Question, Questions};
use crate::{Error, pyjson};

/// Options keep the right-hand cut unless it collapses two of them (`option_fit` false).
pub const LAYOUT: &str = "decima-late-interaction-v1";
/// Options are always kept whole in the option window (`option_fit` true).
pub const LAYOUT_FIT: &str = "decima-late-interaction-v2";

/// The tokenizer this layout needs: token ids without special tokens, and Unicode NFC (upstream's
/// `normalize` runs `unicodedata.normalize("NFC", text)` before the tokenizer sees the text).
pub trait DecimaEncoder: TokenEncoder {
    fn nfc(&self, text: &str) -> String;
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecimaTokens {
    pub cls: u32,
    pub sep: u32,
    pub pad: u32,
}

/// A noul question's options and the text its criteria append to the question.
#[derive(Debug, Clone, Deserialize)]
pub struct NoulText {
    /// The options as the model reads them, the `true` one first.
    pub options: [String; 2],
    /// Appended to the question when a `true` or `false` description is given; `{true}` and
    /// `{false}` are replaced in one pass.
    pub criteria: String,
    /// Written for the side without a description.
    pub missing: String,
}

/// Upstream's request limits, `[min, max]`.
#[derive(Debug, Clone, Deserialize)]
pub struct Limits {
    pub questions: [usize; 2],
    pub choices: [usize; 2],
    pub levels: [usize; 2],
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct DecimaLayout {
    pub state_prefix: String,
    pub option_prefix: String,
    pub question_in_state: bool,
    pub question_in_options: bool,
    /// Longest state row in tokens, `[cls]` and `[sep]` included.
    pub max_state_tokens: usize,
    /// Longest option row in tokens, `[cls]` and `[sep]` included.
    pub max_option_tokens: usize,
    /// The author's `choice_fit` (`decima.json`): every option is kept whole in its row, and only
    /// the question's copy beside it is shortened ([`LAYOUT_FIT`]). Without it ([`LAYOUT`]) the
    /// right-hand cut stays, unless it makes two of a question's option rows identical.
    #[serde(default)]
    pub option_fit: bool,
    pub special_tokens: DecimaTokens,
    pub noul: NoulText,
    pub limits: Limits,
    /// The author's fitted temperature (`decima.json`). The ordinal head of score questions reads
    /// the scores divided by it; choice and noul get it from the calibration layer.
    pub temperature: f64,
}

/// One question's rows, options in upstream's order: a choice's labels, score levels, noul
/// `[yes, no]`.
#[derive(Debug, Clone, PartialEq)]
pub struct DecimaQuestion {
    /// The head that answers it ([`Question::head`]): an ordinal choice reads the score head.
    pub qtype: QType,
    pub state_ids: Vec<u32>,
    pub option_ids: Vec<Vec<u32>>,
    /// The state row was cut to `max_state_tokens`.
    pub state_truncated: bool,
}

/// Every question of a request.
#[derive(Debug, Clone, PartialEq)]
pub struct DecimaRequest {
    pub questions: Vec<DecimaQuestion>,
    /// Tokens in the state's text, before any truncation.
    pub state_tokens: usize,
}

/// `systemone._text`: nothing (or null) as `""`, a string as it is, anything else as
/// `json.dumps(ensure_ascii=False)`.
fn text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(v) => pyjson::dumps(v, false),
    }
}

/// The state as upstream's server reads it: a string as it is, an object or array as
/// `json.dumps(ensure_ascii=False)`. Numbers, booleans and null are rejected.
pub fn state_string(state: &Value) -> Result<String, Error> {
    match state {
        Value::String(s) => Ok(s.clone()),
        Value::Object(_) | Value::Array(_) => Ok(pyjson::dumps(state, false)),
        _ => Err(Error::invalid("state must be a string, object or array")),
    }
}

/// Replace `{true}` and `{false}` in one pass over `template`; values are never rescanned.
fn fill(template: &str, t: &str, f: &str) -> String {
    let mut out = String::with_capacity(template.len() + t.len() + f.len());
    let mut rest = template;
    while let Some(c) = rest.chars().next() {
        if let Some(r) = rest.strip_prefix("{true}") {
            out.push_str(t);
            rest = r;
        } else if let Some(r) = rest.strip_prefix("{false}") {
            out.push_str(f);
            rest = r;
        } else {
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

/// `F.softplus` (beta 1, threshold 20).
fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn log_softmax(z: &[f64]) -> Vec<f64> {
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lse = max + z.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
    z.iter().map(|v| v - lse).collect()
}

/// `DecimaModel.ordinal_log_probs` for one question, on the graph's outputs: `scores` (raw, before
/// the temperature), `g` = `ord_g(z_k)` and `gap` = `ord_gap(z_k)` per level, level 0 first.
///
/// ```text
/// s = scores / T                    expected = Σ softmax(s)_k · k − (K − 1)/2
/// g = mean_k ord_g(z_k) + expected  (= ord_g(mean_k z_k) + expected: ord_g is linear)
/// gaps = softplus(ord_gap(z)) + 1e-3          θ_j = Σ_{i≤j} gaps_i − Σ gaps / 2,  j < K − 1
/// p = [1, σ(g − θ)] − [σ(g − θ), 0]           log p = log_softmax(log(max(p, 1e-7)))
/// ```
pub fn ordinal_log_probs(scores: &[f32], g: &[f32], gap: &[f32], temperature: f64) -> Vec<f64> {
    let k = scores.len();
    if k == 0 {
        return Vec::new();
    }
    let s: Vec<f64> = scores.iter().map(|&x| f64::from(x) / temperature).collect();
    let p = log_softmax(&s);
    let expected = p
        .iter()
        .enumerate()
        .map(|(i, lp)| lp.exp() * i as f64)
        .sum::<f64>()
        - (k as f64 - 1.0) / 2.0;
    let latent = g.iter().map(|&x| f64::from(x)).sum::<f64>() / k as f64 + expected;
    let gaps: Vec<f64> = gap.iter().map(|&x| softplus(f64::from(x)) + 1e-3).collect();
    let half = gaps.iter().sum::<f64>() / 2.0;
    let mut above = Vec::with_capacity(k - 1);
    let mut theta = 0.0;
    for gap in &gaps[..k - 1] {
        theta += gap;
        above.push(sigmoid(latent - (theta - half)));
    }
    let levels: Vec<f64> = (0..k)
        .map(|i| {
            let hi = if i == 0 { 1.0 } else { above[i - 1] };
            let lo = if i + 1 == k { 0.0 } else { above[i] };
            (hi - lo).max(1e-7).ln()
        })
        .collect();
    log_softmax(&levels)
}

impl DecimaLayout {
    /// The layout id a `decision.json` with this configuration declares.
    pub fn layout_id(&self) -> &'static str {
        if self.option_fit { LAYOUT_FIT } else { LAYOUT }
    }

    /// Reject configurations this layout cannot run.
    pub fn validate(&self) -> Result<(), Error> {
        let bad = |msg: String| Err(Error::invalid(msg));
        if self.max_state_tokens < 3 || self.max_option_tokens < 3 {
            return bad(
                "max_state_tokens and max_option_tokens must leave room for a token".into(),
            );
        }
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return bad(format!("temperature {} is not positive", self.temperature));
        }
        let l = &self.limits;
        if [l.questions, l.choices, l.levels]
            .iter()
            .any(|[lo, hi]| *lo == 0 || lo > hi)
        {
            return bad("limits must be [min, max] with 0 < min <= max".into());
        }
        Ok(())
    }

    /// `systemone.to_question`: the question's type, its text, and its options as the model reads
    /// them. Instructions and noul criteria come from the definition as the caller sent it; choice
    /// labels and score levels from the parsed criteria (which keep them as upstream does).
    pub fn question(&self, qid: &str, q: &Question) -> Result<(String, Vec<String>), Error> {
        let bad = |msg: String| Error::invalid(format!("question {qid:?}: {msg}"));
        let def = &q.definition;
        let mut body = match def.get("instructions") {
            None | Some(Value::Null) => qid.to_owned(),
            v => text(v),
        };
        let within = |n: usize, [lo, hi]: [usize; 2]| (lo..=hi).contains(&n);
        let options = match &q.criteria {
            Criteria::Choice(m) => {
                if !within(m.len(), self.limits.choices) {
                    let [lo, hi] = self.limits.choices;
                    return Err(bad(format!(
                        "this model takes {lo} to {hi} choices, got {}",
                        m.len()
                    )));
                }
                // The ordinal head was trained on at most 10 levels (decima bc02983).
                if q.ordinal && !within(m.len(), self.limits.levels) {
                    let [lo, hi] = self.limits.levels;
                    return Err(bad(format!(
                        "an ordinal choice takes {lo} to {hi} labels, got {}",
                        m.len()
                    )));
                }
                m.iter()
                    .map(|(label, v)| {
                        let d = text(Some(v));
                        if strip(&d).is_empty() {
                            label.clone()
                        } else {
                            format!("{label}: {d}")
                        }
                    })
                    .collect()
            }
            Criteria::Score(levels) => {
                if !within(levels.len(), self.limits.levels) {
                    let [lo, hi] = self.limits.levels;
                    return Err(bad(format!(
                        "this model takes {lo} to {hi} score levels, got {}",
                        levels.len()
                    )));
                }
                levels.iter().map(|v| text(Some(v))).collect()
            }
            Criteria::Noul { .. } => {
                // Upstream reads the exact keys "true" and "false" of an object.
                if let Some(Value::Object(m)) = def.get("criteria") {
                    let t = strip(&text(m.get("true"))).to_owned();
                    let f = strip(&text(m.get("false"))).to_owned();
                    if !t.is_empty() || !f.is_empty() {
                        let or_missing = |s: String| {
                            if s.is_empty() {
                                self.noul.missing.clone()
                            } else {
                                s
                            }
                        };
                        body.push_str(&fill(&self.noul.criteria, &or_missing(t), &or_missing(f)));
                    }
                }
                self.noul.options.to_vec()
            }
        };
        Ok((body, options))
    }

    /// `[cls] tok(text) [sep]`, cut to `max_len` tokens, and whether it was cut.
    fn ids(
        &self,
        enc: &dyn DecimaEncoder,
        text: &str,
        max_len: usize,
    ) -> Result<(Vec<u32>, bool), Error> {
        let body = enc.encode(text)?;
        let keep = body.len().min(max_len - 2);
        let mut ids = Vec::with_capacity(keep + 2);
        ids.push(self.special_tokens.cls);
        ids.extend_from_slice(&body[..keep]);
        ids.push(self.special_tokens.sep);
        Ok((ids, keep < body.len()))
    }

    /// `[cls] tok(prefix + normalize(text)) [sep]`, cut to `max_len` tokens, and whether it was cut.
    fn row(
        &self,
        enc: &dyn DecimaEncoder,
        prefix: &str,
        body: &str,
        max_len: usize,
    ) -> Result<(Vec<u32>, bool), Error> {
        let normalized = enc.nfc(body);
        self.ids(enc, &format!("{prefix}{}", strip(&normalized)), max_len)
    }

    /// `len(tok(text)["input_ids"])`: the text's tokens, `[cls]` and `[sep]` included.
    fn tokens(&self, enc: &dyn DecimaEncoder, text: &str) -> Result<usize, Error> {
        Ok(enc.encode(text)?.len() + 2)
    }

    /// `render.choice_texts`: the text of every option row of a question whose text beside the
    /// options is `question` (empty when the layout keeps the question out of option rows).
    pub fn option_texts(
        &self,
        enc: &dyn DecimaEncoder,
        question: &str,
        options: &[String],
    ) -> Result<Vec<String>, Error> {
        let prefix = self.option_prefix.as_str();
        let max_len = self.max_option_tokens;
        let normalize = |s: &str| strip(&enc.nfc(s)).to_owned();
        let full: Vec<String> = options
            .iter()
            .map(|o| format!("{prefix}{}", normalize(strip(&format!("{question} {o}")))))
            .collect();
        if !self.option_fit {
            let mut rows = HashSet::with_capacity(full.len());
            for text in &full {
                rows.insert(self.ids(enc, text, max_len)?.0);
            }
            if rows.len() == full.len() {
                return Ok(full);
            }
        }
        let question = normalize(question);
        let words: Vec<&str> = split(&question).collect();
        let fitted =
            |head: usize, option: &str| format!("{prefix}{} … {option}", words[..head].join(" "));
        let mut out = Vec::with_capacity(options.len());
        for (option, full) in options.iter().zip(full) {
            if self.tokens(enc, &full)? <= max_len {
                out.push(full);
                continue;
            }
            let option = normalize(option);
            // Upstream's binary search for the longest question head that fits.
            let (mut lo, mut hi) = (0, words.len());
            while lo < hi {
                let mid = (lo + hi).div_ceil(2); // (lo + hi + 1) // 2
                if self.tokens(enc, &fitted(mid, &option))? <= max_len {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            out.push(if lo > 0 {
                fitted(lo, &option)
            } else {
                format!("{prefix}{option}")
            });
        }
        Ok(out)
    }

    /// Every question's rows, in request order.
    pub fn encode(
        &self,
        enc: &dyn DecimaEncoder,
        state: &Value,
        questions: &Questions,
    ) -> Result<DecimaRequest, Error> {
        let [lo, hi] = self.limits.questions;
        if !(lo..=hi).contains(&questions.len()) {
            return Err(Error::invalid(format!(
                "this model takes {lo} to {hi} questions, got {}",
                questions.len()
            )));
        }
        let state = state_string(state)?;
        let mut out = Vec::with_capacity(questions.len());
        for (qid, q) in questions {
            let (body, options) = self.question(qid, q)?;
            let state_text = if self.question_in_state && !body.is_empty() {
                format!("{body}\n{state}")
            } else {
                state.clone()
            };
            let (state_ids, state_truncated) =
                self.row(enc, &self.state_prefix, &state_text, self.max_state_tokens)?;
            let in_options = if self.question_in_options {
                body.as_str()
            } else {
                ""
            };
            let option_ids = self
                .option_texts(enc, in_options, &options)?
                .iter()
                .map(|text| {
                    self.ids(enc, text, self.max_option_tokens)
                        .map(|(ids, _)| ids)
                })
                .collect::<Result<_, _>>()?;
            out.push(DecimaQuestion {
                qtype: q.head(),
                state_ids,
                option_ids,
                state_truncated,
            });
        }
        Ok(DecimaRequest {
            questions: out,
            state_tokens: enc.encode(&state)?.len(),
        })
    }

    /// The logits a question's answer is calibrated from, in the order answers use (a choice's
    /// labels, score levels, noul `[false, true]`), from the graph's outputs in upstream's order.
    /// Choice and noul: the raw scores (the calibration layer divides them by the temperature).
    /// Score: the ordinal head's log-probabilities, already tempered (their calibration slot is 1).
    pub fn answer_logits(&self, qtype: QType, scores: &[f32], g: &[f32], gap: &[f32]) -> Vec<f32> {
        match qtype {
            QType::Choice => scores.to_vec(),
            QType::Noul => scores.iter().rev().copied().collect(),
            QType::Score => ordinal_log_probs(scores, g, gap, self.temperature)
                .into_iter()
                .map(|x| x as f32)
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::question::parse_questions;
    use serde_json::json;

    /// One id per char; NFC composes only "e" + U+0301.
    struct Chars;

    impl TokenEncoder for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            Ok(text.chars().map(u32::from).collect())
        }
    }

    impl DecimaEncoder for Chars {
        fn nfc(&self, text: &str) -> String {
            text.replace("e\u{301}", "\u{e9}")
        }
    }

    /// One id per UTF-8 byte: a character can take more than one token.
    struct Bytes;

    impl TokenEncoder for Bytes {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            Ok(text.bytes().map(u32::from).collect())
        }
    }

    impl DecimaEncoder for Bytes {
        fn nfc(&self, text: &str) -> String {
            Chars.nfc(text)
        }
    }

    const CLS: u32 = 0x10_0000;
    const SEP: u32 = 0x10_0001;

    fn layout(max_state: usize, max_option: usize) -> DecimaLayout {
        serde_json::from_value(json!({
            "state_prefix": "q: ", "option_prefix": "p: ",
            "question_in_state": true, "question_in_options": true,
            "max_state_tokens": max_state, "max_option_tokens": max_option,
            "special_tokens": {"cls": CLS, "sep": SEP, "pad": 1},
            "noul": {"options": ["yes", "no"], "criteria": "\nTrue if: {true}\nFalse if: {false}",
                     "missing": "\u{2014}"},
            "limits": {"questions": [1, 256], "choices": [2, 255], "levels": [2, 10]},
            "temperature": 0.5,
        }))
        .unwrap()
    }

    fn read(ids: &[u32]) -> String {
        assert_eq!((ids[0], ids[ids.len() - 1]), (CLS, SEP));
        ids[1..ids.len() - 1]
            .iter()
            .map(|&t| char::from_u32(t).unwrap())
            .collect()
    }

    fn encode(l: &DecimaLayout, state: Value, qs: Value) -> Result<DecimaRequest, Error> {
        l.encode(&Chars, &state, &parse_questions(&qs).unwrap())
    }

    #[test]
    fn maps_questions_like_systemone() {
        let l = layout(512, 64);
        let qs = parse_questions(&json!({
            "c": {"type": "choice", "instructions": {"k": "\u{e9}"},
                  "criteria": {"a": null, "b": " ", "c": "desc", "d": 2.5, "e": {"x": [1, true]}}},
            "s": {"type": "score", "instructions": "", "criteria": ["low", {"what": "mid"}, null]},
            "n": {"type": "noul", "instructions": null, "criteria": {"true": " yes! ", "False": "ignored"}},
            "z": {"type": "noul", "instructions": "Zero?", "criteria": {"true": "", "false": null}},
            "l": {"type": "choice", "instructions": "L", "criteria": ["x", "y", "x"]},
        }))
        .unwrap();
        let q = |id: &str| l.question(id, &qs[id]).unwrap();
        assert_eq!(
            q("c"),
            (
                "{\"k\": \"\u{e9}\"}".into(),
                vec![
                    "a".into(),
                    "b".into(),
                    "c: desc".into(),
                    "d: 2.5".into(),
                    "e: {\"x\": [1, true]}".into()
                ]
            )
        );
        assert_eq!(
            q("s"),
            (
                String::new(),
                vec!["low".into(), "{\"what\": \"mid\"}".into(), String::new()]
            )
        );
        assert_eq!(
            q("n").0,
            "n\nTrue if: yes!\nFalse if: \u{2014}",
            "a null instruction is the id; keys are exact"
        );
        assert_eq!(q("z"), ("Zero?".into(), vec!["yes".into(), "no".into()]));
        assert_eq!(q("l").1, ["x", "y"]);
    }

    #[test]
    fn builds_rows_like_model_py() {
        let l = layout(512, 64);
        let r = encode(
            &l,
            json!({"t": "Caf\u{e9}", "n": 1.0}),
            json!({"q": {"type": "choice", "instructions": " Re\u{301}sume? ", "criteria": {"a": "x", "b": null}}}),
        )
        .unwrap();
        let q = &r.questions[0];
        assert_eq!(
            read(&q.state_ids),
            "q: R\u{e9}sume? \n{\"t\": \"Caf\u{e9}\", \"n\": 1.0}"
        );
        assert_eq!(read(&q.option_ids[0]), "p: R\u{e9}sume?  a: x");
        assert_eq!(read(&q.option_ids[1]), "p: R\u{e9}sume?  b");
        assert!(!q.state_truncated);
        assert_eq!(r.state_tokens, 23);

        // Empty instructions leave the state alone; the option is the level text.
        let r = encode(
            &l,
            json!("  hi  "),
            json!({"q": {"type": "score", "instructions": "", "criteria": ["lo", "hi"]}}),
        )
        .unwrap();
        assert_eq!(read(&r.questions[0].state_ids), "q: hi");
        assert_eq!(read(&r.questions[0].option_ids[1]), "p: hi");
    }

    #[test]
    fn cuts_long_rows() {
        let l = layout(10, 6);
        let r = encode(
            &l,
            json!("abcdefghij"),
            json!({"q": {"type": "choice", "instructions": "Q", "criteria": {"long option": null, "b": null}}}),
        )
        .unwrap();
        let q = &r.questions[0];
        assert_eq!(
            (read(&q.state_ids).as_str(), q.state_truncated),
            ("q: Q\nabc", true)
        );
        // Both options cut to "p: Q" would collapse, so they are fitted: "p: long option", "p: b".
        assert_eq!(read(&q.option_ids[0]), "p: l");
        assert_eq!(read(&q.option_ids[1]), "p: b");
        assert_eq!(q.option_ids[0].len(), 6);
        let r = encode(
            &l,
            json!("ab"),
            json!({"q": {"type": "noul", "instructions": "Q"}}),
        )
        .unwrap();
        assert!(!r.questions[0].state_truncated);
    }

    /// `option_texts` for one question, without (v1) and with (v2) `option_fit`.
    fn texts(
        enc: &dyn DecimaEncoder,
        max_option: usize,
        question: &str,
        options: &[&str],
    ) -> [Vec<String>; 2] {
        let options: Vec<String> = options.iter().map(|&o| o.to_owned()).collect();
        let mut l = layout(512, max_option);
        let v1 = l.option_texts(enc, question, &options).unwrap();
        l.option_fit = true;
        let v2 = l.option_texts(enc, question, &options).unwrap();
        [v1, v2]
    }

    // The expected texts are upstream's `render.choice_texts` (v2.3.0) with the same toy tokenizers.

    #[test]
    fn fits_options_like_render_py() {
        // The right-hand cut would make "yes" and "no" the same row: both layouts fit them, each
        // with the longest question head that fits beside it.
        let want = ["p: Is this … yes", "p: Is this ticket … no"];
        assert_eq!(
            texts(&Chars, 24, "Is this ticket urgent today?", &["yes", "no"]),
            [want, want]
        );
        // Cut rows that stay apart keep v1's rendering; v2 fits them (no question word fits).
        let opts = ["alpha: a long description", "beta: a long description"];
        assert_eq!(
            texts(&Chars, 24, "Q", &opts),
            [
                vec![
                    "p: Q alpha: a long description",
                    "p: Q beta: a long description"
                ],
                vec![
                    "p: alpha: a long description",
                    "p: beta: a long description"
                ],
            ]
        );
        // Two of three collapse: the whole question is fitted, and an option whose full text
        // fits keeps it.
        let opts = [
            "billing: charges and refunds",
            "billing: charges and refunds, VAT",
            "ops",
        ];
        let want = [
            "p: billing: charges and refunds",
            "p: billing: charges and refunds, VAT",
            "p: Pick the team for … ops",
        ];
        assert_eq!(
            texts(&Chars, 32, "Pick the team for this ticket now", &opts),
            [want, want]
        );
    }

    #[test]
    fn fitting_follows_render_py_edges() {
        // Tokens only (v2.3.0 dropped the 62-character shortcut): 16 characters that are 25 tokens
        // are fitted, in v1 too because the cut makes both rows "p: ééééééééé ". The question's one
        // word does not fit beside the option, so the option stands alone.
        let want = ["p: yes", "p: no"];
        assert_eq!(
            texts(&Bytes, 24, &"\u{e9}".repeat(9), &["yes", "no"]),
            [want, want]
        );
        // No question: the option alone, however long.
        let opts = [
            "a long option that does not fit",
            "b long option that does not fit",
        ];
        let want = opts.map(|o| format!("p: {o}"));
        assert_eq!(texts(&Chars, 12, "", &opts), [want.clone(), want]);
        // Words are Python's str.split() of the normalized question, joined by one space.
        let q = " Re\u{301}sume\u{301}\tthe   plan\u{a0}now\u{1f}please ";
        let want = [
            "p: R\u{e9}sum\u{e9} the … yes",
            "p: R\u{e9}sum\u{e9} the … no",
        ];
        assert_eq!(texts(&Chars, 24, q, &["yes", "no"]), [want, want]);
        // No question word fits: the option alone.
        let want = ["p: long option", "p: b"];
        assert_eq!(texts(&Chars, 6, "Q", &["long option", "b"]), [want, want]);
    }

    #[test]
    fn layout_ids() {
        let mut l = layout(512, 64);
        assert_eq!(l.layout_id(), LAYOUT);
        l.option_fit = true;
        assert_eq!(l.layout_id(), LAYOUT_FIT);
    }

    #[test]
    fn rejects_what_upstream_rejects() {
        let l = layout(512, 64);
        let one = json!({"q": {"type": "choice", "instructions": "x", "criteria": ["a", "a"]}});
        assert!(encode(&l, json!("s"), one).is_err());
        let levels: Vec<String> = (0..11).map(|i| i.to_string()).collect();
        let eleven = json!({"q": {"type": "score", "instructions": "x", "criteria": levels}});
        assert!(encode(&l, json!("s"), eleven).is_err());
        let labels: Vec<String> = (0..256).map(|i| format!("c{i}")).collect();
        let many = json!({"q": {"type": "choice", "instructions": "x", "criteria": labels}});
        assert!(encode(&l, json!("s"), many).is_err());
        let noul = json!({"q": {"type": "noul", "instructions": "x"}});
        for state in [json!(1), json!(true), json!(null)] {
            assert!(encode(&l, state, noul.clone()).is_err());
        }
        assert!(encode(&l, json!([1, "a"]), noul).is_ok());
    }

    #[test]
    fn ordinal_choices_read_the_score_head() {
        let l = layout(512, 64);
        let labels: Vec<String> = (0..10).map(|i| format!("c{i}")).collect();
        let ten = json!({"q": {"type": "choice", "instructions": "x", "criteria": labels, "ordinal": true}});
        let r = encode(&l, json!("s"), ten).unwrap();
        assert_eq!(r.questions[0].qtype, QType::Score);
        assert_eq!(r.questions[0].option_ids.len(), 10);
        // The rows are a choice's: only the head differs.
        let plain = json!({"q": {"type": "choice", "instructions": "x", "criteria": labels}});
        assert_eq!(
            encode(&l, json!("s"), plain).unwrap().questions[0].option_ids,
            r.questions[0].option_ids
        );
        let labels: Vec<String> = (0..11).map(|i| format!("c{i}")).collect();
        let eleven = json!({"q": {"type": "choice", "instructions": "x", "criteria": labels, "ordinal": true}});
        assert!(encode(&l, json!("s"), eleven).is_err());
    }

    #[test]
    fn ordinal_head_matches_upstream() {
        // decima-small, "How upset is the customer?" with levels [calm, annoyed, furious] about
        // "My card was charged twice" (ref.py's smoke test, rounded to 4 decimals): the graph's
        // outputs and upstream's log-probabilities.
        let scores = [-4.1834, -2.6412, -1.8191];
        let g = [0.1262, 0.0516, -0.752];
        let gap = [2.6582, 3.2905, 2.3794];
        let lp = ordinal_log_probs(&scores, &g, &gap, 0.935_556_835_803_263_9);
        let want = [-2.0909, -0.3950, -1.5956];
        for (a, b) in lp.iter().zip(want) {
            assert!((a - b).abs() < 1e-3, "{lp:?}");
        }
        let p: f64 = lp.iter().map(|x| x.exp()).sum();
        assert!((p - 1.0).abs() < 1e-12);
    }

    #[test]
    fn answer_logits_in_answer_order() {
        let l = layout(512, 64);
        assert_eq!(
            l.answer_logits(QType::Noul, &[2.0, -1.0], &[0.0; 2], &[0.0; 2]),
            [-1.0, 2.0]
        );
        assert_eq!(
            l.answer_logits(QType::Choice, &[1.0, 2.0, 3.0], &[0.0; 3], &[0.0; 3]),
            [1.0, 2.0, 3.0]
        );
        assert_eq!(
            l.answer_logits(QType::Score, &[0.0; 4], &[0.0; 4], &[0.0; 4])
                .len(),
            4
        );
    }

    #[test]
    fn validates_the_config() {
        assert!(layout(512, 64).validate().is_ok());
        assert!(layout(2, 64).validate().is_err());
        let mut l = layout(512, 64);
        l.temperature = 0.0;
        assert!(l.validate().is_err());
        let mut l = layout(512, 64);
        l.limits.levels = [3, 2];
        assert!(l.validate().is_err());
    }
}
