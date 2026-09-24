//! `decider-slots-v1`: Mapika's decider models (Qwen3.5 decoders), whose rows follow upstream
//! `decider.systemone` + `decider.prompt.build` (`ollaya_convert.families.decider.layout`).
//!
//! Every question becomes one scoring row: the shared context, then the question.
//!
//! ```text
//! enc("Context:\n" + state)[:max_ctx_tokens]
//!   + enc("\n\nQuestion: " + q + "\nOptions:" + "\n(A) o0" + ... + "\nAnswer: (")    <= 10 options
//!   | enc("\n\nQuestion: " + q + "\nOptions:")                                        > 10 options
//!       + concat_j(enc("\n(") + [label_id_j] + enc(") " + o_j)) + enc("\nAnswer: (")
//! ```
//!
//! The model reads its tied LM head at the row's last token (the `(` slot), restricted to the
//! option-label tokens. With `isolated_levels`, a score question becomes one yes/no row per level,
//! and its option logits are `log_sigmoid((l_yes - l_no) / isolated_row_temperature)`: the
//! daemon's softmax at temperature 1 then gives upstream's normalised per-level fit.

use std::fmt::Write;

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::layout::TokenEncoder;
use crate::question::{Criteria, Question, render_criterion};
use crate::{Error, pyjson};

const CONTEXT: &str = "Context:\n";
const TAIL: &str = "\nAnswer: (";
const ISOLATED_OPTIONS: [&str; 2] = ["no", "yes"];

/// The option-label table: `strings[j]` is option j's label and `ids[j]` its single token.
#[derive(Debug, Clone, Deserialize)]
pub struct Labels {
    /// Rows with at most this many options spell their labels in the text (`(A) ...`).
    pub narrow: usize,
    pub strings: Vec<String>,
    pub ids: Vec<u32>,
    /// `enc("\n(")`, which precedes every label token in a wide row.
    pub open_ids: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeciderTokens {
    pub pad: u32,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct DeciderLayout {
    /// Context tokens kept (counted including `Context:\n`); the rest of the state is cut.
    pub max_ctx_tokens: usize,
    pub max_options: usize,
    pub min_options: usize,
    pub max_levels: usize,
    /// Score questions are read one level per row.
    pub isolated_levels: bool,
    /// The temperature inside an isolated level's `log_sigmoid`.
    pub isolated_row_temperature: f64,
    /// JSON arrays of at least this many elements get their positions written in (`_index`).
    pub index_arrays_min_len: usize,
    /// Upstream's option rewriting; not supported (both published configs have it off).
    pub neutralize_none: bool,
    /// One row per question; packed rows are not supported.
    pub independent_rows: bool,
    pub labels: Labels,
    pub special_tokens: DeciderTokens,
}

/// How a question's rows become option logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readout {
    /// One row; option j's logit is label j's logit (choice, noul, listwise score).
    List,
    /// One yes/no row per score level.
    Isolated,
}

impl Readout {
    pub fn name(self) -> &'static str {
        match self {
            Readout::List => "list",
            Readout::Isolated => "iso",
        }
    }
}

/// A row before tokenization: the question text and its options.
#[derive(Debug, Clone, PartialEq)]
pub struct RowText {
    pub question: String,
    pub options: Vec<String>,
}

/// One question's scoring rows. Each row's slot is its last token.
#[derive(Debug, Clone, PartialEq)]
pub struct Scoring {
    pub rows: Vec<Vec<u32>>,
    pub readout: Readout,
    /// The question's option count (levels, for an isolated score).
    pub options: usize,
}

/// The tokenized context every row starts with.
#[derive(Debug, Clone, PartialEq)]
pub struct Context {
    pub ids: Vec<u32>,
    /// Context tokens before the cut (including `Context:\n`).
    pub tokens: usize,
    pub truncated: bool,
}

impl DeciderLayout {
    /// Reject configurations this layout cannot run.
    pub fn validate(&self) -> Result<(), Error> {
        let l = &self.labels;
        let bad = |msg: String| Err(Error::invalid(msg));
        if l.strings.len() != l.ids.len() {
            return bad(format!(
                "labels: {} strings but {} ids",
                l.strings.len(),
                l.ids.len()
            ));
        }
        if l.narrow > l.ids.len() || self.max_options.max(self.max_levels) > l.ids.len() {
            return bad(format!(
                "labels: {} labels cannot cover narrow={}, max_options={}, max_levels={}",
                l.ids.len(),
                l.narrow,
                self.max_options,
                self.max_levels
            ));
        }
        if self.min_options < 1
            || self.min_options > self.max_options
            || self.min_options > self.max_levels
        {
            return bad(format!(
                "min_options={} does not fit max_options={} / max_levels={}",
                self.min_options, self.max_options, self.max_levels
            ));
        }
        if !(self.isolated_row_temperature.is_finite() && self.isolated_row_temperature > 0.0) {
            return bad(format!(
                "isolated_row_temperature must be positive, got {}",
                self.isolated_row_temperature
            ));
        }
        if self.neutralize_none || !self.independent_rows {
            return bad(
                "neutralize_none and packed rows (independent_rows=false) are not supported".into(),
            );
        }
        Ok(())
    }

    /// The state as upstream renders it: strings verbatim, anything else as
    /// `json.dumps(annotate_indices(state), ensure_ascii=False)`.
    pub fn render_state(&self, state: &Value) -> String {
        match state {
            Value::String(s) => s.clone(),
            v => pyjson::dumps(&annotate_indices(v, self.index_arrays_min_len), false),
        }
    }

    /// Tokenize the context once; it is shared by every row.
    pub fn encode_context(&self, enc: &dyn TokenEncoder, state: &Value) -> Result<Context, Error> {
        let mut ids = enc.encode(&format!("{CONTEXT}{}", self.render_state(state)))?;
        let tokens = ids.len();
        ids.truncate(self.max_ctx_tokens);
        Ok(Context {
            ids,
            tokens,
            truncated: tokens > self.max_ctx_tokens,
        })
    }

    /// One question's rows as text, and how they are read.
    pub fn render(&self, qid: &str, q: &Question) -> Result<(Vec<RowText>, Readout), Error> {
        let bad = |msg: String| Error::invalid(format!("question {qid:?}: {msg}"));
        let question = instructions_text(&q.instructions);
        if question.is_empty() {
            return Err(bad(
                "empty 'instructions'; add the text the model should answer".into(),
            ));
        }
        let (max, what) = match q.criteria {
            Criteria::Score(_) => (self.max_levels, "levels"),
            _ => (self.max_options, "options"),
        };
        let k = q.num_options();
        if k > max {
            return Err(Error::TooManyOptions {
                question: qid.to_owned(),
                options: k,
                head_max_len: max,
            });
        }
        if k < self.min_options {
            return Err(bad(format!(
                "the model needs {}..{max} {what}, got {k}",
                self.min_options
            )));
        }
        let options = match &q.criteria {
            Criteria::Choice(m) => m
                .iter()
                .map(|(name, v)| match described(v) {
                    None => name.clone(),
                    Some(d) => format!("{name}: {d}"),
                })
                .collect(),
            Criteria::Noul { r#false, r#true } => {
                let side = |bare: &str, v: &Option<Value>| match v.as_ref().and_then(described) {
                    None => bare.to_owned(),
                    Some(d) => format!("{bare}: {d}"),
                };
                vec![side("no", r#false), side("yes", r#true)]
            }
            Criteria::Score(levels) if self.isolated_levels => {
                let rows = levels
                    .iter()
                    .map(|c| RowText {
                        question: format!(
                            "{question}\nProposed answer: {}\nDoes the proposed answer fit?",
                            strip_level_number(&render_criterion(c))
                        ),
                        options: ISOLATED_OPTIONS.map(str::to_owned).to_vec(),
                    })
                    .collect();
                return Ok((rows, Readout::Isolated));
            }
            Criteria::Score(levels) => levels
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{i}: {}", render_criterion(c)))
                .collect(),
        };
        Ok((vec![RowText { question, options }], Readout::List))
    }

    /// Encode one question's rows after the tokenized context.
    pub fn encode(
        &self,
        enc: &dyn TokenEncoder,
        context: &[u32],
        qid: &str,
        q: &Question,
    ) -> Result<Scoring, Error> {
        let (texts, readout) = self.render(qid, q)?;
        let rows = texts
            .iter()
            .map(|t| {
                let piece = self.piece(enc, t)?;
                let mut ids = Vec::with_capacity(context.len() + piece.len());
                ids.extend_from_slice(context);
                ids.extend_from_slice(&piece);
                Ok(ids)
            })
            .collect::<Result<_, Error>>()?;
        Ok(Scoring {
            rows,
            readout,
            options: q.num_options(),
        })
    }

    /// The question part of a row, ending at the `(` slot.
    fn piece(&self, enc: &dyn TokenEncoder, row: &RowText) -> Result<Vec<u32>, Error> {
        let labels = &self.labels;
        let mut head = format!("\n\nQuestion: {}\nOptions:", row.question);
        if row.options.len() <= labels.narrow {
            for (label, option) in labels.strings.iter().zip(&row.options) {
                let _ = write!(head, "\n({label}) {option}");
            }
            head.push_str(TAIL);
            return enc.encode(&head);
        }
        let mut ids = enc.encode(&head)?;
        for (&label, option) in labels.ids.iter().zip(&row.options) {
            ids.extend_from_slice(&labels.open_ids);
            ids.push(label);
            ids.extend(enc.encode(&format!(") {option}"))?);
        }
        ids.extend(enc.encode(TAIL)?);
        Ok(ids)
    }

    /// Option logits from a question's rows of `label_logits`, in option order.
    pub fn option_logits<'a>(
        &self,
        readout: Readout,
        options: usize,
        rows: impl IntoIterator<Item = &'a [f32]>,
    ) -> Vec<f32> {
        let mut rows = rows.into_iter();
        match readout {
            Readout::List => rows.next().map_or_else(Vec::new, |r| r[..options].to_vec()),
            Readout::Isolated => rows
                .map(|r| {
                    let d = (f64::from(r[1]) - f64::from(r[0])) / self.isolated_row_temperature;
                    log_sigmoid(d) as f32
                })
                .collect(),
        }
    }
}

/// `log(sigmoid(d))`, computed as the reference does.
fn log_sigmoid(d: f64) -> f64 {
    if d > -30.0 {
        -(-d).exp().ln_1p()
    } else {
        d - d.exp().ln_1p()
    }
}

/// A description as upstream tests it: `None` if `null` or `""`.
fn described(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::String(s) if s.is_empty() => None,
        v => Some(render_criterion(v)),
    }
}

/// Instructions as upstream renders them: strings verbatim, anything else as
/// `json.dumps(v, ensure_ascii=False)`.
///
/// The shared question parser renders structured instructions as Laya reads them, with
/// `ensure_ascii` on; the two differ only in `\uXXXX` escapes. A text that parses as a JSON array or
/// object and that `ensure_ascii` rendering reproduces byte for byte is such a rendering, so it is
/// rendered again with `ensure_ascii` off. (A string instruction that is itself exactly an
/// escaped JSON dump is indistinguishable from one, and is unescaped too.)
fn instructions_text(text: &str) -> String {
    if (text.starts_with('{') || text.starts_with('['))
        && text.contains("\\u")
        && let Ok(v) = serde_json::from_str::<Value>(text)
        && pyjson::dumps(&v, true) == text
    {
        return pyjson::dumps(&v, false);
    }
    text.to_owned()
}

/// `annotate_indices`: arrays of at least `min_len` elements get their positions written in. A
/// dict element `e` becomes `{"_index": i, **e}` (its own `_index` wins, the key stays first), any
/// other element `{"_index": i, "value": e}`. Recursive.
pub fn annotate_indices(v: &Value, min_len: usize) -> Value {
    match v {
        Value::Array(items) if items.len() >= min_len => Value::Array(
            items
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let mut m = Map::new();
                    m.insert("_index".into(), Value::from(i));
                    match annotate_indices(e, min_len) {
                        Value::Object(inner) => m.extend(inner),
                        other => {
                            m.insert("value".into(), other);
                        }
                    }
                    Value::Object(m)
                })
                .collect(),
        ),
        Value::Array(items) => {
            Value::Array(items.iter().map(|e| annotate_indices(e, min_len)).collect())
        }
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, e)| (k.clone(), annotate_indices(e, min_len)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// `re.sub(r"^\s*-?\d+\s*:\s*", "", level, count=1)` with Python's Unicode `\s` and `\d`.
pub fn strip_level_number(level: &str) -> &str {
    fn skip(s: &str, f: fn(char) -> bool) -> (usize, &str) {
        let n = s.find(|c: char| !f(c)).unwrap_or(s.len());
        (n, &s[n..])
    }
    let (_, rest) = skip(level, is_py_space);
    let rest = rest.strip_prefix('-').unwrap_or(rest);
    let (digits, rest) = skip(rest, is_py_digit);
    if digits == 0 {
        return level;
    }
    let (_, rest) = skip(rest, is_py_space);
    match rest.strip_prefix(':') {
        Some(rest) => skip(rest, is_py_space).1,
        None => level,
    }
}

/// `\s` in a Python `str` pattern (`str.isspace`): Unicode `White_Space` plus U+001C..U+001F.
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// First code point of every run of ten decimal digits (general category `Nd`), Unicode 15.0 as
/// in the reference's CPython 3.12. Every `Nd` character is in one of these runs.
const DIGIT_ZEROS: [u32; 68] = [
    0x0030, 0x0660, 0x06F0, 0x07C0, 0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6,
    0x0D66, 0x0DE6, 0x0E50, 0x0ED0, 0x0F20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80,
    0x1A90, 0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0,
    0xFF10, 0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0,
    0x11650, 0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60,
    0x16AC0, 0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0,
    0x1E950, 0x1FBF0,
];

/// `\d` in a Python `str` pattern: a decimal digit of any script.
fn is_py_digit(c: char) -> bool {
    let cp = u32::from(c);
    let i = DIGIT_ZEROS.partition_point(|&z| z <= cp);
    i > 0 && cp - DIGIT_ZEROS[i - 1] < 10
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
        ids.iter().map(|&i| char::from_u32(i).unwrap()).collect()
    }

    fn layout(isolated: bool) -> DeciderLayout {
        let strings: Vec<String> = (b'A'..=b'Z').map(|c| char::from(c).to_string()).collect();
        serde_json::from_value(json!({
            "max_ctx_tokens": 20,
            "max_options": 26,
            "min_options": 2,
            "max_levels": 10,
            "isolated_levels": isolated,
            "isolated_row_temperature": 2.0,
            "index_arrays_min_len": 3,
            "neutralize_none": false,
            "independent_rows": true,
            "labels": {"narrow": 10, "ids": (1000..1026).collect::<Vec<u32>>(), "strings": strings,
                       "open_ids": [10, 40]},
            "special_tokens": {"pad": 0},
        }))
        .unwrap()
    }

    fn question(def: Value) -> Question {
        Question::parse("q", &def).unwrap()
    }

    #[test]
    fn annotates_long_arrays() {
        let l = layout(true);
        let state = json!({"a": [1, {"x": 2, "_index": "mine"}, [3, 4, 5]], "b": [1, 2], "c": "é"});
        assert_eq!(
            l.render_state(&state),
            "{\"a\": [{\"_index\": 0, \"value\": 1}, {\"_index\": \"mine\", \"x\": 2}, \
             {\"_index\": 2, \"value\": [{\"_index\": 0, \"value\": 3}, {\"_index\": 1, \"value\": 4}, \
             {\"_index\": 2, \"value\": 5}]}], \"b\": [1, 2], \"c\": \"é\"}"
        );
        assert_eq!(l.render_state(&json!("[1, 2, 3]")), "[1, 2, 3]");
    }

    #[test]
    fn cuts_the_context() {
        let l = layout(true);
        let c = l.encode_context(&Chars, &json!("0123456789")).unwrap();
        assert_eq!(text(&c.ids), "Context:\n0123456789");
        assert!(!c.truncated);
        let c = l
            .encode_context(&Chars, &json!("0123456789abcdef"))
            .unwrap();
        assert_eq!((c.ids.len(), c.tokens, c.truncated), (20, 25, true));
    }

    #[test]
    fn renders_rows() {
        let l = layout(true);
        let choice = question(json!({"type": "choice", "instructions": "Which?",
            "criteria": {"a": null, "b": "", "c": {"k": [1, "ü"]}}}));
        let s = l.encode(&Chars, &[7], "q", &choice).unwrap();
        assert_eq!((s.readout, s.options, s.rows.len()), (Readout::List, 3, 1));
        assert_eq!(
            text(&s.rows[0]),
            "\u{7}\n\nQuestion: Which?\nOptions:\n(A) a\n(B) b\n(C) c: {\"k\": [1, \"ü\"]}\nAnswer: ("
        );

        let noul = question(json!({"type": "noul", "instructions": {"ask": "ü?"},
            "criteria": {"true": "yes it is", "false": ""}}));
        let (rows, readout) = l.render("q", &noul).unwrap();
        assert_eq!(readout, Readout::List);
        assert_eq!(rows[0].question, "{\"ask\": \"ü?\"}");
        assert_eq!(rows[0].options, ["no", "yes: yes it is"]);

        let score = question(json!({"type": "score", "instructions": "How bad?",
            "criteria": ["0: fine", " \u{0663} : bad", "-2 meh", 5]}));
        let (rows, readout) = l.render("q", &score).unwrap();
        assert_eq!(readout, Readout::Isolated);
        let levels: Vec<&str> = rows
            .iter()
            .map(|r| r.question.split('\n').nth(1).unwrap())
            .collect();
        assert_eq!(
            levels,
            [
                "Proposed answer: fine",
                "Proposed answer: bad",
                "Proposed answer: -2 meh",
                "Proposed answer: 5"
            ]
        );
        assert_eq!(rows[0].options, ["no", "yes"]);
        let (rows, readout) = layout(false).render("q", &score).unwrap();
        assert_eq!(readout, Readout::List);
        assert_eq!(rows[0].options[1], "1:  \u{0663} : bad");
    }

    #[test]
    fn wide_rows_use_label_tokens() {
        let l = layout(true);
        let names: Vec<String> = (0..11).map(|i| format!("o{i}")).collect();
        let q = question(json!({"type": "choice", "instructions": "Q", "criteria": names}));
        let s = l.encode(&Chars, &[], "q", &q).unwrap();
        let head = Chars.encode("\n\nQuestion: Q\nOptions:").unwrap();
        let row = &s.rows[0];
        assert_eq!(&row[..head.len()], head.as_slice());
        assert_eq!(&row[head.len()..head.len() + 3], [10, 40, 1000]);
        assert_eq!(text(&row[head.len() + 3..head.len() + 7]), ") o0");
        assert!(text(&row[row.len() - 10..]).ends_with("\nAnswer: ("));
        assert_eq!(row.iter().filter(|&&t| t >= 1000).count(), 11);
    }

    #[test]
    fn enforces_option_limits() {
        let l = layout(true);
        let one = question(json!({"type": "choice", "instructions": "Q", "criteria": ["a"]}));
        assert!(matches!(l.render("q1", &one), Err(Error::Invalid(m)) if m.contains("\"q1\"")));
        let many: Vec<String> = (0..27).map(|i| i.to_string()).collect();
        let many = question(json!({"type": "choice", "instructions": "Q", "criteria": many}));
        assert!(matches!(
            l.render("q2", &many),
            Err(Error::TooManyOptions { question, options: 27, head_max_len: 26 }) if question == "q2"
        ));
        let levels =
            question(json!({"type": "score", "instructions": "Q", "criteria": vec!["x"; 11]}));
        assert!(matches!(
            l.render("q", &levels),
            Err(Error::TooManyOptions {
                options: 11,
                head_max_len: 10,
                ..
            })
        ));
        let empty = question(json!({"type": "noul", "instructions": ""}));
        assert!(l.render("q", &empty).is_err());
    }

    #[test]
    fn strips_level_numbers_like_python() {
        for (level, want) in [
            ("3: high", "high"),
            ("\t-12 :\u{a0}x", "x"),
            ("\u{1c}\u{0967}:y", "y"),
            ("3 high", "3 high"),
            (": x", ": x"),
            ("-: x", "-: x"),
            ("1:", ""),
            ("x 1: y", "x 1: y"),
            ("\u{00b2}: x", "\u{00b2}: x"),
        ] {
            assert_eq!(strip_level_number(level), want, "{level:?}");
        }
        assert!(is_py_digit('9') && is_py_digit('\u{1fbf9}') && !is_py_digit('\u{1fbfa}'));
        assert!(!is_py_digit('/') && !is_py_digit(':'));
    }

    #[test]
    fn recovers_structured_instructions() {
        assert_eq!(
            instructions_text("{\"a\": \"\\u00fc\\n\"}"),
            "{\"a\": \"ü\\n\"}"
        );
        // Not the ensure_ascii rendering of a value: kept as is.
        assert_eq!(
            instructions_text("{\"a\":\"\\u00fc\"}"),
            "{\"a\":\"\\u00fc\"}"
        );
        assert_eq!(
            instructions_text("Is \\u00fc {here}?"),
            "Is \\u00fc {here}?"
        );
    }

    #[test]
    fn reads_option_logits() {
        let l = layout(true);
        let rows = [[1.0f32, 3.0, 9.0], [0.0, -70.0, 0.0]];
        let it = || rows.iter().map(|r| r.as_slice());
        assert_eq!(l.option_logits(Readout::List, 2, it()), [1.0, 3.0]);
        let z = l.option_logits(Readout::Isolated, 2, it());
        assert!((f64::from(z[0]) - (1.0f64 / (1.0 + (-1.0f64).exp())).ln()).abs() < 1e-6);
        assert!((f64::from(z[1]) + 35.0).abs() < 1e-6);
    }
}
