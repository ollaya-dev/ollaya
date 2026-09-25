//! `kev-pointer-v1`: jaredpalmer's Kev models (a LoRA on a Qwen3.5 base plus a pointer head), whose
//! rows follow upstream `kev.api.to_record` + `kev.model.encode` + `rows_of`
//! (`ollaya_convert.families.kev.layout`).
//!
//! Every question becomes one causal row: the shared state, the question, one span per option, and
//! the decide token.
//!
//! ```text
//! [state] user(render(state))[:max_state_tokens - 1]
//!   [question] user(render(instructions))
//!   ([option_open] user(option_j) [option_close])*      opt_pos[j] = the option_close position
//!   [decide]                                             decide_pos = the last position
//! ```
//!
//! `user(text)` rewrites every `<|name|>` to `<¦name¦>` before tokenizing, so text cannot produce
//! the delimiter tokens. The graph scores option j as `k(h[opt_pos[j]]) · q(h[decide_pos]) / 16`;
//! those scores are the option logits for every question type.

use std::borrow::Cow;
use std::fmt::Write;

use serde::Deserialize;
use serde_json::{Number, Value};

use crate::layout::TokenEncoder;
use crate::question::{Criteria, Question};
use crate::{Error, pyjson};

/// Delimiter token ids, as `decision.json` declares them.
#[derive(Debug, Clone, Deserialize)]
pub struct KevTokens {
    /// Opens the state (`<|fim_prefix|>`).
    pub state: u32,
    /// Opens the question (`<|fim_middle|>`).
    pub question: u32,
    /// Opens an option (`<|box_start|>`).
    pub option_open: u32,
    /// Closes an option (`<|box_end|>`); the option is scored here.
    pub option_close: u32,
    /// Ends the row (`<|fim_suffix|>`); the question is read here.
    pub decide: u32,
    pub pad: u32,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct KevLayout {
    /// State tokens kept, counting the state token itself.
    pub max_state_tokens: usize,
    /// A longer row is rejected (upstream `ContextOverflow`).
    pub max_row_tokens: usize,
    pub min_options: usize,
    pub max_options: usize,
    pub special_tokens: KevTokens,
}

/// The tokenized state every row starts with (state token included).
#[derive(Debug, Clone, PartialEq)]
pub struct KevState {
    pub ids: Vec<u32>,
    /// Tokens of the rendered state before the cut.
    pub tokens: usize,
    pub truncated: bool,
}

/// One question's row.
#[derive(Debug, Clone, PartialEq)]
pub struct KevRow {
    pub ids: Vec<u32>,
    /// Position of the decide token (the last one).
    pub decide: usize,
    /// Position of each option's closing token, in option order.
    pub opts: Vec<usize>,
}

impl KevLayout {
    /// Reject configurations this layout cannot run.
    pub fn validate(&self) -> Result<(), Error> {
        if self.max_state_tokens == 0 || self.max_row_tokens == 0 {
            return Err(Error::invalid(format!(
                "max_state_tokens={} and max_row_tokens={} must be positive",
                self.max_state_tokens, self.max_row_tokens
            )));
        }
        if self.min_options < 1 || self.min_options > self.max_options {
            return Err(Error::invalid(format!(
                "min_options={} does not fit max_options={}",
                self.min_options, self.max_options
            )));
        }
        Ok(())
    }

    /// `user(text)`: tokenized with every `<|name|>` defused.
    fn user(&self, enc: &dyn TokenEncoder, text: &str) -> Result<Vec<u32>, Error> {
        enc.encode(&escape_delimiters(text))
    }

    /// Tokenize the state once; it is shared by every row.
    pub fn encode_state(&self, enc: &dyn TokenEncoder, state: &Value) -> Result<KevState, Error> {
        let body = self.user(enc, &render(state))?;
        let keep = self.max_state_tokens - 1;
        let mut ids = Vec::with_capacity(body.len().min(keep) + 1);
        ids.push(self.special_tokens.state);
        ids.extend_from_slice(&body[..body.len().min(keep)]);
        Ok(KevState {
            ids,
            tokens: body.len(),
            truncated: body.len() > keep,
        })
    }

    /// One question's instruction and option texts, validated as upstream's request model does.
    pub fn render_question(&self, qid: &str, q: &Question) -> Result<(String, Vec<String>), Error> {
        let bad = |msg: String| Error::invalid(format!("question {qid:?}: {msg}"));
        let criteria = q.definition.get("criteria").filter(|c| !c.is_null());
        let options: Vec<String> = match &q.criteria {
            Criteria::Choice(m) => {
                if !criteria.is_some_and(Value::is_object) {
                    return Err(bad(
                        "this model takes choice criteria as an object of label -> description; \
                         write a list of labels as {\"label\": null, ...}"
                            .into(),
                    ));
                }
                m.iter()
                    .map(|(name, d)| option_text(name, Some(d)))
                    .collect()
            }
            Criteria::Score(levels) => levels.iter().map(render).collect(),
            // Upstream reads the exact keys `false` and `true`.
            Criteria::Noul { .. } => {
                let side = |key: &str| criteria.and_then(|c| c.get(key));
                vec![
                    option_text("no", side("false")),
                    option_text("yes", side("true")),
                ]
            }
        };
        if options.len() > self.max_options {
            return Err(Error::TooManyOptions {
                question: qid.to_owned(),
                options: options.len(),
                head_max_len: self.max_options,
            });
        }
        if options.len() < self.min_options {
            return Err(bad(format!(
                "the model needs {}..{} options, got {}",
                self.min_options,
                self.max_options,
                options.len()
            )));
        }
        let instructions = q.definition.get("instructions").unwrap_or(&Value::Null);
        Ok((render(instructions), options))
    }

    /// Encode one question's row after the tokenized state.
    pub fn encode(
        &self,
        enc: &dyn TokenEncoder,
        state: &[u32],
        qid: &str,
        q: &Question,
    ) -> Result<KevRow, Error> {
        let (instructions, options) = self.render_question(qid, q)?;
        let t = &self.special_tokens;
        let mut ids = Vec::with_capacity(state.len() + 64);
        ids.extend_from_slice(state);
        ids.push(t.question);
        ids.extend(self.user(enc, &instructions)?);
        let mut opts = Vec::with_capacity(options.len());
        for option in &options {
            ids.push(t.option_open);
            ids.extend(self.user(enc, option)?);
            opts.push(ids.len());
            ids.push(t.option_close);
        }
        ids.push(t.decide);
        if ids.len() > self.max_row_tokens {
            return Err(Error::invalid(format!(
                "question {qid:?}: the row is {} tokens ({} of them the state); the model's limit is {}",
                ids.len(),
                state.len(),
                self.max_row_tokens
            )));
        }
        Ok(KevRow {
            decide: ids.len() - 1,
            ids,
            opts,
        })
    }
}

/// `option_text(name, d)`: the name alone when `d` is missing, null or `""`.
fn option_text(name: &str, description: Option<&Value>) -> String {
    match description {
        None | Some(Value::Null) => name.to_owned(),
        Some(Value::String(s)) if s.is_empty() => name.to_owned(),
        Some(d) => format!("{name}: {}", render(d)),
    }
}

/// `kev.api.render`: null as `""`, scalars as Python's `str()`, lists as `- item` lines and objects
/// as `key: value` lines, nested two spaces per level.
pub fn render(v: &Value) -> String {
    render_at(v, 0)
}

fn render_at(v: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => py_number(n),
        Value::Array(items) => {
            let mut out = String::new();
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push('\n');
                }
                let item = render_at(x, indent + 1);
                let _ = write!(out, "{pad}- {}", item.trim_start_matches(is_py_space));
            }
            out
        }
        Value::Object(m) => {
            let mut out = String::new();
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push('\n');
                }
                match x {
                    Value::Object(_) | Value::Array(_) => {
                        let _ = write!(out, "{pad}{k}:\n{}", render_at(x, indent + 1));
                    }
                    x => {
                        let _ = write!(out, "{pad}{k}: {}", render_at(x, 0));
                    }
                }
            }
            out
        }
    }
}

/// Python's `str()` of a JSON number: integers as written, floats as `repr`.
fn py_number(n: &Number) -> String {
    if let Some(i) = n.as_i64() {
        i.to_string()
    } else if let Some(u) = n.as_u64() {
        u.to_string()
    } else {
        pyjson::float_repr(n.as_f64().unwrap_or(f64::NAN))
    }
}

/// `str.isspace`, which `str.lstrip()` strips: Unicode `White_Space` plus U+001C..U+001F.
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", text)`.
pub fn escape_delimiters(text: &str) -> Cow<'_, str> {
    if !text.contains("<|") {
        return Cow::Borrowed(text);
    }
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len() + 8);
    let mut copied = 0;
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'<' && bytes[i + 1] == b'|' {
            let start = i + 2;
            let end = start
                + bytes[start..]
                    .iter()
                    .take_while(|b| b.is_ascii_alphanumeric() || **b == b'_')
                    .count();
            if end > start && bytes[end..].starts_with(b"|>") {
                out.push_str(&text[copied..i]);
                out.push_str("<¦");
                out.push_str(&text[start..end]);
                out.push_str("¦>");
                i = end + 2;
                copied = i;
                continue;
            }
        }
        i += 1;
    }
    out.push_str(&text[copied..]);
    Cow::Owned(out)
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

    fn layout() -> KevLayout {
        serde_json::from_value(json!({
            "max_state_tokens": 8,
            "max_row_tokens": 40,
            "min_options": 1,
            "max_options": 3,
            "special_tokens": {"state": 0x2460, "question": 0x2461, "option_open": 0x2462,
                               "option_close": 0x2463, "decide": 0x2464, "pad": 0},
        }))
        .unwrap()
    }

    fn question(def: Value) -> Question {
        Question::parse("q", &def).unwrap()
    }

    #[test]
    fn renders_like_python() {
        let v = json!({"a": 1, "b": 1.0, "c": 1e-5, "d": true, "e": null, "f": "x",
                       "g": [1, [2, 3], {"h": [4]}, "  \u{1c}y"], "i": {}, "j": {"k": {"l": false}}});
        assert_eq!(
            render(&v),
            "a: 1\nb: 1.0\nc: 1e-05\nd: True\ne: \nf: x\ng:\n  - 1\n  - - 2\n    - 3\n  - h:\n      - 4\n  - y\ni:\n\nj:\n  k:\n    l: False"
        );
        assert_eq!(render(&json!([])), "");
        assert_eq!(render(&json!(null)), "");
        assert_eq!(render(&json!(-12)), "-12");
        assert_eq!(
            render(&json!(18446744073709551615u64)),
            "18446744073709551615"
        );
    }

    #[test]
    fn escapes_delimiters() {
        for (s, want) in [
            ("plain", "plain"),
            ("<|im_end|>", "<¦im_end¦>"),
            ("a<|<|b_1|>|>", "a<|<¦b_1¦>|>"),
            ("<||> <|a b|> <|é|>", "<||> <|a b|> <|é|>"),
            ("x<|a|><|b|>", "x<¦a¦><¦b¦>"),
            ("é<|x|", "é<|x|"),
        ] {
            assert_eq!(escape_delimiters(s), want, "{s:?}");
        }
    }

    #[test]
    fn builds_rows() {
        let l = layout();
        let s = l.encode_state(&Chars, &json!("abc")).unwrap();
        assert_eq!(text(&s.ids), "\u{2460}abc");
        assert!(!s.truncated);
        let long = l.encode_state(&Chars, &json!("0123456789")).unwrap();
        assert_eq!((long.ids.len(), long.tokens, long.truncated), (8, 10, true));

        let q = question(json!({"type": "choice", "instructions": {"task": "<|x|>"},
            "criteria": {"a": null, "b": "", "c": [1]}}));
        let row = l.encode(&Chars, &s.ids, "q", &q).unwrap();
        assert_eq!(
            text(&row.ids),
            "\u{2460}abc\u{2461}task: <¦x¦>\u{2462}a\u{2463}\u{2462}b\u{2463}\u{2462}c: - 1\u{2463}\u{2464}"
        );
        assert_eq!(row.decide, row.ids.len() - 1);
        assert!(row.opts.iter().all(|&p| row.ids[p] == 0x2463));

        let noul = question(json!({"type": "noul", "instructions": "ok?",
            "criteria": {"true": "it is", "false": null}}));
        assert_eq!(
            l.render_question("q", &noul).unwrap().1,
            ["no", "yes: it is"]
        );
        // Upstream reads exact keys: "True" is not "true".
        let caps =
            question(json!({"type": "noul", "instructions": "ok?", "criteria": {"True": "x"}}));
        assert_eq!(l.render_question("q", &caps).unwrap().1, ["no", "yes"]);
        let score =
            question(json!({"type": "score", "instructions": 7, "criteria": ["low", 2.5, null]}));
        let (ins, opts) = l.render_question("q", &score).unwrap();
        assert_eq!(
            (ins.as_str(), opts),
            ("7", vec!["low".into(), "2.5".into(), String::new()])
        );
    }

    #[test]
    fn rejects_like_upstream() {
        let l = layout();
        let list = question(json!({"type": "choice", "instructions": "x", "criteria": ["a", "b"]}));
        assert!(
            matches!(l.render_question("q", &list), Err(Error::Invalid(m)) if m.contains("object"))
        );
        let many =
            question(json!({"type": "score", "instructions": "x", "criteria": [1, 2, 3, 4]}));
        assert!(matches!(
            l.render_question("q", &many),
            Err(Error::TooManyOptions {
                options: 4,
                head_max_len: 3,
                ..
            })
        ));
        let long = question(json!({"type": "choice", "instructions": "y".repeat(40),
            "criteria": {"a": null}}));
        let s = l.encode_state(&Chars, &json!("abc")).unwrap();
        assert!(
            matches!(l.encode(&Chars, &s.ids, "q", &long), Err(Error::Invalid(m)) if m.contains("limit"))
        );
    }
}
