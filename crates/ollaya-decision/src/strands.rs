//! `strands-pointer-v1` and `-v2`: Strands Agents' Strands Decider (a LoRA on a Qwen3.5 or Gemma 4
//! base plus a pointer head), whose rows follow upstream `strands_decider.prompting` and
//! `SystemOneEngine._fit` (`ollaya_convert.families.strands.layout`). v2 differs in one token: the
//! state starts with `special_tokens.bos`, as Gemma's tokenizer adds it.
//!
//! Every question becomes one causal row: the state, then the question with one line per option.
//!
//! ```text
//! state    = "<state>\n" render(state) "\n</state>\n"
//! question = "<question type=\"{type}\">\n" header "\n" render(instructions) "\n<options>\n"
//!            line_1 "\n" ... line_k "\n</options>\n</question>\n<answer>"
//! line_j   = "{j}. {name}" or "{j}. {name} — {collapse(text)}"
//! ```
//!
//! `render` is `""` for null, the stripped string, or `json.dumps(indent=2)`. Questions are fitted
//! in runs of `max_batch`: with `L` the run's longest question, each keeps its last
//! `min(L, int(max_length * max_question_fraction))` tokens and the state its first
//! `max_length` minus that. The graph scores option j at the last token of its line against the
//! last token of the row (`<answer>`); those scores are the option logits for every type.

use serde::Deserialize;
use serde_json::Value;

use crate::kev::KevRow;
use crate::question::Questions;
use crate::{Error, pyjson};

/// Token ids and each token's `(start, end)` character offsets.
pub type Encoded = (Vec<u32>, Vec<(usize, usize)>);

/// Tokenizes text without special tokens, with each token's character offsets
/// (`tok(text, add_special_tokens=False, return_offsets_mapping=True)`).
pub trait OffsetEncoder {
    fn encode_offsets(&self, text: &str) -> Result<Encoded, Error>;
}

/// The line above each type's instructions.
#[derive(Debug, Clone, Deserialize)]
pub struct Headers {
    pub noul: String,
    pub choice: String,
    pub score: String,
}

/// What a noul's two lines say when the request leaves a side out.
#[derive(Debug, Clone, Deserialize)]
pub struct NoulDefaults {
    pub r#false: String,
    pub r#true: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Templates {
    pub headers: Headers,
    pub noul_defaults: NoulDefaults,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StrandsTokens {
    pub pad: u32,
    /// `strands-pointer-v2`: the token the tokenizer prepends to the state (Gemma's `<bos>`,
    /// upstream `force_bos`); it counts against the state's budget as upstream's truncation does.
    #[serde(default)]
    pub bos: Option<u32>,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct StrandsLayout {
    /// The window a row is fitted to.
    pub max_length: usize,
    /// The largest share of the window a question may keep.
    pub max_question_fraction: f64,
    /// Questions fitted together (upstream `EngineConfig.max_batch`).
    pub max_batch: usize,
    /// `[min, max]` options of a choice question.
    pub choice_options: [usize; 2],
    /// `[min, max]` levels of a score question.
    pub score_levels: [usize; 2],
    pub special_tokens: StrandsTokens,
    pub templates: Templates,
}

/// One request's rows, one per question, in request order.
#[derive(Debug, Clone, PartialEq)]
pub struct StrandsEncoding {
    pub rows: Vec<KevRow>,
    /// Tokens of the rendered state before any cut.
    pub state_tokens: usize,
    pub state_truncated: bool,
}

/// Python's `str.isspace`: Rust's White_Space plus the four information separators (U+001C..U+001F).
fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `render_content`: null as `""`, a string stripped, anything else as `json.dumps(indent=2)`.
fn render(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.trim_matches(py_isspace).to_owned(),
        v => pyjson::dumps_indent(v, false, 2),
    }
}

/// `" ".join(s.split())`.
fn collapse(s: &str) -> String {
    s.split(py_isspace)
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Upstream's `Content`: a string, an object or an array.
fn is_content(v: &Value) -> bool {
    matches!(v, Value::String(_) | Value::Object(_) | Value::Array(_))
}

impl StrandsLayout {
    /// Reject configurations this layout cannot run.
    pub fn validate(&self) -> Result<(), Error> {
        let range = |[lo, hi]: [usize; 2]| lo >= 1 && lo <= hi;
        if self.max_length == 0
            || self.max_batch == 0
            || !(self.max_question_fraction > 0.0 && self.max_question_fraction <= 1.0)
            || !range(self.choice_options)
            || !range(self.score_levels)
        {
            return Err(Error::invalid(format!(
                "max_length={}, max_batch={}, max_question_fraction={}, choice_options={:?} and \
                 score_levels={:?} are inconsistent",
                self.max_length,
                self.max_batch,
                self.max_question_fraction,
                self.choice_options,
                self.score_levels
            )));
        }
        Ok(())
    }

    /// The most tokens a question keeps: `max(1, int(max_length * max_question_fraction))`.
    fn reserve_cap(&self) -> usize {
        ((self.max_length as f64 * self.max_question_fraction) as usize).max(1)
    }

    /// One question's text and each option line's character span in it, validated as upstream's
    /// request model does. `def` is the definition as the caller sent it.
    pub fn question(&self, qid: &str, def: &Value) -> Result<(String, Vec<(usize, usize)>), Error> {
        let bad = |msg: &str| Error::invalid(format!("question {qid:?}: {msg}"));
        let instructions = def.get("instructions").unwrap_or(&Value::Null);
        if !is_content(instructions) {
            return Err(bad(
                "this model takes instructions as a string, an object or an array",
            ));
        }
        let crit = def.get("criteria").unwrap_or(&Value::Null);
        let described =
            |m: &serde_json::Map<String, Value>| m.values().all(|v| v.is_null() || is_content(v));
        let h = &self.templates.headers;
        let (kind, header, pairs): (&str, &str, Vec<(String, String)>) = match def
            .get("type")
            .and_then(Value::as_str)
        {
            Some("noul") => {
                let empty = serde_json::Map::new();
                let c = match crit {
                    Value::Null => &empty,
                    Value::Object(m) if m.keys().all(|k| k == "true" || k == "false") => m,
                    _ => {
                        return Err(bad(
                            "this model takes noul criteria with only \"true\" and \"false\" keys",
                        ));
                    }
                };
                if !described(c) {
                    return Err(bad("criteria must be strings, objects, arrays or null"));
                }
                let d = &self.templates.noul_defaults;
                let side = |key: &str, default: &str| match c.get(key) {
                    None => default.to_owned(),
                    Some(v) => render(v),
                };
                let pairs = vec![
                    ("false".to_owned(), side("false", &d.r#false)),
                    ("true".to_owned(), side("true", &d.r#true)),
                ];
                ("noul", h.noul.as_str(), pairs)
            }
            Some("choice") => {
                let [lo, hi] = self.choice_options;
                let Value::Object(m) = crit else {
                    return Err(bad(
                        "this model takes choice criteria as an object of label -> description; \
                             write a list of labels as {\"label\": null, ...}",
                    ));
                };
                if m.len() > hi {
                    return Err(Error::TooManyOptions {
                        question: qid.to_owned(),
                        options: m.len(),
                        head_max_len: hi,
                    });
                }
                if m.len() < lo {
                    return Err(bad(&format!("this model needs {lo} to {hi} options")));
                }
                if !described(m) {
                    return Err(bad("criteria must be strings, objects, arrays or null"));
                }
                let pairs = m.iter().map(|(k, v)| (k.clone(), render(v))).collect();
                ("choice", h.choice.as_str(), pairs)
            }
            Some("score") => {
                let [lo, hi] = self.score_levels;
                let levels: Option<Vec<&str>> = crit
                    .as_array()
                    .and_then(|l| l.iter().map(Value::as_str).collect());
                let levels = match levels {
                    Some(l) if (lo..=hi).contains(&l.len()) => l,
                    _ => {
                        return Err(bad(&format!(
                            "this model takes score criteria as a list of {lo} to {hi} strings"
                        )));
                    }
                };
                let pairs = levels
                    .iter()
                    .enumerate()
                    .map(|(i, l)| (i.to_string(), (*l).to_owned()))
                    .collect();
                ("score", h.score.as_str(), pairs)
            }
            _ => return Err(bad("unknown type; use one of choice, noul, score")),
        };

        let mut text = format!(
            "<question type=\"{kind}\">\n{header}\n{}\n<options>\n",
            render(instructions)
        );
        let mut cursor = text.chars().count();
        let mut spans = Vec::with_capacity(pairs.len());
        for (j, (name, desc)) in pairs.iter().enumerate() {
            let desc = collapse(desc);
            let line = if desc.is_empty() {
                format!("{}. {name}", j + 1)
            } else {
                format!("{}. {name} \u{2014} {desc}", j + 1)
            };
            if j > 0 {
                text.push('\n');
            }
            let len = line.chars().count();
            spans.push((cursor, cursor + len));
            cursor += len + 1;
            text.push_str(&line);
        }
        text.push_str("\n</options>\n</question>\n<answer>");
        Ok((text, spans))
    }

    /// Every question's row, fitted to the window.
    pub fn encode(
        &self,
        enc: &dyn OffsetEncoder,
        state: &Value,
        questions: &Questions,
    ) -> Result<StrandsEncoding, Error> {
        if !is_content(state) {
            return Err(Error::invalid(
                "this model takes the state as a string, an object or an array",
            ));
        }
        let texts = questions
            .iter()
            .map(|(qid, q)| Ok((qid.as_str(), self.question(qid, &q.definition)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        let (state_ids, _) =
            enc.encode_offsets(&format!("<state>\n{}\n</state>\n", render(state)))?;
        let state_ids: Vec<u32> = self
            .special_tokens
            .bos
            .into_iter()
            .chain(state_ids)
            .collect();
        let cap = self.reserve_cap();
        let mut rows = Vec::with_capacity(texts.len());
        let mut truncated = false;
        for run in texts.chunks(self.max_batch) {
            let encoded = run
                .iter()
                .map(|(_, (text, _))| enc.encode_offsets(text))
                .collect::<Result<Vec<_>, Error>>()?;
            let longest = encoded.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);
            let reserve = longest.min(cap);
            let budget = self.max_length.saturating_sub(reserve).max(1);
            let s = &state_ids[..state_ids.len().min(budget)];
            truncated |= state_ids.len() > budget;
            for ((qid, (_, spans)), (ids, offsets)) in run.iter().zip(&encoded) {
                let cut = ids.len().saturating_sub(reserve);
                let (ids, offsets) = (&ids[cut..], &offsets[cut..]);
                let opts = spans
                    .iter()
                    .map(|&(a, b)| {
                        offsets
                            .iter()
                            .rposition(|&(lo, hi)| hi > lo && lo >= a && hi <= b)
                            .map(|j| s.len() + j)
                            .ok_or_else(|| {
                                Error::invalid(format!(
                                    "question {qid:?}: the window cuts an option's line; shorten the \
                                     question or its options"
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let mut row = Vec::with_capacity(s.len() + ids.len());
                row.extend_from_slice(s);
                row.extend_from_slice(ids);
                rows.push(KevRow {
                    decide: row.len() - 1,
                    ids: row,
                    opts,
                });
            }
        }
        Ok(StrandsEncoding {
            rows,
            state_tokens: state_ids.len(),
            state_truncated: truncated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One token per character, offsets in characters.
    struct Chars;

    impl OffsetEncoder for Chars {
        fn encode_offsets(&self, text: &str) -> Result<Encoded, Error> {
            Ok(text
                .chars()
                .enumerate()
                .map(|(i, c)| (c as u32, (i, i + 1)))
                .unzip())
        }
    }

    fn layout(max_length: usize) -> StrandsLayout {
        StrandsLayout {
            max_length,
            max_question_fraction: 0.75,
            max_batch: 2,
            choice_options: [2, 255],
            score_levels: [2, 10],
            special_tokens: StrandsTokens { pad: 0, bos: None },
            templates: Templates {
                headers: Headers {
                    noul: "N.".into(),
                    choice: "C.".into(),
                    score: "S.".into(),
                },
                noul_defaults: NoulDefaults {
                    r#false: "does not hold".into(),
                    r#true: "holds".into(),
                },
            },
        }
    }

    fn questions(v: Value) -> Questions {
        crate::parse_questions(&v).unwrap()
    }

    #[test]
    fn questions_follow_upstream_prompting() {
        let l = layout(4096);
        let (text, spans) = l
            .question(
                "q",
                &json!({"type": "choice", "instructions": "  Which?\n",
                        "criteria": {"a": " two\n  lines ", "b": null, "c": {"k": [1, 2.5]}}}),
            )
            .unwrap();
        assert_eq!(
            text,
            "<question type=\"choice\">\nC.\nWhich?\n<options>\n1. a \u{2014} two lines\n2. b\n\
             3. c \u{2014} { \"k\": [ 1, 2.5 ] }\n</options>\n</question>\n<answer>"
        );
        let lines: Vec<String> = spans
            .iter()
            .map(|&(a, b)| text.chars().skip(a).take(b - a).collect())
            .collect();
        assert_eq!(lines[1], "2. b");
        let (noul, _) = l
            .question(
                "n",
                &json!({"type": "noul", "instructions": ["x"], "criteria": {"true": null}}),
            )
            .unwrap();
        assert!(noul.contains("1. false \u{2014} does not hold\n2. true\n</options>"));
        assert!(noul.contains("N.\n[\n  \"x\"\n]\n"));
    }

    #[test]
    fn v2_prepends_bos_to_the_state() {
        let qs = questions(json!({"q": {"type": "noul", "instructions": "ok?"}}));
        let mut l = layout(4096);
        l.special_tokens.bos = Some(7);
        let e = l.encode(&Chars, &json!("a state"), &qs).unwrap();
        let plain = layout(4096).encode(&Chars, &json!("a state"), &qs).unwrap();
        let (row, base) = (&e.rows[0], &plain.rows[0]);
        assert_eq!(row.ids[0], 7);
        assert_eq!(row.ids[1..], base.ids[..]);
        assert_eq!(row.decide, base.decide + 1);
        assert_eq!(
            row.opts.iter().map(|o| o - 1).collect::<Vec<_>>(),
            base.opts
        );
        assert_eq!(e.state_tokens, plain.state_tokens + 1);
        // In a tight window BOS counts against the state's budget, as upstream's truncation counts it.
        let long = json!("s".repeat(500));
        let mut t = layout(120);
        t.special_tokens.bos = Some(7);
        let a = t.encode(&Chars, &long, &qs).unwrap();
        let b = layout(120).encode(&Chars, &long, &qs).unwrap();
        assert_eq!(a.rows[0].ids.len(), b.rows[0].ids.len());
        assert_eq!(a.rows[0].ids[0], 7);
        assert!(a.state_truncated);
    }

    #[test]
    fn upstream_rejections() {
        let l = layout(4096);
        let reject = |def: Value| l.question("q", &def).is_err();
        assert!(reject(json!({"type": "noul", "instructions": 5})));
        assert!(reject(
            json!({"type": "noul", "instructions": "q", "criteria": {"True": "y"}})
        ));
        assert!(reject(
            json!({"type": "choice", "instructions": "q", "criteria": {"a": null}})
        ));
        assert!(reject(
            json!({"type": "choice", "instructions": "q", "criteria": ["a", "b"]})
        ));
        assert!(reject(
            json!({"type": "choice", "instructions": "q", "criteria": {"a": 1, "b": null}})
        ));
        assert!(reject(
            json!({"type": "score", "instructions": "q", "criteria": ["only"]})
        ));
        assert!(reject(
            json!({"type": "score", "instructions": "q", "criteria": ["a", 1]})
        ));
        let ok = questions(json!({"q": {"type": "noul", "instructions": "q"}}));
        assert!(l.encode(&Chars, &json!(3), &ok).is_err());
        assert!(l.encode(&Chars, &json!(""), &ok).is_ok());
    }

    #[test]
    fn rows_fit_the_window_per_run() {
        // 3 questions in runs of 2: the first run's reserve is its longest question.
        let qs = questions(json!({
            "a": {"type": "noul", "instructions": "short"},
            "b": {"type": "noul", "instructions": "a longer question text"},
            "c": {"type": "noul", "instructions": "c"},
        }));
        let l = layout(120);
        let e = l.encode(&Chars, &json!("s".repeat(100)), &qs).unwrap();
        let state_len = "<state>\n\n</state>\n".len() + 100;
        assert_eq!(e.state_tokens, state_len);
        let qlen = |id: &str| {
            l.question(id, &qs[id].definition)
                .unwrap()
                .0
                .chars()
                .count()
        };
        // reserve = min(the run's longest question, 90); the state keeps 120 - reserve
        let row = |id: &str, longest: usize| {
            let reserve = longest.min(90);
            (120 - reserve).min(state_len) + qlen(id).min(reserve)
        };
        let run1 = qlen("a").max(qlen("b"));
        assert_eq!(e.rows[0].ids.len(), row("a", run1));
        assert_eq!(e.rows[1].ids.len(), row("b", run1));
        assert_eq!(e.rows[2].ids.len(), row("c", qlen("c")));
        assert!(e.state_truncated);
        for r in &e.rows {
            assert_eq!(r.decide, r.ids.len() - 1);
            assert_eq!(r.opts.len(), 2);
            assert!(r.opts[0] < r.opts[1] && r.opts[1] < r.decide);
        }
        // a question longer than the cap loses its front, and its options while they fit
        let long = questions(json!({"q": {"type": "noul", "instructions": "w ".repeat(200)}}));
        let e = l.encode(&Chars, &json!("s"), &long).unwrap();
        assert_eq!(e.rows[0].ids.len(), (120 - 90).min(e.state_tokens) + 90);
        let many: serde_json::Map<String, Value> =
            (0..40).map(|i| (format!("o{i}"), json!(null))).collect();
        let cut =
            questions(json!({"q": {"type": "choice", "instructions": "q", "criteria": many}}));
        assert!(l.encode(&Chars, &json!("s"), &cut).is_err());
    }
}
