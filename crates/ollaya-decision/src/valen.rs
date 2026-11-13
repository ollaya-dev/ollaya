//! `valen-decision-1.5-v1`: Valen-Team's Valen-Decision-1.5 (a Qwen3.5 vision-language backbone
//! with block-structured full attention and a Mixer candidate head), whose one sequence per
//! request follows upstream `Compiler._compile_shared` in the model repository's `compiler.py`
//! (`ollaya_convert.families.valen.layout`).
//!
//! ```text
//! ids = tok(user_open ⧺ strip(image ⧺ state) ⧺ user_close)                      the shared prefix
//!       ⧺ tok(questions_open)
//!       ⧺ for question i: tok(question(n, type) ⧺ instructions ⧺ candidates)    instruction span
//!                         ⧺ for each candidate: tok([key ⧺ key_sep] ⧺ description)   candidate span
//!                                               ⧺ tok(candidate_end)
//!       ⧺ tok(readouts_open)
//!       ⧺ for question i: tok(readout(n))  (its last token is the decision position)
//!                         ⧺ tok(readout_end)
//! ```
//!
//! The state is a string as is, anything else `json.dumps(ensure_ascii=False, sort_keys=True)`;
//! `strip` is the chat template's `|trim`; `image` is `image_open ⧺ image_token × n ⧺ image_close`
//! for a request with an image of `n` visual tokens. Every piece after the prefix is tokenized on
//! its own, and a span is the tokens whose characters overlap the instructions (or the
//! description), read from the tokenizer's offsets as upstream does.
//!
//! noul has two fixed candidates, `true` then `false`, and its criteria are appended to the
//! instructions; a choice's candidates are `key: description` in the caller's order; a score's
//! are its level descriptions alone. Nothing is truncated: a sequence over `max_tokens` is
//! rejected, and so is any text holding one of the tokenizer's special tokens.
//!
//! Each token also has an attention block (upstream's `bbox_blocks` mask: the prefix, the shared
//! glue, and one block for every decision question together) and interleaved M-RoPE positions.

use serde::Deserialize;
use serde_json::Value;

use crate::layout::TokenEncoder;
use crate::question::QType;
use crate::{Error, pyjson, pyrepr};

pub const LAYOUT: &str = "valen-decision-1.5-v1";

/// Attention blocks, as upstream `block_attention.py` numbers them.
pub const PREFIX: i64 = -1;
pub const COMMON: i64 = 0;
/// Every decision question: under `bbox_blocks` they share one block.
pub const QUESTIONS: i64 = 1;
pub const PADDING: i64 = -2;

/// Token ids and each token's `[start, end)` character span in the text.
pub type Offsets = (Vec<u32>, Vec<(usize, usize)>);

/// A tokenizer that also reports each token's character span in the text (Python's
/// `return_offsets_mapping`).
pub trait OffsetEncoder: TokenEncoder {
    fn encode_offsets(&self, text: &str) -> Result<Offsets, Error>;
}

/// The fixed noul candidates and the labels its criteria are appended under.
#[derive(Debug, Clone, Deserialize)]
pub struct NoulPrompt {
    pub r#true: String,
    pub r#false: String,
    pub true_criterion: String,
    pub false_criterion: String,
}

/// The fixed strings of the sequence; `question` takes `{n}` and `{type}`, `readout` `{n}`.
#[derive(Debug, Clone, Deserialize)]
pub struct Prompt {
    pub user_open: String,
    pub user_close: String,
    pub image_open: String,
    pub image_token: String,
    pub image_close: String,
    pub questions_open: String,
    pub question: String,
    pub candidates: String,
    pub key_sep: String,
    pub candidate_end: String,
    pub readouts_open: String,
    pub readout: String,
    pub readout_end: String,
    pub noul: NoulPrompt,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Tokens {
    pub image: u32,
    pub pad: u32,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct Valen15Layout {
    pub prompt: Prompt,
    /// The tokenizer's special tokens: text that holds one is rejected.
    pub reserved_tokens: Vec<String>,
    pub max_tokens: usize,
    pub min_levels: usize,
    pub max_options: usize,
    pub tokens: Tokens,
    pub hidden_size: usize,
}

/// One question in the sequence: spans are `[start, end)` positions in [`Valen15Row::ids`].
#[derive(Debug, Clone, PartialEq)]
pub struct Valen15Question {
    pub qid: String,
    pub qtype: QType,
    /// Candidate names, in the order the head scores them.
    pub keys: Vec<String>,
    pub instruction: (usize, usize),
    pub candidates: Vec<(usize, usize)>,
    pub decision: usize,
}

/// A request's sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct Valen15Row {
    pub ids: Vec<u32>,
    /// Tokens of the shared prefix (the state and its image).
    pub prefix: usize,
    pub blocks: Vec<i64>,
    /// Interleaved M-RoPE positions (t, h, w) of every token.
    pub positions: [Vec<i64>; 3],
    pub questions: Vec<Valen15Question>,
}

/// The state as the model reads it: a string as is, anything else as sorted-key JSON.
pub fn state_text(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => pyjson::dumps_sorted(other),
    }
}

type Candidates = Vec<(String, String)>;

impl Valen15Layout {
    pub fn validate(&self) -> Result<(), Error> {
        let p = &self.prompt;
        if self.max_tokens == 0
            || self.hidden_size == 0
            || self.max_options == 0
            || p.image_token.is_empty()
            || p.readout.is_empty()
        {
            return Err(Error::invalid(
                "decision.json: max_tokens, hidden_size, max_options and the prompt are required",
            ));
        }
        Ok(())
    }

    fn clean(&self, what: &str, text: &str) -> Result<(), Error> {
        match self.reserved_tokens.iter().find(|t| text.contains(*t)) {
            Some(t) => Err(Error::invalid(format!(
                "{what} holds {t:?}, a control token of this model's tokenizer"
            ))),
            None => Ok(()),
        }
    }

    /// upstream `fold_noul_criteria`, then `schema.candidates`: the type, the instructions as the
    /// model reads them and `(key, description)` per candidate.
    fn candidates(&self, qid: &str, def: &Value) -> Result<(QType, String, Candidates), Error> {
        let bad = |msg: String| Error::invalid(format!("question {qid:?}: {msg}"));
        let q = def
            .as_object()
            .ok_or_else(|| bad("must be an object".into()))?;
        let kind = q.get("type").and_then(Value::as_str);
        let criteria = q.get("criteria").unwrap_or(&Value::Null);
        let text = || {
            q.get("instructions")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("this model takes instructions as text".into()))
        };
        let n = &self.prompt.noul;
        let mut instructions = String::new();
        if kind == Some("noul") && pyrepr::truthy(criteria) {
            let both = criteria
                .as_object()
                .filter(|m| m.len() == 2)
                .and_then(|m| Some((m.get("true")?, m.get("false")?)));
            let Some((t, f)) = both else {
                return Err(bad(
                    "this model takes noul criteria as an object with exactly the keys \"true\" and \"false\""
                        .into(),
                ));
            };
            instructions = format!(
                "{}{}{}{}{}",
                text()?,
                n.true_criterion,
                pyrepr::str(t),
                n.false_criterion,
                pyrepr::str(f)
            );
        }
        if instructions.is_empty() {
            instructions = text()?.to_owned();
        }
        if pyrepr::strip(&instructions).is_empty() {
            return Err(bad("instructions must not be blank".into()));
        }
        let described = |v: &Value| v.as_str().map(str::to_owned);
        let (qtype, pairs) = match kind {
            Some("noul") => (
                QType::Noul,
                vec![
                    ("true".to_owned(), Some(n.r#true.clone())),
                    ("false".to_owned(), Some(n.r#false.clone())),
                ],
            ),
            Some("choice") => {
                let m = criteria
                    .as_object()
                    .filter(|m| (1..=self.max_options).contains(&m.len()))
                    .ok_or_else(|| {
                        bad(format!(
                            "this model takes choice criteria as an object of 1 to {} label -> description entries",
                            self.max_options
                        ))
                    })?;
                (
                    QType::Choice,
                    m.iter().map(|(k, v)| (k.clone(), described(v))).collect(),
                )
            }
            Some("score") => {
                let levels = criteria
                    .as_array()
                    .filter(|l| (self.min_levels..=self.max_options).contains(&l.len()))
                    .ok_or_else(|| {
                        bad(format!(
                            "this model takes score criteria as a list of {} to {} level descriptions",
                            self.min_levels, self.max_options
                        ))
                    })?;
                (
                    QType::Score,
                    levels
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (i.to_string(), described(v)))
                        .collect(),
                )
            }
            _ => return Err(bad("type must be noul, choice or score".into())),
        };
        let pairs = pairs
            .into_iter()
            .map(|(k, v)| match v {
                Some(v) if !k.is_empty() && !pyrepr::strip(&v).is_empty() => Ok((k, v)),
                _ => Err(bad(
                    "every option needs a name and a description that is non-blank text".into(),
                )),
            })
            .collect::<Result<Candidates, Error>>()?;
        Ok((qtype, instructions, pairs))
    }

    /// The request's sequence, every question in request order. `image_grid` is the image's
    /// merged (rows, columns) grid of visual tokens, `None` for a request without an image.
    pub fn encode(
        &self,
        tok: &dyn OffsetEncoder,
        state: &Value,
        defs: &[(&str, &Value)],
        image_grid: Option<(usize, usize)>,
    ) -> Result<Valen15Row, Error> {
        if defs.is_empty() {
            return Err(Error::invalid("questions must be a non-empty object"));
        }
        let p = &self.prompt;
        let mut selected = Vec::with_capacity(defs.len());
        for (qid, def) in defs {
            let (qtype, instructions, pairs) = self.candidates(qid, def)?;
            selected.push((*qid, qtype, instructions, pairs));
        }
        let text = state_text(state);
        self.clean("the state", &text)?;
        let image = match image_grid {
            Some((rows, cols)) => format!(
                "{}{}{}",
                p.image_open,
                p.image_token.repeat(rows * cols),
                p.image_close
            ),
            None => String::new(),
        };
        let content = format!("{image}{text}");
        let mut ids = tok.encode(&format!(
            "{}{}{}",
            p.user_open,
            pyrepr::strip(&content),
            p.user_close
        ))?;
        let prefix = ids.len();
        let mut blocks = vec![PREFIX; prefix];

        let glue = |ids: &mut Vec<u32>, blocks: &mut Vec<i64>, piece: &str, block: i64| {
            let new = tok.encode(piece)?;
            blocks.extend(std::iter::repeat_n(block, new.len()));
            ids.extend(new);
            Ok::<(), Error>(())
        };
        // `lead ⧺ body`, and the span of the tokens whose characters overlap `body`.
        let spanned =
            |ids: &mut Vec<u32>, blocks: &mut Vec<i64>, lead: &str, body: &str, tail: &str| {
                let (new, offsets) = tok.encode_offsets(&format!("{lead}{body}{tail}"))?;
                let from = lead.chars().count();
                let to = from + body.chars().count();
                let inside = |&(a, b): &(usize, usize)| a < to && b > from;
                let (Some(first), Some(last)) = (
                    offsets.iter().position(inside),
                    offsets.iter().rposition(inside),
                ) else {
                    return Err(Error::invalid(
                        "a question's instructions or an option's description holds no token",
                    ));
                };
                let start = ids.len();
                blocks.extend(std::iter::repeat_n(QUESTIONS, new.len()));
                ids.extend(new);
                Ok::<(usize, usize), Error>((start + first, start + last + 1))
            };

        glue(&mut ids, &mut blocks, &p.questions_open, COMMON)?;
        let mut questions = Vec::with_capacity(selected.len());
        for (i, (qid, qtype, instructions, pairs)) in selected.into_iter().enumerate() {
            let what = format!("question {qid:?}");
            self.clean(&what, &instructions)?;
            for (key, description) in &pairs {
                self.clean(&what, key)?;
                self.clean(&what, description)?;
            }
            let lead = p
                .question
                .replace("{n}", &(i + 1).to_string())
                .replace("{type}", qtype.name());
            let instruction = spanned(&mut ids, &mut blocks, &lead, &instructions, &p.candidates)?;
            let mut spans = Vec::with_capacity(pairs.len());
            let mut keys = Vec::with_capacity(pairs.len());
            for (key, description) in pairs {
                let lead = match qtype {
                    QType::Score => String::new(),
                    _ => format!("{key}{}", p.key_sep),
                };
                spans.push(spanned(&mut ids, &mut blocks, &lead, &description, "")?);
                glue(&mut ids, &mut blocks, &p.candidate_end, QUESTIONS)?;
                keys.push(key);
            }
            questions.push(Valen15Question {
                qid: qid.to_owned(),
                qtype,
                keys,
                instruction,
                candidates: spans,
                decision: 0,
            });
        }
        glue(&mut ids, &mut blocks, &p.readouts_open, COMMON)?;
        for (i, q) in questions.iter_mut().enumerate() {
            let readout = p.readout.replace("{n}", &(i + 1).to_string());
            glue(&mut ids, &mut blocks, &readout, QUESTIONS)?;
            q.decision = ids.len() - 1;
            glue(&mut ids, &mut blocks, &p.readout_end, QUESTIONS)?;
        }
        if ids.len() > self.max_tokens {
            return Err(Error::invalid(format!(
                "the request takes {} tokens; this model reads up to {} and truncates nothing",
                ids.len(),
                self.max_tokens
            )));
        }
        let positions = self.positions(&ids, image_grid);
        Ok(Valen15Row {
            ids,
            prefix,
            blocks,
            positions,
            questions,
        })
    }

    /// Interleaved M-RoPE positions: text counts up on all three axes; the image's merged cell
    /// (i, j), starting at position s, is (s, s + i, s + j), and the text after it continues at
    /// s + max(rows, columns) (upstream `get_rope_index`).
    fn positions(&self, ids: &[u32], image_grid: Option<(usize, usize)>) -> [Vec<i64>; 3] {
        let mut pos: [Vec<i64>; 3] = std::array::from_fn(|_| Vec::with_capacity(ids.len()));
        let (mut cur, mut k) = (0i64, 0);
        while k < ids.len() {
            match image_grid {
                Some((rows, cols)) if ids[k] == self.tokens.image => {
                    for i in 0..rows as i64 {
                        for j in 0..cols as i64 {
                            pos[0].push(cur);
                            pos[1].push(cur + i);
                            pos[2].push(cur + j);
                        }
                    }
                    cur += rows.max(cols) as i64;
                    k += rows * cols;
                }
                _ => {
                    for axis in &mut pos {
                        axis.push(cur);
                    }
                    cur += 1;
                    k += 1;
                }
            }
        }
        for axis in &mut pos {
            axis.truncate(ids.len());
        }
        pos
    }
}

impl Valen15Question {
    /// `logits` (one per candidate, in the head's order) in the order answers use: noul
    /// `[false, true]`; a choice's labels and a score's levels are already in the caller's order.
    pub fn in_question_order(&self, logits: &[f32]) -> Result<Vec<f32>, Error> {
        if logits.len() != self.keys.len() {
            return Err(Error::invalid(format!(
                "question {:?}: {} logits for {} options",
                self.qid,
                logits.len(),
                self.keys.len()
            )));
        }
        Ok(match self.qtype {
            QType::Noul => vec![logits[1], logits[0]],
            QType::Choice | QType::Score => logits.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const IMAGE: u32 = 2_000_000;

    /// One token per char; `<|x|>` and `<|img|>` are one token each (ids past the last `char`, so
    /// `text` below prints them as `#`).
    struct Tok;
    impl OffsetEncoder for Tok {
        fn encode_offsets(&self, text: &str) -> Result<Offsets, Error> {
            let (mut ids, mut offsets) = (Vec::new(), Vec::new());
            let (mut rest, mut at) = (text, 0);
            while let Some(c) = rest.chars().next() {
                let special = [("<|x|>", 2_000_001), ("<|img|>", IMAGE)]
                    .into_iter()
                    .find_map(|(s, id)| rest.strip_prefix(s).map(|r| (r, id, s.chars().count())));
                if let Some((r, id, n)) = special {
                    ids.push(id);
                    offsets.push((at, at + n));
                    rest = r;
                    at += n;
                    continue;
                }
                ids.push(c as u32);
                offsets.push((at, at + 1));
                rest = &rest[c.len_utf8()..];
                at += 1;
            }
            Ok((ids, offsets))
        }
    }
    impl TokenEncoder for Tok {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            Ok(self.encode_offsets(text)?.0)
        }
    }

    fn layout(max_tokens: usize) -> Valen15Layout {
        Valen15Layout {
            prompt: Prompt {
                user_open: "<|x|>U\n".into(),
                user_close: "<|x|>\n".into(),
                image_open: "[".into(),
                image_token: "<|img|>".into(),
                image_close: "]".into(),
                questions_open: "<|x|>Q\n".into(),
                question: "Q{n} {type}: ".into(),
                candidates: "\nC:\n".into(),
                key_sep: ": ".into(),
                candidate_end: "\n".into(),
                readouts_open: "<|x|>A\n".into(),
                readout: "D{n}:".into(),
                readout_end: "\n".into(),
                noul: NoulPrompt {
                    r#true: "yes".into(),
                    r#false: "no".into(),
                    true_criterion: "\nT: ".into(),
                    false_criterion: "\nF: ".into(),
                },
            },
            reserved_tokens: vec!["<|x|>".into(), "<|img|>".into()],
            max_tokens,
            min_levels: 2,
            max_options: 255,
            tokens: Tokens {
                image: IMAGE,
                pad: 0,
            },
            hidden_size: 8,
        }
    }

    fn defs(qs: &Value) -> Vec<(&str, &Value)> {
        qs.as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect()
    }

    fn text(row: &Valen15Row, span: (usize, usize)) -> String {
        row.ids[span.0..span.1]
            .iter()
            .map(|&t| char::from_u32(t).unwrap_or('#'))
            .collect()
    }

    #[test]
    fn spans_blocks_and_readouts() {
        let qs = json!({
            "team": {"type": "choice", "instructions": "Which?", "criteria": {"tech": "bugs", "bill": "money"}},
            "ok": {"type": "noul", "instructions": "Fine?", "criteria": {"false": null, "true": [1, "a"]}},
            "lvl": {"type": "score", "instructions": "How?", "criteria": ["low", "high"]},
        });
        let row = layout(10_000)
            .encode(&Tok, &json!("  hi there\n"), &defs(&qs), None)
            .unwrap();
        let [team, ok, lvl] = &row.questions[..] else {
            panic!()
        };
        // the chat template trims the state
        assert_eq!(text(&row, (0, row.prefix)), "#U\nhi there#\n");
        assert_eq!(text(&row, team.instruction), "Which?");
        assert_eq!(team.keys, ["tech", "bill"]);
        assert_eq!(text(&row, team.candidates[0]), "bugs");
        assert_eq!(text(&row, team.candidates[1]), "money");
        // noul: fixed candidates, the criteria appended with Python's str()
        assert_eq!(text(&row, ok.instruction), "Fine?\nT: [1, 'a']\nF: None");
        assert_eq!(ok.keys, ["true", "false"]);
        assert_eq!(text(&row, ok.candidates[0]), "yes");
        // score: the description alone
        assert_eq!(text(&row, lvl.candidates[1]), "high");
        assert_eq!(text(&row, (lvl.decision - 2, lvl.decision + 1)), "D3:");
        let all = text(&row, (0, row.ids.len()));
        assert!(all.contains("#Q\nQ1 choice: Which?\nC:\ntech: bugs\nbill: money\nQ2 noul: "));
        assert!(all.ends_with("#A\nD1:\nD2:\nD3:\n"));
        // blocks: the prefix, the glue, one block for every question and readout
        assert!(row.blocks[..row.prefix].iter().all(|&b| b == PREFIX));
        let glue: String = row
            .ids
            .iter()
            .zip(&row.blocks)
            .filter(|&(_, &b)| b == COMMON)
            .map(|(&t, _)| char::from_u32(t).unwrap_or('#'))
            .collect();
        assert_eq!(glue, "#Q\n#A\n");
        assert_eq!(row.blocks.len(), row.ids.len());
        // text-only positions count up on every axis
        let want: Vec<i64> = (0..row.ids.len() as i64).collect();
        assert!(row.positions.iter().all(|p| *p == want));
    }

    #[test]
    fn image_tokens_and_positions() {
        let qs = json!({"q": {"type": "noul", "instructions": "x"}});
        let l = layout(10_000);
        let row = l
            .encode(&Tok, &json!("s "), &defs(&qs), Some((2, 3)))
            .unwrap();
        // "#U\n[" then six image tokens, "]s" (trimmed), "#\n"
        assert_eq!(&row.ids[4..10], &[IMAGE; 6]);
        assert_eq!(row.prefix, 4 + 6 + 4);
        assert_eq!(row.positions[0][..12], [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 7, 8]);
        assert_eq!(row.positions[1][4..10], [4, 4, 4, 5, 5, 5]);
        assert_eq!(row.positions[2][4..10], [4, 5, 6, 4, 5, 6]);
        assert_eq!(row.positions[1][10..12], [7, 8]);
        // an image and an empty state: the image alone
        let empty = l
            .encode(&Tok, &json!(""), &defs(&qs), Some((1, 1)))
            .unwrap();
        assert_eq!(text(&empty, (0, empty.prefix)), "#U\n[#]#\n");
    }

    #[test]
    fn structured_states_are_sorted_json() {
        let qs = json!({"q": {"type": "noul", "instructions": "x"}});
        let row = layout(10_000)
            .encode(&Tok, &json!({"z": 1, "a": ["é", null]}), &defs(&qs), None)
            .unwrap();
        assert_eq!(
            text(&row, (3, row.prefix - 2)),
            r#"{"a": ["é", null], "z": 1}"#
        );
    }

    #[test]
    fn rejects_what_upstream_rejects() {
        let l = layout(10_000);
        let s = json!("s");
        for q in [
            json!({"type": "choice", "instructions": "x", "criteria": ["a", "b"]}),
            json!({"type": "choice", "instructions": "x", "criteria": {}}),
            json!({"type": "choice", "instructions": "x", "criteria": {"a": null}}),
            json!({"type": "choice", "instructions": "x", "criteria": {"a": " "}}),
            json!({"type": "choice", "instructions": "x", "criteria": {"": "a"}}),
            json!({"type": "choice", "instructions": "x", "criteria": {"a": "<|x|>"}}),
            json!({"type": "score", "instructions": "x", "criteria": ["only"]}),
            json!({"type": "score", "instructions": "x", "criteria": ["a", 2]}),
            json!({"type": "noul", "instructions": " \n"}),
            json!({"type": "noul", "instructions": {"a": 1}}),
            json!({"type": "noul", "instructions": "x", "criteria": {"true": "t"}}),
            json!({"type": "noul", "instructions": "x", "criteria": {"True": "t", "False": "f"}}),
            json!({"type": "noul", "instructions": "x<|x|>"}),
            json!({"type": "bbox", "instructions": "x"}),
        ] {
            assert!(l.encode(&Tok, &s, &[("q", &q)], None).is_err(), "{q}");
        }
        let ok = json!({"type": "noul", "instructions": "x", "criteria": {}});
        assert!(l.encode(&Tok, &s, &[("q", &ok)], None).is_ok());
        assert!(
            l.encode(&Tok, &json!("a <|img|>"), &[("q", &ok)], None)
                .is_err()
        );
        // nothing is truncated
        let full = l.encode(&Tok, &s, &[("q", &ok)], None).unwrap();
        assert!(
            layout(full.ids.len())
                .encode(&Tok, &s, &[("q", &ok)], None)
                .is_ok()
        );
        assert!(
            layout(full.ids.len() - 1)
                .encode(&Tok, &s, &[("q", &ok)], None)
                .is_err()
        );
    }

    #[test]
    fn logits_come_back_in_question_order() {
        let qs = json!({
            "n": {"type": "noul", "instructions": "x"},
            "c": {"type": "choice", "instructions": "x", "criteria": {"b": "B", "a": "A"}},
        });
        let row = layout(10_000)
            .encode(&Tok, &json!("s"), &defs(&qs), None)
            .unwrap();
        assert_eq!(
            row.questions[0].in_question_order(&[0.9, 0.1]).unwrap(),
            [0.1, 0.9]
        );
        assert_eq!(
            row.questions[1].in_question_order(&[1.0, 2.0]).unwrap(),
            [1.0, 2.0]
        );
        assert!(row.questions[1].in_question_order(&[1.0]).is_err());
    }
}
