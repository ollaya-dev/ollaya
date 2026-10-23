//! `jev-style-v1`: chaoliangUNSW/Jev-Style-2B-Decision-v3 (Apache-2.0), the "render v2" input
//! (`macjev-render-v2-long-options`, layout `sb`) and its verdict readout.
//!
//! A port of the model's own self-contained runtime `jev_style_decision_gguf.py` (the "Shared core",
//! byte-identical across its torch / mlx / gguf runtimes), `release_config.json` -> `runtime_parity`.
//! Unlike `jevk5-v1` / `llm-logits-v1` this model is **not** a last-token label-logit readout:
//!
//!   * no chat template, no system prompt, no answer letters;
//!   * the input is plain text, token segments encoded separately so slot positions are exact:
//!
//!     ```text
//!     state prefix   State:\n<state>\n\n
//!     short form     Question [<type>]: <question>\nOptions:\n
//!     (q+opts+slots   - <option 1> ->\n ... - <option K> ->\n
//!      <= 2048 tok)
//!     overflow form  Question [<type>]: <question>\nOptions:\n
//!     (otherwise)    Option 1: <option 1>\n ... Option K: <option K>\n        (the catalogue)
//!                    Judge each numbered option in the complete catalogue above:\n
//!                    Option 1 ->\n ... Option K ->\n                           (the rubric)
//!     ```
//!
//!   * the score of option k is `logit(" yes") - logit(" no")` read at the k-th ` ->` slot, and the
//!     probabilities are `softmax(scores / T)` in canonical option order, with ONE global temperature
//!     (`readout_config.json` `temperatures.global` = 0.8278650620942867).
//!
//! The engine ([`crate::...`] via the llama runner) owns the decode: this module renders the token
//! ids, the per-option verdict slots and the attention blocks; it does not read logits.
//!
//! The attention is block-causal in the model's 6 full-attention layers (every block attends to all
//! earlier tokens and itself, no causal mask inside a block); the Gated-DeltaNet layers are recurrent.
//! The blocks here tile `[0, len(ids))`: the state is cut into 2048-token blocks, the short question
//! is one block, and in the overflow form the catalogue and the rubric are each cut into 2048 blocks.

use serde::Deserialize;
use serde_json::Value;

use crate::Error;
use crate::question::{Criteria, QType, Question};

pub const LAYOUT: &str = "jev-style-v1";
/// Attention block size (a processing unit, not a content limit).
pub const BLOCK: usize = 2048;
/// The complete input (state + question + options + readout) must fit this many tokens.
pub const CONTEXT_LIMIT: usize = 25_600;

/// `decision.json` of a `jev-style-v1` model: the three readout token ids (checked against the GGUF
/// at load, as `readout_config.json` `slot_tokens`) and the one global calibration temperature.
#[derive(Debug, Clone, Deserialize)]
pub struct JevStyleConfig {
    pub layout: String,
    /// ` yes` token id (9542 in v3).
    pub yes: u32,
    /// ` no` token id (874).
    pub no: u32,
    /// ` ->` verdict-slot token id (1411).
    pub arrow: u32,
}

impl JevStyleConfig {
    pub fn validate(&self) -> Result<(), Error> {
        if self.layout != LAYOUT {
            return Err(Error::invalid(format!(
                "decision.json: layout {:?} is not {LAYOUT}",
                self.layout
            )));
        }
        Ok(())
    }
}

/// One rendered question: token ids, the verdict slots (one per option, canonical order), the
/// attention blocks (`[start, stop)` pairs tiling `[0, ids.len())`) and the state-prefix length.
#[derive(Debug, Clone, PartialEq)]
pub struct Rendered {
    pub qtype: QType,
    /// Option identifiers in canonical order = the order probabilities are returned.
    pub names: Vec<String>,
    pub ids: Vec<u32>,
    pub prefix_len: usize,
    /// Prompt position of each option's ` ->` slot, canonical order.
    pub slots: Vec<usize>,
    pub blocks: Vec<(usize, usize)>,
    pub catalogue_overflow: bool,
}

impl Rendered {
    pub fn input_tokens(&self) -> usize {
        self.ids.len()
    }
}

/// `serialize_state`: a string state passes through; any other JSON value is `json.dumps`ed
/// (`ensure_ascii=False`).
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        v => crate::pyjson::dumps(v, false),
    }
}

/// Canonical option identifiers in probability order: a choice's labels, a score's level
/// indices, or `["false", "true"]` for a noul. The shared parser ([`crate::question::Question`])
/// has already validated and ordered them (and accepts a choice as a list of labels).
fn canonical_names(q: &Question) -> Vec<String> {
    match &q.criteria {
        Criteria::Choice(m) => m.keys().cloned().collect(),
        Criteria::Score(levels) => (0..levels.len()).map(|i| i.to_string()).collect(),
        Criteria::Noul { .. } => vec!["false".into(), "true".into()],
    }
}

/// `[(s, min(s + BLOCK, stop)) for s in range(start, stop, BLOCK)]`.
fn blocks(start: usize, stop: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut s = start;
    while s < stop {
        out.push((s, (s + BLOCK).min(stop)));
        s += BLOCK;
    }
    out
}

impl JevStyleConfig {
    /// Render one wire question (`{type, instructions, criteria}`) into token ids, verdict slots and
    /// attention blocks. `tokenize(text)` encodes with no BOS/EOS and special tokens split (never
    /// control tokens), exactly as the runtime's `TextEncoder` does. Validation and the budget raise
    /// [`Error`]; a tokenizer failure is the caller's `E`.
    pub fn render<E: From<Error>>(
        &self,
        qid: &str,
        q: &Question,
        state_prefix: &[u32],
        mut tokenize: impl FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Rendered, E> {
        let bad = |m: String| E::from(Error::invalid(format!("question {qid:?}: {m}")));
        // The shared parser accepts structured instructions (json.dumps) and 1..N score levels;
        // jev-style keeps the author's stricter limits: a non-empty string and 2..=10 levels.
        let ins = match q.definition.get("instructions") {
            Some(Value::String(s)) if !s.trim().is_empty() => s.clone(),
            _ => {
                return Err(bad(
                    "question text (instructions) must be a non-empty string".into(),
                ));
            }
        };
        if q.qtype == QType::Score && !(2..=10).contains(&q.num_options()) {
            return Err(bad("score needs a list of 2..10 level descriptions".into()));
        }
        let qtype = q.qtype;
        let names = canonical_names(q);
        let bodies = q.render_options();

        let head = tokenize(&format!("Question [{}]: {ins}\nOptions:\n", qtype.name()))?;
        let opts: Vec<Vec<u32>> = bodies
            .iter()
            .map(|o| tokenize(o))
            .collect::<Result<_, _>>()?;
        let dash = tokenize("- ")?;
        let newline = tokenize("\n")?;
        let k = opts.len();
        if k < 1 {
            return Err(bad("at least one option is required".into()));
        }

        // short form: head, then `- <opt> ->\n` per option; the slot is the ` ->` position.
        let mut short = head.clone();
        let mut rel = Vec::with_capacity(k);
        for o in &opts {
            short.extend_from_slice(&dash);
            short.extend_from_slice(o);
            short.push(self.arrow);
            rel.push(short.len() - 1);
            short.extend_from_slice(&newline);
        }
        let overflow = short.len() > BLOCK;

        let prefix_len = state_prefix.len();
        let (ids, slots, blks) = if !overflow {
            let mut ids = state_prefix.to_vec();
            ids.extend_from_slice(&short);
            let slots = rel.iter().map(|&s| prefix_len + s).collect();
            let mut blks = blocks(0, prefix_len);
            blks.push((prefix_len, ids.len()));
            (ids, slots, blks)
        } else {
            // overflow form: a numbered catalogue, then a rubric of `Option k ->` slots.
            let mut catalogue = head.clone();
            for (pos, o) in opts.iter().enumerate() {
                catalogue.extend(tokenize(&format!("Option {}: ", pos + 1))?);
                catalogue.extend_from_slice(o);
                catalogue.extend_from_slice(&newline);
            }
            let doc_end = prefix_len + catalogue.len();
            let mut rubric =
                tokenize("Judge each numbered option in the complete catalogue above:\n")?;
            let mut rel = Vec::with_capacity(k);
            for pos in 0..k {
                rubric.extend(tokenize(&format!("Option {}", pos + 1))?);
                rubric.push(self.arrow);
                rel.push(rubric.len() - 1);
                rubric.extend_from_slice(&newline);
            }
            let mut ids = state_prefix.to_vec();
            ids.extend_from_slice(&catalogue);
            ids.extend_from_slice(&rubric);
            let slots = rel.iter().map(|&s| doc_end + s).collect();
            let mut blks = blocks(0, prefix_len);
            blks.extend(blocks(prefix_len, doc_end));
            blks.extend(blocks(doc_end, ids.len()));
            (ids, slots, blks)
        };

        if ids.len() > CONTEXT_LIMIT {
            return Err(bad(format!(
                "the complete input needs {} tokens (state {prefix_len} + question/options/readout \
                 {}); the limit is {CONTEXT_LIMIT}. Nothing was truncated: shorten the state, the \
                 question or the options.",
                ids.len(),
                ids.len() - prefix_len
            )));
        }
        Ok(Rendered {
            qtype,
            names,
            ids,
            prefix_len,
            slots,
            blocks: blks,
            catalogue_overflow: overflow,
        })
    }

    /// The state prefix `State:\n<state>\n\n`, tokenised once and shared across a request's questions.
    pub fn state_prefix<E>(
        &self,
        state: &Value,
        mut tokenize: impl FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Vec<u32>, E> {
        let mut ids = tokenize("State:\n")?;
        ids.extend(tokenize(&serialize_state(state))?);
        ids.extend(tokenize("\n\n")?);
        Ok(ids)
    }
}

/// `softmax(scores / T)` in f64, canonical option order (the reference computation).
pub fn probabilities(scores: &[f64], temperature: f64) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores
        .iter()
        .map(|&s| ((s - max) / temperature).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const ARROW: u32 = 1411;

    fn config() -> JevStyleConfig {
        JevStyleConfig {
            layout: LAYOUT.into(),
            yes: 9542,
            no: 874,
            arrow: ARROW,
        }
    }

    /// A character-level tokenizer: one token per char, except ` ->` (the verdict slot) which is the
    /// single configured arrow id. Good enough to assert slot positions and reconstruct the text.
    fn tok(text: &str) -> Result<Vec<u32>, Error> {
        let mut ids = Vec::new();
        let mut rest = text;
        while let Some(c) = rest.chars().next() {
            if let Some(stripped) = rest.strip_prefix(" ->") {
                ids.push(ARROW);
                rest = stripped;
            } else {
                ids.push(c as u32);
                rest = &rest[c.len_utf8()..];
            }
        }
        Ok(ids)
    }

    fn text_of(ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| {
                if i == ARROW {
                    " ->".to_string()
                } else {
                    char::from_u32(i).unwrap().to_string()
                }
            })
            .collect()
    }

    fn render(c: &JevStyleConfig, state: &Value, qid: &str, def: &Value) -> Rendered {
        let prefix = c.state_prefix(state, tok).unwrap();
        let q = crate::question::Question::parse(qid, def).unwrap();
        c.render::<Error>(qid, &q, &prefix, tok).unwrap()
    }

    #[test]
    fn short_form_choice_matches_reference() {
        let c = config();
        let r = render(
            &c,
            &json!("the order was billed twice"),
            "team",
            &json!({"type": "choice", "instructions": "Which team?",
                    "criteria": {"billing": "cards", "tech": null, "sales": ""}}),
        );
        assert!(!r.catalogue_overflow);
        assert_eq!(r.names, ["billing", "tech", "sales"]);
        assert_eq!(
            text_of(&r.ids),
            "State:\nthe order was billed twice\n\n\
             Question [choice]: Which team?\nOptions:\n\
             - billing: cards ->\n- tech ->\n- sales ->\n"
        );
        // Each slot sits on a ` ->` arrow token.
        for &s in &r.slots {
            assert_eq!(r.ids[s], ARROW);
        }
        assert_eq!(r.slots.len(), 3);
    }

    #[test]
    fn noul_and_score_bodies_match_reference() {
        let c = config();
        let n = render(
            &c,
            &json!("x"),
            "q",
            &json!({"type": "noul", "instructions": "Finished?"}),
        );
        assert_eq!(n.names, ["false", "true"]);
        assert!(text_of(&n.ids).contains(
            "- false: no, the statement does not hold ->\n- true: yes, the statement holds ->\n"
        ));

        let s = render(
            &c,
            &json!("x"),
            "q",
            &json!({"type": "score", "instructions": "How bad?", "criteria": ["calm", "tense", 2]}),
        );
        assert_eq!(s.names, ["0", "1", "2"]);
        assert!(
            text_of(&s.ids).contains("- level 0: calm ->\n- level 1: tense ->\n- level 2: 2 ->\n")
        );
    }

    #[test]
    fn state_is_serialized_like_json_dumps() {
        let c = config();
        let r = render(
            &c,
            &json!({"n": 1.5, "msg": "hi"}),
            "q",
            &json!({"type": "noul", "instructions": "ok?"}),
        );
        assert!(text_of(&r.ids).starts_with("State:\n{\"n\": 1.5, \"msg\": \"hi\"}\n\n"));
    }

    #[test]
    fn overflow_switches_to_catalogue_and_rubric() {
        let c = config();
        // Many long options so the short block exceeds BLOCK tokens.
        let crit: serde_json::Map<String, Value> = (0..200)
            .map(|i| {
                (
                    format!("option_number_{i:03}_with_a_fairly_long_name"),
                    Value::Null,
                )
            })
            .collect();
        let r = render(
            &c,
            &json!("s"),
            "q",
            &json!({"type": "choice", "instructions": "pick", "criteria": crit}),
        );
        assert!(r.catalogue_overflow);
        assert_eq!(r.slots.len(), 200);
        let t = text_of(&r.ids);
        assert!(t.contains("Option 1: option_number_000"));
        assert!(t.contains("Judge each numbered option in the complete catalogue above:\n"));
        assert!(t.contains("Option 1 ->\n"));
        for &s in &r.slots {
            assert_eq!(r.ids[s], ARROW);
        }
    }

    #[test]
    fn blocks_tile_the_whole_input() {
        let c = config();
        let r = render(
            &c,
            &json!("hello"),
            "q",
            &json!({"type": "noul", "instructions": "ok?"}),
        );
        assert_eq!(r.blocks.first().unwrap().0, 0);
        assert_eq!(r.blocks.last().unwrap().1, r.ids.len());
        for w in r.blocks.windows(2) {
            assert_eq!(w[0].1, w[1].0); // contiguous, no gaps/overlaps
            assert!(w[0].1 - w[0].0 <= BLOCK);
        }
    }

    #[test]
    fn probabilities_softmax() {
        let p = probabilities(&[2.0, 0.0], 1.0);
        assert!((p[0] + p[1] - 1.0).abs() < 1e-12);
        assert!(p[0] > p[1]);
    }
}
