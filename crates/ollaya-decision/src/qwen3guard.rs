//! `qwen3guard-gen-v1`: Qwen3Guard-Gen read as a fixed-preset decision model
//! (`ollaya_convert.families.qwen3guard.ref`).
//!
//! Qwen3Guard's task (policy, categories, output format) lives in its chat template, so it answers
//! its own questions only: the preset in `decision.json`, which the registry also ships as the
//! model's built-in questions. The state is the text to judge. Two causal rows, read at their last
//! token, restricted to the first tokens of the labels the model writes:
//!
//! ```text
//! r0 = tok(prompt_head + state + prompt_tail + "Safety:")                          -> Safe / Cont / Unsafe
//! r1 = tok(prompt_head + state + prompt_tail + "Safety: Unsafe\nCategories:")      -> the 10 categories
//! ```
//!
//! The graph returns 13 candidate logits per row: `[0:3]` safety, `[3:13]` categories.

use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::Value;

use crate::layout::{TokenEncoder, serialize_state};
use crate::question::{Criteria, Question, Questions, parse_questions};
use crate::{Error, QType};

const SAFETY_ROW: &str = "Safety:";
const CATEGORY_ROW: &str = "Safety: Unsafe\nCategories:";
/// The safety candidates, in graph order.
const SAFETY: [&str; 3] = ["safe", "controversial", "unsafe"];

#[derive(Debug, Clone, Deserialize)]
pub struct Templates {
    /// The chat template up to the user message.
    pub prompt_head: String,
    /// The chat template after the user message, through the empty think block.
    pub prompt_tail: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CandidateSet {
    pub labels: Vec<String>,
    pub ids: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Candidates {
    pub safety: CandidateSet,
    pub category: CandidateSet,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GuardTokens {
    pub pad: u32,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct GuardConfig {
    pub preset_only: bool,
    /// The preset, as a question schema.
    pub questions: Value,
    pub templates: Templates,
    pub candidates: Candidates,
    /// Longest row, in tokens; longer input is rejected, never cut.
    pub max_len: usize,
    pub special_tokens: GuardTokens,
}

/// How a preset question is read from the candidate logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readout {
    /// `safety`: r0 `[safe, controversial, unsafe]`.
    Safety,
    /// `unsafe`: r0 `[logsumexp(safe, controversial), unsafe]`.
    Unsafe,
    /// `unsafe_strict`: r0 `[safe, logsumexp(controversial, unsafe)]`.
    UnsafeStrict,
    /// `category`: r1 categories.
    Category,
}

impl Readout {
    fn of(qid: &str) -> Option<Self> {
        match qid {
            "safety" => Some(Readout::Safety),
            "unsafe" => Some(Readout::Unsafe),
            "unsafe_strict" => Some(Readout::UnsafeStrict),
            "category" => Some(Readout::Category),
            _ => None,
        }
    }

    /// The row this question reads: 0 (safety line) or 1 (categories line).
    pub fn row(self) -> usize {
        match self {
            Readout::Category => 1,
            _ => 0,
        }
    }
}

/// The layout, validated, with its preset parsed.
#[derive(Debug, Clone)]
pub struct GuardLayout {
    pub config: GuardConfig,
    pub preset: Questions,
    readouts: IndexMap<String, Readout>,
}

/// One request's rows: the safety row always, the category row when `category` is asked.
#[derive(Debug, Clone, PartialEq)]
pub struct GuardRows {
    pub rows: Vec<Vec<u32>>,
    /// Readout per question, in request order.
    pub readouts: Vec<Readout>,
}

impl GuardLayout {
    pub fn new(config: GuardConfig) -> Result<Self, Error> {
        let bad = |msg: String| Err(Error::invalid(msg));
        if !config.preset_only {
            return bad(
                "qwen3guard-gen-v1 serves its preset only (preset_only must be true)".into(),
            );
        }
        let (s, c) = (&config.candidates.safety, &config.candidates.category);
        if s.labels != SAFETY || s.ids.len() != SAFETY.len() {
            return bad(format!(
                "candidates.safety must be {SAFETY:?} with one id each"
            ));
        }
        if c.labels.is_empty() || c.labels.len() != c.ids.len() {
            return bad("candidates.category needs one id per label".into());
        }
        let preset = parse_questions(&config.questions)?;
        let mut readouts = IndexMap::new();
        for (qid, q) in &preset {
            let readout = Readout::of(qid)
                .ok_or_else(|| Error::invalid(format!("unknown preset question {qid:?}")))?;
            let expected = match readout {
                Readout::Safety => (QType::Choice, SAFETY.len()),
                Readout::Category => (QType::Choice, c.labels.len()),
                Readout::Unsafe | Readout::UnsafeStrict => (QType::Noul, 2),
            };
            if (q.qtype, q.num_options()) != expected {
                return bad(format!(
                    "preset question {qid:?} must be {} with {} options",
                    expected.0.name(),
                    expected.1
                ));
            }
            readouts.insert(qid.clone(), readout);
        }
        Ok(GuardLayout {
            config,
            preset,
            readouts,
        })
    }

    /// The readout of every question, or an error if any is not one of the preset's own.
    pub fn plan(&self, questions: &Questions) -> Result<Vec<Readout>, Error> {
        questions
            .iter()
            .map(|(qid, q)| match self.preset.get(qid) {
                Some(p) if same_question(p, q) => Ok(self.readouts[qid]),
                _ => Err(Error::invalid(format!(
                    "question {qid:?}: this model answers only its built-in questions ({}); \
                     omit 'questions' to get all of them, or send a subset unchanged",
                    self.preset
                        .keys()
                        .map(String::as_str)
                        .collect::<Vec<_>>()
                        .join(", ")
                ))),
            })
            .collect()
    }

    /// Candidate ids in graph order: safety, then categories.
    pub fn candidate_ids(&self) -> impl Iterator<Item = u32> + '_ {
        let c = &self.config.candidates;
        c.safety.ids.iter().chain(&c.category.ids).copied()
    }

    /// The prompt around `text` as the chat template renders it.
    fn prompt(&self, text: &str) -> String {
        let t = &self.config.templates;
        format!("{}{text}{}", t.prompt_head, t.prompt_tail)
    }

    /// Tokens the prompt adds around an empty state (the safety row).
    pub fn overhead(&self, enc: &dyn TokenEncoder) -> Result<usize, Error> {
        Ok(enc
            .encode(&format!("{}{SAFETY_ROW}", self.prompt("")))?
            .len())
    }

    /// The rows one request needs. The category row is built only when it is read.
    pub fn encode(
        &self,
        enc: &dyn TokenEncoder,
        state: &Value,
        questions: &Questions,
    ) -> Result<GuardRows, Error> {
        let readouts = self.plan(questions)?;
        let base = self.prompt(&serialize_state(state));
        let mut rows = vec![enc.encode(&format!("{base}{SAFETY_ROW}"))?];
        if readouts.contains(&Readout::Category) {
            rows.push(enc.encode(&format!("{base}{CATEGORY_ROW}"))?);
        }
        if let Some(n) = rows.iter().map(Vec::len).max()
            && n > self.config.max_len
        {
            return Err(Error::invalid(format!(
                "the state is too long for this model: its prompt is {n} tokens, the limit is {}",
                self.config.max_len
            )));
        }
        Ok(GuardRows { rows, readouts })
    }

    /// A question's option logits from its row's candidate logits (`[safety..., categories...]`).
    pub fn option_logits(&self, readout: Readout, cand: &[f32]) -> Vec<f32> {
        let z = |i: usize| f64::from(cand[i]);
        match readout {
            Readout::Safety => cand[..SAFETY.len()].to_vec(),
            Readout::Unsafe => vec![logsumexp(z(0), z(1)) as f32, cand[2]],
            Readout::UnsafeStrict => vec![cand[0], logsumexp(z(1), z(2)) as f32],
            Readout::Category => cand[SAFETY.len()..].to_vec(),
        }
    }
}

/// Same type, instructions and criteria, options in the same order (the logits come in the
/// preset's order). Spelling differences that parse the same, such as a list of choice labels or
/// an empty noul criteria object, are the same question.
fn same_question(preset: &Question, q: &Question) -> bool {
    let criteria = match (&preset.criteria, &q.criteria) {
        (Criteria::Choice(a), Criteria::Choice(b)) => a.iter().eq(b.iter()),
        (a, b) => a == b,
    };
    preset.qtype == q.qtype && preset.instructions == q.instructions && criteria
}

fn logsumexp(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One id per char.
    struct Chars;

    impl TokenEncoder for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            Ok(text.chars().map(u32::from).collect())
        }
    }

    fn preset() -> Value {
        json!({
            "safety": {"type": "choice", "instructions": "level",
                       "criteria": {"safe": null, "controversial": null, "unsafe": null}},
            "unsafe": {"type": "noul", "instructions": "unsafe"},
            "unsafe_strict": {"type": "noul", "instructions": "unsafe or controversial"},
            "category": {"type": "choice", "instructions": "category",
                         "criteria": {"none": null, "violent": null}},
        })
    }

    fn layout(max_len: usize) -> GuardLayout {
        GuardLayout::new(
            serde_json::from_value(json!({
                "preset_only": true,
                "questions": preset(),
                "templates": {"prompt_head": "<", "prompt_tail": ">"},
                "candidates": {
                    "safety": {"labels": ["safe", "controversial", "unsafe"], "ids": [1, 2, 3]},
                    "category": {"labels": ["none", "violent"], "ids": [4, 5]},
                },
                "max_len": max_len,
                "special_tokens": {"pad": 0},
            }))
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn builds_the_rows_it_needs() {
        let l = layout(100);
        let all = parse_questions(&preset()).unwrap();
        let r = l.encode(&Chars, &json!({"k": "é"}), &all).unwrap();
        let text: Vec<String> = r
            .rows
            .iter()
            .map(|row| row.iter().map(|&c| char::from_u32(c).unwrap()).collect())
            .collect();
        assert_eq!(
            text,
            [
                "<{\"k\": \"é\"}>Safety:",
                "<{\"k\": \"é\"}>Safety: Unsafe\nCategories:"
            ]
        );
        assert_eq!(
            r.readouts,
            [
                Readout::Safety,
                Readout::Unsafe,
                Readout::UnsafeStrict,
                Readout::Category
            ]
        );
        let one = parse_questions(&json!({"unsafe": preset()["unsafe"]})).unwrap();
        let r = l.encode(&Chars, &json!("hi"), &one).unwrap();
        assert_eq!(
            (r.rows.len(), r.readouts.as_slice()),
            (1, &[Readout::Unsafe][..])
        );
        assert_eq!(l.overhead(&Chars).unwrap(), "<>Safety:".len());
        assert!(layout(5).encode(&Chars, &json!("hi"), &one).is_err());
    }

    #[test]
    fn rejects_other_questions() {
        let l = layout(100);
        for q in [
            json!({"custom": {"type": "noul", "instructions": "Is it spam?"}}),
            json!({"unsafe": {"type": "noul", "instructions": "Is it unsafe?"}}),
            json!({"safety": {"type": "choice", "instructions": "level", "criteria": ["safe", "unsafe"]}}),
            json!({"safety": {"type": "choice", "instructions": "level",
                              "criteria": ["unsafe", "safe", "controversial"]}}),
        ] {
            let q = parse_questions(&q).unwrap();
            let err = l.encode(&Chars, &json!("x"), &q).unwrap_err().to_string();
            assert!(
                err.contains("built-in questions (safety, unsafe, unsafe_strict, category)"),
                "{err}"
            );
        }
        // The same questions spelled differently.
        let same = parse_questions(&json!({
            "unsafe": {"type": "noul", "instructions": "unsafe", "criteria": {}},
            "safety": {"type": "choice", "instructions": "level", "criteria": ["safe", "controversial", "unsafe"]},
        }))
        .unwrap();
        assert_eq!(l.plan(&same).unwrap(), [Readout::Unsafe, Readout::Safety]);
    }

    #[test]
    fn reads_option_logits() {
        let l = layout(100);
        let cand = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(l.option_logits(Readout::Safety, &cand), [1.0, 2.0, 3.0]);
        assert_eq!(l.option_logits(Readout::Category, &cand), [4.0, 5.0]);
        let u = l.option_logits(Readout::Unsafe, &cand);
        assert!((f64::from(u[0]) - (1f64.exp() + 2f64.exp()).ln()).abs() < 1e-6 && u[1] == 3.0);
        let s = l.option_logits(Readout::UnsafeStrict, &cand);
        assert!(s[0] == 1.0 && (f64::from(s[1]) - (2f64.exp() + 3f64.exp()).ln()).abs() < 1e-6);
    }
}
