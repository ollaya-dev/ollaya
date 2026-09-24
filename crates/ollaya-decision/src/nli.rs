//! `nli-pairs-v1`: zero-shot NLI cross-encoders, scored as `transformers`'
//! `pipeline("zero-shot-classification")` scores them (`ollaya_convert.families.nli.ref`).
//!
//! Each option becomes a hypothesis through the model's templates, and each (state, hypothesis)
//! pair is one encoder row:
//!
//! ```text
//! [CLS] <state> [SEP] <hypothesis> [SEP]
//! ```
//!
//! The state is cut from the right to fit `max_len` (`only_first`). Option logits are the rows'
//! entailment logits; a noul without both descriptions is one row, read as
//! `[not_entailment, entailment]`.

use serde::Deserialize;
use serde_json::Value;

use crate::Error;
use crate::layout::TokenEncoder;
use crate::question::{Criteria, Question, render_criterion};

/// `[CLS]`, `[SEP]` and `[PAD]` ids.
#[derive(Debug, Clone, Deserialize)]
pub struct PairTokens {
    pub cls: u32,
    pub sep: u32,
    pub pad: u32,
}

/// Special tokens around one pair: `[CLS]`, and `[SEP]` after each side.
const PAIR_SPECIALS: usize = 3;

/// Columns of the graph's `scores`.
#[derive(Debug, Clone, Deserialize)]
pub struct Classes {
    pub entailment: usize,
    pub not_entailment: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChoiceTemplates {
    pub with_description: String,
    pub without_description: String,
}

/// Hypothesis templates, one set per model.
#[derive(Debug, Clone, Deserialize)]
pub struct Templates {
    pub choice: ChoiceTemplates,
    pub score: String,
    pub noul_pair: String,
    pub noul_single: String,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct NliLayout {
    pub max_len: usize,
    pub special_tokens: PairTokens,
    pub classes: Classes,
    pub templates: Templates,
}

/// How a question's rows become option logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// One row per option (noul: `[false, true]`); each option's logit is its entailment logit.
    Entail,
    /// One row whose `[not_entailment, entailment]` are the noul's `[false, true]` logits.
    Single,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Entail => "entail",
            Mode::Single => "single",
        }
    }
}

/// One question's encoder rows, in option order.
#[derive(Debug, Clone, PartialEq)]
pub struct Pairs {
    pub rows: Vec<Vec<u32>>,
    pub mode: Mode,
    /// The state was cut to fit at least one row.
    pub state_truncated: bool,
}

const CHOICE_FIELDS: &[&str] = &["instructions", "label", "description", "option"];
const SCORE_FIELDS: &[&str] = &["instructions", "level", "description"];
const NOUL_PAIR_FIELDS: &[&str] = &["instructions", "description"];
const NOUL_SINGLE_FIELDS: &[&str] = &["instructions"];

impl NliLayout {
    /// Reject templates that use a field their question type does not provide.
    pub fn validate(&self) -> Result<(), Error> {
        let t = &self.templates;
        for (template, names) in [
            (&t.choice.with_description, CHOICE_FIELDS),
            (&t.choice.without_description, CHOICE_FIELDS),
            (&t.score, SCORE_FIELDS),
            (&t.noul_pair, NOUL_PAIR_FIELDS),
            (&t.noul_single, NOUL_SINGLE_FIELDS),
        ] {
            let fields: Vec<(&str, &str)> = names.iter().map(|&n| (n, "")).collect();
            fill(template, &fields)?;
        }
        if self.classes.entailment == self.classes.not_entailment {
            return Err(Error::invalid(
                "classes: entailment and not_entailment share a column",
            ));
        }
        Ok(())
    }

    /// Hypotheses for one question, in option order, and how their rows are read.
    pub fn hypotheses(&self, q: &Question) -> Result<(Vec<String>, Mode), Error> {
        let tpl = &self.templates;
        let ins = q.instructions.as_str();
        match &q.criteria {
            Criteria::Choice(options) => {
                let hyps = options
                    .iter()
                    .map(|(label, v)| {
                        let desc = if v.is_null() {
                            String::new()
                        } else {
                            render_criterion(v)
                        };
                        let (template, option) = if desc.is_empty() {
                            (&tpl.choice.without_description, label.clone())
                        } else {
                            (&tpl.choice.with_description, format!("{label}: {desc}"))
                        };
                        fill(
                            template,
                            &[
                                ("instructions", ins),
                                ("label", label),
                                ("description", &desc),
                                ("option", &option),
                            ],
                        )
                    })
                    .collect::<Result<_, _>>()?;
                Ok((hyps, Mode::Entail))
            }
            Criteria::Score(levels) => {
                let hyps = levels
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        fill(
                            &tpl.score,
                            &[
                                ("instructions", ins),
                                ("level", &i.to_string()),
                                ("description", &render_criterion(c)),
                            ],
                        )
                    })
                    .collect::<Result<_, _>>()?;
                Ok((hyps, Mode::Entail))
            }
            Criteria::Noul { r#false, r#true } => match (described(r#false), described(r#true)) {
                (Some(f), Some(t)) => {
                    let hyps = [f, t]
                        .iter()
                        .map(|d| fill(&tpl.noul_pair, &[("instructions", ins), ("description", d)]))
                        .collect::<Result<_, _>>()?;
                    Ok((hyps, Mode::Entail))
                }
                _ => Ok((
                    vec![fill(&tpl.noul_single, &[("instructions", ins)])?],
                    Mode::Single,
                )),
            },
        }
    }

    /// Encode one question's rows against the tokenized state (`premise`), as
    /// `tok(premise, hypothesis, truncation="only_first", max_length=max_len)` does.
    pub fn encode(
        &self,
        enc: &dyn TokenEncoder,
        premise: &[u32],
        qid: &str,
        q: &Question,
    ) -> Result<Pairs, Error> {
        let (hypotheses, mode) = self.hypotheses(q)?;
        let mut rows = Vec::with_capacity(hypotheses.len());
        let mut state_truncated = false;
        for hypothesis in &hypotheses {
            let h = enc.encode(hypothesis)?;
            let keep = match self.max_len.checked_sub(h.len() + PAIR_SPECIALS) {
                Some(room) if premise.len() <= room => premise.len(),
                // tokenizers never cuts the premise away entirely (`SequenceTooShort`).
                Some(room) if room > 0 => {
                    state_truncated = true;
                    room
                }
                _ => {
                    return Err(Error::invalid(format!(
                        "question {qid:?}: its hypothesis is {} tokens and leaves no room for the state \
                         (the model reads {} tokens per state and hypothesis); shorten the instructions or options",
                        h.len(),
                        self.max_len
                    )));
                }
            };
            let mut ids = Vec::with_capacity(keep + h.len() + PAIR_SPECIALS);
            ids.push(self.special_tokens.cls);
            ids.extend_from_slice(&premise[..keep]);
            ids.push(self.special_tokens.sep);
            ids.extend_from_slice(&h);
            ids.push(self.special_tokens.sep);
            rows.push(ids);
        }
        Ok(Pairs {
            rows,
            mode,
            state_truncated,
        })
    }

    /// Option logits from a question's rows of `scores` (class logits per row), in option order.
    pub fn option_logits<'a>(
        &self,
        mode: Mode,
        rows: impl IntoIterator<Item = &'a [f32]>,
    ) -> Vec<f32> {
        let (e, n) = (self.classes.entailment, self.classes.not_entailment);
        let mut rows = rows.into_iter();
        match mode {
            Mode::Entail => rows.map(|r| r[e]).collect(),
            Mode::Single => rows.next().map_or_else(Vec::new, |r| vec![r[n], r[e]]),
        }
    }
}

/// A description as the reference tests it: `None` if missing, `null` or `""`.
fn described(value: &Option<Value>) -> Option<String> {
    match value {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(v) => Some(render_criterion(v)),
    }
}

/// Substitute `{field}` placeholders in one left-to-right pass, as Python's
/// `re.sub(r"\{(\w+)\}", ...)` does: braces inside a substituted value are never expanded.
fn fill(template: &str, fields: &[(&str, &str)]) -> Result<String, Error> {
    let mut out = String::with_capacity(template.len() + 64);
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let end = after
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(after.len());
        if end > 0 && after[end..].starts_with('}') {
            let name = &after[..end];
            let value = fields
                .iter()
                .find_map(|&(k, v)| (k == name).then_some(v))
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "template {template:?} uses unknown field {{{name}}}"
                    ))
                })?;
            out.push_str(value);
            rest = &after[end + 1..];
        } else {
            out.push('{');
            rest = after;
        }
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One token per whitespace-separated word: id = word length.
    struct Words;

    impl TokenEncoder for Words {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            Ok(text.split_whitespace().map(|w| w.len() as u32).collect())
        }
    }

    fn layout(max_len: usize, deberta: bool) -> NliLayout {
        let templates = if deberta {
            json!({"choice": {"with_description": "The answer to \"{instructions}\" is: {option}",
                              "without_description": "The answer to \"{instructions}\" is: {option}"},
                   "score": "The answer to \"{instructions}\" is: {description}",
                   "noul_pair": "{description}", "noul_single": "{instructions}"})
        } else {
            json!({"choice": {"with_description": "{description}",
                              "without_description": "This text is about {label}."},
                   "score": "{description}", "noul_pair": "{description}", "noul_single": "{instructions}"})
        };
        serde_json::from_value(json!({
            "max_len": max_len,
            "special_tokens": {"cls": 101, "sep": 102, "pad": 0},
            "classes": {"entailment": 0, "not_entailment": 1},
            "templates": templates,
        }))
        .unwrap()
    }

    fn question(def: Value) -> Question {
        Question::parse("q", &def).unwrap()
    }

    #[test]
    fn fills_in_one_pass() {
        let f = |t: &str| fill(t, &[("instructions", "{option}"), ("option", "x")]).unwrap();
        assert_eq!(f("{instructions} / {option}"), "{option} / x");
        assert_eq!(
            f("{{option}} {} { option} {option"),
            "{x} {} { option} {option"
        );
        assert!(fill("{label}", &[("instructions", "")]).is_err());
    }

    #[test]
    fn hypotheses_follow_the_templates() {
        let choice = question(json!({"type": "choice", "instructions": "Which team?",
            "criteria": {"billing": "payments", "other": "", "tier": {"sla_h": 24}}}));
        let (h, mode) = layout(512, true).hypotheses(&choice).unwrap();
        assert_eq!(mode, Mode::Entail);
        assert_eq!(
            h,
            [
                "The answer to \"Which team?\" is: billing: payments",
                "The answer to \"Which team?\" is: other",
                "The answer to \"Which team?\" is: tier: {\"sla_h\": 24}",
            ]
        );
        let (h, _) = layout(512, false).hypotheses(&choice).unwrap();
        assert_eq!(
            h,
            ["payments", "This text is about other.", "{\"sla_h\": 24}"]
        );

        let score =
            question(json!({"type": "score", "instructions": "How bad?", "criteria": ["fine", 3]}));
        let (h, _) = layout(512, true).hypotheses(&score).unwrap();
        assert_eq!(
            h,
            [
                "The answer to \"How bad?\" is: fine",
                "The answer to \"How bad?\" is: 3"
            ]
        );

        let pair = question(json!({"type": "noul", "instructions": "Angry?",
            "criteria": {"TRUE": "The customer is angry.", "false": "The customer is calm."}}));
        assert_eq!(
            layout(512, true).hypotheses(&pair).unwrap(),
            (
                vec![
                    "The customer is calm.".into(),
                    "The customer is angry.".into()
                ],
                Mode::Entail
            )
        );
        let single = question(json!({"type": "noul", "instructions": "It is spam.",
            "criteria": {"true": "yes", "false": ""}}));
        assert_eq!(
            layout(512, false).hypotheses(&single).unwrap(),
            (vec!["It is spam.".into()], Mode::Single)
        );
    }

    #[test]
    fn truncates_only_the_state() {
        // Hypothesis "h h h" is 3 tokens, so rows are [CLS] state [SEP] h h h [SEP].
        let l = layout(10, false);
        let q = question(json!({"type": "noul", "instructions": "h h h"}));
        let pairs = l.encode(&Words, &[7, 8], "q", &q).unwrap();
        assert_eq!(pairs.rows, [vec![101, 7, 8, 102, 1, 1, 1, 102]]);
        assert!(!pairs.state_truncated);
        let pairs = l.encode(&Words, &[1, 2, 3, 4, 5, 6], "q", &q).unwrap();
        assert_eq!(pairs.rows, [vec![101, 1, 2, 3, 4, 102, 1, 1, 1, 102]]);
        assert!(pairs.state_truncated);
    }

    #[test]
    fn rejects_like_tokenizers_only_first() {
        // max_len 10: a 7-token hypothesis fills it exactly, which only an empty state fits.
        let l = layout(10, false);
        let q =
            |n: usize| question(json!({"type": "noul", "instructions": vec!["h"; n].join(" ")}));
        assert_eq!(
            l.encode(&Words, &[5; 9], "q", &q(6)).unwrap().rows[0].len(),
            10
        );
        assert_eq!(l.encode(&Words, &[], "q", &q(7)).unwrap().rows[0].len(), 10);
        assert!(l.encode(&Words, &[5], "q", &q(7)).is_err());
        assert!(l.encode(&Words, &[], "q", &q(8)).is_err());
    }

    #[test]
    fn reads_option_logits() {
        let l = layout(512, true);
        let rows = [[1.0, -1.0], [0.5, 2.0]];
        let it = || rows.iter().map(|r| r.as_slice());
        assert_eq!(l.option_logits(Mode::Entail, it()), [1.0, 0.5]);
        assert_eq!(l.option_logits(Mode::Single, it()), [-1.0, 1.0]);
    }

    #[test]
    fn validates_templates() {
        let mut l = layout(512, true);
        assert!(l.validate().is_ok());
        l.templates.noul_single = "{description}".into();
        assert!(l.validate().is_err());
    }
}
