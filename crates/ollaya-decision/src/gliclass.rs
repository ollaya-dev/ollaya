//! `gliclass-uni-v1`: GLiClass uni-encoder rows (upstream `UniEncoderZeroShotClassificationPipeline`).
//!
//! One row per question. Labels, prompt and state share the sequence, labels first:
//!
//! ```text
//! [CLS] <<LABEL>>label0<<LABEL>>label1 ... <<SEP>><prompt><state> [SEP]
//! ```
//!
//! The row is cut from the right to `max_len`, so the state goes first. The graph pools each label
//! from its `<<LABEL>>` span. GLiClass has no typed questions: [`Call`] is Ollaya's mapping onto
//! its `labels` / `prompt` call (`ollaya_convert.families.gliclass.ref`).

use serde::Deserialize;
use serde_json::Value;

use crate::Error;
use crate::layout::{TokenEncoder, serialize_state};
use crate::question::{Criteria, Question, render_criterion};

/// Special token ids and marker texts, as the model's tokenizer defines them.
#[derive(Debug, Clone, Deserialize)]
pub struct GliclassTokens {
    pub cls: u32,
    pub sep: u32,
    pub pad: u32,
    /// `<<LABEL>>`: opens each label's span.
    pub label: u32,
    pub label_text: String,
    /// `<<SEP>>`: ends the labels.
    pub text_sep_text: String,
}

#[derive(Debug, Clone)]
pub struct GliclassLayout {
    pub max_len: usize,
    pub special: GliclassTokens,
    /// Marker texts replaced by `" "` in every user string, in this order.
    pub sanitize: Vec<String>,
}

/// How a question's label logits become option logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// choice / score: one label per option, softmax (upstream `single-label`).
    Softmax,
    /// noul with both descriptions: labels `[false, true]`, softmax.
    Pair,
    /// Any other noul: one label (the instructions), sigmoid (upstream `multi-label`).
    Sigmoid,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Softmax => "softmax",
            Mode::Pair => "pair",
            Mode::Sigmoid => "sigmoid",
        }
    }

    /// Option logits whose softmax is upstream's score: the label logits, or `[0, z]` for a
    /// sigmoid label, since softmax(`[0, z]`) = `[1 − σ(z), σ(z)]` (noul order is `[false, true]`).
    pub fn option_logits(self, label_logits: &[f32]) -> Vec<f32> {
        match self {
            Mode::Sigmoid => vec![0.0, label_logits[0]],
            Mode::Softmax | Mode::Pair => label_logits.to_vec(),
        }
    }
}

/// The upstream pipeline call a question maps to (`ref.question_call`), before sanitising.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub labels: Vec<String>,
    pub prompt: String,
    pub mode: Mode,
}

impl Call {
    pub fn new(q: &Question) -> Self {
        if let Criteria::Noul { r#false, r#true } = &q.criteria {
            return match (described(r#false), described(r#true)) {
                (Some(f), Some(t)) => Call {
                    labels: vec![f, t],
                    prompt: q.instructions.clone(),
                    mode: Mode::Pair,
                },
                _ => Call {
                    labels: vec![q.instructions.clone()],
                    prompt: String::new(),
                    mode: Mode::Sigmoid,
                },
            };
        }
        Call {
            labels: q.render_options(),
            prompt: q.instructions.clone(),
            mode: Mode::Softmax,
        }
    }
}

/// A noul description, when given (`not in (None, "")`).
fn described(value: &Option<Value>) -> Option<String> {
    match value {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(v) => Some(render_criterion(v)),
    }
}

/// One question's encoder row.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub ids: Vec<u32>,
    /// Position of each label's `<<LABEL>>`, in label order.
    pub markers: Vec<usize>,
    pub mode: Mode,
    /// The row was cut to `max_len`. The state is last, so it is cut first.
    pub state_truncated: bool,
}

impl GliclassLayout {
    /// Replace every marker text with `" "` (`ref.sanitize`).
    pub fn sanitize(&self, text: &str) -> String {
        self.sanitize
            .iter()
            .fold(text.to_owned(), |t, m| t.replace(m.as_str(), " "))
    }

    /// The state as the model reads it, shared by every question.
    pub fn state_text(&self, state: &Value) -> String {
        self.sanitize(&serialize_state(state))
    }

    /// Upstream's input text (`prepare_input`, `prompt_first`): labels, `<<SEP>>`, prompt, state.
    pub fn text(&self, state_text: &str, call: &Call) -> String {
        let mut text = String::new();
        for label in &call.labels {
            text.push_str(&self.special.label_text);
            text.push_str(&self.sanitize(label));
        }
        text.push_str(&self.special.text_sep_text);
        text.push_str(&self.sanitize(&call.prompt));
        text.push_str(state_text);
        text
    }

    /// `[CLS] text [SEP]`, cut from the right to `max_len`, with the final `[SEP]` kept.
    pub fn encode(
        &self,
        enc: &dyn TokenEncoder,
        state_text: &str,
        q: &Question,
    ) -> Result<Row, Error> {
        let call = Call::new(q);
        let body = enc.encode(&self.text(state_text, &call))?;
        let room = self.max_len.saturating_sub(2);
        let kept = body.len().min(room);

        let mut ids = Vec::with_capacity(kept + 2);
        ids.push(self.special.cls);
        ids.extend_from_slice(&body[..kept]);
        ids.push(self.special.sep);
        let markers: Vec<usize> = (0..ids.len())
            .filter(|&i| ids[i] == self.special.label)
            .collect();

        if markers.len() < call.labels.len() {
            return Err(Error::TooManyOptions {
                question: String::new(),
                options: q.num_options(),
                head_max_len: self.max_len,
            });
        }
        if markers.len() > call.labels.len() {
            // The reference rejects this too: a spelled-out marker would be read as a label.
            return Err(Error::invalid(format!(
                "the instructions and state together form the model's {} marker; rephrase them",
                self.special.label_text
            )));
        }
        Ok(Row {
            ids,
            markers,
            mode: call.mode,
            state_truncated: kept < body.len() && !state_text.is_empty(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const LABEL: u32 = 900;
    const SEP: u32 = 901;

    /// One id per char, with the two markers as single tokens.
    struct Chars;

    impl TokenEncoder for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            let mut ids = Vec::new();
            let mut rest = text;
            while let Some(c) = rest.chars().next() {
                if let Some(r) = rest.strip_prefix("<<LABEL>>") {
                    ids.push(LABEL);
                    rest = r;
                } else if let Some(r) = rest.strip_prefix("<<SEP>>") {
                    ids.push(SEP);
                    rest = r;
                } else {
                    ids.push(c as u32);
                    rest = &rest[c.len_utf8()..];
                }
            }
            Ok(ids)
        }
    }

    fn layout(max_len: usize) -> GliclassLayout {
        GliclassLayout {
            max_len,
            special: GliclassTokens {
                cls: 1,
                sep: 2,
                pad: 0,
                label: LABEL,
                label_text: "<<LABEL>>".into(),
                text_sep_text: "<<SEP>>".into(),
            },
            sanitize: vec!["<<LABEL>>".into(), "<<SEP>>".into(), "<<EXAMPLE>>".into()],
        }
    }

    fn q(def: Value) -> Question {
        Question::parse("q", &def).unwrap()
    }

    #[test]
    fn maps_questions_like_the_reference() {
        let choice = q(json!({"type": "choice", "instructions": "Team?",
                              "criteria": {"billing": "payments", "other": ""}}));
        assert_eq!(
            Call::new(&choice),
            Call {
                labels: vec!["billing: payments".into(), "other".into()],
                prompt: "Team?".into(),
                mode: Mode::Softmax
            }
        );
        let score =
            q(json!({"type": "score", "instructions": "How?", "criteria": ["low", "high"]}));
        assert_eq!(Call::new(&score).labels, ["level 0: low", "level 1: high"]);
        let pair = q(json!({"type": "noul", "instructions": "Angry?",
                            "criteria": {"True": "shouting", "false": false}}));
        assert_eq!(
            Call::new(&pair),
            Call {
                labels: vec!["false".into(), "shouting".into()],
                prompt: "Angry?".into(),
                mode: Mode::Pair
            }
        );
        for crit in [
            json!(null),
            json!({"true": "yes"}),
            json!({"true": "yes", "false": ""}),
        ] {
            let single =
                q(json!({"type": "noul", "instructions": {"is": "refund"}, "criteria": crit}));
            assert_eq!(
                Call::new(&single),
                Call {
                    labels: vec!["{\"is\": \"refund\"}".into()],
                    prompt: String::new(),
                    mode: Mode::Sigmoid
                }
            );
        }
        assert_eq!(Mode::Sigmoid.option_logits(&[1.5]), [0.0, 1.5]);
        assert_eq!(Mode::Pair.option_logits(&[1.0, 2.0]), [1.0, 2.0]);
    }

    #[test]
    fn builds_and_cuts_rows() {
        let l = layout(64);
        let state = l.state_text(&json!("s<<SEP>>t"));
        assert_eq!(state, "s t");
        let choice = q(
            json!({"type": "choice", "instructions": "p<<EXAMPLE>>", "criteria": ["a<<LABEL>>", "b"]}),
        );
        assert_eq!(
            l.text(&state, &Call::new(&choice)),
            "<<LABEL>>a <<LABEL>>b<<SEP>>p s t"
        );
        let row = l.encode(&Chars, &state, &choice).unwrap();
        assert_eq!(row.markers, [1, 4]);
        assert_eq!((row.ids[0], *row.ids.last().unwrap()), (1, 2));
        assert!(!row.state_truncated);

        // The state is cut first; the final [SEP] stays.
        let long = "x".repeat(100);
        let row = layout(16).encode(&Chars, &long, &choice).unwrap();
        assert_eq!(row.ids.len(), 16);
        assert_eq!(row.ids[15], 2);
        assert!(row.state_truncated);
    }

    #[test]
    fn rejects_labels_that_do_not_fit() {
        let many =
            q(json!({"type": "choice", "instructions": "p", "criteria": ["aaaa", "bbbb", "cccc"]}));
        match layout(10).encode(&Chars, "", &many) {
            Err(Error::TooManyOptions {
                options: 3,
                head_max_len: 10,
                ..
            }) => {}
            other => panic!("{other:?}"),
        }
        let spelled = q(json!({"type": "choice", "instructions": "p <<LAB", "criteria": ["a"]}));
        assert!(matches!(
            layout(64).encode(&Chars, "EL>> s", &spelled),
            Err(Error::Invalid(_))
        ));
    }
}
