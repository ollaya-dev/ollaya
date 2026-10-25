//! `gliclass-kv-v1`: GLiClass v4's decoder-KV rows (upstream `DecoderKVZeroShotClassificationPipeline`,
//! `ollaya_convert.families.gliclass_kv.layout`).
//!
//! One decoder row per question: the context (prompt, then the state, no separator), then the label
//! section between two `<<SEP>>` markers, each label followed by its `<<LABEL>>`:
//!
//! ```text
//! <prompt><state> <<SEP>> label0 <<LABEL>> label1 <<LABEL>> ... <<SEP>>
//! ```
//!
//! The context is cut from the right to `max_len` minus the label section (at least one token). The
//! graph reads label j at its `<<LABEL>>` and the text at the final `<<SEP>>`, all within the label
//! section, and returns calibrated label logits. GLiClass has no typed questions: the mapping onto its
//! `labels` / `prompt` call is the uni-encoder family's [`Call`].

use serde::Deserialize;
use serde_json::Value;

use crate::Error;
use crate::gliclass::{Call, Mode};
use crate::layout::{TokenEncoder, serialize_state};
use crate::question::Question;

pub const LAYOUT: &str = "gliclass-kv-v1";

#[derive(Debug, Clone, Deserialize)]
pub struct GliclassKvTokens {
    pub pad: u32,
    /// `<<LABEL>>`: closes each label.
    pub label: u32,
    pub label_text: String,
    /// `<<SEP>>`: opens and closes the label section.
    pub sep: u32,
    pub sep_text: String,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct GliclassKvLayout {
    pub max_len: usize,
    pub special_tokens: GliclassKvTokens,
    /// Marker texts replaced by `" "` in every user string, in this order.
    pub sanitize: Vec<String>,
}

/// One question's decoder row.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub ids: Vec<u32>,
    /// Position of each label's `<<LABEL>>`, in label order.
    pub markers: Vec<usize>,
    /// First token of the label section (after the `<<SEP>>` that follows the context).
    pub section_start: usize,
    /// The final `<<SEP>>`, the row's last token.
    pub sep: usize,
    pub mode: Mode,
    /// The context was cut to fit the window.
    pub state_truncated: bool,
}

impl GliclassKvLayout {
    pub fn validate(&self) -> Result<(), Error> {
        if self.max_len < 8
            || self.special_tokens.label_text.is_empty()
            || self.special_tokens.sep_text.is_empty()
        {
            return Err(Error::invalid(
                "decision.json: max_len (>= 8), special_tokens.label_text and sep_text are required",
            ));
        }
        Ok(())
    }

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

    /// The label section's text: `<<SEP>>` label0 `<<LABEL>>` ... `<<SEP>>`.
    pub fn label_text(&self, call: &Call) -> String {
        let t = &self.special_tokens;
        let mut s = t.sep_text.clone();
        for label in &call.labels {
            s.push_str(&self.sanitize(label));
            s.push_str(&t.label_text);
        }
        s.push_str(&t.sep_text);
        s
    }

    pub fn encode(
        &self,
        enc: &dyn TokenEncoder,
        state_text: &str,
        qid: &str,
        q: &Question,
    ) -> Result<Row, Error> {
        let call = Call::new(q);
        let labels = enc.encode(&self.label_text(&call))?;
        // The window must hold the whole label section and at least one context token; a longer
        // section would exceed the graph's section axis as well.
        if labels.len() >= self.max_len {
            return Err(Error::TooManyOptions {
                question: String::new(),
                options: q.num_options(),
                head_max_len: self.max_len,
            });
        }
        let sep = Some(&self.special_tokens.sep);
        if labels.first() != sep || labels.last() != sep {
            return Err(Error::invalid(format!(
                "question {qid:?}: the label section does not start and end with the {} marker",
                self.special_tokens.sep_text
            )));
        }
        let context = enc.encode(&format!("{}{state_text}", self.sanitize(&call.prompt)))?;
        let budget = self.max_len - labels.len();
        let kept = context.len().min(budget);
        let mut ids = Vec::with_capacity(kept + labels.len());
        ids.extend_from_slice(&context[..kept]);
        let section_start = ids.len() + 1;
        ids.extend_from_slice(&labels);
        let markers: Vec<usize> = (section_start..ids.len())
            .filter(|&i| ids[i] == self.special_tokens.label)
            .collect();
        if markers.len() != call.labels.len() {
            // The reference reads a spelled-out marker as a label boundary; sanitising keeps user text
            // out of here, so this is a tokenizer surprise, not a request error.
            return Err(Error::invalid(format!(
                "question {qid:?}: the labels tokenize to {} {} markers for {} labels",
                markers.len(),
                self.special_tokens.label_text,
                call.labels.len()
            )));
        }
        Ok(Row {
            sep: ids.len() - 1,
            ids,
            markers,
            section_start,
            mode: call.mode,
            state_truncated: kept < context.len(),
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
            let mut out = Vec::new();
            let mut rest = text;
            while !rest.is_empty() {
                if let Some(r) = rest.strip_prefix("<<LABEL>>") {
                    out.push(LABEL);
                    rest = r;
                } else if let Some(r) = rest.strip_prefix("<<SEP>>") {
                    out.push(SEP);
                    rest = r;
                } else {
                    let c = rest.chars().next().unwrap();
                    out.push(c as u32);
                    rest = &rest[c.len_utf8()..];
                }
            }
            Ok(out)
        }
    }

    fn layout(max_len: usize) -> GliclassKvLayout {
        GliclassKvLayout {
            max_len,
            special_tokens: GliclassKvTokens {
                pad: 0,
                label: LABEL,
                label_text: "<<LABEL>>".into(),
                sep: SEP,
                sep_text: "<<SEP>>".into(),
            },
            sanitize: vec!["<<LABEL>>".into(), "<<SEP>>".into(), "<<EXAMPLE>>".into()],
        }
    }

    fn q(def: Value) -> Question {
        Question::parse("q", &def).unwrap()
    }

    #[test]
    fn builds_rows_like_prepare_inputs() {
        let l = layout(4096);
        let state = l.state_text(&json!("a <<SEP>> state"));
        assert_eq!(state, "a   state");
        let row = l
            .encode(
                &Chars,
                &state,
                "q",
                &q(json!({"type": "choice", "instructions": "Which?", "criteria": {"x": "one", "y": null}})),
            )
            .unwrap();
        let text: String = row
            .ids
            .iter()
            .map(|&i| match i {
                LABEL => "<L>".to_owned(),
                SEP => "<S>".to_owned(),
                i => char::from_u32(i).unwrap().to_string(),
            })
            .collect();
        assert_eq!(text, "Which?a   state<S>x: one<L>y<L><S>");
        assert_eq!(row.section_start, "Which?a   state".len() + 1);
        assert_eq!(row.sep, row.ids.len() - 1);
        assert_eq!(row.markers.len(), 2);
        assert!(
            row.markers
                .iter()
                .all(|&m| row.ids[m] == LABEL && m >= row.section_start)
        );
        assert_eq!(row.mode, Mode::Softmax);
        assert!(!row.state_truncated);
        // a one-label noul: no prompt, the instructions as the label
        let noul = l
            .encode(
                &Chars,
                &state,
                "n",
                &q(json!({"type": "noul", "instructions": "ok?"})),
            )
            .unwrap();
        assert_eq!(noul.mode, Mode::Sigmoid);
        assert_eq!(&noul.ids[..state.len()], &Chars.encode(&state).unwrap()[..]);
    }

    #[test]
    fn cuts_the_context_to_the_window() {
        let l = layout(20);
        let state = "s".repeat(50);
        let row = l
            .encode(
                &Chars,
                &state,
                "q",
                &q(json!({"type": "noul", "instructions": "ok?"})),
            )
            .unwrap();
        // labels = <S> o k ? <L> <S> = 6 tokens; the context keeps 14
        assert_eq!(row.ids.len(), 20);
        assert_eq!(row.section_start, 15);
        assert!(row.state_truncated);
    }

    #[test]
    fn a_label_section_that_fills_the_window_is_too_many_options() {
        // labels = <S> o k ? <L> <S> = 6 tokens: a 7-token window keeps one context token, a
        // 6-token window has no room for the context
        let question = q(json!({"type": "noul", "instructions": "ok?"}));
        let row = layout(7).encode(&Chars, "state", "q", &question).unwrap();
        assert_eq!(row.ids.len(), 7);
        assert_eq!(row.section_start, 2);
        assert!(matches!(
            layout(6).encode(&Chars, "state", "q", &question),
            Err(Error::TooManyOptions {
                options: 2,
                head_max_len: 6,
                ..
            })
        ));
    }
}
