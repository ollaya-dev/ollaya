//! Sequence layouts: how a (state, question) pair becomes encoder input.
//!
//! `laya-markers-v1` is Laya's layout (`laya.common.build_sequence`):
//!
//! ```text
//! [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
//! ```
//!
//! Each option is scored at its own `[MASK]`. Instructions, options and state share `max_len`;
//! options share a `head_max_len` budget.

use serde::Deserialize;
use serde_json::Value;

use crate::question::Question;
use crate::{Error, pyjson};

/// Tokenizes text without special tokens (`tok(text, add_special_tokens=False)`).
pub trait TokenEncoder {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error>;
}

/// Special token ids and the mask token's text, as the model's tokenizer defines them.
#[derive(Debug, Clone, Deserialize)]
pub struct SpecialTokens {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub pad: u32,
    pub mask_text: String,
}

/// Tokens per option before the option budget is applied (`[:48]` in laya).
const MAX_OPTION_TOKENS: usize = 48;
/// Minimum room left for instructions once options are placed.
const MIN_HEAD_ROOM: usize = 16;
/// Instructions are never cut below this many tokens.
const MIN_INSTRUCTION_TOKENS: usize = 8;
/// When options overflow the budget, each keeps at least this many tokens.
const MIN_TOKENS_PER_OPTION: usize = 4;

#[derive(Debug, Clone)]
pub struct LayaLayout {
    pub max_len: usize,
    pub head_max_len: usize,
    pub special: SpecialTokens,
}

/// One question's encoder input.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    pub ids: Vec<u32>,
    /// Position of each option's `[MASK]`, in option order.
    pub markers: Vec<usize>,
}

/// The state as the model reads it: strings verbatim, anything else as `json.dumps`.
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        v => pyjson::dumps(v, false),
    }
}

impl LayaLayout {
    /// Tokenize the (already serialized) state once; it is shared by every question.
    pub fn encode_state(
        &self,
        enc: &dyn TokenEncoder,
        state_text: &str,
    ) -> Result<Vec<u32>, Error> {
        enc.encode(&state_text.replace(&self.special.mask_text, " "))
    }

    pub fn encode(
        &self,
        enc: &dyn TokenEncoder,
        state_ids: &[u32],
        q: &Question,
    ) -> Result<Encoded, Error> {
        let mask_text = &self.special.mask_text;
        let options = q.render_options();

        let head_text = format!(
            "{} question: {}",
            q.qtype.name(),
            q.instructions.replace(mask_text, " ")
        );
        let mut head_ids = enc.encode(&head_text)?;

        let mut opt_ids = Vec::with_capacity(options.len());
        for opt in &options {
            let mut ids = vec![self.special.mask];
            let toks = enc.encode(&format!(" {}", opt.replace(mask_text, " ")))?;
            ids.extend(toks.into_iter().take(MAX_OPTION_TOKENS));
            opt_ids.push(ids);
        }

        // Budgets are signed: options can overflow head_max_len before truncation.
        let head_max = self.head_max_len as isize;
        let mut opt_budget = head_max - opt_ids.iter().map(|o| o.len() as isize).sum::<isize>();
        if opt_budget < MIN_HEAD_ROOM as isize {
            let per = ((self.head_max_len.saturating_sub(MIN_HEAD_ROOM)) / opt_ids.len().max(1))
                .max(MIN_TOKENS_PER_OPTION);
            for o in &mut opt_ids {
                o.truncate(per);
            }
            opt_budget = head_max - opt_ids.iter().map(|o| o.len() as isize).sum::<isize>();
        }
        head_ids.truncate(opt_budget.max(MIN_INSTRUCTION_TOKENS as isize) as usize);

        let mut ids = Vec::with_capacity(self.max_len);
        ids.push(self.special.cls);
        ids.extend_from_slice(&head_ids);
        ids.push(self.special.sep);
        let mut markers = Vec::with_capacity(opt_ids.len());
        for o in &opt_ids {
            markers.push(ids.len());
            ids.extend_from_slice(o);
        }
        ids.push(self.special.sep);
        let room = self.max_len.saturating_sub(ids.len() + 1);
        ids.extend_from_slice(&state_ids[..room.min(state_ids.len())]);
        ids.push(self.special.sep);
        ids.truncate(self.max_len);
        markers.retain(|&m| m < self.max_len);

        if markers.len() != options.len() {
            return Err(Error::TooManyOptions {
                options: options.len(),
                head_max_len: self.head_max_len,
            });
        }
        Ok(Encoded { ids, markers })
    }
}
