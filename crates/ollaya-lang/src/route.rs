//! The detection-based part of laya's `Router.route`: which checkpoint reads a state, and why.
//!
//! Explicit `model`, `task` and `lang` overrides and the `lang_guess` hook take precedence and
//! are the caller's to apply; this is what decides when none of them does. The reasons are laya's
//! word for word, since API users see them in the `routing` field.

use serde_json::Value;

use crate::analysis::{Analysis, analyse};
use crate::pystr;
use crate::script::Script;

/// The checkpoint laya's `Router()` falls back to.
pub const DEFAULT_MODEL: &str = "english";

/// Subtags that mean the English checkpoint can read the text.
const ENGLISH_SUBTAGS: [&str; 3] = ["en", "eng", "english"];

/// Where a request goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// The English checkpoint (`english`).
    English,
    /// The multilingual checkpoint (`multilingual`).
    Multilingual,
    /// The router's default: nothing identified the language.
    Default,
}

impl Target {
    /// The checkpoint name, given the router's default.
    pub fn model(self, default: &str) -> &str {
        match self {
            Target::English => "english",
            Target::Multilingual => "multilingual",
            Target::Default => default,
        }
    }
}

/// A routing decision.
#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    pub target: Target,
    /// Why, in laya's words.
    pub reason: String,
    /// The detection the decision rests on (laya's `detection`).
    pub analysis: Analysis,
}

/// Route `state` as laya's stock `Router()` does, whose default is [`DEFAULT_MODEL`].
pub fn route(state: &Value) -> Route {
    route_with_default(state, DEFAULT_MODEL)
}

/// Route `state` for a router whose default checkpoint is `default` (a canonical name such as
/// `english` or `multilingual`), which the reasons for [`Target::Default`] name.
pub fn route_with_default(state: &Value, default: &str) -> Route {
    let analysis = analyse(state);
    let (target, reason) = decide(&analysis, default);
    Route {
        target,
        reason,
        analysis,
    }
}

fn decide(a: &Analysis, default: &str) -> (Target, String) {
    match a.script {
        Script::Unknown => (
            Target::Default,
            format!("no letters detected in state; using default ({default})"),
        ),
        Script::Latin if !a.is_english => match a.language {
            Some(lang) => (
                Target::Multilingual,
                format!("Latin script but language looks like '{lang}', not English"),
            ),
            // No list covers the language: the non-English letters alone decide.
            None => (
                Target::Multilingual,
                format!(
                    "Latin script, language not identified but {}% non-English letters; \
                     not safe for the English checkpoint",
                    pystr::percent(a.diacritic_rate)
                ),
            ),
        },
        // Too short, or only content words: no evidence of English either.
        Script::Latin if a.language_undecided => (
            Target::Default,
            format!(
                "Latin script, language not identified and no non-English letters; \
                 using default ({default})"
            ),
        ),
        Script::Latin => (Target::English, "English Latin text".to_owned()),
        script => (
            Target::Multilingual,
            format!(
                "non-Latin script ({script}, {}% of letters); the English checkpoint cannot read it",
                pystr::percent(a.non_latin_fraction)
            ),
        ),
    }
}

/// The checkpoint a language code routes to (laya's `_english_from_code`), or `None` when the
/// code identifies nothing.
///
/// Takes the forms a caller has to hand: `en`, `EN`, `en-US`, POSIX `en_US` and `en_US.UTF-8`.
/// The primary subtag decides, and every non-English one routes to the multilingual checkpoint.
/// laya routes an explicit `lang` that yields `None` to the multilingual checkpoint, while a
/// `lang_guess` hint that yields `None` falls through to detection.
pub fn route_by_lang(code: &str) -> Option<Target> {
    let code = code.trim_matches(pystr::is_space).to_lowercase();
    let code = code.split('.').next().unwrap_or_default();
    let primary = code.split(['_', '-']).next().unwrap_or_default();
    if primary.is_empty() {
        return None;
    }
    Some(if ENGLISH_SUBTAGS.contains(&primary) {
        Target::English
    } else {
        Target::Multilingual
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn default_is_named_in_the_reason() {
        let r = route_with_default(&json!("Quero cancelar"), "multilingual");
        assert_eq!(r.target, Target::Default);
        assert_eq!(r.target.model("multilingual"), "multilingual");
        assert_eq!(
            r.reason,
            "Latin script, language not identified and no non-English letters; \
             using default (multilingual)"
        );
    }

    #[test]
    fn non_latin_reason_carries_the_share() {
        let r = route(&json!(
            "登录页面一直报错 500，我们下午两点要演示，请尽快处理。"
        ));
        assert_eq!(r.target, Target::Multilingual);
        assert_eq!(
            r.reason,
            "non-Latin script (han, 100% of letters); the English checkpoint cannot read it"
        );
    }

    #[test]
    fn language_codes() {
        assert_eq!(route_by_lang("en_US.UTF-8"), Some(Target::English));
        assert_eq!(route_by_lang(" English "), Some(Target::English));
        assert_eq!(route_by_lang("pt-BR"), Some(Target::Multilingual));
        assert_eq!(route_by_lang("-en"), None);
        assert_eq!(route_by_lang(""), None);
    }
}
