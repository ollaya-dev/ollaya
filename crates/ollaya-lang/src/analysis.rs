//! The full detection result for a state (`laya.lang.analyse`).
//!
//! Text without letters is treated as English, and any dominant script other than Latin as not
//! English. Latin text is English when it is identified as English, or when nothing identifies it
//! and it does not look non-English: undecided is not English, but short English stays English.

use serde::{Serialize, Serializer};
use serde_json::Value;

use crate::latin::latin_profile;
use crate::pystr;
use crate::script::{Script, ScriptCounts};
use crate::text::state_text;

/// What detection found in a state. Serializes as laya's `detection` object, key for key.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Analysis {
    /// Dominant script of the state's letters.
    pub script: Script,
    /// Share of letters per script: Latin first, then in order of appearance. Empty without
    /// letters.
    #[serde(serialize_with = "profile_map")]
    pub script_profile: Vec<(Script, f64)>,
    /// Best-effort language of Latin text; `None` when undecided or not Latin.
    pub language: Option<&'static str>,
    /// Whether the English checkpoint can be expected to read the state.
    pub is_english: bool,
    /// No language was identified (always so outside Latin script).
    pub language_undecided: bool,
    /// Share of letters English does not use in Latin text, rounded to 4 places; 0 otherwise.
    pub diacritic_rate: f64,
    /// Share of letters outside Latin script, rounded to 4 places.
    pub non_latin_fraction: f64,
}

fn profile_map<S: Serializer>(profile: &[(Script, f64)], s: S) -> Result<S::Ok, S::Error> {
    s.collect_map(
        profile
            .iter()
            .map(|(script, share)| (script.as_str(), share)),
    )
}

/// Detect the script and, for Latin text, the language of `state` (a string, object or array).
pub fn analyse(state: &Value) -> Analysis {
    let text = state_text(state);
    let counts = ScriptCounts::of(&text);
    let script = counts.dominant();
    let non_latin_fraction = pystr::round(counts.non_latin_share(), 4);
    let undecided = Analysis {
        script,
        script_profile: counts.profile(),
        language: None,
        is_english: false,
        language_undecided: true,
        diacritic_rate: 0.0,
        non_latin_fraction,
    };
    match script {
        Script::Unknown => Analysis {
            is_english: true,
            ..undecided
        },
        Script::Latin => {
            let latin = latin_profile(&text);
            let language_undecided = latin.language.is_none();
            Analysis {
                language: latin.language,
                is_english: latin.language == Some("en")
                    || (language_undecided && !latin.looks_non_english),
                language_undecided,
                diacritic_rate: pystr::round(latin.diacritic_rate, 4),
                ..undecided
            }
        }
        _ => undecided,
    }
}

/// Whether the English checkpoint can be expected to read `state`.
pub fn is_english(state: &Value) -> bool {
    analyse(state).is_english
}
