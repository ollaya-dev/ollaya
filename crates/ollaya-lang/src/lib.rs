//! Language and script detection that routes requests between Laya checkpoints.
//!
//! A port of `laya.lang` and of the detection-based part of `laya.router.Router.route` (laya
//! 0.3.7). The English checkpoint collapses on scripts it has no tokens for and degrades on other
//! Latin-script languages, so routing answers one question: is this English Latin text?
//!
//! 1. [`state_text`] flattens a state to its string leaves, joined and cut at 4000 characters.
//! 2. [`detect_script`] attributes every letter to a script. Anything but Latin goes to the
//!    multilingual checkpoint.
//! 3. For Latin text, [`latin_profile`] guesses the language from function words and measures the
//!    share of letters English does not use. The text counts as English when it is identified as
//!    English, or when nothing identifies it and it has too few such letters.
//!
//! Results match laya exactly: characters are classified as CPython's `str.isalpha` and `re` see
//! them, rates are rounded as Python's `round` does, and the reasons are laya's word for word,
//! since they reach API users in the `routing` field.
//!
//! ```
//! use ollaya_lang::{Target, route};
//! use serde_json::json;
//!
//! let r = route(&json!({"message": "Bonjour, je veux annuler mon abonnement et être remboursé"}));
//! assert_eq!(r.target, Target::Multilingual);
//! assert_eq!(r.reason, "Latin script but language looks like 'fr', not English");
//! ```

mod analysis;
mod latin;
mod pystr;
mod route;
mod script;
mod tables;
mod text;

pub use analysis::{Analysis, analyse, is_english};
pub use latin::{LatinProfile, NON_EN_DIACRITIC_RATE, guess_latin_language, latin_profile};
pub use route::{DEFAULT_MODEL, Route, Target, route, route_by_lang, route_with_default};
pub use script::{Script, detect_script, script_profile};
pub use text::{MAX_STATE_CHARS, state_text};
