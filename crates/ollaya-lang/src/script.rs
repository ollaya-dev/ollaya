//! Script detection (`laya.lang.detect_script` and `script_profile`).
//!
//! Every letter is attributed to a script by code point: Latin (below U+0250, Latin Extended
//! Additional and fullwidth Latin), one of the Unicode blocks laya lists, or [`Script::Other`] when
//! no listed block claims it, so an unlisted script still counts as non-Latin. The dominant script
//! has the most letters; ties go to the script seen first, and Latin wins only outright. Text
//! without letters is [`Script::Unknown`].

use std::fmt;
use std::iter;

use serde::{Serialize, Serializer};

use crate::pystr;

/// A script, named as laya names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    Latin,
    Greek,
    Cyrillic,
    Armenian,
    Hebrew,
    Arabic,
    Devanagari,
    Bengali,
    Gurmukhi,
    Gujarati,
    Oriya,
    Tamil,
    Telugu,
    Kannada,
    Malayalam,
    Sinhala,
    Thai,
    Lao,
    Tibetan,
    Myanmar,
    Georgian,
    Ethiopic,
    Khmer,
    Hangul,
    Kana,
    Han,
    /// Letters of a script no listed block covers.
    Other,
    /// No letters at all.
    Unknown,
}

impl Script {
    pub fn as_str(self) -> &'static str {
        match self {
            Script::Latin => "latin",
            Script::Greek => "greek",
            Script::Cyrillic => "cyrillic",
            Script::Armenian => "armenian",
            Script::Hebrew => "hebrew",
            Script::Arabic => "arabic",
            Script::Devanagari => "devanagari",
            Script::Bengali => "bengali",
            Script::Gurmukhi => "gurmukhi",
            Script::Gujarati => "gujarati",
            Script::Oriya => "oriya",
            Script::Tamil => "tamil",
            Script::Telugu => "telugu",
            Script::Kannada => "kannada",
            Script::Malayalam => "malayalam",
            Script::Sinhala => "sinhala",
            Script::Thai => "thai",
            Script::Lao => "lao",
            Script::Tibetan => "tibetan",
            Script::Myanmar => "myanmar",
            Script::Georgian => "georgian",
            Script::Ethiopic => "ethiopic",
            Script::Khmer => "khmer",
            Script::Hangul => "hangul",
            Script::Kana => "kana",
            Script::Han => "han",
            Script::Other => "other",
            Script::Unknown => "unknown",
        }
    }

    /// The script of a letter (`str.isalpha`), or `None` for anything else.
    fn of(c: char) -> Option<Script> {
        if !pystr::is_alpha(c) {
            return None;
        }
        let cp = u32::from(c);
        if is_latin(cp) {
            return Some(Script::Latin);
        }
        let block = BLOCKS
            .iter()
            .find(|(_, ranges)| ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(&cp)));
        Some(block.map_or(Script::Other, |&(script, _)| script))
    }
}

impl fmt::Display for Script {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for Script {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

fn is_latin(cp: u32) -> bool {
    cp < 0x0250
        || (0x1E00..=0x1EFF).contains(&cp)
        || (0xFF21..=0xFF3A).contains(&cp)
        || (0xFF41..=0xFF5A).contains(&cp)
}

/// The blocks of each non-Latin script (`_SCRIPT_RANGES`), in laya's order.
const BLOCKS: &[(Script, &[(u32, u32)])] = &[
    (Script::Greek, &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    (
        Script::Cyrillic,
        &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    ),
    (Script::Armenian, &[(0x0530, 0x058F)]),
    (Script::Hebrew, &[(0x0590, 0x05FF)]),
    (
        Script::Arabic,
        &[
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ],
    ),
    (Script::Devanagari, &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    (Script::Bengali, &[(0x0980, 0x09FF)]),
    (Script::Gurmukhi, &[(0x0A00, 0x0A7F)]),
    (Script::Gujarati, &[(0x0A80, 0x0AFF)]),
    (Script::Oriya, &[(0x0B00, 0x0B7F)]),
    (Script::Tamil, &[(0x0B80, 0x0BFF)]),
    (Script::Telugu, &[(0x0C00, 0x0C7F)]),
    (Script::Kannada, &[(0x0C80, 0x0CFF)]),
    (Script::Malayalam, &[(0x0D00, 0x0D7F)]),
    (Script::Sinhala, &[(0x0D80, 0x0DFF)]),
    (Script::Thai, &[(0x0E00, 0x0E7F)]),
    (Script::Lao, &[(0x0E80, 0x0EFF)]),
    (Script::Tibetan, &[(0x0F00, 0x0FFF)]),
    (Script::Myanmar, &[(0x1000, 0x109F)]),
    (Script::Georgian, &[(0x10A0, 0x10FF)]),
    (Script::Ethiopic, &[(0x1200, 0x137F)]),
    (Script::Khmer, &[(0x1780, 0x17FF)]),
    (
        Script::Hangul,
        &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    ),
    (
        Script::Kana,
        &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    ),
    (
        Script::Han,
        &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    ),
];

/// Letters per script.
#[derive(Debug, Default)]
pub(crate) struct ScriptCounts {
    latin: usize,
    /// Non-Latin scripts in the order their first letter appears, as laya's dict keeps them.
    others: Vec<(Script, usize)>,
}

impl ScriptCounts {
    pub(crate) fn of(text: &str) -> Self {
        let mut counts = Self::default();
        for script in text.chars().filter_map(Script::of) {
            if script == Script::Latin {
                counts.latin += 1;
            } else if let Some((_, n)) = counts.others.iter_mut().find(|(s, _)| *s == script) {
                *n += 1;
            } else {
                counts.others.push((script, 1));
            }
        }
        counts
    }

    fn total(&self) -> usize {
        self.latin + self.others.iter().map(|&(_, n)| n).sum::<usize>()
    }

    /// The script with the most letters. laya takes Python's `max` over a dict where Latin is
    /// inserted last, and `max` keeps the first maximum: ties go to the non-Latin script seen
    /// first, and Latin wins only outright.
    pub(crate) fn dominant(&self) -> Script {
        let latin = (Script::Latin, self.latin);
        let mut best = (Script::Unknown, 0);
        for &(script, n) in self.others.iter().chain(iter::once(&latin)) {
            if n > best.1 {
                best = (script, n);
            }
        }
        best.0
    }

    /// Share of letters per script: Latin first, then the others in order of appearance. Empty
    /// without letters.
    pub(crate) fn profile(&self) -> Vec<(Script, f64)> {
        let total = self.total() as f64;
        let latin = (self.latin > 0).then_some((Script::Latin, self.latin));
        latin
            .into_iter()
            .chain(self.others.iter().copied())
            .map(|(script, n)| (script, n as f64 / total))
            .collect()
    }

    /// Share of letters outside Latin script, unrounded; 0 without letters.
    pub(crate) fn non_latin_share(&self) -> f64 {
        match self.total() {
            0 => 0.0,
            total => 1.0 - self.latin as f64 / total as f64,
        }
    }
}

/// Dominant script of `text`, or [`Script::Unknown`] when it has no letters.
pub fn detect_script(text: &str) -> Script {
    ScriptCounts::of(text).dominant()
}

/// Share of `text`'s letters in each script present: Latin first, then the others in order of
/// appearance. Empty when there are no letters.
pub fn script_profile(text: &str) -> Vec<(Script, f64)> {
    ScriptCounts::of(text).profile()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letters_only() {
        assert_eq!(detect_script("42 !? 😀"), Script::Unknown);
        // Devanagari vowel signs are marks, not letters, as in `str.isalpha`
        assert_eq!(script_profile("कि"), [(Script::Devanagari, 1.0)]);
    }

    #[test]
    fn ties_go_to_the_first_non_latin_script() {
        assert_eq!(detect_script("ab 中文"), Script::Han);
        assert_eq!(detect_script("中文 ab"), Script::Han);
        assert_eq!(detect_script("αβ ᏣᎳ"), Script::Greek);
        assert_eq!(detect_script("ᏣᎳ αβ"), Script::Other);
        assert_eq!(detect_script("abc αβ"), Script::Latin);
    }

    #[test]
    fn profile_lists_latin_first() {
        let profile = script_profile("αβ a");
        assert_eq!(
            profile,
            [(Script::Latin, 1.0 / 3.0), (Script::Greek, 2.0 / 3.0)]
        );
    }
}
