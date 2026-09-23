//! The Latin-script language guess (`laya.lang.latin_profile`).
//!
//! Best effort by design. Identifiers are removed first (`github.com`, `user@acme.com`, `v1.2.3`:
//! a dot or `@` joining word characters), because their pieces collide with function words. The
//! remaining words are looked up in per-language function-word lists. A non-English language is
//! named only when it matched a word no other list claims and beats English by two hits, or, when
//! the rate of letters English does not use reaches [`NON_EN_DIACRITIC_RATE`], has two hits and no
//! fewer than English. Otherwise English is named if any English function word matched and the
//! rate stays below the threshold. Text of fewer than four words names nothing.

use serde::Serialize;

use crate::pystr;
use crate::tables::{DIACRITICS, LANGUAGES, STOPWORDS};

/// A diacritic rate at or above this is evidence the text is not English, even when no
/// function-word list matches it.
pub const NON_EN_DIACRITIC_RATE: f64 = 0.02;

/// Index of English in `LANGUAGES`.
const ENGLISH: usize = 0;

/// Fewer words than this name no language.
const MIN_WORDS: usize = 4;

/// Evidence behind the Latin-script language guess.
///
/// "Undecided" (`language: None`) and "English" are different answers: only English, or undecided
/// text that does not look non-English, is safe for the English checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LatinProfile {
    /// The language named, if any: `en`, `fr`, `de`, `es`, `pt`, `it`, `nl` or `ro`.
    pub language: Option<&'static str>,
    /// Words that are English function words (0 below four words).
    pub english_hits: usize,
    /// Share of the lowercased text's characters that English does not use, unrounded.
    pub diacritic_rate: f64,
    /// `diacritic_rate` reaches [`NON_EN_DIACRITIC_RATE`].
    pub looks_non_english: bool,
}

/// The language evidence in `text`; see the module docs.
pub fn latin_profile(text: &str) -> LatinProfile {
    // Python counts over `text.lower()`. Lowercasing char by char skips only the final-sigma
    // rule, which picks between two Greek letters and changes neither count.
    let (diacritics, len) = text
        .chars()
        .flat_map(char::to_lowercase)
        .fold((0usize, 0usize), |(d, n), c| {
            (d + usize::from(DIACRITICS.binary_search(&c).is_ok()), n + 1)
        });
    let diacritic_rate = diacritics as f64 / len.max(1) as f64;
    let looks_non_english = diacritic_rate >= NON_EN_DIACRITIC_RATE;

    let stripped = strip_identifiers(text);
    let words: Vec<String> = words(&stripped).map(str::to_lowercase).collect();
    if words.len() < MIN_WORDS {
        return LatinProfile {
            language: None,
            english_hits: 0,
            diacritic_rate,
            looks_non_english,
        };
    }

    let mut hits = [0usize; LANGUAGES.len()];
    // Languages that matched a word no other list claims: only these may be named.
    let mut evidenced = 0u8;
    for word in &words {
        let langs = stopword_languages(word);
        for (lang, n) in hits.iter_mut().enumerate() {
            *n += usize::from(langs & (1 << lang) != 0);
        }
        if langs.count_ones() == 1 {
            evidenced |= langs;
        }
    }
    let english = hits[ENGLISH];
    // The evidenced non-English language with the most hits; ties keep the earlier one, as
    // Python's `max` does.
    let mut best: Option<(usize, usize)> = None;
    for lang in (0..LANGUAGES.len()).filter(|&l| l != ENGLISH && evidenced & (1 << l) != 0) {
        if best.is_none_or(|(_, n)| hits[lang] > n) {
            best = Some((lang, hits[lang]));
        }
    }

    let language = match best {
        // a clear margin over English function words (laya's `max(2, en + 2)`)
        Some((lang, n)) if n >= english + 2 => Some(LANGUAGES[lang]),
        // the letters already say "not English": two hits and no fewer than English suffice
        Some((lang, n)) if looks_non_english && n >= english.max(2) => Some(LANGUAGES[lang]),
        _ if english > 0 && !looks_non_english => Some(LANGUAGES[ENGLISH]),
        _ => None,
    };
    LatinProfile {
        language,
        english_hits: english,
        diacritic_rate,
        looks_non_english,
    }
}

/// Best-effort language of Latin-script text, or `None` when undecided (`guess_latin_language`).
pub fn guess_latin_language(text: &str) -> Option<&'static str> {
    latin_profile(text).language
}

/// The languages whose function-word list holds `word`, one bit per `LANGUAGES` entry.
fn stopword_languages(word: &str) -> u8 {
    STOPWORDS
        .binary_search_by(|&(w, _)| w.cmp(word))
        .map_or(0, |i| STOPWORDS[i].1)
}

/// `_IDENTIFIER.sub(" ", text)` for `[\w-]*(?:[.@][\w-]+)+`: every run of word characters and
/// hyphens joined by `.` or `@` becomes one space. A trailing dot (`arrivato.`) stays, since the
/// pattern needs word characters on both sides of it.
fn strip_identifiers(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let part = |i: usize| chars.get(i).is_some_and(|&c| c == '-' || pystr::is_word(c));
    let run_end = |mut i: usize| {
        while part(i) {
            i += 1;
        }
        i
    };
    // A `.` or `@` at `i` that joins the run before it to a following word character.
    let joins = |i: usize| matches!(chars.get(i), Some('.' | '@')) && part(i + 1);

    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let mut end = run_end(i);
        if joins(end) {
            while joins(end) {
                end = run_end(end + 1);
            }
            out.push(' ');
            i = end;
        } else {
            // No match starts anywhere up to `end`, where the same test would fail again.
            let stop = (end + 1).min(chars.len());
            out.extend(&chars[i..stop]);
            i = stop;
        }
    }
    out
}

/// `_WORD.findall`: maximal runs of `[^\W\d_]`.
fn words(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c| !pystr::is_word_letter(c))
        .filter(|w| !w.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_is_the_first_language() {
        assert_eq!(LANGUAGES[ENGLISH], "en");
    }

    #[test]
    fn identifiers_are_stripped_as_python_does() {
        let cases = [
            (
                "a..b a.b.c -x.y- @que .la para.es le@la e.g. i.e.",
                "a.               .  .",
            ),
            (
                "See github.com/org/repo, v1.2.3 and U.S.A. for",
                "See  /org/repo,   and  . for",
            ),
            ("Il pacco non è arrivato.", "Il pacco non è arrivato."),
            ("x-y.z_w foo_bar.baz ..x a.", "    .  a."),
            ("user@acme.com,la@que.es", " , "),
        ];
        for (text, want) in cases {
            assert_eq!(strip_identifiers(text), want, "{text:?}");
        }
    }

    #[test]
    fn words_keep_numeric_letters_and_drop_digits() {
        let got: Vec<&str> = words("the² x²y Ⅻ ½ 42 a_b").collect();
        assert_eq!(got, ["the²", "x²y", "Ⅻ", "½", "a", "b"]);
    }
}
