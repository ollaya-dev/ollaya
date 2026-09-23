//! The parts of Python's `str`, `re` and float formatting that detection depends on.
//!
//! laya runs on CPython, so a character is a letter when `str.isalpha` says so: general category
//! `L*` in CPython's Unicode version. Rust's `char::is_alphabetic` differs (it admits combining
//! vowel signs and letter numbers, and follows a newer Unicode), so the classes come from CPython
//! itself, in `tables.rs`.

use crate::tables::CHAR_CLASSES;

/// How CPython classifies a code point, as far as `str.isalpha` and `re` are concerned. Code
/// points without a class are neither letters nor word characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CharClass {
    /// `str.isalpha`: general category `L*`.
    Alpha,
    /// Numeric but not a decimal digit (`²`, `½`, `Ⅻ`): `re` counts it as a word letter.
    Numeric,
    /// A decimal digit: `\d`.
    Decimal,
}

fn class(c: char) -> Option<CharClass> {
    let cp = u32::from(c);
    let i = CHAR_CLASSES.partition_point(|&(_, hi, _)| hi < cp);
    CHAR_CLASSES
        .get(i)
        .filter(|&&(lo, _, _)| lo <= cp)
        .map(|&(_, _, class)| class)
}

/// `str.isalpha()`.
pub(crate) fn is_alpha(c: char) -> bool {
    class(c) == Some(CharClass::Alpha)
}

/// `\w` in a `str` pattern: `str.isalnum()` or `_`.
pub(crate) fn is_word(c: char) -> bool {
    c == '_' || class(c).is_some()
}

/// `[^\W\d_]`: a word character that is neither a decimal digit nor `_`.
pub(crate) fn is_word_letter(c: char) -> bool {
    matches!(class(c), Some(CharClass::Alpha | CharClass::Numeric))
}

/// `str.isspace()`: Unicode `White_Space` plus the separators U+001C..U+001F, which CPython
/// counts as whitespace for their bidirectional class.
pub(crate) fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `round(x, ndigits)`: the double nearest to `x` correctly rounded to `ndigits` decimals, ties
/// to even on the exact binary value. Rust's fixed-precision formatting rounds the same way.
pub(crate) fn round(x: f64, ndigits: usize) -> f64 {
    format!("{x:.ndigits$}").parse().unwrap_or(x)
}

/// `"%.0f" % (100 * x)`: a share as a whole percentage, ties to even.
pub(crate) fn percent(x: f64) -> String {
    format!("{:.0}", 100.0 * x)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use serde_json::Value;

    use super::*;

    fn cpython() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/chars.json")).unwrap()
    }

    fn code_points(v: &Value, key: &str) -> HashSet<u32> {
        let list = v[key].as_array().unwrap();
        list.iter().map(|cp| cp.as_u64().unwrap() as u32).collect()
    }

    #[test]
    fn classes_match_cpython_on_both_sides_of_every_boundary() {
        let py = cpython();
        let (alpha, letter, word) = (
            code_points(&py, "alpha"),
            code_points(&py, "word_letter"),
            code_points(&py, "word"),
        );
        let probes = code_points(&py, "probes");
        assert!(probes.len() >= 2 * CHAR_CLASSES.len());
        for cp in probes {
            let c = char::from_u32(cp).unwrap();
            assert_eq!(is_alpha(c), alpha.contains(&cp), "isalpha U+{cp:04X}");
            assert_eq!(
                is_word_letter(c),
                letter.contains(&cp),
                "[^\\W\\d_] U+{cp:04X}"
            );
            assert_eq!(is_word(c), word.contains(&cp), "\\w U+{cp:04X}");
        }
    }

    #[test]
    fn isspace_matches_cpython_everywhere() {
        let spaces = code_points(&cpython(), "space");
        for c in (0..=0x10FFFF).filter_map(char::from_u32) {
            assert_eq!(is_space(c), spaces.contains(&u32::from(c)), "{c:?}");
        }
    }

    #[test]
    fn table_is_sorted_and_merged() {
        for w in CHAR_CLASSES.windows(2) {
            let ((lo, hi, a), (next, _, b)) = (w[0], w[1]);
            assert!(lo <= hi && hi < next, "U+{lo:04X}");
            assert!(!(a == b && hi + 1 == next), "unmerged at U+{hi:04X}");
        }
    }

    #[test]
    fn rounding_ties_go_to_even_on_the_exact_value() {
        assert_eq!(round(1.0 / 32.0, 4), 0.0312);
        assert_eq!(round(3.0 / 32.0, 4), 0.0938);
        assert_eq!(round(1.0 - 31.0 / 32.0, 4), 0.0312);
        assert_eq!(percent(0.025), "2");
        assert_eq!(percent(0.125), "12");
        assert_eq!(percent(0.375), "38");
        assert_eq!(percent(0.035), "4");
    }
}
