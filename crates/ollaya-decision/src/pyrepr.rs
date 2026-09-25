//! Python `str()` and `repr()` of JSON values, exactly as CPython 3.12 prints them.
//!
//! Von renders structured states and criteria with Python `str()` (`f"{key}: {value}"`), not
//! `json.dumps`: `{'k': 'v', 'n': 1, 'b': True, 'x': None}`. Strings inside containers are quoted
//! and escaped by `repr()` (CPython's `unicode_repr`), whose idea of a printable character comes
//! from CPython's Unicode tables (`printable.rs`).
//!
//! The same divergence as in [`crate::pyjson`] applies: integers beyond the 64-bit range arrive
//! from `serde_json` as floats, so they render in float form where Python would print every digit.

use std::fmt::Write;

use serde_json::{Number, Value};

use crate::printable::PRINTABLE;
use crate::pyjson::float_repr;

/// `str(value)` for a value `json.loads` produced: a string as it is, anything else as `repr`.
pub fn str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        v => repr(v),
    }
}

/// `repr(value)` for a value `json.loads` produced.
pub fn repr(value: &Value) -> String {
    let mut out = String::new();
    write_repr(&mut out, value);
    out
}

/// `bool(value)`: `None`, `False`, zero, and empty strings, lists and dicts are false.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// `str.strip()`: removes `str.isspace()` characters, which are Unicode `White_Space` plus the
/// separators U+001C..U+001F.
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.isprintable()` for one character.
pub fn is_printable(c: char) -> bool {
    let cp = u32::from(c);
    let i = PRINTABLE.partition_point(|&(_, last)| last < cp);
    PRINTABLE.get(i).is_some_and(|&(first, _)| first <= cp)
}

fn write_repr(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Number(n) => write_number(out, n),
        Value::String(s) => write_str(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_repr(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str(out, k);
                out.push_str(": ");
                write_repr(out, v);
            }
            out.push('}');
        }
    }
}

fn write_number(out: &mut String, n: &Number) {
    if let Some(i) = n.as_i64() {
        let _ = write!(out, "{i}");
    } else if let Some(u) = n.as_u64() {
        let _ = write!(out, "{u}");
    } else {
        out.push_str(&float_repr(n.as_f64().unwrap_or(f64::NAN)));
    }
}

/// CPython's `unicode_repr`.
fn write_str(out: &mut String, s: &str) {
    // Single quotes, unless the text has a single quote and no double quote.
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for c in s.chars() {
        match c {
            c if c == quote || c == '\\' => {
                out.push('\\');
                out.push(c);
            }
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c < ' ' || c == '\u{7f}' => {
                let _ = write!(out, "\\x{:02x}", u32::from(c));
            }
            c if c.is_ascii() || is_printable(c) => out.push(c),
            c => {
                let n = u32::from(c);
                let _ = if n <= 0xff {
                    write!(out, "\\x{n:02x}")
                } else if n <= 0xffff {
                    write!(out, "\\u{n:04x}")
                } else {
                    write!(out, "\\U{n:08x}")
                };
            }
        }
    }
    out.push(quote);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn repr_matches_python() {
        // Expected strings are CPython 3.12 `repr(json.loads(...))`.
        let v = json!({"k": "v", "n": 1, "f": 19.99, "e": 1e-5, "b": true, "x": null,
                       "l": ["a", 2.0, false, [], {}]});
        assert_eq!(
            repr(&v),
            "{'k': 'v', 'n': 1, 'f': 19.99, 'e': 1e-05, 'b': True, 'x': None, 'l': ['a', 2.0, False, [], {}]}"
        );
        for (s, want) in [
            ("plain", "'plain'"),
            ("it's", "\"it's\""),
            ("it's \"q\"", "'it\\'s \"q\"'"),
            ("say \"hi\"", "'say \"hi\"'"),
            ("a\\b", "'a\\\\b'"),
            ("t\tn\nr\r", "'t\\tn\\nr\\r'"),
            ("\u{0}\u{1b}\u{7f}", "'\\x00\\x1b\\x7f'"),
            ("\u{a0}\u{ad}é", "'\\xa0\\xadé'"),
            ("👨\u{200d}👩 ok", "'👨\\u200d👩 ok'"),
            ("\u{2028}\u{3000}\u{e000}", "'\\u2028\\u3000\\ue000'"),
            ("\u{e0001}\u{10ffff}", "'\\U000e0001\\U0010ffff'"),
            ("Türkçe 中文 😡", "'Türkçe 中文 😡'"),
        ] {
            assert_eq!(repr(&json!(s)), want, "{s:?}");
        }
    }

    #[test]
    fn str_keeps_top_level_strings() {
        assert_eq!(str(&json!("it's")), "it's");
        assert_eq!(str(&json!(null)), "None");
        assert_eq!(str(&json!(["it's"])), "[\"it's\"]");
        assert_eq!(str(&json!(-3)), "-3");
        assert_eq!(str(&json!(18446744073709551615u64)), "18446744073709551615");
    }

    #[test]
    fn truthiness_and_strip() {
        for v in [
            json!(null),
            json!(false),
            json!(0),
            json!(0.0),
            json!(""),
            json!([]),
            json!({}),
        ] {
            assert!(!truthy(&v), "{v}");
        }
        for v in [
            json!(true),
            json!(-1),
            json!(0.5),
            json!(" "),
            json!([0]),
            json!({"a": 0}),
        ] {
            assert!(truthy(&v), "{v}");
        }
        assert_eq!(strip("\u{1c}\u{a0} x y \u{3000}\n"), "x y");
        assert_eq!(strip("\u{200b}x"), "\u{200b}x");
    }

    #[test]
    fn printable_table_edges() {
        assert!(is_printable(' ') && is_printable('~') && is_printable('é'));
        assert!(!is_printable('\u{7f}') && !is_printable('\u{a0}') && !is_printable('\u{ad}'));
        assert!(!is_printable('\u{378}'), "unassigned in Unicode 15.0");
        assert!(is_printable('\u{e0100}') && !is_printable('\u{e01f0}'));
        assert!(!is_printable('\u{10ffff}'));
    }
}
