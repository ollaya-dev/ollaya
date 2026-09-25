//! `json.dumps` exactly as CPython renders it.
//!
//! Laya's encoder was trained on text produced by Python's `json.dumps` (structured states,
//! rubric-valued criteria, non-string instructions), so the runtime must reproduce that text
//! byte for byte: `", "` / `": "` separators, Python's float repr (`1.0`, `1e-05`, `1e+16`),
//! and Python's escaping rules with and without `ensure_ascii`.
//!
//! One known divergence: integers beyond the 64-bit range arrive from `serde_json` as floats, so
//! they render in float form where Python would print every digit.

use std::fmt::Write;

use serde_json::{Number, Value};

/// `json.dumps(value, ensure_ascii=ensure_ascii)` with the default separators.
pub fn dumps(value: &Value, ensure_ascii: bool) -> String {
    let mut out = String::new();
    let style = Style {
        ascii: ensure_ascii,
        sort_keys: false,
        item_sep: ", ",
        key_sep: ": ",
    };
    write_value(&mut out, value, &style);
    out
}

/// `json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`: compact, with
/// object keys in code point order (the order Python sorts `str` keys in).
pub fn dumps_canonical(value: &Value) -> String {
    let mut out = String::new();
    let style = Style {
        ascii: false,
        sort_keys: true,
        item_sep: ",",
        key_sep: ":",
    };
    write_value(&mut out, value, &style);
    out
}

struct Style {
    ascii: bool,
    sort_keys: bool,
    item_sep: &'static str,
    key_sep: &'static str,
}

fn write_value(out: &mut String, value: &Value, style: &Style) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => write_number(out, n),
        Value::String(s) => write_str(out, s, style.ascii),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(style.item_sep);
                }
                write_value(out, item, style);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if style.sort_keys {
                // UTF-8 byte order is code point order, which is how Python compares `str`.
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (i, (k, v)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(style.item_sep);
                }
                write_str(out, k, style.ascii);
                out.push_str(style.key_sep);
                write_value(out, v, style);
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

/// Python's `repr(float)`, which is also what `json.dumps` writes for floats.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // `{:e}` gives the shortest round-trip digits, the same digits Python's repr uses.
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci
        .split_once('e')
        .expect("LowerExp always has an exponent");
    let exp: i32 = exp.parse().expect("LowerExp exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1; // value = 0.<digits> * 10^decpt

    let mut out = String::new();
    if x < 0.0 {
        out.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    } else if decpt <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-decpt) as usize));
        out.push_str(&digits);
    } else if decpt as usize >= digits.len() {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', decpt as usize - digits.len()));
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    out
}

fn write_str(out: &mut String, s: &str, ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            // ensure_ascii escapes everything outside printable ASCII (space..tilde).
            c if ascii && (c as u32) > 0x7e => {
                let n = c as u32;
                if n > 0xffff {
                    let n = n - 0x10000;
                    let _ = write!(
                        out,
                        "\\u{:04x}\\u{:04x}",
                        0xd800 | (n >> 10),
                        0xdc00 | (n & 0x3ff)
                    );
                } else {
                    let _ = write!(out, "\\u{n:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_match_python_repr() {
        // Expected strings are Python 3.12 `repr(x)`.
        let cases: &[(f64, &str)] = &[
            (1.0, "1.0"),
            (19.99, "19.99"),
            (0.1, "0.1"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-05, "1.5e-05"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e20, "1.2345e+20"),
            (123.456, "123.456"),
            (-2.5, "-2.5"),
            (-0.0, "-0.0"),
            (1e-310, "1e-310"),
            (f64::MAX, "1.7976931348623157e+308"),
            (0.1 + 0.2, "0.30000000000000004"),
        ];
        for (x, want) in cases {
            assert_eq!(float_repr(*x), *want, "repr({x:e})");
        }
    }

    #[test]
    fn dumps_matches_python() {
        let v = json!({"a": [1, 2.0, true, null], "é": "x\"y\\z\n\u{1}", "emoji": "😡", "del": "\u{7f}"});
        // json.dumps(v, ensure_ascii=False)
        assert_eq!(
            dumps(&v, false),
            "{\"a\": [1, 2.0, true, null], \"é\": \"x\\\"y\\\\z\\n\\u0001\", \"emoji\": \"😡\", \"del\": \"\u{7f}\"}"
        );
        // json.dumps(v)
        assert_eq!(
            dumps(&v, true),
            "{\"a\": [1, 2.0, true, null], \"\\u00e9\": \"x\\\"y\\\\z\\n\\u0001\", \"emoji\": \"\\ud83d\\ude21\", \"del\": \"\\u007f\"}"
        );
        assert_eq!(dumps(&json!({}), false), "{}");
        assert_eq!(dumps(&json!([]), false), "[]");
    }

    #[test]
    fn canonical_matches_python() {
        // json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        let v = json!({"key": "b", "description": {"z": [1, 2.5, null], "é": "x\n", "A": true, "a": 1e-5}});
        assert_eq!(
            dumps_canonical(&v),
            r#"{"description":{"A":true,"a":1e-05,"z":[1,2.5,null],"é":"x\n"},"key":"b"}"#
        );
        assert_eq!(dumps_canonical(&json!("a\"b")), r#""a\"b""#);
        assert_eq!(dumps_canonical(&json!(null)), "null");
        assert_eq!(dumps_canonical(&json!({})), "{}");
    }
}
