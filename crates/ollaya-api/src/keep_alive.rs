//! `keep_alive`: how long a model stays loaded after a request (`docs/api.md` §6).
//!
//! Ollama's semantics: a Go duration string (`"5m"`, `"1h30m"`, `"300ms"`), a number of seconds,
//! `0` to unload right away, or any negative value to keep the model loaded indefinitely. Numeric
//! strings (`"300"`, `"-1"`) are accepted too, so one parser serves JSON and `OLLAYA_KEEP_ALIVE`.

use std::fmt;
use std::str::FromStr;
use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

/// Environment variable holding the server's default `keep_alive`.
pub const KEEP_ALIVE_ENV: &str = "OLLAYA_KEEP_ALIVE";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeepAlive {
    /// Stay loaded this long after the request finishes; zero unloads right away.
    For(Duration),
    /// Stay loaded until the server stops or the model is explicitly unloaded.
    Forever,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KeepAliveError {
    #[error(
        "invalid keep_alive {0:?}; use a duration such as \"5m\", \"1h30m\" or \"300ms\", a number of seconds, 0 to unload or -1 to keep loaded"
    )]
    Invalid(String),
    #[error("keep_alive {0:?} is too large")]
    TooLarge(String),
    #[error("keep_alive must be a finite number of seconds")]
    NotFinite,
    #[error("keep_alive must be a duration string or a number of seconds")]
    WrongType,
}

const NANOS_PER_SEC: u128 = 1_000_000_000;

impl KeepAlive {
    /// `5m`, the default when neither the request nor `OLLAYA_KEEP_ALIVE` sets one.
    pub const DEFAULT: KeepAlive = KeepAlive::For(Duration::from_secs(300));
    /// Unload as soon as the request finishes.
    pub const UNLOAD: KeepAlive = KeepAlive::For(Duration::ZERO);

    /// Parse a string: a Go duration or a (possibly fractional, possibly negative) number of
    /// seconds. Surrounding whitespace is ignored.
    pub fn parse(input: &str) -> Result<Self, KeepAliveError> {
        let s = input.trim();
        if is_number(s) {
            let secs: f64 = s
                .parse()
                .map_err(|_| KeepAliveError::Invalid(input.to_owned()))?;
            return KeepAlive::from_seconds(secs).map_err(|e| match e {
                KeepAliveError::TooLarge(_) => KeepAliveError::TooLarge(input.to_owned()),
                other => other,
            });
        }
        let (negative, nanos) = parse_go_duration(s).map_err(|e| match e {
            GoError::Invalid => KeepAliveError::Invalid(input.to_owned()),
            GoError::Overflow => KeepAliveError::TooLarge(input.to_owned()),
        })?;
        if negative && nanos > 0 {
            Ok(KeepAlive::Forever)
        } else {
            Ok(KeepAlive::For(nanos_to_duration(nanos)))
        }
    }

    /// A number of seconds: negative keeps the model loaded, zero unloads it.
    pub fn from_seconds(secs: f64) -> Result<Self, KeepAliveError> {
        if !secs.is_finite() {
            return Err(KeepAliveError::NotFinite);
        }
        if secs < 0.0 {
            return Ok(KeepAlive::Forever);
        }
        if secs * 1e9 > u64::MAX as f64 {
            return Err(KeepAliveError::TooLarge(secs.to_string()));
        }
        Duration::try_from_secs_f64(secs)
            .map(KeepAlive::For)
            .map_err(|_| KeepAliveError::TooLarge(secs.to_string()))
    }

    /// A JSON request value: `null` means absent.
    pub fn from_json(value: &Value) -> Result<Option<Self>, KeepAliveError> {
        match value {
            Value::Null => Ok(None),
            Value::Number(n) => n
                .as_f64()
                .ok_or(KeepAliveError::NotFinite)
                .and_then(KeepAlive::from_seconds)
                .map(Some),
            Value::String(s) => KeepAlive::parse(s).map(Some),
            _ => Err(KeepAliveError::WrongType),
        }
    }

    /// `OLLAYA_KEEP_ALIVE`, else [`KeepAlive::DEFAULT`].
    pub fn from_env() -> Result<Self, KeepAliveError> {
        match std::env::var(KEEP_ALIVE_ENV) {
            Ok(v) if !v.trim().is_empty() => KeepAlive::parse(&v),
            _ => Ok(KeepAlive::DEFAULT),
        }
    }

    /// The duration, or `None` for [`KeepAlive::Forever`].
    pub fn duration(&self) -> Option<Duration> {
        match self {
            KeepAlive::For(d) => Some(*d),
            KeepAlive::Forever => None,
        }
    }

    /// Whether the model unloads as soon as the request finishes.
    pub fn unloads_immediately(&self) -> bool {
        *self == KeepAlive::UNLOAD
    }
}

impl Default for KeepAlive {
    fn default() -> Self {
        KeepAlive::DEFAULT
    }
}

impl FromStr for KeepAlive {
    type Err = KeepAliveError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        KeepAlive::parse(s)
    }
}

/// Compact Go-style text that [`KeepAlive::parse`] reads back: `5m`, `1h30m`, `1.5s`, `300ms`,
/// `0s`; `-1` for [`KeepAlive::Forever`].
impl fmt::Display for KeepAlive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeepAlive::Forever => f.write_str("-1"),
            KeepAlive::For(d) => f.write_str(&format_duration(*d)),
        }
    }
}

/// `Forever` as `-1`, durations as their [`Display`](fmt::Display) string.
impl Serialize for KeepAlive {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            KeepAlive::Forever => s.serialize_i64(-1),
            KeepAlive::For(_) => s.collect_str(self),
        }
    }
}

impl<'de> Deserialize<'de> for KeepAlive {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(d)?;
        match KeepAlive::from_json(&value) {
            Ok(Some(k)) => Ok(k),
            Ok(None) => Err(serde::de::Error::custom(KeepAliveError::WrongType)),
            Err(e) => Err(serde::de::Error::custom(e)),
        }
    }
}

fn nanos_to_duration(nanos: u128) -> Duration {
    Duration::new(
        (nanos / NANOS_PER_SEC) as u64,
        (nanos % NANOS_PER_SEC) as u32,
    )
}

fn format_duration(d: Duration) -> String {
    let n = d.as_nanos();
    if n == 0 {
        return "0s".into();
    }
    if n < NANOS_PER_SEC {
        return if n.is_multiple_of(1_000_000) {
            format!("{}ms", n / 1_000_000)
        } else if n.is_multiple_of(1_000) {
            format!("{}us", n / 1_000)
        } else {
            format!("{n}ns")
        };
    }
    let hour = 3600 * NANOS_PER_SEC;
    let minute = 60 * NANOS_PER_SEC;
    let (h, rest) = (n / hour, n % hour);
    let (m, rest) = (rest / minute, rest % minute);
    let mut out = String::new();
    if h > 0 {
        out.push_str(&format!("{h}h"));
    }
    if m > 0 {
        out.push_str(&format!("{m}m"));
    }
    if rest > 0 {
        let (secs, frac) = (rest / NANOS_PER_SEC, rest % NANOS_PER_SEC);
        if frac == 0 {
            out.push_str(&format!("{secs}s"));
        } else {
            let frac = format!("{frac:09}");
            out.push_str(&format!("{secs}.{}s", frac.trim_end_matches('0')));
        }
    }
    out
}

/// `[+-]?digits[.digits]` or `[+-]?.digits`, with at least one digit.
fn is_number(s: &str) -> bool {
    let body = s.strip_prefix(['-', '+']).unwrap_or(s);
    let mut digits = 0;
    let mut dots = 0;
    for c in body.chars() {
        match c {
            '0'..='9' => digits += 1,
            '.' => dots += 1,
            _ => return false,
        }
    }
    digits > 0 && dots <= 1
}

enum GoError {
    Invalid,
    Overflow,
}

/// Go's `time.ParseDuration`: returns (negative, magnitude in nanoseconds).
fn parse_go_duration(s: &str) -> Result<(bool, u128), GoError> {
    let (negative, mut rest) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    if rest.is_empty() {
        return Err(GoError::Invalid);
    }
    let mut total: u128 = 0;
    while !rest.is_empty() {
        let int_len = rest.bytes().take_while(u8::is_ascii_digit).count();
        let (int_part, after) = rest.split_at(int_len);
        let (frac_part, after) = match after.strip_prefix('.') {
            Some(a) => {
                let n = a.bytes().take_while(u8::is_ascii_digit).count();
                a.split_at(n)
            }
            None => ("", after),
        };
        if int_part.is_empty() && frac_part.is_empty() {
            return Err(GoError::Invalid);
        }
        let unit_len = after
            .char_indices()
            .find(|&(_, c)| c.is_ascii_digit() || c == '.')
            .map_or(after.len(), |(i, _)| i);
        let (unit, next) = after.split_at(unit_len);
        let scale: u128 = match unit {
            "ns" => 1,
            "us" | "\u{b5}s" | "\u{3bc}s" => 1_000,
            "ms" => 1_000_000,
            "s" => NANOS_PER_SEC,
            "m" => 60 * NANOS_PER_SEC,
            "h" => 3600 * NANOS_PER_SEC,
            _ => return Err(GoError::Invalid),
        };
        let int: u128 = if int_part.is_empty() {
            0
        } else {
            int_part.parse().map_err(|_| GoError::Overflow)?
        };
        let mut value = int.checked_mul(scale).ok_or(GoError::Overflow)?;
        // Fractional digits beyond nanosecond precision cannot change the result.
        let frac_digits = &frac_part[..frac_part.len().min(18)];
        if !frac_digits.is_empty() {
            let frac: u128 = frac_digits.parse().map_err(|_| GoError::Invalid)?;
            value += frac * scale / 10u128.pow(frac_digits.len() as u32);
        }
        total = total.checked_add(value).ok_or(GoError::Overflow)?;
        if total > u128::from(u64::MAX) {
            return Err(GoError::Overflow);
        }
        rest = next;
    }
    Ok((negative, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn secs(s: u64) -> KeepAlive {
        KeepAlive::For(Duration::from_secs(s))
    }

    #[test]
    fn parses_go_durations() {
        let cases = [
            ("5m", secs(300)),
            ("1h30m", secs(5400)),
            ("1.5h", secs(5400)),
            ("300ms", KeepAlive::For(Duration::from_millis(300))),
            (
                "2h45m10.5s",
                KeepAlive::For(Duration::from_millis(9_910_500)),
            ),
            ("1us", KeepAlive::For(Duration::from_micros(1))),
            ("1\u{b5}s", KeepAlive::For(Duration::from_micros(1))),
            ("1\u{3bc}s", KeepAlive::For(Duration::from_micros(1))),
            ("42ns", KeepAlive::For(Duration::from_nanos(42))),
            (".5s", KeepAlive::For(Duration::from_millis(500))),
            ("1.s", secs(1)),
            ("+10s", secs(10)),
            ("0s", KeepAlive::UNLOAD),
            ("-0s", KeepAlive::UNLOAD),
            ("-5m", KeepAlive::Forever),
            (" 5m ", secs(300)),
        ];
        for (input, want) in cases {
            assert_eq!(KeepAlive::parse(input), Ok(want), "{input:?}");
        }
    }

    #[test]
    fn parses_numbers_of_seconds() {
        assert_eq!(KeepAlive::parse("300"), Ok(secs(300)));
        assert_eq!(KeepAlive::parse("0"), Ok(KeepAlive::UNLOAD));
        assert_eq!(KeepAlive::parse("-1"), Ok(KeepAlive::Forever));
        assert_eq!(
            KeepAlive::parse("1.5"),
            Ok(KeepAlive::For(Duration::from_millis(1500)))
        );
        assert_eq!(KeepAlive::from_seconds(-0.5), Ok(KeepAlive::Forever));
        assert_eq!(KeepAlive::from_json(&json!(0)), Ok(Some(KeepAlive::UNLOAD)));
        assert_eq!(
            KeepAlive::from_json(&json!(-1)),
            Ok(Some(KeepAlive::Forever))
        );
        assert_eq!(KeepAlive::from_json(&json!(600)), Ok(Some(secs(600))));
        assert_eq!(KeepAlive::from_json(&json!(null)), Ok(None));
        assert_eq!(
            KeepAlive::from_json(&json!(true)),
            Err(KeepAliveError::WrongType)
        );
    }

    #[test]
    fn rejects_bad_input() {
        for bad in [
            "", "   ", "5x", "m", "1.2.3s", "5m3", "-", "+", ".s", "1 m", "1e3s", "five",
        ] {
            assert!(
                matches!(KeepAlive::parse(bad), Err(KeepAliveError::Invalid(_))),
                "{bad:?}"
            );
        }
        assert!(matches!(
            KeepAlive::parse("99999999999999999999h"),
            Err(KeepAliveError::TooLarge(_))
        ));
        // u64::MAX nanoseconds is about 5,124,095 hours.
        assert_eq!(KeepAlive::parse("5000000h"), Ok(secs(5_000_000 * 3600)));
        assert!(matches!(
            KeepAlive::parse("6000000h"),
            Err(KeepAliveError::TooLarge(_))
        ));
        assert!(matches!(
            KeepAlive::from_seconds(1e20),
            Err(KeepAliveError::TooLarge(_))
        ));
        assert_eq!(
            KeepAlive::from_seconds(f64::NAN),
            Err(KeepAliveError::NotFinite)
        );
    }

    #[test]
    fn displays_compactly_and_round_trips() {
        let cases = [
            (secs(300), "5m"),
            (secs(5400), "1h30m"),
            (secs(3600), "1h"),
            (KeepAlive::For(Duration::from_millis(1500)), "1.5s"),
            (KeepAlive::For(Duration::from_millis(90_250)), "1m30.25s"),
            (KeepAlive::For(Duration::from_millis(300)), "300ms"),
            (KeepAlive::For(Duration::from_micros(7)), "7us"),
            (KeepAlive::For(Duration::from_nanos(42)), "42ns"),
            (KeepAlive::UNLOAD, "0s"),
            (KeepAlive::Forever, "-1"),
        ];
        for (k, text) in cases {
            assert_eq!(k.to_string(), text);
            assert_eq!(KeepAlive::parse(text), Ok(k), "{text}");
        }
    }

    #[test]
    fn serde_round_trip() {
        assert_eq!(serde_json::to_value(secs(600)).unwrap(), json!("10m"));
        assert_eq!(serde_json::to_value(KeepAlive::Forever).unwrap(), json!(-1));
        assert_eq!(
            serde_json::to_value(KeepAlive::UNLOAD).unwrap(),
            json!("0s")
        );
        for v in [json!("10m"), json!(-1), json!(0), json!(2.5), json!("-1")] {
            let k: KeepAlive = serde_json::from_value(v.clone()).unwrap();
            let back: KeepAlive = serde_json::from_value(serde_json::to_value(k).unwrap()).unwrap();
            assert_eq!(k, back, "{v}");
        }
        assert!(serde_json::from_value::<KeepAlive>(json!("soon")).is_err());
    }
}
