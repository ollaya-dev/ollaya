//! Temperature calibration: raw option logits to calibrated probabilities.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use crate::question::QType;

/// A fitted temperature below this sharpens logits instead of softening them. Laya's shipped
/// `choice:11+` bucket is 0.10, which would publish a 0.24 top probability as 0.99, so such
/// values are clamped (as `laya.common.clamp_temperature` does).
pub const TEMP_MIN: f64 = 0.5;
pub const TEMP_MAX: f64 = 5.0;

/// The `calibration` layer: one temperature per question type, optionally refined per
/// (type, option-count bucket), or a temperature that depends on the input.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CalibrationFile {
    #[serde(default)]
    pub temperature: Vec<Value>,
    #[serde(default)]
    pub temperature_by_options: HashMap<String, Value>,
    /// An input-conditioned temperature (see [`TemperatureMap`]). When it is valid and of a known
    /// kind it replaces the fixed temperatures, which stay as the fallback otherwise.
    #[serde(default)]
    pub temperature_map: Option<Value>,
}

/// Temperatures as they are applied: clamped, with neutral fallbacks for invalid entries.
#[derive(Debug, Clone)]
pub struct Calibration {
    by_type: [f64; 3],
    by_options: HashMap<String, f64>,
    map: Option<TemperatureMap>,
}

impl Default for Calibration {
    fn default() -> Self {
        Calibration {
            by_type: [1.0; 3],
            by_options: HashMap::new(),
            map: None,
        }
    }
}

/// A temperature computed per question from the question's own logits and the request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemperatureMap {
    /// `von-entropy-length-v1`, Von's `OptionMarkerBackend._effective_temperature`:
    ///
    /// ```text
    /// T = clamp(bias + entropy·H + log_tokens·log10(max(state_tokens, 1))/4 + n_options·K/8, lo, hi)
    /// ```
    ///
    /// `H` is the normalised entropy of `softmax(logits)` (0 for one option), `K` the number of
    /// options and `state_tokens` the state's length in tokens. The map's own bounds apply, not
    /// [`TEMP_MIN`] / [`TEMP_MAX`].
    EntropyLength {
        bias: f64,
        entropy: f64,
        log_tokens: f64,
        n_options: f64,
        lo: f64,
        hi: f64,
    },
}

impl TemperatureMap {
    /// A map of a kind this build knows, validated as upstream validates it
    /// (`_validate_calibration_map`): every coefficient a number, missing ones 0, at least one
    /// present, bounds defaulting to [0.5, 12] with `lo <= hi`. Anything else is `None`, and the
    /// fixed temperatures apply.
    pub fn parse(value: &Value) -> Option<Self> {
        let map = value.as_object()?;
        match map.get("kind")?.as_str()? {
            "von-entropy-length-v1" => {
                // `Some(None)` for a missing key, `None` for one that is not a number.
                let get = |key: &str| match map.get(key) {
                    None => Some(None),
                    Some(v) => v.as_f64().filter(|x| x.is_finite()).map(Some),
                };
                let (bias, entropy) = (get("bias")?, get("entropy")?);
                let (log_tokens, n_options) = (get("log_tokens")?, get("n_options")?);
                if [bias, entropy, log_tokens, n_options]
                    .iter()
                    .all(Option::is_none)
                {
                    return None;
                }
                let (lo, hi) = (get("lo")?.unwrap_or(0.5), get("hi")?.unwrap_or(12.0));
                (lo <= hi).then_some(TemperatureMap::EntropyLength {
                    bias: bias.unwrap_or(0.0),
                    entropy: entropy.unwrap_or(0.0),
                    log_tokens: log_tokens.unwrap_or(0.0),
                    n_options: n_options.unwrap_or(0.0),
                    lo,
                    hi,
                })
            }
            _ => None,
        }
    }

    /// The temperature for one question's option logits.
    pub fn temperature(&self, logits: &[f32], state_tokens: usize) -> f64 {
        match *self {
            TemperatureMap::EntropyLength {
                bias,
                entropy,
                log_tokens,
                n_options,
                lo,
                hi,
            } => {
                let k = logits.len();
                let h = if k > 1 {
                    let p = softmax(logits.iter().map(|&z| f64::from(z)));
                    -p.iter().map(|&p| p * p.max(1e-12).ln()).sum::<f64>() / (k as f64).ln()
                } else {
                    0.0
                };
                let tokens = state_tokens.max(1) as f64;
                let t = bias
                    + entropy * h
                    + log_tokens * (tokens.log10() / 4.0)
                    + n_options * (k as f64 / 8.0);
                t.max(lo).min(hi)
            }
        }
    }
}

pub fn clamp_temperature(t: &Value) -> f64 {
    match t.as_f64() {
        Some(t) if t.is_finite() => t.clamp(TEMP_MIN, TEMP_MAX),
        _ => 1.0,
    }
}

/// `laya.common.temp_bucket`: "<type>:<2|3-5|6-10|11+>".
pub fn bucket(qtype: QType, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{size}", qtype.name())
}

impl Calibration {
    pub fn from_file(file: &CalibrationFile) -> Self {
        let mut by_type = [1.0; 3];
        for (slot, t) in by_type.iter_mut().zip(&file.temperature) {
            *slot = clamp_temperature(t);
        }
        let by_options = file
            .temperature_by_options
            .iter()
            .map(|(k, v)| (k.clone(), clamp_temperature(v)))
            .collect();
        Calibration {
            by_type,
            by_options,
            map: file
                .temperature_map
                .as_ref()
                .and_then(TemperatureMap::parse),
        }
    }

    /// The input-conditioned temperature, when the calibration has a valid one.
    pub fn map(&self) -> Option<&TemperatureMap> {
        self.map.as_ref()
    }

    /// The fixed temperature for a question type and option count.
    pub fn temperature(&self, qtype: QType, k: usize) -> f64 {
        self.by_options
            .get(&bucket(qtype, k))
            .copied()
            .unwrap_or(self.by_type[qtype.index()])
    }

    /// The temperature applied to one question's logits. `state_tokens` (the request's state
    /// length in tokens) only matters to an input-conditioned map.
    pub fn temperature_for(&self, qtype: QType, logits: &[f32], state_tokens: usize) -> f64 {
        match &self.map {
            // von divides by `max(T, 1e-4)`.
            Some(map) => map.temperature(logits, state_tokens).max(1e-4),
            None => self.temperature(qtype, logits.len()),
        }
    }

    /// Calibrated option probabilities from raw logits.
    pub fn probabilities(&self, qtype: QType, logits: &[f32], state_tokens: usize) -> Vec<f64> {
        let t = self.temperature_for(qtype, logits, state_tokens);
        softmax(logits.iter().map(|&z| f64::from(z) / t))
    }
}

pub fn softmax(z: impl Iterator<Item = f64> + Clone) -> Vec<f64> {
    let max = z.clone().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.map(|v| (v - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.into_iter().map(|v| v / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn von() -> CalibrationFile {
        serde_json::from_value(json!({
            "temperature": [2.2, 2.2, 2.2],
            "temperature_by_options": {},
            "temperature_map": {"kind": "von-entropy-length-v1", "formula": "…", "bias": 0.2056,
                                "entropy": -3.255, "log_tokens": 20.2391, "n_options": -5.2263,
                                "lo": 0.3, "hi": 12.0},
        }))
        .unwrap()
    }

    #[test]
    fn entropy_length_map_matches_von() {
        let cal = Calibration::from_file(&von());
        // goldens-von.jsonl, preset/triage/tr_billing, "intent": the reference's option logits
        // (f32 values), temperature and probabilities.
        let z = [
            -6.360236644744873f64,
            -0.39759331941604614,
            -4.1223626136779785,
            -5.925295352935791,
            -3.9016213417053223,
            6.17419958114624,
        ]
        .map(|x| x as f32);
        let t = cal.temperature_for(QType::Choice, &z, 47);
        assert!((t - 4.7253352031915625).abs() < 1e-9, "{t}");
        // Clamped at lo for a long, confident question; at hi for a short, uncertain one.
        assert_eq!(cal.temperature_for(QType::Noul, &[-20.0, 20.0], 1), 0.3);
        assert_eq!(cal.temperature_for(QType::Noul, &[0.0, 0.0], 100_000), 12.0);
        // One option: no entropy term.
        let one = cal.temperature_for(QType::Choice, &[1.0], 10);
        assert!((one - (0.2056 + 20.2391 * 0.25 - 5.2263 / 8.0)).abs() < 1e-12);
        let p = cal.probabilities(QType::Choice, &z, 47);
        assert!((p[5] - 0.6141246730929746).abs() < 1e-9, "{p:?}");
    }

    #[test]
    fn invalid_or_unknown_maps_fall_back_to_temperatures() {
        for map in [
            json!({"kind": "something-else", "bias": 1.0}),
            json!({"kind": "von-entropy-length-v1", "bias": "high"}),
            json!({"kind": "von-entropy-length-v1", "lo": 0.1}),
            json!({"kind": "von-entropy-length-v1", "bias": 1.0, "lo": 3.0, "hi": 2.0}),
            json!({"bias": 1.0}),
            json!("von-entropy-length-v1"),
        ] {
            let mut file = von();
            file.temperature_map = Some(map.clone());
            let cal = Calibration::from_file(&file);
            assert!(cal.map().is_none(), "{map}");
            assert_eq!(cal.temperature_for(QType::Choice, &[0.0, 1.0], 10), 2.2);
        }
        let partial = json!({"kind": "von-entropy-length-v1", "bias": 1.0});
        assert_eq!(
            TemperatureMap::parse(&partial),
            Some(TemperatureMap::EntropyLength {
                bias: 1.0,
                entropy: 0.0,
                log_tokens: 0.0,
                n_options: 0.0,
                lo: 0.5,
                hi: 12.0
            })
        );
    }

    #[test]
    fn fixed_temperatures_ignore_the_state() {
        let file: CalibrationFile = serde_json::from_value(json!({
            "temperature": [0.1, 2.0, 9.0], "temperature_by_options": {"choice:3-5": 1.5},
        }))
        .unwrap();
        let cal = Calibration::from_file(&file);
        assert_eq!(
            cal.temperature_for(QType::Choice, &[0.0, 1.0], 5000),
            TEMP_MIN
        );
        assert_eq!(cal.temperature_for(QType::Choice, &[0.0, 1.0, 2.0], 1), 1.5);
        assert_eq!(cal.temperature_for(QType::Noul, &[0.0, 1.0], 1), TEMP_MAX);
        assert_eq!(
            cal.probabilities(QType::Score, &[0.0, 2.0], 1),
            cal.probabilities(QType::Score, &[0.0, 2.0], 9999)
        );
    }
}
