//! Temperature calibration: raw option logits to calibrated probabilities.

use std::collections::HashMap;

use serde::Deserialize;

use crate::question::QType;

/// A fitted temperature below this sharpens logits instead of softening them. Laya's shipped
/// `choice:11+` bucket is 0.10, which would publish a 0.24 top probability as 0.99, so such
/// values are clamped (as `laya.common.clamp_temperature` does).
pub const TEMP_MIN: f64 = 0.5;
pub const TEMP_MAX: f64 = 5.0;

/// The `calibration` layer: one temperature per question type, optionally refined per
/// (type, option-count bucket).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CalibrationFile {
    #[serde(default)]
    pub temperature: Vec<serde_json::Value>,
    #[serde(default)]
    pub temperature_by_options: HashMap<String, serde_json::Value>,
}

/// Temperatures as they are applied: clamped, with neutral fallbacks for invalid entries.
#[derive(Debug, Clone)]
pub struct Calibration {
    by_type: [f64; 3],
    by_options: HashMap<String, f64>,
}

impl Default for Calibration {
    fn default() -> Self {
        Calibration {
            by_type: [1.0; 3],
            by_options: HashMap::new(),
        }
    }
}

pub fn clamp_temperature(t: &serde_json::Value) -> f64 {
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
        }
    }

    pub fn temperature(&self, qtype: QType, k: usize) -> f64 {
        self.by_options
            .get(&bucket(qtype, k))
            .copied()
            .unwrap_or(self.by_type[qtype.index()])
    }

    /// Calibrated option probabilities from raw logits.
    pub fn probabilities(&self, qtype: QType, logits: &[f32]) -> Vec<f64> {
        let t = self.temperature(qtype, logits.len());
        softmax(logits.iter().map(|&z| f64::from(z) / t))
    }
}

pub fn softmax(z: impl Iterator<Item = f64> + Clone) -> Vec<f64> {
    let max = z.clone().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.map(|v| (v - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.into_iter().map(|v| v / sum).collect()
}
