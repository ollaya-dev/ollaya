//! Answers: calibrated probabilities to typed results, in the wire shapes clients expect.
//!
//! Two confidences exist:
//! * `confidence` (TypeSafe's): the top probability normalized over K options, (K·pmax − 1)/(K − 1).
//!   This is what `/v1/systemone` returns for every model, so thresholds transfer across models.
//! * laya's entropy confidence, 1 − H(p)/log K, kept for laya-compatible output.

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::calibration::{Calibration, softmax};
use crate::question::{Criteria, QType, Question};

/// Round like Python's `round(x, 4)`: the correctly rounded decimal, back as a float.
pub fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap_or(x)
}

/// A decided question, before it is rendered into a wire format.
#[derive(Debug, Clone)]
pub struct Answer {
    pub qtype: QType,
    /// Calibrated probability per option, in option order.
    pub probabilities: Vec<f64>,
    /// Probability that acting on this answer is appropriate, when the model has an act head.
    pub act_probability: Option<f64>,
}

impl Answer {
    pub fn new(
        q: &Question,
        calibration: &Calibration,
        logits: &[f32],
        act_logits: Option<&[f32]>,
    ) -> Self {
        let act_probability = act_logits.map(|a| softmax(a.iter().map(|&v| f64::from(v)))[0]);
        Answer {
            qtype: q.qtype,
            probabilities: calibration.probabilities(q.qtype, logits),
            act_probability,
        }
    }

    fn argmax(&self) -> usize {
        self.probabilities
            .iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |best, (i, &p)| {
                if p > best.1 { (i, p) } else { best }
            })
            .0
    }

    /// TypeSafe's normalized top-probability confidence.
    pub fn confidence(&self) -> f64 {
        let k = self.probabilities.len() as f64;
        if k < 2.0 {
            return 1.0;
        }
        let pmax = self.probabilities[self.argmax()];
        ((k * pmax - 1.0) / (k - 1.0)).clamp(0.0, 1.0)
    }

    /// laya's normalized-entropy confidence, 1 − H(p)/log K.
    pub fn entropy_confidence(&self) -> f64 {
        let k = self.probabilities.len();
        if k < 2 {
            return 1.0;
        }
        let h: f64 = -self
            .probabilities
            .iter()
            .map(|&p| p * p.clamp(1e-12, 1.0).ln())
            .sum::<f64>();
        (1.0 - h / (k as f64).ln()).clamp(0.0, 1.0)
    }

    fn expected_score(&self) -> f64 {
        self.probabilities
            .iter()
            .enumerate()
            .map(|(i, p)| i as f64 * p)
            .sum()
    }

    fn noul(&self) -> f64 {
        self.probabilities[1]
    }

    fn option_probabilities(&self, q: &Question) -> Map<String, Value> {
        let labels: Vec<String> = match &q.criteria {
            Criteria::Choice(m) => m.keys().cloned().collect(),
            _ => (0..self.probabilities.len())
                .map(|i| i.to_string())
                .collect(),
        };
        labels
            .into_iter()
            .zip(&self.probabilities)
            .map(|(l, &p)| (l, json!(round4(p))))
            .collect()
    }

    /// TypeSafe `/v1/systemone` shape. Noul answers carry no confidence there.
    pub fn to_typesafe(&self, q: &Question) -> Value {
        match &q.criteria {
            Criteria::Choice(m) => json!({
                "type": "choice",
                "choice": m.get_index(self.argmax()).map(|(k, _)| k.clone()),
                "confidence": round4(self.confidence()),
                "probabilities": self.option_probabilities(q),
            }),
            Criteria::Score(levels) => json!({
                "type": "score",
                "score": round4(self.expected_score()),
                "confidence": round4(self.confidence()),
                "legend": legend(levels),
                "probabilities": self.option_probabilities(q),
            }),
            Criteria::Noul { .. } => json!({"type": "noul", "noul": round4(self.noul())}),
        }
    }

    /// `laya` `system_one` shape, field for field.
    pub fn to_laya(&self, q: &Question) -> Value {
        let action = json!({"act_probability": self.act_probability.map(round4)});
        match &q.criteria {
            Criteria::Choice(m) => json!({
                "type": "choice",
                "choice": m.get_index(self.argmax()).map(|(k, _)| k.clone()),
                "probabilities": self.option_probabilities(q),
                "confidence": round4(self.entropy_confidence()),
                "action": action,
            }),
            Criteria::Score(levels) => json!({
                "type": "score",
                "score": round4(self.expected_score()),
                "legend": legend(levels),
                "probabilities": self.option_probabilities(q),
                "confidence": round4(self.entropy_confidence()),
                "action": action,
            }),
            Criteria::Noul { .. } => {
                let p = self.noul();
                json!({
                    "type": "noul",
                    "noul": round4(p),
                    "confidence": round4(p.max(1.0 - p)),
                    "action": action,
                })
            }
        }
    }
}

fn legend(levels: &[Value]) -> IndexMap<String, Value> {
    levels
        .iter()
        .enumerate()
        .map(|(i, c)| (i.to_string(), c.clone()))
        .collect()
}
