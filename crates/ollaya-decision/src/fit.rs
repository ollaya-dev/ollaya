//! Fitting temperatures: the `calibration` layer's numbers from a model's own option logits.
//!
//! Temperature scaling (Guo et al., 2017): one `T` per group of questions minimises the mean
//! cross-entropy of `softmax(z / T)` against the targets. The targets may be soft (a probability
//! per option); one-hot targets give the usual negative log-likelihood. As a function of `1 / T`
//! the loss is convex, so a golden-section search on `1 / T` finds the minimum.

use crate::calibration::softmax;

/// One question: the option logits before any temperature, and the target distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub logits: Vec<f32>,
    pub target: Vec<f64>,
}

impl Sample {
    fn probabilities(&self, t: f64) -> Vec<f64> {
        softmax(self.logits.iter().map(|&z| f64::from(z) / t))
    }
}

/// Mean cross-entropy of `softmax(z / T)` against the targets.
pub fn cross_entropy(samples: &[Sample], t: f64) -> f64 {
    let total: f64 = samples
        .iter()
        .map(|s| {
            let p = s.probabilities(t);
            -s.target
                .iter()
                .zip(&p)
                .filter(|(q, _)| **q > 0.0)
                .map(|(q, p)| q * p.max(1e-300).ln())
                .sum::<f64>()
        })
        .sum();
    total / samples.len().max(1) as f64
}

/// The temperature in `[lo, hi]` that minimises [`cross_entropy`].
pub fn fit_temperature(samples: &[Sample], [lo, hi]: [f64; 2]) -> f64 {
    // Golden-section search on b = 1 / T, where the loss is convex.
    let loss = |b: f64| cross_entropy(samples, 1.0 / b);
    let r = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (1.0 / hi, 1.0 / lo);
    let mut c = b - r * (b - a);
    let mut d = a + r * (b - a);
    let (mut fc, mut fd) = (loss(c), loss(d));
    for _ in 0..200 {
        if (b - a).abs() <= 1e-12 * b.abs() {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - r * (b - a);
            fc = loss(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + r * (b - a);
            fd = loss(d);
        }
    }
    (1.0 / ((a + b) / 2.0)).clamp(lo, hi)
}

fn argmax(x: &[f64]) -> usize {
    x.iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0
}

/// The share of questions whose top option is the target's top option.
pub fn accuracy(samples: &[Sample]) -> f64 {
    let hits = samples
        .iter()
        .filter(|s| argmax(&s.probabilities(1.0)) == argmax(&s.target))
        .count();
    hits as f64 / samples.len().max(1) as f64
}

/// Expected calibration error of the top option over `bins` equal-width confidence bins, against
/// the target's top option.
pub fn ece(samples: &[Sample], t: f64, bins: usize) -> f64 {
    let bins = bins.max(1);
    let mut conf = vec![0.0; bins];
    let mut hit = vec![0.0; bins];
    let mut n = vec![0usize; bins];
    for s in samples {
        let p = s.probabilities(t);
        let top = argmax(&p);
        let i = ((p[top] * bins as f64) as usize).min(bins - 1);
        conf[i] += p[top];
        hit[i] += f64::from(u8::from(top == argmax(&s.target)));
        n[i] += 1;
    }
    let total = samples.len().max(1) as f64;
    (0..bins)
        .filter(|&i| n[i] > 0)
        .map(|i| (conf[i] - hit[i]).abs() / total)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Logits whose calibrated temperature is known: drawn so that softmax(z / 4) is the truth.
    fn tempered(t_true: f64, n: usize) -> Vec<Sample> {
        let mut x: u64 = 7;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let logits: Vec<f32> = (0..4).map(|_| (next() * 40.0 - 20.0) as f32).collect();
                // The expected outcome under the true distribution: soft targets.
                let target = softmax(logits.iter().map(|&z| f64::from(z) / t_true));
                Sample { logits, target }
            })
            .collect()
    }

    #[test]
    fn soft_targets_recover_the_true_temperature() {
        for t in [0.7, 4.0, 91.43] {
            let samples = tempered(t, 400);
            let fit = fit_temperature(&samples, [0.05, 1000.0]);
            assert!((fit / t - 1.0).abs() < 1e-6, "{t}: {fit}");
            assert!(cross_entropy(&samples, fit) <= cross_entropy(&samples, t * 1.1));
            assert!(ece(&samples, fit, 10) < ece(&samples, t * 3.0, 10));
        }
    }

    #[test]
    fn the_fit_stays_in_range_and_metrics_are_plain() {
        // Always right: the loss falls as T does, so the fit sits at the lower bound.
        let sure = vec![
            Sample {
                logits: vec![0.0, 1.0],
                target: vec![0.0, 1.0],
            };
            10
        ];
        assert!((fit_temperature(&sure, [0.5, 50.0]) - 0.5).abs() < 1e-9);
        assert_eq!(accuracy(&sure), 1.0);
        // A coin: the fit sits at the upper bound, and a 50/50 answer is calibrated.
        let coin = [
            Sample {
                logits: vec![0.0, 1.0],
                target: vec![1.0, 0.0],
            },
            Sample {
                logits: vec![0.0, 1.0],
                target: vec![0.0, 1.0],
            },
        ];
        assert!((fit_temperature(&coin, [0.5, 50.0]) - 50.0).abs() < 1e-6);
        assert_eq!(accuracy(&coin), 0.5);
        assert!(ece(&coin, 1e6, 10) < 1e-6);
        let ln2 = std::f64::consts::LN_2;
        assert!((cross_entropy(&coin, 1e9) - ln2).abs() < 1e-9);
    }
}
