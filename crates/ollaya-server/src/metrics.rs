//! `GET /metrics`: the Prometheus text format, written out by hand for a handful of series.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Mutex;
use std::time::Duration;

use crate::scheduler::RunningInfo;

/// Upper bounds in seconds, from a warm small model to a slow load.
const BUCKETS: [f64; 13] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 120.0,
];

#[derive(Default)]
pub struct Metrics(Mutex<Series>);

#[derive(Default)]
struct Series {
    /// Decisions past model resolution and not yet answered, by model.
    in_flight: BTreeMap<String, u64>,
    /// Of those, the ones handed to the runner, which runs one forward pass at a time.
    forwarded: BTreeMap<String, u64>,
    decisions: BTreeMap<(String, String), u64>,
    eval: BTreeMap<String, Histogram>,
    load: BTreeMap<String, Histogram>,
}

#[derive(Default)]
struct Histogram {
    counts: [u64; BUCKETS.len()],
    count: u64,
    sum: f64,
}

impl Histogram {
    fn observe(&mut self, value: Duration) {
        let seconds = value.as_secs_f64();
        for (count, bound) in self.counts.iter_mut().zip(BUCKETS) {
            if seconds <= bound {
                *count += 1;
            }
        }
        self.count += 1;
        self.sum += seconds;
    }
}

/// Decrements its gauge when the request leaves that stage, however it leaves.
pub struct Stage<'a> {
    metrics: &'a Metrics,
    model: String,
    forwarded: bool,
}

impl Drop for Stage<'_> {
    fn drop(&mut self) {
        let mut s = self.metrics.0.lock().unwrap();
        let gauge = if self.forwarded {
            &mut s.forwarded
        } else {
            &mut s.in_flight
        };
        if let Some(n) = gauge.get_mut(&self.model) {
            *n = n.saturating_sub(1);
        }
    }
}

impl Metrics {
    fn stage(&self, model: &str, forwarded: bool) -> Stage<'_> {
        let mut s = self.0.lock().unwrap();
        let gauge = if forwarded {
            &mut s.forwarded
        } else {
            &mut s.in_flight
        };
        *gauge.entry(model.to_owned()).or_default() += 1;
        Stage {
            metrics: self,
            model: model.to_owned(),
            forwarded,
        }
    }

    /// A decision for `model` is waiting for a load, a slot or the runner, or running.
    pub fn in_flight(&self, model: &str) -> Stage<'_> {
        self.stage(model, false)
    }

    /// The decision is with the runner: running, or queued behind the pass that is.
    pub fn forwarded(&self, model: &str) -> Stage<'_> {
        self.stage(model, true)
    }

    pub fn timings(&self, model: &str, load: Duration, eval: Duration) {
        let mut s = self.0.lock().unwrap();
        s.load.entry(model.to_owned()).or_default().observe(load);
        s.eval.entry(model.to_owned()).or_default().observe(eval);
    }

    /// `code` is the error code, or `OK` for an answer.
    pub fn decided(&self, model: &str, code: &str) {
        let mut s = self.0.lock().unwrap();
        *s.decisions
            .entry((model.to_owned(), code.to_owned()))
            .or_default() += 1;
    }

    /// Each metric family as one group: its `# HELP` and `# TYPE`, then its samples.
    pub fn render(&self, running: &[RunningInfo]) -> String {
        let s = self.0.lock().unwrap();
        let mut out = String::new();
        family(
            &mut out,
            "ollaya_requests_in_flight",
            "gauge",
            "Decision requests waiting or running, by model.",
        );
        for (model, n) in &s.in_flight {
            let _ = writeln!(
                out,
                "ollaya_requests_in_flight{{model=\"{}\"}} {n}",
                label(model)
            );
        }
        family(
            &mut out,
            "ollaya_requests_queued",
            "gauge",
            "Decision requests waiting: for a load, a slot, or the runner's current forward pass.",
        );
        for (model, n) in &s.in_flight {
            let running = u64::from(s.forwarded.get(model).is_some_and(|f| *f > 0));
            let _ = writeln!(
                out,
                "ollaya_requests_queued{{model=\"{}\"}} {}",
                label(model),
                n.saturating_sub(running)
            );
        }
        family(
            &mut out,
            "ollaya_model_loaded",
            "gauge",
            "1 for each loaded model, with the device it runs on.",
        );
        for r in running {
            let _ = writeln!(
                out,
                "ollaya_model_loaded{{model=\"{}\",device=\"{}\"}} 1",
                label(&r.name),
                label(&r.device)
            );
        }
        family(
            &mut out,
            "ollaya_model_memory_bytes",
            "gauge",
            "Size of each loaded model.",
        );
        for r in running {
            let _ = writeln!(
                out,
                "ollaya_model_memory_bytes{{model=\"{}\"}} {}",
                label(&r.name),
                r.size
            );
        }
        family(
            &mut out,
            "ollaya_decisions_total",
            "counter",
            "Decision requests by model and error code (OK for an answer).",
        );
        for ((model, code), n) in &s.decisions {
            let _ = writeln!(
                out,
                "ollaya_decisions_total{{model=\"{}\",code=\"{}\"}} {n}",
                label(model),
                label(code)
            );
        }
        for (name, help, series) in [
            (
                "ollaya_eval_duration_seconds",
                "Time in the runner per answered decision.",
                &s.eval,
            ),
            (
                "ollaya_load_duration_seconds",
                "Time an answered decision waited for its model to load.",
                &s.load,
            ),
        ] {
            family(&mut out, name, "histogram", help);
            for (model, h) in series {
                let model = label(model);
                for (count, bound) in h.counts.iter().zip(BUCKETS) {
                    let _ = writeln!(
                        out,
                        "{name}_bucket{{model=\"{model}\",le=\"{bound}\"}} {count}"
                    );
                }
                let _ = writeln!(
                    out,
                    "{name}_bucket{{model=\"{model}\",le=\"+Inf\"}} {}",
                    h.count
                );
                let _ = writeln!(out, "{name}_sum{{model=\"{model}\"}} {}", h.sum);
                let _ = writeln!(out, "{name}_count{{model=\"{model}\"}} {}", h.count);
            }
        }
        out
    }
}

fn family(out: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
}

/// A label value, escaped as the text format requires.
fn label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_depth_counts_what_waits_behind_the_running_pass() {
        let m = Metrics::default();
        let (_a, _b, _c) = (m.in_flight("m:1"), m.in_flight("m:1"), m.in_flight("m:1"));
        let running = m.forwarded("m:1");
        let _queued_in_runner = m.forwarded("m:1");
        m.timings("m:1", Duration::ZERO, Duration::from_millis(20));
        m.decided("m:1", "OK");
        m.decided("m:1", "OK");
        m.decided("x\"y", "MODEL_NOT_FOUND");
        let text = m.render(&[]);
        for line in [
            "ollaya_requests_in_flight{model=\"m:1\"} 3",
            "ollaya_requests_queued{model=\"m:1\"} 2",
            "ollaya_decisions_total{model=\"m:1\",code=\"OK\"} 2",
            "ollaya_decisions_total{model=\"x\\\"y\",code=\"MODEL_NOT_FOUND\"} 1",
            "ollaya_eval_duration_seconds_bucket{model=\"m:1\",le=\"0.01\"} 0",
            "ollaya_eval_duration_seconds_bucket{model=\"m:1\",le=\"0.025\"} 1",
            "ollaya_eval_duration_seconds_bucket{model=\"m:1\",le=\"+Inf\"} 1",
            "ollaya_load_duration_seconds_bucket{model=\"m:1\",le=\"0.005\"} 1",
        ] {
            assert!(
                text.lines().any(|l| l == line),
                "missing {line:?} in\n{text}"
            );
        }
        drop(running);
        drop(_queued_in_runner);
        assert!(
            m.render(&[])
                .contains("ollaya_requests_queued{model=\"m:1\"} 3")
        );
    }

    #[test]
    fn each_family_is_one_group() {
        let m = Metrics::default();
        let (_a, _b) = (m.in_flight("a:1"), m.in_flight("b:1"));
        m.timings("a:1", Duration::ZERO, Duration::from_millis(5));
        m.decided("a:1", "OK");
        let running = [RunningInfo {
            name: "a:1".into(),
            digest: String::new(),
            device: "cpu".into(),
            precision: String::new(),
            size: 1,
            loaded_at: std::time::UNIX_EPOCH,
            expires_at: None,
        }];
        let mut seen = Vec::new();
        let mut current = "";
        let text = m.render(&running);
        for line in text.lines() {
            if let Some(name) = line.strip_prefix("# TYPE ") {
                current = name.split(' ').next().unwrap();
                assert!(!seen.contains(&current), "{current} split in two");
                seen.push(current);
            } else if !line.starts_with('#') {
                let sample = line.split('{').next().unwrap();
                let family = ["_bucket", "_sum", "_count"]
                    .iter()
                    .find_map(|s| sample.strip_suffix(s))
                    .filter(|f| f.ends_with("_seconds"))
                    .unwrap_or(sample);
                assert_eq!(family, current, "{line} outside its family");
            }
        }
        assert_eq!(seen.len(), 7);
    }
}
