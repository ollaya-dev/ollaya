//! The port against laya itself, on fixtures from `python -m ollaya_convert.lang_goldens`.

use std::path::Path;

use ollaya_lang::{
    DEFAULT_MODEL, Target, detect_script, is_english, latin_profile, route, route_by_lang,
    route_with_default, state_text,
};
use serde_json::{Value, json};

/// Floats in the recorded objects may differ by this much, since serde_json's default parser can
/// read a literal one ulp off. The `floats` field pins them bit for bit.
const TOLERANCE: f64 = 1e-12;

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Equal up to [`TOLERANCE`] on floats and exactly otherwise, with object keys in the same order
/// and integers staying integers, since the `routing` field must serialize as laya's does.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) if x.is_f64() && y.is_f64() => {
            (x.as_f64().unwrap() - y.as_f64().unwrap()).abs() <= TOLERANCE
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| same(x, y))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y)
                    .all(|((kx, vx), (ky, vy))| kx == ky && same(vx, vy))
        }
        _ => a == b,
    }
}

/// The floats of a detection result, in key order.
fn floats(value: &Value, out: &mut Vec<f64>) {
    match value {
        Value::Object(map) => map.values().for_each(|v| floats(v, out)),
        Value::Number(n) if n.is_f64() => out.extend(n.as_f64()),
        _ => {}
    }
}

#[test]
fn every_state_analyses_and_routes_as_laya_does() {
    let mut failures = Vec::new();
    let mut states = 0;
    for line in fixture("route.jsonl").lines() {
        let rec: Value = serde_json::from_str(line).unwrap();
        let (id, state) = (&rec["id"], &rec["state"]);
        states += 1;

        let mut check = |what: &str, got: Value, want: &Value| {
            if !same(&got, want) {
                failures.push(format!("{id} {what}:\n  got  {got}\n  want {want}"));
            }
        };
        let want = &rec["analysis"];
        let text = state_text(state);
        let latin = json!(latin_profile(&text));
        check("script", json!(detect_script(&text)), &want["script"]);
        check("is_english", json!(is_english(state)), &want["is_english"]);
        check("latin", latin.clone(), &rec["latin"]);

        let r = route(state);
        let analysis = json!(r.analysis);
        check("analysis", analysis.clone(), want);

        // Every float bit for bit, against Python's repr of it.
        let mut got = Vec::new();
        floats(&analysis, &mut got);
        floats(&latin, &mut got);
        let python = rec["floats"].as_array().unwrap().iter();
        let python: Vec<f64> = python
            .map(|x| x.as_str().unwrap().parse().unwrap())
            .collect();
        let bits = |xs: &[f64]| json!(xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        check("floats", bits(&got), &bits(&python));

        check(
            "route",
            json!({"model": r.target.model(DEFAULT_MODEL), "reason": r.reason}),
            &rec["route"],
        );
        // `route_ml` is only recorded where the default changes the decision.
        let r = route_with_default(state, "multilingual");
        check(
            "route_ml",
            json!({"model": r.target.model("multilingual"), "reason": r.reason}),
            rec.get("route_ml").unwrap_or(&rec["route"]),
        );
    }
    assert!(states >= 1000, "only {states} states in the fixture");
    assert!(
        failures.is_empty(),
        "{} mismatches over {states} states:\n{}",
        failures.len(),
        failures[..failures.len().min(20)].join("\n")
    );
}

#[test]
fn language_codes_route_as_laya_does() {
    for line in fixture("lang_code.jsonl").lines() {
        let rec: Value = serde_json::from_str(line).unwrap();
        let code = rec["code"].as_str().unwrap();
        let want = rec["english"].as_bool().map(|english| match english {
            true => Target::English,
            false => Target::Multilingual,
        });
        assert_eq!(route_by_lang(code), want, "{code:?}");
    }
}
