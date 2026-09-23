//! Every example in `docs/api.md` against the types and validators of this crate.
//!
//! Blocks are tagged in the document with an HTML comment:
//! * `<!-- json: Type -->`: a response body; it must round-trip through `Type` byte for byte
//!   (field order, number formatting), because the server serializes exactly these types.
//! * `<!-- ndjson: ProgressResponse -->`: one progress line or error line per line.
//! * `<!-- curl: Type [invalid] -->`: the `-d '...'` body must pass that endpoint's validator
//!   (or fail it, when marked `invalid`, and the next `ErrorBody` example must be the exact
//!   error the validator produces).

use ollaya_api::decide::{Answer, LayaExtra};
use ollaya_api::validate::{self, Body, Issues};
use ollaya_api::{
    CopyRequest, CreateRequest, DecideRequest, DecideResponse, DeleteRequest, ErrorBody, ErrorCode,
    ModelList, ProgressResponse, PsResponse, PullRequest, RouterInfo, ShowResponse,
    SystemOneRequest, SystemOneResponse, TagsResponse, VersionResponse,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

const DOC: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/api.md"));

#[derive(Debug)]
struct Example {
    kind: String,
    ty: String,
    invalid: bool,
    text: String,
    line: usize,
}

fn examples() -> Vec<Example> {
    let lines: Vec<&str> = DOC.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim();
        if let Some(inner) = t.strip_prefix("<!--").and_then(|s| s.strip_suffix("-->")) {
            let (kind, rest) = inner.trim().split_once(':').expect("marker kind");
            let mut words = rest.split_whitespace();
            let ty = words.next().expect("marker type").to_owned();
            let invalid = words.next() == Some("invalid");
            let fence = i + 1;
            assert!(
                lines[fence].trim_start().starts_with("```"),
                "line {}: marker not followed by a code fence",
                i + 1
            );
            let mut j = fence + 1;
            let mut body = Vec::new();
            while lines[j].trim() != "```" {
                body.push(lines[j]);
                j += 1;
            }
            out.push(Example {
                kind: kind.trim().to_owned(),
                ty,
                invalid,
                text: body.join("\n"),
                line: i + 1,
            });
            i = j;
        }
        i += 1;
    }
    out
}

/// The response round-trips through `T` to the same bytes.
fn exact<T: DeserializeOwned + Serialize>(text: &str, ctx: &str) -> T {
    let value: Value = serde_json::from_str(text).unwrap_or_else(|e| panic!("{ctx}: {e}"));
    let typed: T = serde_json::from_value(value.clone()).unwrap_or_else(|e| panic!("{ctx}: {e}"));
    assert_eq!(
        serde_json::to_string(&typed).unwrap(),
        serde_json::to_string(&value).unwrap(),
        "{ctx}: the example does not match what the type serializes"
    );
    typed
}

fn curl_body(ex: &Example) -> Body {
    let start = ex
        .text
        .find("-d '")
        .unwrap_or_else(|| panic!("line {}: no -d", ex.line))
        + 4;
    let end = start + ex.text[start..].find('\'').expect("closing quote");
    match serde_json::from_str(&ex.text[start..end]) {
        Ok(Value::Object(obj)) => obj,
        other => panic!("line {}: body is not a JSON object: {other:?}", ex.line),
    }
}

/// A request example: the validator's verdict, and agreement with plain serde for valid ones.
fn request<T>(ex: &Example, validator: fn(Body) -> Result<T, Issues>) -> Option<ErrorBody>
where
    T: DeserializeOwned + Serialize + PartialEq + std::fmt::Debug,
{
    let body = curl_body(ex);
    let ctx = format!("line {} ({})", ex.line, ex.ty);
    match (validator(body.clone()), ex.invalid) {
        (Ok(typed), false) => {
            let again: T = serde_json::from_value(serde_json::to_value(&typed).unwrap()).unwrap();
            assert_eq!(again, typed, "{ctx}: typed round trip");
            let direct: T = serde_json::from_value(Value::Object(body))
                .unwrap_or_else(|e| panic!("{ctx}: serde rejects a valid body: {e}"));
            assert_eq!(direct, typed, "{ctx}: serde and the validator disagree");
            None
        }
        (Err(issues), true) => Some(ErrorBody::invalid_request(issues)),
        (Ok(_), true) => panic!("{ctx}: marked invalid but passes validation"),
        (Err(issues), false) => panic!("{ctx}: fails validation: {issues:#?}"),
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// Numbers in an answer agree with the contract's formulas (4-decimal rounding allowed).
fn check_answer(a: &Answer, laya: Option<&LayaExtra>, ctx: &str) {
    let normalized = |p: &[f64]| {
        let k = p.len() as f64;
        let pmax = p.iter().copied().fold(f64::MIN, f64::max);
        ((k * pmax - 1.0) / (k - 1.0)).clamp(0.0, 1.0)
    };
    let entropy = |p: &[f64]| {
        let h: f64 = -p.iter().map(|&x| x * x.max(1e-12).ln()).sum::<f64>();
        1.0 - h / (p.len() as f64).ln()
    };
    let probs = match a {
        Answer::Choice(c) => {
            let p: Vec<f64> = c.probabilities.values().copied().collect();
            let best = c
                .probabilities
                .iter()
                .max_by(|x, y| x.1.total_cmp(y.1))
                .unwrap();
            assert_eq!(&c.choice, best.0, "{ctx}: choice is not the argmax");
            assert!(
                close(c.confidence, normalized(&p), 1e-3),
                "{ctx}: confidence"
            );
            p
        }
        Answer::Score(s) => {
            let p: Vec<f64> = s.probabilities.values().copied().collect();
            let keys: Vec<String> = (0..p.len()).map(|i| i.to_string()).collect();
            assert!(
                s.probabilities.keys().eq(keys.iter()),
                "{ctx}: probability keys"
            );
            assert!(s.legend.keys().eq(keys.iter()), "{ctx}: legend keys");
            let expected: f64 = p.iter().enumerate().map(|(i, x)| i as f64 * x).sum();
            assert!(close(s.score, expected, 1e-3), "{ctx}: score");
            assert!(
                close(s.confidence, normalized(&p), 1e-3),
                "{ctx}: confidence"
            );
            p
        }
        Answer::Noul(n) => {
            assert!((0.0..=1.0).contains(&n.noul), "{ctx}: noul");
            vec![1.0 - n.noul, n.noul]
        }
    };
    if !matches!(a, Answer::Noul(_)) {
        assert!(
            close(probs.iter().sum(), 1.0, 1e-3),
            "{ctx}: probabilities sum"
        );
    }
    if let Some(l) = laya {
        let want = match a {
            Answer::Noul(n) => n.noul.max(1.0 - n.noul),
            _ => entropy(&probs),
        };
        assert!(close(l.confidence, want, 2e-3), "{ctx}: laya confidence");
    }
}

#[test]
fn every_example_matches_the_contract() {
    let examples = examples();
    let mut pending: Option<(usize, ErrorBody)> = None;
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for ex in &examples {
        *counts.entry(ex.kind.as_str()).or_default() += 1;
        let ctx = format!("docs/api.md line {} ({}: {})", ex.line, ex.kind, ex.ty);
        match (ex.kind.as_str(), ex.ty.as_str()) {
            ("json", "ErrorBody") => {
                let body: ErrorBody = exact(&ex.text, &ctx);
                assert_eq!(
                    body.code.has_detail(),
                    body.detail.is_some(),
                    "{ctx}: detail"
                );
                if let Some(detail) = &body.detail {
                    assert_eq!(
                        body.error,
                        ollaya_api::error::issues_message(detail),
                        "{ctx}: error is not built from detail"
                    );
                }
                if let Some((line, expected)) = pending.take() {
                    assert_eq!(
                        serde_json::to_string(&body).unwrap(),
                        serde_json::to_string(&expected).unwrap(),
                        "{ctx}: not what the validator returns for the request on line {line}"
                    );
                }
            }
            ("json", "DecideResponse") => {
                let r: DecideResponse = exact(&ex.text, &ctx);
                for (id, a) in &r.answers {
                    check_answer(&a.answer, a.laya.as_ref(), &format!("{ctx} {id}"));
                }
                // Additive: a TypeSafe client reads the same text as a SystemOneResponse.
                let as_typesafe: SystemOneResponse = serde_json::from_str(&ex.text).unwrap();
                assert_eq!(as_typesafe, r.clone().into_system_one(), "{ctx}");
                if let Some(routing) = &r.routing {
                    assert_eq!(routing.model, r.model, "{ctx}: routing.model");
                }
            }
            ("json", "SystemOneResponse") => {
                let r: SystemOneResponse = exact(&ex.text, &ctx);
                for (id, a) in &r.answers {
                    check_answer(a, None, &format!("{ctx} {id}"));
                }
            }
            ("json", "VersionResponse") => drop(exact::<VersionResponse>(&ex.text, &ctx)),
            ("json", "TagsResponse") => drop(exact::<TagsResponse>(&ex.text, &ctx)),
            ("json", "ShowResponse") => drop(exact::<ShowResponse>(&ex.text, &ctx)),
            ("json", "RouterInfo") => drop(exact::<RouterInfo>(&ex.text, &ctx)),
            ("json", "PsResponse") => drop(exact::<PsResponse>(&ex.text, &ctx)),
            ("json", "ModelList") => drop(exact::<ModelList>(&ex.text, &ctx)),
            ("json", "ProgressResponse") => drop(exact::<ProgressResponse>(&ex.text, &ctx)),
            ("ndjson", "ProgressResponse") => {
                let lines: Vec<&str> = ex.text.lines().collect();
                for (n, line) in lines.iter().enumerate() {
                    let ctx = format!("{ctx}, stream line {}", n + 1);
                    let value: Value = serde_json::from_str(line).unwrap();
                    let last = n + 1 == lines.len();
                    if value.get("error").is_some() {
                        exact::<ErrorBody>(line, &ctx);
                        assert!(last, "{ctx}: an error line ends the stream");
                    } else {
                        let p: ProgressResponse = exact(line, &ctx);
                        assert_eq!(p.is_success(), last, "{ctx}: success is the last line");
                    }
                }
            }
            ("curl", ty) => {
                let err = match ty {
                    "DecideRequest" => request::<DecideRequest>(ex, validate::decide_request),
                    "SystemOneRequest" => {
                        request::<SystemOneRequest>(ex, validate::system_one_request)
                    }
                    "PullRequest" => request::<PullRequest>(ex, validate::pull_request),
                    "DeleteRequest" => request::<DeleteRequest>(ex, validate::delete_request),
                    "CopyRequest" => request::<CopyRequest>(ex, validate::copy_request),
                    "CreateRequest" => request::<CreateRequest>(ex, validate::create_request),
                    other => panic!("{ctx}: unknown request type {other}"),
                };
                if let Some(e) = err {
                    pending = Some((ex.line, e));
                }
            }
            (kind, ty) => panic!("{ctx}: unknown example tag {kind}: {ty}"),
        }
    }
    assert!(
        pending.is_none(),
        "an invalid request has no ErrorBody example after it"
    );
    // Guard against markers silently disappearing.
    assert!(counts.get("json").copied().unwrap_or(0) >= 16, "{counts:?}");
    assert!(
        counts.get("ndjson").copied().unwrap_or(0) >= 3,
        "{counts:?}"
    );
    assert!(counts.get("curl").copied().unwrap_or(0) >= 10, "{counts:?}");
}

#[test]
fn every_json_block_is_tagged() {
    let lines: Vec<&str> = DOC.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if line.trim() == "```json" {
            let prev = lines[..i]
                .iter()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap();
            assert!(
                prev.trim().starts_with("<!--"),
                "docs/api.md line {}: untagged JSON example",
                i + 1
            );
        }
    }
}

/// The error constructors the daemon uses produce the documented bodies.
#[test]
fn documented_errors_come_from_the_constructors() {
    let bodies: Vec<ErrorBody> = examples()
        .iter()
        .flat_map(|ex| match (ex.kind.as_str(), ex.ty.as_str()) {
            ("json", "ErrorBody") => vec![serde_json::from_str(&ex.text).unwrap()],
            ("ndjson", _) => ex
                .text
                .lines()
                .filter(|l| l.contains("\"error\""))
                .map(|l| serde_json::from_str(l).unwrap())
                .collect(),
            _ => vec![],
        })
        .collect();
    let find = |code: ErrorCode| {
        bodies
            .iter()
            .find(|b| b.code == code)
            .unwrap_or_else(|| panic!("no {code} example"))
            .clone()
    };
    assert_eq!(
        find(ErrorCode::ModelNotFound),
        ErrorBody::model_not_found("laya:xl")
    );
    assert_eq!(
        find(ErrorCode::TooManyOptions),
        ErrorBody::too_many_options("intent", "choice", 140, "laya:en")
    );
    assert_eq!(
        find(ErrorCode::DigestMismatch),
        ErrorBody::digest_mismatch(
            "sha256:8d32a80bb199bcd4ff10abc28d651fe576fb59f86b24039402e49be9e01578c2"
        )
    );
    // Ollama tools match this sentence.
    assert!(
        ErrorBody::model_not_found("x:latest")
            .error
            .ends_with("not found, try pulling it first")
    );
}

/// `docs/api.md` §4.2 lists exactly the codes of `ErrorCode::ALL`, with the same statuses.
#[test]
fn error_code_table_matches_the_enum() {
    let section = DOC
        .split("### 4.2 Error codes")
        .nth(1)
        .unwrap()
        .split("###")
        .next()
        .unwrap();
    let rows: Vec<(String, u16)> = section
        .lines()
        .filter_map(|l| {
            let cells: Vec<&str> = l.split('|').map(str::trim).collect();
            let code = cells.get(1)?.strip_prefix('`')?.strip_suffix('`')?;
            Some((code.to_owned(), cells.get(2)?.parse().ok()?))
        })
        .collect();
    let expected: Vec<(String, u16)> = ErrorCode::ALL
        .iter()
        .map(|c| (c.as_str().to_owned(), c.status()))
        .collect();
    assert_eq!(rows, expected);
}
