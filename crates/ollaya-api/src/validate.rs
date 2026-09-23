//! Boundary validation (`docs/api.md` §4.4, §5).
//!
//! Handlers parse the raw body with [`parse_body`] (`400 INVALID_JSON`, `413 REQUEST_TOO_LARGE`)
//! and turn it into a typed request with one of the per-endpoint functions. These collect *every*
//! problem as a [`ValidationIssue`], which becomes a `422 INVALID_REQUEST` body with
//! [`ErrorBody::invalid_request`]. Past this point internal code trusts the types.
//!
//! Two checks stay with the daemon because they need its state:
//! * model-name grammar (`ollaya-registry` owns it): report failures with
//!   [`ValidationIssue::model_name`];
//! * `questions` absent for a model without embedded questions: report
//!   [`ValidationIssue::missing`] at `["body", "questions"]`.
//!
//! Unknown fields are ignored everywhere; `null` means absent.

use serde_json::{Map, Value};

use crate::decide::{
    ChoiceCriteria, ChoiceQuestion, DecideRequest, Extra, NoulCriteria, NoulQuestion, Question,
    Questions, ScoreQuestion, SystemOneRequest,
};
use crate::error::{ErrorBody, Loc, ValidationIssue, body_loc};
use crate::keep_alive::KeepAlive;
use crate::models::{
    CalibrationSpec, CopyRequest, CreateParameters, CreateRequest, DeleteRequest, License,
    PullRequest, ShowRequest,
};
use crate::{
    MAX_BODY_BYTES, MAX_CHOICE_OPTIONS, MAX_QUESTIONS, MAX_SCORE_LEVELS, MIN_CHOICE_OPTIONS,
    MIN_QUESTIONS, MIN_SCORE_LEVELS,
};

/// A request body: always a JSON object.
pub type Body = Map<String, Value>;
pub type Issues = Vec<ValidationIssue>;

/// Parse a raw request body. The `Content-Type` is not consulted (Ollama behaviour).
pub fn parse_body(bytes: &[u8]) -> Result<Body, ErrorBody> {
    if bytes.len() > MAX_BODY_BYTES {
        return Err(ErrorBody::request_too_large());
    }
    if bytes.iter().all(u8::is_ascii_whitespace) {
        return Err(ErrorBody::invalid_json("missing request body"));
    }
    match serde_json::from_slice::<Value>(bytes) {
        Ok(Value::Object(obj)) => Ok(obj),
        Ok(_) => Err(ErrorBody::invalid_json(
            "request body must be a JSON object",
        )),
        Err(e) => Err(ErrorBody::invalid_json(format!("invalid JSON: {e}"))),
    }
}

/// [`parse_body`] then `validate`, with issues turned into a `422` body.
pub fn body<T>(
    bytes: &[u8],
    validate: impl FnOnce(Body) -> Result<T, Issues>,
) -> Result<T, ErrorBody> {
    validate(parse_body(bytes)?).map_err(ErrorBody::invalid_request)
}

/// `POST /api/decide`.
pub fn decide_request(mut obj: Body) -> Result<DecideRequest, Issues> {
    let mut issues = Issues::new();
    let model = required_name(&obj, "model", &mut issues);
    let has_state = check_state(&obj, false, &mut issues);
    let questions = match get(&obj, "questions") {
        Some(v) => {
            if !has_state {
                issues.push(ValidationIssue::missing(body_loc(&["state"])));
            }
            questions(v, body_loc(&["questions"]), &mut issues)
        }
        None => None,
    };
    let keep_alive = match get(&obj, "keep_alive") {
        None => None,
        Some(v) => match KeepAlive::from_json(v) {
            Ok(k) => k,
            Err(e) => {
                issues.push(ValidationIssue::keep_alive(
                    body_loc(&["keep_alive"]),
                    e.to_string(),
                ));
                None
            }
        },
    };
    let extras = extras(get(&obj, "extras"), &mut issues);
    match get(&obj, "stream") {
        None | Some(Value::Bool(false)) => {}
        Some(Value::Bool(true)) => {
            issues.push(ValidationIssue::stream_unsupported(body_loc(&["stream"])))
        }
        Some(_) => issues.push(ValidationIssue::wrong_type(
            body_loc(&["stream"]),
            "bool_type",
        )),
    }
    let (Some(model), true) = (model, issues.is_empty()) else {
        return Err(issues);
    };
    Ok(DecideRequest {
        model,
        state: take(&mut obj, "state"),
        questions,
        keep_alive,
        extras,
    })
}

/// `POST /v1/systemone` and `POST /v1/decisions`. Native fields are ignored.
pub fn system_one_request(mut obj: Body) -> Result<SystemOneRequest, Issues> {
    let mut issues = Issues::new();
    let model = required_name(&obj, "model", &mut issues);
    check_state(&obj, true, &mut issues);
    let questions =
        get(&obj, "questions").and_then(|v| questions(v, body_loc(&["questions"]), &mut issues));
    let (Some(model), true) = (model, issues.is_empty()) else {
        return Err(issues);
    };
    Ok(SystemOneRequest {
        model,
        state: take(&mut obj, "state").unwrap_or(Value::Null),
        questions,
    })
}

/// `POST /api/show`.
pub fn show_request(obj: Body) -> Result<ShowRequest, Issues> {
    let mut issues = Issues::new();
    match required_name(&obj, "model", &mut issues) {
        Some(model) => Ok(ShowRequest { model }),
        None => Err(issues),
    }
}

/// `DELETE /api/delete`.
pub fn delete_request(obj: Body) -> Result<DeleteRequest, Issues> {
    let mut issues = Issues::new();
    match required_name(&obj, "model", &mut issues) {
        Some(model) => Ok(DeleteRequest { model }),
        None => Err(issues),
    }
}

/// `POST /api/copy`.
pub fn copy_request(obj: Body) -> Result<CopyRequest, Issues> {
    let mut issues = Issues::new();
    let source = required_name(&obj, "source", &mut issues);
    let destination = required_name(&obj, "destination", &mut issues);
    match (source, destination) {
        (Some(source), Some(destination)) => Ok(CopyRequest {
            source,
            destination,
        }),
        _ => Err(issues),
    }
}

/// `POST /api/pull`.
pub fn pull_request(obj: Body) -> Result<PullRequest, Issues> {
    let mut issues = Issues::new();
    let model = required_name(&obj, "model", &mut issues);
    let insecure = optional_bool(&obj, "insecure", &mut issues).unwrap_or(false);
    let stream = optional_bool(&obj, "stream", &mut issues);
    match (model, issues.is_empty()) {
        (Some(model), true) => Ok(PullRequest {
            model,
            insecure,
            stream,
        }),
        _ => Err(issues),
    }
}

/// `POST /api/create`.
pub fn create_request(obj: Body) -> Result<CreateRequest, Issues> {
    let mut issues = Issues::new();
    let model = required_name(&obj, "model", &mut issues);
    let from = required_name(&obj, "from", &mut issues);
    let questions =
        get(&obj, "questions").and_then(|v| questions(v, body_loc(&["questions"]), &mut issues));
    let calibration = get(&obj, "calibration").and_then(|v| calibration(v, &mut issues));
    let parameters = get(&obj, "parameters").and_then(|v| parameters(v, &mut issues));
    let license = get(&obj, "license").and_then(|v| license(v, &mut issues));
    let description = match get(&obj, "description") {
        None => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => {
            issues.push(ValidationIssue::wrong_type(
                body_loc(&["description"]),
                "string_type",
            ));
            None
        }
    };
    let stream = optional_bool(&obj, "stream", &mut issues);
    match (model, from, issues.is_empty()) {
        (Some(model), Some(from), true) => Ok(CreateRequest {
            model,
            from,
            questions,
            calibration,
            parameters,
            license,
            description,
            stream,
        }),
        _ => Err(issues),
    }
}

/// A `questions` object: 1–256 questions, each valid.
pub fn questions(value: &Value, base: Vec<Loc>, issues: &mut Issues) -> Option<Questions> {
    let Value::Object(map) = value else {
        issues.push(ValidationIssue::wrong_type(base, "dict_type"));
        return None;
    };
    let before = issues.len();
    if map.len() < MIN_QUESTIONS {
        issues.push(ValidationIssue::too_short(
            base.clone(),
            "Dictionary",
            MIN_QUESTIONS,
            map.len(),
        ));
    } else if map.len() > MAX_QUESTIONS {
        issues.push(ValidationIssue::too_long(
            base.clone(),
            "Dictionary",
            MAX_QUESTIONS,
            map.len(),
        ));
    }
    let mut out = Questions::with_capacity(map.len());
    for (id, def) in map {
        if let Some(q) = question(id, def, &base, issues) {
            out.insert(id.clone(), q);
        }
    }
    (issues.len() == before).then_some(out)
}

fn question(id: &str, def: &Value, base: &[Loc], issues: &mut Issues) -> Option<Question> {
    let qloc = at(base, [Loc::key(id)]);
    let Value::Object(q) = def else {
        issues.push(ValidationIssue::wrong_type(qloc, "dict_type"));
        return None;
    };
    let tag = match q.get("type") {
        Some(Value::String(t)) => t.as_str(),
        _ => {
            issues.push(ValidationIssue::union_tag_not_found(qloc));
            return None;
        }
    };
    if !matches!(tag, "choice" | "score" | "noul") {
        issues.push(ValidationIssue::union_tag_invalid(qloc, tag));
        return None;
    }
    let tloc = at(&qloc, [Loc::key(tag)]);
    let before = issues.len();
    let instructions = match q.get("instructions") {
        None | Some(Value::Null) => None,
        Some(v) if is_json_content(v) => Some(v.clone()),
        Some(_) => {
            issues.push(ValidationIssue::json_type(
                at(&tloc, ["instructions".into()]),
                true,
            ));
            None
        }
    };
    let cloc = at(&tloc, ["criteria".into()]);
    let criteria = q.get("criteria").filter(|c| !c.is_null());
    let question = match tag {
        "choice" => choice_criteria(criteria, &cloc, issues).map(|criteria| {
            Question::Choice(ChoiceQuestion {
                instructions,
                criteria,
            })
        }),
        "score" => score_criteria(criteria, &cloc, issues).map(|criteria| {
            Question::Score(ScoreQuestion {
                instructions,
                criteria,
            })
        }),
        _ => noul_criteria(criteria, &cloc, issues).map(|criteria| {
            Question::Noul(NoulQuestion {
                instructions,
                criteria,
            })
        }),
    };
    if issues.len() == before {
        question
    } else {
        None
    }
}

fn choice_criteria(
    criteria: Option<&Value>,
    cloc: &[Loc],
    issues: &mut Issues,
) -> Option<ChoiceCriteria> {
    match criteria {
        None => {
            issues.push(ValidationIssue::missing(cloc.to_vec()));
            None
        }
        Some(Value::Object(m)) => {
            count(
                cloc,
                "Dictionary",
                m.len(),
                MIN_CHOICE_OPTIONS,
                MAX_CHOICE_OPTIONS,
                issues,
            );
            for (label, description) in m {
                if !description.is_null() && !is_json_content(description) {
                    issues.push(ValidationIssue::json_type(
                        at(cloc, [Loc::key(label.as_str())]),
                        true,
                    ));
                }
            }
            Some(ChoiceCriteria::Map(
                m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            ))
        }
        Some(Value::Array(items)) => {
            let mut labels = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                match item {
                    Value::String(s) => labels.push(s.clone()),
                    _ => issues.push(ValidationIssue::wrong_type(
                        at(cloc, [Loc::from(i)]),
                        "string_type",
                    )),
                }
            }
            let criteria = ChoiceCriteria::Labels(labels);
            let distinct = criteria.labels().len();
            count(
                cloc,
                "List",
                distinct,
                MIN_CHOICE_OPTIONS,
                MAX_CHOICE_OPTIONS,
                issues,
            );
            Some(criteria)
        }
        Some(_) => {
            issues.push(ValidationIssue::wrong_type(cloc.to_vec(), "dict_type"));
            None
        }
    }
}

fn score_criteria(
    criteria: Option<&Value>,
    cloc: &[Loc],
    issues: &mut Issues,
) -> Option<Vec<Value>> {
    match criteria {
        None => {
            issues.push(ValidationIssue::missing(cloc.to_vec()));
            None
        }
        Some(Value::Array(levels)) => {
            count(
                cloc,
                "List",
                levels.len(),
                MIN_SCORE_LEVELS,
                MAX_SCORE_LEVELS,
                issues,
            );
            for (i, level) in levels.iter().enumerate() {
                if !is_json_content(level) {
                    issues.push(ValidationIssue::json_type(at(cloc, [Loc::from(i)]), false));
                }
            }
            Some(levels.clone())
        }
        Some(_) => {
            issues.push(ValidationIssue::wrong_type(cloc.to_vec(), "list_type"));
            None
        }
    }
}

fn noul_criteria(
    criteria: Option<&Value>,
    cloc: &[Loc],
    issues: &mut Issues,
) -> Option<Option<NoulCriteria>> {
    match criteria {
        None => Some(None),
        Some(Value::Object(m)) => {
            let mut side = |key: &str| match m.get(key) {
                None | Some(Value::Null) => None,
                Some(v) if is_json_content(v) => Some(v.clone()),
                Some(_) => {
                    issues.push(ValidationIssue::json_type(at(cloc, [Loc::key(key)]), true));
                    None
                }
            };
            let when_true = side("true");
            let when_false = side("false");
            Some(Some(NoulCriteria {
                when_true,
                when_false,
            }))
        }
        Some(_) => {
            issues.push(ValidationIssue::wrong_type(cloc.to_vec(), "dict_type"));
            None
        }
    }
}

fn extras(value: Option<&Value>, issues: &mut Issues) -> Vec<Extra> {
    let base = body_loc(&["extras"]);
    let Some(value) = value else {
        return Vec::new();
    };
    let Value::Array(items) = value else {
        issues.push(ValidationIssue::wrong_type(base, "list_type"));
        return Vec::new();
    };
    let expected = Extra::ALL
        .iter()
        .map(|e| format!("'{}'", e.as_str()))
        .collect::<Vec<_>>()
        .join(", ");
    let mut out = Vec::new();
    for (i, item) in items.iter().enumerate() {
        match item {
            Value::String(s) => match Extra::parse(s) {
                Some(e) if !out.contains(&e) => out.push(e),
                Some(_) => {}
                None => issues.push(ValidationIssue::enumeration(
                    at(&base, [Loc::from(i)]),
                    &expected,
                )),
            },
            _ => issues.push(ValidationIssue::wrong_type(
                at(&base, [Loc::from(i)]),
                "string_type",
            )),
        }
    }
    out
}

fn calibration(value: &Value, issues: &mut Issues) -> Option<CalibrationSpec> {
    let base = body_loc(&["calibration"]);
    let Value::Object(m) = value else {
        issues.push(ValidationIssue::wrong_type(base, "dict_type"));
        return None;
    };
    let before = issues.len();
    let mut spec = CalibrationSpec::default();
    match m.get("temperature").filter(|v| !v.is_null()) {
        None => {}
        Some(Value::Array(ts)) => {
            let tloc = at(&base, ["temperature".into()]);
            if ts.len() > 3 {
                issues.push(ValidationIssue::too_long(tloc.clone(), "List", 3, ts.len()));
            }
            for (i, t) in ts.iter().enumerate() {
                match t.as_f64().filter(|t| t.is_finite()) {
                    Some(t) => spec.temperature.push(t),
                    None => issues.push(ValidationIssue::wrong_type(
                        at(&tloc, [Loc::from(i)]),
                        "float_type",
                    )),
                }
            }
        }
        Some(_) => issues.push(ValidationIssue::wrong_type(
            at(&base, ["temperature".into()]),
            "list_type",
        )),
    }
    match m.get("temperature_by_options").filter(|v| !v.is_null()) {
        None => {}
        Some(Value::Object(buckets)) => {
            let bloc = at(&base, ["temperature_by_options".into()]);
            for (key, t) in buckets {
                let kloc = at(&bloc, [Loc::key(key.as_str())]);
                if !valid_bucket(key) {
                    issues.push(ValidationIssue::calibration(
                        kloc,
                        "keys are \"<choice|score|noul>:<2|3-5|6-10|11+>\"",
                    ));
                    continue;
                }
                match t.as_f64().filter(|t| t.is_finite()) {
                    Some(t) => {
                        spec.temperature_by_options.insert(key.clone(), t);
                    }
                    None => issues.push(ValidationIssue::wrong_type(kloc, "float_type")),
                }
            }
        }
        Some(_) => issues.push(ValidationIssue::wrong_type(
            at(&base, ["temperature_by_options".into()]),
            "dict_type",
        )),
    }
    (issues.len() == before).then_some(spec)
}

fn valid_bucket(key: &str) -> bool {
    matches!(
        key.split_once(':'),
        Some(("choice" | "score" | "noul", "2" | "3-5" | "6-10" | "11+"))
    )
}

fn parameters(value: &Value, issues: &mut Issues) -> Option<CreateParameters> {
    let base = body_loc(&["parameters"]);
    let Value::Object(m) = value else {
        issues.push(ValidationIssue::wrong_type(base, "dict_type"));
        return None;
    };
    let before = issues.len();
    let mut params = CreateParameters::default();
    for (key, v) in m {
        let ploc = at(&base, [Loc::key(key.as_str())]);
        match (key.as_str(), v.as_str()) {
            ("precision", Some(p @ ("fp16" | "fp32"))) => params.precision = Some(p.to_owned()),
            ("precision", _) => issues.push(ValidationIssue::parameter(
                ploc,
                "precision must be \"fp16\" or \"fp32\"",
            )),
            _ => issues.push(ValidationIssue::parameter(
                ploc,
                format!("unknown parameter {key:?}; known parameters: precision"),
            )),
        }
    }
    (issues.len() == before).then_some(params)
}

fn license(value: &Value, issues: &mut Issues) -> Option<License> {
    let base = body_loc(&["license"]);
    match value {
        Value::String(s) => Some(License::One(s.clone())),
        Value::Array(items) => {
            let before = issues.len();
            let texts: Vec<String> = items
                .iter()
                .enumerate()
                .filter_map(|(i, t)| match t {
                    Value::String(s) => Some(s.clone()),
                    _ => {
                        issues.push(ValidationIssue::wrong_type(
                            at(&base, [Loc::from(i)]),
                            "string_type",
                        ));
                        None
                    }
                })
                .collect();
            (issues.len() == before).then_some(License::Many(texts))
        }
        _ => {
            issues.push(ValidationIssue::wrong_type(base, "string_type"));
            None
        }
    }
}

/// Checks `state`; returns whether one is present. `required` is the `/v1/*` rule.
fn check_state(obj: &Body, required: bool, issues: &mut Issues) -> bool {
    match get(obj, "state") {
        None => {
            if required {
                issues.push(ValidationIssue::missing(body_loc(&["state"])));
            }
            false
        }
        Some(v) if is_json_content(v) => true,
        Some(_) => {
            issues.push(ValidationIssue::state_type(body_loc(&["state"])));
            true
        }
    }
}

fn required_name(obj: &Body, key: &str, issues: &mut Issues) -> Option<String> {
    let loc = body_loc(&[key]);
    match get(obj, key) {
        None => {
            issues.push(ValidationIssue::missing(loc));
            None
        }
        Some(Value::String(s)) if s.trim().is_empty() => {
            issues.push(ValidationIssue::string_too_short(loc));
            None
        }
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => {
            issues.push(ValidationIssue::wrong_type(loc, "string_type"));
            None
        }
    }
}

fn optional_bool(obj: &Body, key: &str, issues: &mut Issues) -> Option<bool> {
    match get(obj, key) {
        None => None,
        Some(Value::Bool(b)) => Some(*b),
        Some(_) => {
            issues.push(ValidationIssue::wrong_type(body_loc(&[key]), "bool_type"));
            None
        }
    }
}

fn count(loc: &[Loc], field_type: &str, n: usize, min: usize, max: usize, issues: &mut Issues) {
    if n < min {
        issues.push(ValidationIssue::too_short(loc.to_vec(), field_type, min, n));
    } else if n > max {
        issues.push(ValidationIssue::too_long(loc.to_vec(), field_type, max, n));
    }
}

/// TypeSafe's `str | dict | list`.
fn is_json_content(v: &Value) -> bool {
    matches!(v, Value::String(_) | Value::Object(_) | Value::Array(_))
}

/// A field, with `null` read as absent.
fn get<'a>(obj: &'a Body, key: &str) -> Option<&'a Value> {
    obj.get(key).filter(|v| !v.is_null())
}

fn take(obj: &mut Body, key: &str) -> Option<Value> {
    obj.remove(key).filter(|v| !v.is_null())
}

fn at(base: &[Loc], more: impl IntoIterator<Item = Loc>) -> Vec<Loc> {
    base.iter().cloned().chain(more).collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::ErrorCode;

    fn obj(v: Value) -> Body {
        match v {
            Value::Object(m) => m,
            other => panic!("{other}"),
        }
    }

    fn kinds(issues: &[ValidationIssue]) -> Vec<(String, String)> {
        issues.iter().map(|i| (i.path(), i.kind.clone())).collect()
    }

    fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    #[test]
    fn parse_body_errors() {
        assert_eq!(parse_body(b"").unwrap_err().code, ErrorCode::InvalidJson);
        assert_eq!(
            parse_body(b"  \n").unwrap_err().error,
            "missing request body"
        );
        assert_eq!(parse_body(b"[1]").unwrap_err().code, ErrorCode::InvalidJson);
        assert_eq!(
            parse_body(b"{oops").unwrap_err().code,
            ErrorCode::InvalidJson
        );
        let big = vec![b' '; MAX_BODY_BYTES + 1];
        assert_eq!(
            parse_body(&big).unwrap_err().code,
            ErrorCode::RequestTooLarge
        );
        assert!(parse_body(br#"{"model":"laya"}"#).is_ok());
    }

    #[test]
    fn accepts_typesafe_sdk_bodies() {
        // What the SDK sends for Noul()/Choice()/Score() without instructions, plus extra_body.
        let req = system_one_request(obj(json!({
            "state": {"message": "Please help.", "subject": "Duplicate charge"},
            "model": "laya",
            "questions": {
                "billing": {"type": "noul"},
                "tone": {"type": "choice", "criteria": {"angry": null, "calm": {"desc": "polite"}}},
                "urgency": {"type": "score", "criteria": ["Can wait", ["this", "week"], {"level": "today"}]},
                "spam": {"type": "noul", "instructions": ["a", "b"], "criteria": {"true": "Ads", "extra": 1}},
            },
            "some_extra_body_field": 1,
        })))
        .unwrap();
        let qs = req.questions.unwrap();
        assert_eq!(
            qs.keys().collect::<Vec<_>>(),
            ["billing", "tone", "urgency", "spam"]
        );
        let Question::Noul(spam) = &qs["spam"] else {
            panic!()
        };
        assert_eq!(
            spam.criteria,
            Some(NoulCriteria {
                when_true: Some(json!("Ads")),
                when_false: None
            })
        );
    }

    #[test]
    fn reports_every_issue_with_typesafe_locations() {
        let issues = system_one_request(obj(json!({
            "model": "",
            "state": 42,
            "questions": {
                "a": "nope",
                "b": {"instructions": "x"},
                "c": {"type": "maybe"},
                "d": {"type": "choice", "instructions": 3, "criteria": {"only": "one"}},
                "e": {"type": "choice", "criteria": ["x", "x", 7]},
                "f": {"type": "score", "criteria": ["a", 1, null]},
                "g": {"type": "score", "criteria": ["l0","l1","l2","l3","l4","l5","l6","l7","l8","l9","l10"]},
                "h": {"type": "noul", "criteria": {"true": 1}},
                "i": {"type": "noul", "criteria": "yes"},
                "j": {"type": "choice"},
                "k": {"type": "score", "criteria": {"0": "low"}},
            },
        })))
        .unwrap_err();
        assert_eq!(
            kinds(&issues),
            pairs(&[
                ("model", "string_too_short"),
                ("state", "state_type"),
                ("questions.a", "dict_type"),
                ("questions.b", "union_tag_not_found"),
                ("questions.c", "union_tag_invalid"),
                ("questions.d.choice.instructions", "json_type"),
                ("questions.d.choice.criteria", "too_short"),
                ("questions.e.choice.criteria.2", "string_type"),
                ("questions.e.choice.criteria", "too_short"),
                ("questions.f.score.criteria.1", "json_type"),
                ("questions.f.score.criteria.2", "json_type"),
                ("questions.g.score.criteria", "too_long"),
                ("questions.h.noul.criteria.true", "json_type"),
                ("questions.i.noul.criteria", "dict_type"),
                ("questions.j.choice.criteria", "missing"),
                ("questions.k.score.criteria", "list_type"),
            ])
        );
    }

    #[test]
    fn question_and_option_limits() {
        let many: Map<String, Value> = (0..=MAX_QUESTIONS)
            .map(|i| (format!("q{i}"), json!({"type": "noul"})))
            .collect();
        let issues = system_one_request(obj(json!({"model": "m", "state": "", "questions": many})))
            .unwrap_err();
        assert_eq!(kinds(&issues), pairs(&[("questions", "too_long")]));

        let empty = system_one_request(obj(json!({"model": "m", "state": "", "questions": {}})))
            .unwrap_err();
        assert_eq!(
            empty[0].msg,
            "Dictionary should have at least 1 item after validation, not 0"
        );

        let labels: Map<String, Value> = (0..=MAX_CHOICE_OPTIONS)
            .map(|i| (i.to_string(), Value::Null))
            .collect();
        let issues = system_one_request(obj(json!({"model": "m", "state": "s",
            "questions": {"q": {"type": "choice", "criteria": labels}}})))
        .unwrap_err();
        assert_eq!(
            issues[0].msg,
            "Dictionary should have at most 255 items after validation, not 256"
        );

        let ok: Map<String, Value> = (0..MAX_CHOICE_OPTIONS)
            .map(|i| (i.to_string(), Value::Null))
            .collect();
        assert!(
            system_one_request(obj(json!({"model": "m", "state": "s",
                "questions": {"q": {"type": "choice", "criteria": ok}}})))
            .is_ok()
        );
    }

    #[test]
    fn decide_native_options_and_lifecycle() {
        let req = decide_request(obj(json!({"model": "laya", "keep_alive": 0}))).unwrap();
        assert!(req.is_lifecycle());
        assert_eq!(req.keep_alive, Some(KeepAlive::UNLOAD));

        let req = decide_request(obj(json!({
            "model": "laya", "state": "hi", "questions": {"q": {"type": "noul"}},
            "keep_alive": "10m", "extras": ["laya", "laya"], "stream": false,
        })))
        .unwrap();
        assert_eq!(req.extras, vec![Extra::Laya]);
        assert!(!req.is_lifecycle());

        let issues = decide_request(obj(json!({
            "model": 7, "questions": {"q": {"type": "noul"}},
            "keep_alive": "soon", "extras": ["logits", 1], "stream": "yes",
        })))
        .unwrap_err();
        assert_eq!(
            kinds(&issues),
            pairs(&[
                ("model", "string_type"),
                ("state", "missing"),
                ("keep_alive", "keep_alive"),
                ("extras.0", "enum"),
                ("extras.1", "string_type"),
                ("stream", "bool_type"),
            ])
        );
        assert_eq!(issues[3].msg, "Input should be 'laya'");
    }

    #[test]
    fn management_requests() {
        assert_eq!(
            pull_request(obj(json!({"model": "laya", "stream": false}))).unwrap(),
            PullRequest {
                model: "laya".into(),
                insecure: false,
                stream: Some(false)
            }
        );
        assert_eq!(
            kinds(&pull_request(obj(json!({"insecure": "no"}))).unwrap_err()),
            pairs(&[("model", "missing"), ("insecure", "bool_type")])
        );
        assert_eq!(
            kinds(&copy_request(obj(json!({"source": "a"}))).unwrap_err()),
            pairs(&[("destination", "missing")])
        );
        assert!(show_request(obj(json!({"model": null}))).is_err());
        assert!(delete_request(obj(json!({"model": "x"}))).is_ok());

        let issues = create_request(obj(json!({
            "model": "triage", "from": "laya:en",
            "calibration": {"temperature": [1.0, "hot"], "temperature_by_options": {"choice:7": 1.2, "score:3-5": 0.9}},
            "parameters": {"precision": "int4", "num_ctx": 4096},
            "license": ["MIT", 3],
            "stream": 1,
        })))
        .unwrap_err();
        assert_eq!(
            kinds(&issues),
            pairs(&[
                ("calibration.temperature.1", "float_type"),
                ("calibration.temperature_by_options.choice:7", "calibration"),
                ("parameters.precision", "parameter"),
                ("parameters.num_ctx", "parameter"),
                ("license.1", "string_type"),
                ("stream", "bool_type"),
            ])
        );

        let ok = create_request(obj(json!({
            "model": "triage", "from": "laya:en",
            "questions": {"d": {"type": "choice", "criteria": ["a", "b"]}},
            "calibration": {"temperature": [1.0, 1.1, 0.9], "temperature_by_options": {"choice:11+": 1.4}},
            "parameters": {"precision": "fp32"}, "description": "Ticket triage",
            "license": "Apache-2.0",
        })))
        .unwrap();
        assert_eq!(ok.parameters.unwrap().precision.as_deref(), Some("fp32"));
        assert_eq!(ok.description.as_deref(), Some("Ticket triage"));
    }

    #[test]
    fn body_helper_builds_422() {
        let err = body(br#"{"model":"laya"}"#, system_one_request).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidRequest);
        assert_eq!(err.status(), 422);
        assert_eq!(err.error, "state: Field required");
        let err = body(b"nope", system_one_request).unwrap_err();
        assert_eq!(err.status(), 400);
    }
}
