//! Wire compatibility with TypeSafe, using the examples embedded in the official SDK's generated
//! schema (`typesafe_sdk/_schemas/models.py`, typesafe-sdk 0.7.1, generated from
//! `https://api.typesafe.ai/openapi.json`) and the SDK's error-message extraction.

use ollaya_api::error::body_loc;
use ollaya_api::validate::{self, Body};
use ollaya_api::{
    Answer, ErrorBody, ErrorCode, ModelList, SystemOneRequest, SystemOneResponse, ValidationIssue,
};
use serde_json::{Value, json};

fn obj(v: Value) -> Body {
    match v {
        Value::Object(m) => m,
        other => panic!("{other}"),
    }
}

/// Serialize and compare byte for byte (field order included).
fn same_text<T: serde::Serialize>(typed: &T, expected: &Value) {
    assert_eq!(
        serde_json::to_string(typed).unwrap(),
        serde_json::to_string(expected).unwrap()
    );
}

#[test]
fn answer_examples_round_trip_in_schema_field_order() {
    // Field order is the order of the pydantic model's fields.
    let choice = json!({"type": "choice", "choice": "angry", "confidence": 0.9,
        "probabilities": {"angry": 0.8, "calm": 0.1, "excited": 0.1}});
    let score = json!({"type": "score", "score": 1.7, "confidence": 0.9,
        "legend": {"0": "Can wait", "1": "Needs attention this week", "2": "Needs attention today"},
        "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}});
    let noul = json!({"type": "noul", "noul": 0.98});
    for example in [choice, score, noul] {
        let answer: Answer = serde_json::from_value(example.clone()).unwrap();
        same_text(&answer, &example);
    }
}

#[test]
fn response_and_model_list_examples() {
    let response = json!({
        "model": "jev-latest",
        "answers": {"billing": {"noul": 0.98, "type": "noul"}},
        "usage": {"input_tokens": 120, "output_tokens": 12},
    });
    let typed: SystemOneResponse = serde_json::from_value(response.clone()).unwrap();
    assert_eq!(serde_json::to_value(&typed).unwrap(), response);

    let models = json!({"models": [
        {"description": "General-purpose system one model.", "name": "jev-latest", "release_date": "2026-09-15"}
    ]});
    let typed: ModelList = serde_json::from_value(models.clone()).unwrap();
    assert_eq!(serde_json::to_value(&typed).unwrap(), models);
    same_text(
        &typed,
        &json!({"models": [{"name": "jev-latest", "description": "General-purpose system one model.", "release_date": "2026-09-15"}]}),
    );
}

#[test]
fn request_examples_validate() {
    let questions =
        json!({"billing": {"instructions": "Is this message about billing?", "type": "noul"}});
    for state in [
        json!("I was charged twice. Please help."),
        json!({"message": "Please help.", "subject": "Duplicate charge"}),
    ] {
        let body = obj(json!({"state": state, "model": "jev-latest", "questions": questions}));
        let req = validate::system_one_request(body).unwrap();
        assert_eq!(req.model, "jev-latest");
        let direct: SystemOneRequest = serde_json::from_value(
            json!({"state": state, "model": "jev-latest", "questions": questions}),
        )
        .unwrap();
        assert_eq!(direct, req);
    }
}

#[test]
fn question_examples_validate() {
    let choice = json!({"type": "choice", "instructions": "What is the tone of this message?",
        "criteria": {"angry": "An upset or hostile message", "calm": "A neutral or polite message",
                     "excited": "An enthusiastic or eager message"}});
    let score = json!({"type": "score", "instructions": "How urgent is this message?",
        "criteria": ["Can wait", "Needs attention this week", "Needs attention today"]});
    let mut questions = serde_json::Map::new();
    questions.insert("tone".into(), choice);
    questions.insert("urgency".into(), score);
    // Every NoulQuestion `instructions` example, with and without criteria.
    for (i, instructions) in [
        json!("Is this message spam?"),
        json!("This message contains unsolicited advertising."),
        json!({"task": "Identify unsolicited advertising."}),
    ]
    .into_iter()
    .enumerate()
    {
        questions.insert(
            format!("spam{i}"),
            json!({"type": "noul", "instructions": instructions,
            "criteria": {"false": "A legitimate conversation", "true": "Unsolicited advertising"}}),
        );
    }
    questions.insert(
        "criteria_example".into(),
        json!({"type": "noul", "criteria": {"true": "The message is unsolicited advertising.",
            "false": "The message is a legitimate conversation."}}),
    );
    let req = validate::system_one_request(obj(json!({
        "model": "laya", "state": "Buy now!", "questions": questions,
    })))
    .unwrap();
    assert_eq!(req.questions.unwrap().len(), 6);
}

#[test]
fn http_validation_error_example() {
    // TypeSafe's 422 body, as the SDK receives it.
    let body =
        br#"{"detail": [{"loc": ["body", "state"], "msg": "Field required", "type": "missing"}]}"#;
    let parsed = ErrorBody::from_response(422, body);
    assert_eq!(parsed.error, "state: Field required");
    assert_eq!(parsed.code, ErrorCode::InvalidRequest);
    assert_eq!(parsed.detail.as_ref().unwrap().len(), 1);

    // Ollaya's issue for the same condition is identical to TypeSafe's entry.
    let ours = validate::system_one_request(obj(
        json!({"model": "laya", "questions": {"q": {"type": "noul"}}}),
    ))
    .unwrap_err();
    assert_eq!(
        serde_json::to_value(&ours).unwrap(),
        json!([{"loc": ["body", "state"], "msg": "Field required", "type": "missing"}])
    );
    assert_eq!(ours[0], ValidationIssue::missing(body_loc(&["state"])));
}

#[test]
fn validation_error_example_parses() {
    let example = json!({
        "loc": ["body", "questions", "urgency", "score", "criteria"],
        "msg": "Field required",
        "type": "missing",
        "input": {"type": "score"},
        "ctx": {"min_length": 1},
    });
    let issue: ValidationIssue = serde_json::from_value(example).unwrap();
    assert_eq!(issue.path(), "questions.urgency.score.criteria");
    assert_eq!(issue.ctx.unwrap()["min_length"], 1);

    // Ollaya puts the discriminator tag into `loc` the same way.
    let ours = validate::system_one_request(obj(json!({
        "model": "laya", "state": "x", "questions": {"urgency": {"type": "score"}},
    })))
    .unwrap_err();
    assert_eq!(ours[0].path(), "questions.urgency.score.criteria");
    assert_eq!(ours[0].kind, "missing");
}

/// Every error body Ollaya sends gives the SDK a readable message through `extract_message`,
/// which reads a string `error` before anything else.
#[test]
fn sdk_reads_every_error_body() {
    for code in ErrorCode::ALL {
        let body = if code.has_detail() {
            ErrorBody::with_issues(
                code.clone(),
                vec![ValidationIssue::missing(body_loc(&["x"]))],
            )
        } else {
            ErrorBody::new(code.clone(), format!("{code} happened"))
        };
        let v = serde_json::to_value(&body).unwrap();
        assert!(v["error"].is_string(), "{code}");
        let bytes = serde_json::to_vec(&body).unwrap();
        assert_eq!(
            ErrorBody::from_response(code.status(), &bytes),
            body,
            "{code}"
        );
    }
}
