//! Questions, answers, and the bodies of `POST /api/decide` and `POST /v1/systemone`
//! (`docs/api.md` §5, §7.3, §8).
//!
//! Answer types serialize to TypeSafe's wire shapes field for field and in the same order, so a
//! [`SystemOneResponse`] is exactly what TypeSafe returns. [`DecideResponse`] adds native fields
//! only, which a TypeSafe client ignores.

use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::keep_alive::KeepAlive;

/// Question id -> question, in the caller's order.
pub type Questions = IndexMap<String, Question>;

/// A typed question (TypeSafe's discriminated union on `type`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Choice(ChoiceQuestion),
    Score(ScoreQuestion),
    Noul(NoulQuestion),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceQuestion {
    /// String, object or array; absent means the model reads the question id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Value>,
    pub criteria: ChoiceCriteria,
}

/// Choice criteria: TypeSafe's label -> description map, or (laya extension) a list of labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChoiceCriteria {
    Map(IndexMap<String, Value>),
    Labels(Vec<String>),
}

impl ChoiceCriteria {
    /// Labels in option order, duplicates of a list collapsed onto their first position.
    pub fn labels(&self) -> Vec<&str> {
        match self {
            ChoiceCriteria::Map(m) => m.keys().map(String::as_str).collect(),
            ChoiceCriteria::Labels(l) => {
                let mut seen = indexmap::IndexSet::new();
                for label in l {
                    seen.insert(label.as_str());
                }
                seen.into_iter().collect()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreQuestion {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Value>,
    /// Level descriptions, level 0 first.
    pub criteria: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoulQuestion {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<NoulCriteria>,
}

/// What counts as true and false. Keys are exact, as in TypeSafe.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(rename = "true", default, skip_serializing_if = "Option::is_none")]
    pub when_true: Option<Value>,
    #[serde(rename = "false", default, skip_serializing_if = "Option::is_none")]
    pub when_false: Option<Value>,
}

impl Question {
    pub fn type_name(&self) -> &'static str {
        match self {
            Question::Choice(_) => "choice",
            Question::Score(_) => "score",
            Question::Noul(_) => "noul",
        }
    }

    pub fn instructions(&self) -> Option<&Value> {
        match self {
            Question::Choice(q) => q.instructions.as_ref(),
            Question::Score(q) => q.instructions.as_ref(),
            Question::Noul(q) => q.instructions.as_ref(),
        }
    }

    /// Number of options the model scores.
    pub fn num_options(&self) -> usize {
        match self {
            Question::Choice(q) => q.criteria.labels().len(),
            Question::Score(q) => q.criteria.len(),
            Question::Noul(_) => 2,
        }
    }

    /// The question as the engine reads it (`ollaya_decision::Question::parse` input), with the
    /// contract's rule for missing instructions applied: the question id is read in their place.
    pub fn to_engine(&self, id: &str) -> Value {
        let instructions = self
            .instructions()
            .cloned()
            .unwrap_or_else(|| Value::String(id.to_owned()));
        let mut obj = Map::new();
        obj.insert("type".into(), Value::String(self.type_name().into()));
        obj.insert("instructions".into(), instructions);
        let criteria = match self {
            Question::Choice(q) => serde_json::to_value(&q.criteria).ok(),
            Question::Score(q) => Some(Value::Array(q.criteria.clone())),
            Question::Noul(q) => q.criteria.as_ref().map(|c| {
                let mut m = Map::new();
                if let Some(t) = &c.when_true {
                    m.insert("true".into(), t.clone());
                }
                if let Some(f) = &c.when_false {
                    m.insert("false".into(), f.clone());
                }
                Value::Object(m)
            }),
        };
        if let Some(c) = criteria {
            obj.insert("criteria".into(), c);
        }
        Value::Object(obj)
    }
}

/// Every question in engine form, ready for `ollaya_decision::parse_questions`.
pub fn engine_questions(questions: &Questions) -> Value {
    Value::Object(
        questions
            .iter()
            .map(|(id, q)| (id.clone(), q.to_engine(id)))
            .collect(),
    )
}

/// `POST /v1/systemone` and `/v1/decisions` body: TypeSafe's `SystemOneRequest`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneRequest {
    pub model: String,
    /// A JSON string, object or array.
    pub state: Value,
    /// Absent only for models with embedded questions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub questions: Option<Questions>,
}

/// Named outputs that `extras` adds to every answer of `/api/decide`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Extra {
    /// Adds `laya: {confidence, act_probability}`.
    Laya,
}

impl Extra {
    pub const ALL: [Extra; 1] = [Extra::Laya];

    pub fn as_str(self) -> &'static str {
        match self {
            Extra::Laya => "laya",
        }
    }

    pub fn parse(s: &str) -> Option<Extra> {
        Extra::ALL.into_iter().find(|e| e.as_str() == s)
    }
}

/// `POST /api/decide` body: the systemone body plus native options. Without `state` (and
/// `questions`) it loads or unloads the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecideRequest {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub questions: Option<Questions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<KeepAlive>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extras: Vec<Extra>,
}

impl DecideRequest {
    pub fn new(model: impl Into<String>, state: Value, questions: Option<Questions>) -> Self {
        DecideRequest {
            model: model.into(),
            state: Some(state),
            questions,
            keep_alive: None,
            extras: Vec::new(),
        }
    }

    /// Load `model` now and keep it for `keep_alive` (default when `None`).
    pub fn load(model: impl Into<String>, keep_alive: Option<KeepAlive>) -> Self {
        DecideRequest {
            model: model.into(),
            state: None,
            questions: None,
            keep_alive,
            extras: Vec::new(),
        }
    }

    /// Unload `model` once its in-flight requests finish.
    pub fn unload(model: impl Into<String>) -> Self {
        DecideRequest::load(model, Some(KeepAlive::UNLOAD))
    }

    /// Whether this request only loads or unloads the model.
    pub fn is_lifecycle(&self) -> bool {
        self.state.is_none()
    }

    pub fn wants(&self, extra: Extra) -> bool {
        self.extras.contains(&extra)
    }
}

impl From<SystemOneRequest> for DecideRequest {
    fn from(r: SystemOneRequest) -> Self {
        DecideRequest::new(r.model, r.state, r.questions)
    }
}

/// TypeSafe's `Usage`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Encoder tokens read, summed over questions, special tokens included.
    pub input_tokens: u64,
    /// Always 0: decision models do not generate.
    pub output_tokens: u64,
}

/// One answer, in TypeSafe's shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
    Noul(NoulAnswer),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceAnswer {
    pub choice: String,
    pub confidence: f64,
    pub probabilities: IndexMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreAnswer {
    pub score: f64,
    pub confidence: f64,
    pub legend: IndexMap<String, Value>,
    pub probabilities: IndexMap<String, f64>,
}

/// No `confidence`, as in TypeSafe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoulAnswer {
    pub noul: f64,
}

impl Answer {
    pub fn type_name(&self) -> &'static str {
        match self {
            Answer::Choice(_) => "choice",
            Answer::Score(_) => "score",
            Answer::Noul(_) => "noul",
        }
    }

    /// Render a decided question with `ollaya_decision`'s TypeSafe rendering.
    pub fn from_decision(
        answer: &ollaya_decision::Answer,
        question: &ollaya_decision::Question,
    ) -> Result<Answer, serde_json::Error> {
        serde_json::from_value(answer.to_typesafe(question))
    }
}

/// `extras: ["laya"]`: laya's own values, namespaced so they never redefine TypeSafe's fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayaExtra {
    /// 1 − H(p)/ln K for choice and score; max(p, 1 − p) for noul.
    pub confidence: f64,
    /// From the act head; `null` for models without one.
    pub act_probability: Option<f64>,
}

impl LayaExtra {
    /// Take laya's values from `ollaya_decision`'s laya rendering.
    pub fn from_decision(
        answer: &ollaya_decision::Answer,
        question: &ollaya_decision::Question,
    ) -> Result<LayaExtra, serde_json::Error> {
        #[derive(Deserialize)]
        struct Laya {
            confidence: f64,
            action: Action,
        }
        #[derive(Deserialize)]
        struct Action {
            act_probability: Option<f64>,
        }
        let laya: Laya = serde_json::from_value(answer.to_laya(question))?;
        Ok(LayaExtra {
            confidence: laya.confidence,
            act_probability: laya.action.act_probability,
        })
    }
}

/// An `/api/decide` answer: TypeSafe's answer plus the requested extras.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecideAnswer {
    #[serde(flatten)]
    pub answer: Answer,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub laya: Option<LayaExtra>,
}

impl From<Answer> for DecideAnswer {
    fn from(answer: Answer) -> Self {
        DecideAnswer { answer, laya: None }
    }
}

/// TypeSafe's `SystemOneResponse`: exactly `model`, `answers`, `usage`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneResponse {
    /// The model that answered (a router's target).
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
}

/// The routing decision of a router model (`docs/api.md` §9).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Routing {
    /// The router that was requested, e.g. `laya:latest`.
    pub router: String,
    /// The target that answered, e.g. `laya:en`.
    pub model: String,
    /// Route key, e.g. `english`. Stable: branch on this.
    pub route: String,
    /// Why, in words. Informative only.
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DoneReason {
    Decide,
    Load,
    Unload,
}

/// `POST /api/decide` response: TypeSafe's response plus native fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecideResponse {
    pub model: String,
    pub answers: IndexMap<String, DecideAnswer>,
    pub usage: Usage,
    pub routing: Option<Routing>,
    pub state_truncated: bool,
    pub done_reason: DoneReason,
    pub created_at: DateTime<Utc>,
    /// Nanoseconds from receiving the request to producing the response.
    pub total_duration: u64,
    /// Nanoseconds this request waited for the model to load.
    pub load_duration: u64,
    /// Nanoseconds in the runner.
    pub eval_duration: u64,
}

impl DecideResponse {
    /// The `/v1/systemone` view of this response: native fields and extras dropped.
    pub fn into_system_one(self) -> SystemOneResponse {
        SystemOneResponse {
            model: self.model,
            answers: self
                .answers
                .into_iter()
                .map(|(id, a)| (id, a.answer))
                .collect(),
            usage: self.usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn missing_instructions_read_the_question_id() {
        let qs: Questions = serde_json::from_value(json!({
            "tone": {"type": "choice", "criteria": {"angry": null, "calm": "A polite message"}},
            "is_spam": {"type": "noul", "criteria": {"true": "Advertising"}},
            "urgency": {"type": "score", "instructions": "", "criteria": ["low", "high"]},
        }))
        .unwrap();
        let parsed = ollaya_decision::parse_questions(&engine_questions(&qs)).unwrap();
        assert_eq!(parsed["tone"].instructions, "tone");
        assert_eq!(parsed["is_spam"].instructions, "is_spam");
        assert_eq!(
            parsed["urgency"].instructions, "",
            "explicit strings are kept"
        );
        assert_eq!(
            parsed["is_spam"].render_options(),
            vec![
                "false: no, the statement does not hold",
                "true: Advertising"
            ]
        );
    }

    #[test]
    fn answers_render_through_ollaya_decision() {
        let q = ollaya_decision::Question::parse(
            "dept",
            &json!({"type": "choice", "instructions": "x", "criteria": {"a": null, "b": null, "c": null}}),
        )
        .unwrap();
        let cal = ollaya_decision::Calibration::default();
        let decided =
            ollaya_decision::Answer::new(&q, &cal, &[2.0, 0.5, -1.0], Some(&[1.5, -0.5]), 10);
        let answer = Answer::from_decision(&decided, &q).unwrap();
        assert_eq!(
            serde_json::to_value(&answer).unwrap(),
            decided.to_typesafe(&q)
        );
        let Answer::Choice(c) = &answer else {
            panic!("{answer:?}")
        };
        assert_eq!(c.choice, "a");
        assert_eq!(c.probabilities.keys().collect::<Vec<_>>(), ["a", "b", "c"]);

        let laya = LayaExtra::from_decision(&decided, &q).unwrap();
        let expected = decided.to_laya(&q);
        assert_eq!(json!(laya.confidence), expected["confidence"]);
        assert_eq!(
            json!(laya.act_probability),
            expected["action"]["act_probability"]
        );

        let noul =
            ollaya_decision::Question::parse("n", &json!({"type": "noul", "instructions": "x"}))
                .unwrap();
        let decided = ollaya_decision::Answer::new(&noul, &cal, &[0.0, 1.0], None, 10);
        let answer = DecideAnswer {
            answer: Answer::from_decision(&decided, &noul).unwrap(),
            laya: Some(LayaExtra::from_decision(&decided, &noul).unwrap()),
        };
        let v = serde_json::to_value(&answer).unwrap();
        assert_eq!(v["type"], "noul");
        assert!(v.get("confidence").is_none(), "noul has no confidence");
        assert_eq!(v["laya"]["act_probability"], Value::Null);
    }

    #[test]
    fn decide_answer_flattens_in_typesafe_order() {
        let a = DecideAnswer {
            answer: Answer::Score(ScoreAnswer {
                score: 1.5,
                confidence: 0.25,
                legend: [("0".into(), json!("low")), ("1".into(), json!("high"))]
                    .into_iter()
                    .collect(),
                probabilities: [("0".into(), 0.5), ("1".into(), 0.5)].into_iter().collect(),
            }),
            laya: Some(LayaExtra {
                confidence: 0.0,
                act_probability: Some(0.5),
            }),
        };
        let text = serde_json::to_string(&a).unwrap();
        assert_eq!(
            text,
            r#"{"type":"score","score":1.5,"confidence":0.25,"legend":{"0":"low","1":"high"},"probabilities":{"0":0.5,"1":0.5},"laya":{"confidence":0.0,"act_probability":0.5}}"#
        );
        assert_eq!(serde_json::from_str::<DecideAnswer>(&text).unwrap(), a);
    }
}
