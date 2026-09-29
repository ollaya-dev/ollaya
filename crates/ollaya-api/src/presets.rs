//! Presets: named question sets. Six are built in: the five that `laya.presets` ships, plus
//! Ollaya's own `agent`, which reviews a command an AI agent is about to run. The CLI (`--preset`),
//! the MCP server and the desktop app all use these. Custom presets live in the daemon's model
//! store (`presets/<name>.json`) and are managed through `/api/presets` (`docs/api.md` §7.11).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::decide::Questions;

pub const NAMES: [&str; 6] = ["triage", "email", "guard", "moderation", "router", "agent"];

/// One line on what each built-in set decides.
pub fn describe(name: &str) -> Option<&'static str> {
    Some(match name {
        "triage" => "Support tickets: intent, urgency, frustration, refund request, churn risk",
        "email" => "Email: category, spam, phishing, urgency, needs a reply",
        "guard" => "Prompt screening: jailbreak, prompt injection, sensitive data, harm, topic",
        "moderation" => "Moderation: toxic, harassment, threat, spam, severity",
        "router" => "Routing: difficulty, domain, needs tools, sensitive",
        "agent" => "A command an AI agent is about to run: action, on task, risk, destructive",
        _ => return None,
    })
}

pub fn is_builtin(name: &str) -> bool {
    NAMES.contains(&name)
}

/// Custom preset names: 1 to 64 of `a-z`, `0-9`, `-`, `_`, starting with a letter or digit.
/// Lowercase only, so a name means the same file on every filesystem.
pub fn valid_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name.as_bytes()[0].is_ascii_alphanumeric()
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
}

/// One `GET /api/presets` entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetInfo {
    pub name: String,
    /// Built into Ollaya; it cannot be changed or deleted.
    pub builtin: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Question ids, in order.
    pub questions: Vec<String>,
    /// When a custom preset was last written; `null` for built-in ones.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<DateTime<Utc>>,
}

/// `GET /api/presets`: built-in presets first, then custom ones by name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetsResponse {
    pub presets: Vec<PresetInfo>,
}

/// `POST /api/presets/create`: create or replace a custom preset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreatePresetRequest {
    pub name: String,
    pub questions: Questions,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// `POST /api/presets/show` and `DELETE /api/presets/delete`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresetRequest {
    pub name: String,
}

/// `POST /api/presets/show`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetResponse {
    pub name: String,
    pub builtin: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub questions: Questions,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<DateTime<Utc>>,
}

pub fn get(name: &str) -> Option<Value> {
    let text = match name {
        "triage" => include_str!("presets/triage.json"),
        "email" => include_str!("presets/email.json"),
        "guard" => include_str!("presets/guard.json"),
        "moderation" => include_str!("presets/moderation.json"),
        "router" => include_str!("presets/router.json"),
        "agent" => include_str!("presets/agent.json"),
        _ => return None,
    };
    Some(serde_json::from_str(text).expect("presets are valid JSON"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn names() {
        for ok in ["a", "my-triage", "v2_routing", "0day"] {
            assert!(super::valid_name(ok), "{ok}");
        }
        for bad in [
            "",
            "-x",
            "_x",
            "Triage",
            "a b",
            "a/b",
            "..",
            "a.json",
            &"x".repeat(65),
        ] {
            assert!(!super::valid_name(bad), "{bad}");
        }
        for name in super::NAMES {
            assert!(super::valid_name(name) && super::describe(name).is_some());
        }
    }

    #[test]
    fn every_preset_parses_as_questions() {
        for name in super::NAMES {
            let q = super::get(name).unwrap();
            serde_json::from_value::<crate::Questions>(q).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }
}
