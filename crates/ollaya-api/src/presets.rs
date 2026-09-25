//! Built-in question sets: the five that `laya.presets` ships, plus Ollaya's own `agent`, which
//! reviews a command an AI agent is about to run. The CLI (`--preset`), the MCP server and the
//! desktop app all use these.

use serde_json::Value;

pub const NAMES: [&str; 6] = ["triage", "email", "guard", "moderation", "router", "agent"];

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
    fn every_preset_parses_as_questions() {
        for name in super::NAMES {
            let q = super::get(name).unwrap();
            serde_json::from_value::<crate::Questions>(q).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }
}
