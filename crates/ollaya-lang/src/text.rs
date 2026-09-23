//! The text detection reads from a state (`laya.lang.state_text`).
//!
//! Only string leaves count: keys are usually English identifiers that say nothing about the
//! content, and numbers, booleans and nulls carry no letters. Leaves are joined with single spaces
//! in document order, anything nested deeper than six containers is ignored, and the result is cut
//! at [`MAX_STATE_CHARS`] characters (code points, as Python slices).

use serde_json::Value;

/// Characters of a state that detection reads.
pub const MAX_STATE_CHARS: usize = 4000;

/// Containers below this depth are not read.
const MAX_DEPTH: usize = 6;

/// The state's string leaves, joined with spaces and cut at [`MAX_STATE_CHARS`].
pub fn state_text(state: &Value) -> String {
    let mut leaves = Vec::new();
    collect_strings(state, 0, &mut leaves);
    leaves.join(" ").chars().take(MAX_STATE_CHARS).collect()
}

fn collect_strings<'a>(value: &'a Value, depth: usize, out: &mut Vec<&'a str>) {
    if depth > MAX_DEPTH {
        return;
    }
    match value {
        Value::String(s) => out.push(s),
        Value::Array(items) => items
            .iter()
            .for_each(|v| collect_strings(v, depth + 1, out)),
        Value::Object(map) => map
            .values()
            .for_each(|v| collect_strings(v, depth + 1, out)),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn joins_string_leaves_in_order_and_skips_keys_and_scalars() {
        let state = json!({"subject": "Hi", "n": 3, "tags": ["a", "", null, true], "body": "x"});
        assert_eq!(state_text(&state), "Hi a  x");
    }

    #[test]
    fn stops_six_containers_deep() {
        let mut state = json!("deep");
        for _ in 0..6 {
            state = json!([state]);
        }
        assert_eq!(state_text(&state), "deep");
        assert_eq!(state_text(&json!([state])), "");
    }

    #[test]
    fn cuts_at_the_character_limit() {
        let text = state_text(&json!(["é".repeat(3998), "abc"]));
        assert_eq!(text, "é".repeat(3998) + " a");
    }
}
