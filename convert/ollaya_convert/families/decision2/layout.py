"""`decision2-endpoint-v1`: the request -> token rows layout of vllm-sr's Decision 2.0 models, written the way
the Rust port is (HF `tokenizers` only, no author code). `goldens.py` checks it id for id against the author's
`question_to_row` + `encode` (`decision2/_vendor/dev2model`, prompt version
`decision2-segmented-options-global-query-v1`).

The rows are Decision 1.0's (`decision.layout`): the same prefix, option and suffix segments, one causal row per
question. The options and the checks are the author's `question_to_row` and `system_one`:

    state         text, an object or an array (not null, not a number)
    instructions  nonempty text, an object or an array
    choice        criteria an object of 2..255 entries, in the caller's order; keys nonempty; a description is
                  text, an object, an array or null
    noul          criteria null or an object with only "false" and "true". A missing side is "No" / "Yes" and the
                  options are [false, true]; with both sides given they keep the caller's order. A description
                  is text, an object or an array
    score         criteria a list of 2..10 levels, each text, an object or an array; option j has key str(j)

Nested values are JSON (finite numbers). Any invalid question rejects the request (upstream answers the others
with an error), and so does a row longer than `max_row_tokens` (Kai 8,192, the others 16,384): nothing is
truncated.

The graph returns the head's raw logits in option order. The engine returns a noul question's as [false, true]
and adds Kai's per-level score offsets (`score_bias`, keyed by the level count) before the temperature (1).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from ..decision.layout import MAX_OPTIONS, MAX_SCORE_LEVELS, MIN_OPTIONS, DecisionLayout, LayoutError

PROMPT_VERSION = "decision2-segmented-options-global-query-v1"
NOUL_DEFAULTS = {"false": "No", "true": "Yes"}
TYPES = ("choice", "noul", "score")


def json_payload(value, nullable: bool = False, nonempty_text: bool = False) -> bool:
    """The author's `_api_json_payload`: text, an object or an array of JSON values (null when `nullable`)."""
    if value is None:
        return nullable
    if not isinstance(value, (str, dict, list)) or (nonempty_text and value == ""):
        return False

    def ok(part) -> bool:
        if part is None or isinstance(part, (str, bool, int)):
            return True
        if isinstance(part, float):
            return math.isfinite(part)
        if isinstance(part, list):
            return all(ok(child) for child in part)
        if isinstance(part, dict):
            return all(isinstance(k, str) and ok(child) for k, child in part.items())
        return False

    return ok(value)


def options(q: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The author's question_to_row options: [{"key", "description"}] in candidate order."""
    kind = q.get("type")
    if kind not in TYPES:
        raise LayoutError("unsupported or malformed question type")
    if not json_payload(q.get("instructions"), nonempty_text=True):
        raise LayoutError("missing question instructions")
    crit = q.get("criteria")
    if kind == "score":
        if not isinstance(crit, list) or not MIN_OPTIONS <= len(crit) <= MAX_SCORE_LEVELS:
            raise LayoutError("score criteria must be an ordered list of 2..10 levels")
        if not all(json_payload(d) for d in crit):
            raise LayoutError("score criteria descriptions must be text or structured data")
        return [{"key": str(i), "description": d} for i, d in enumerate(crit)]
    if kind == "noul":
        crit = {} if crit is None else crit
        if not isinstance(crit, dict) or set(crit) - {"false", "true"}:
            raise LayoutError("noul requires only false and true criteria")
        if len(crit) < 2:
            crit = {k: crit.get(k, NOUL_DEFAULTS[k]) for k in ("false", "true")}
    if not isinstance(crit, dict) or not MIN_OPTIONS <= len(crit) <= MAX_OPTIONS:
        raise LayoutError("choice/noul criteria must be an object with 2..255 options")
    if not all(isinstance(k, str) and k and json_payload(d, nullable=kind == "choice") for k, d in crit.items()):
        raise LayoutError("choice/noul criteria need nonempty string keys and valid descriptions")
    return [{"key": k, "description": d} for k, d in crit.items()]


class Decision2Layout(DecisionLayout):
    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        self.max_row = decision["max_row_tokens"]
        self.score_bias = {int(k): v for k, v in (decision.get("score_bias") or {}).items()}
        assert decision["prompt_version"] == PROMPT_VERSION

    def row(self, state, q: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(q, dict):
            raise LayoutError("each question must be an object")
        return self.build(state, q["type"], q["instructions"], options(q))

    def encode(self, state, questions: Dict[str, Any]):
        if isinstance(questions, dict) and any(not k for k in questions):
            raise LayoutError("question ids must be nonempty")
        if not json_payload(state):
            raise LayoutError("state must be text, an object, or an array")
        return super().encode(state, questions)

    def option_logits(self, q: Dict[str, Any], logits: List[float]) -> List[float]:
        """The option logits as the engine returns them: a noul question's as [false, true], a score question's
        plus the offsets of its level count."""
        keys = [o["key"] for o in options(q)]
        if q["type"] == "noul":
            return [logits[keys.index("false")], logits[keys.index("true")]]
        if q["type"] == "score" and len(logits) in self.score_bias:
            return [float(z) + b for z, b in zip(logits, self.score_bias[len(logits)])]
        return list(logits)
