"""`decision-endpoint-v1`: the request -> token rows layout of llm-semantic-router's Decision 1.0 models,
written the way the Rust port is (HF `tokenizers` only, no author code). `parity.py` and `goldens.py`
check it id for id against the author's `question_row` + `encode` (prompt version
`structured-segmented-candidate-endpoints-global-query-v2`).

One causal row per question. Three kinds of segment are tokenized separately and concatenated, so no
BPE merge crosses a boundary:

    prefix  = "Context:\\n" + payload(state) + "\\n\\nTask type: " + type + "\\nQuestion:\\n"
              + payload(instructions) + "\\nOptions:"
    option  = "\\n<option>\\n" + canonical({"key": key, "description": description}) + "\\n</option>"
    suffix  = "\\n\\nSelect the single option best supported by the context and instructions.\\nDecision:"
    row     = enc(prefix) + enc(option_1) + ... + enc(option_k) + enc(suffix)
    cand_pos[j] = the last token of option j;  query_pos = the last token of the row

    payload(v)   = v if v is a str else canonical(v)
    canonical(v) = json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

Options per type (the author's `question_row`):
    choice  criteria is an object of 2..255 entries; key -> description in key order. A null description
            renders as JSON null, or (models with `choice_null_description = "key"`, Nox) as the key.
    noul    criteria is null or an object whose keys are only "true"/"false" (exact spelling);
            options [false, true], description criteria.get(key, default); a key given as null stays null.
    score   criteria is a list of 2..10 levels; option j has key str(j) and the level as description.

The graph returns the head's raw candidate logits; the server divides by the temperature and softmaxes.
Rows longer than 16,384 tokens reject the whole request: nothing is truncated.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

MAX_OPTIONS = 255
MAX_SCORE_LEVELS = 10
MIN_OPTIONS = 2
MAX_ROW_TOKENS = 16384
PROMPT_VERSION = "structured-segmented-candidate-endpoints-global-query-v2"
NOUL_DEFAULTS = {"false": "The answer to the question is no.", "true": "The answer to the question is yes."}
SUFFIX = "\n\nSelect the single option best supported by the context and instructions.\nDecision:"


class LayoutError(ValueError):
    """The request is invalid for this model (HTTP 400/422)."""


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload(value) -> str:
    return value if isinstance(value, str) else canonical(value)


def options(q: Dict[str, Any], choice_null_description: str = "null") -> List[Dict[str, Any]]:
    """The author's question_row options: [{"key", "description"}] in candidate order."""
    kind = q.get("type")
    crit = q.get("criteria")
    if kind == "noul":
        crit = {} if crit is None else crit
        if not isinstance(crit, dict) or set(crit) - {"true", "false"}:
            raise LayoutError("noul criteria may contain only true and false")
        return [{"key": k, "description": crit.get(k, NOUL_DEFAULTS[k])} for k in ("false", "true")]
    if kind == "score":
        if not isinstance(crit, list) or not MIN_OPTIONS <= len(crit) <= MAX_SCORE_LEVELS:
            raise LayoutError("score requires an ordered list of 2..10 criteria")
        return [{"key": str(i), "description": v} for i, v in enumerate(crit)]
    if kind == "choice":
        if not isinstance(crit, dict) or not MIN_OPTIONS <= len(crit) <= MAX_OPTIONS:
            raise LayoutError("choice requires a mapping of 2..255 criteria")
        by_key = choice_null_description == "key"
        return [{"key": k, "description": k if (by_key and v is None) else v} for k, v in crit.items()]
    raise LayoutError("unknown question type %r" % (kind,))


class DecisionLayout:
    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        self.max_row = decision["max_row_tokens"]
        self.choice_null = decision["choice_null_description"]
        assert decision["prompt_version"] == PROMPT_VERSION

    def enc(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def row(self, state, q: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(q, dict):
            raise LayoutError("each question must be an object")
        if "instructions" not in q:
            raise LayoutError("instructions is required")
        opts = options(q, self.choice_null)
        prefix = "Context:\n%s\n\nTask type: %s\nQuestion:\n%s\nOptions:" % (payload(state), q["type"],
                                                                              payload(q["instructions"]))
        ids = self.enc(prefix)
        cand = []
        for o in opts:
            ids += self.enc("\n<option>\n" + canonical({"key": o["key"], "description": o["description"]})
                            + "\n</option>")
            cand.append(len(ids) - 1)
        ids += self.enc(SUFFIX)
        if len(ids) > self.max_row:
            raise LayoutError("%d tokens exceeds max_length=%d; no truncation allowed" % (len(ids), self.max_row))
        return {"ids": ids, "query": len(ids) - 1, "cands": cand}

    def encode(self, state, questions: Dict[str, Any]):
        """-> (rows, meta): one row per question in request order; any invalid question rejects the request."""
        if not isinstance(questions, dict) or not questions:
            raise LayoutError("questions must be a nonempty mapping")
        rows, meta = [], []
        for qid, q in questions.items():
            r = self.row(state, q)
            rows.append(r)
            meta.append({"qid": qid, "type": q["type"], "k": len(r["cands"])})
        return rows, meta
