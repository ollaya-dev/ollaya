"""`nimble-codes-v1`: the request -> token rows layout of Bespoke-Nimble-9B-v2, written the way the Rust
port is (HF `tokenizers` and `decision.json` only, no transformers, no upstream code). `parity.py` and
`goldens.py` check it id for id against upstream `serving_schema.prepare_prompts` on the schema that
`ref._schema` maps from the TypeSafe questions.

One chat row per question. Every row carries the whole request (state and every field); only the
requested field name at the end differs:

    content = safe_json({"context": serialize(state), "schema": fields}) + "\\n\\nRequested field: " + safe_json(qid)
    row     = tok(pre + content + post)        # special tokens parsed; the chat template around the user turn
    fields  = [{"name": qid, "description": serialize(instructions),
                "choices": [{"code": codes[j], "value": value_j, "description": text_j}, ...]}, ...]

    noul:   values [false, true] (JSON booleans), texts criteria["false"] or "No", criteria["true"] or "Yes"
    choice: values = the labels, texts = the description or, when it is null, the label
    score:  values "0".."n-1", texts = the levels
    serialize(x) = x if a string, else json.dumps(x, ensure_ascii=False)
    safe_json(x) = json.dumps(x, ensure_ascii=False) with "<" and ">" written as \\u003c and \\u003e

Bespoke-Nimble-9B-v3 (`mapping: "decision-index"`, the author's `decision_index_engine`): a choice or noul
side without a description gets no "description" key (no label, no Yes/No), noul criteria keyed exactly
{yes, no} read as {true, false}, and a blank string state is sent as `json.dumps(state)`.

`safe_json` keeps user text from forming a chat control token, so the whole row is tokenized with
special tokens parsed, as upstream does. Codes are A..Z, then 229 two-letter codes (`serving_config.json`);
a request whose widest question has more than 26 options uses the "short" system prompt ("short code"
instead of "one-letter code") for every row. Readout: the next-token logits of the question's first k
codes at the row's last token.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

NOUL_DEFAULTS = {"false": "No", "true": "Yes"}


class LayoutError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


def serialize(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, allow_nan=False)


def safe_json(v) -> str:
    return json.dumps(v, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e")


def fields_of(questions: Dict[str, Any], max_options: int, mapping: str = "openjev", noul_defaults=None):
    """-> (fields without codes, meta). Mirrors ref._schema (or ref_v3.prepare) + upstream validate_schema.
    A text of None leaves the choice without a description."""
    di = mapping == "decision-index"
    noul_defaults = NOUL_DEFAULTS if noul_defaults is None else noul_defaults
    if not isinstance(questions, dict) or not questions:
        raise LayoutError("questions must contain at least one question")
    fields, meta = [], []
    for qid, q in questions.items():
        if not qid.strip():
            raise LayoutError("question ids must not be blank")
        if "instructions" not in q:
            raise LayoutError("question %r: instructions are required" % qid)
        desc = serialize(q["instructions"])
        if not desc.strip():
            raise LayoutError("question %r: instructions must not be blank" % qid)
        t, crit = q.get("type"), q.get("criteria")
        if t == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise LayoutError("question %r: noul criteria must be an object" % qid)
            c = crit or {}
            yes_no = di and set(c) == {"yes", "no"}
            if set(c) - {"false", "true"} and not yes_no:
                raise LayoutError("question %r: noul criteria take only \"true\" and \"false\"" % qid)
            sides = (("no", "false"), ("yes", "true")) if yes_no else (("false", "false"), ("true", "true"))
            values = [False, True]
            texts = [serialize(c[k]) if c.get(k) is not None else (None if di else noul_defaults[d]) for k, d in sides]
        elif t == "choice":
            labels = list(crit) if isinstance(crit, dict) else list(dict.fromkeys(crit))
            descs = crit if isinstance(crit, dict) else {}
            if any(not label.strip() for label in labels):
                raise LayoutError("question %r: choice labels must not be blank" % qid)
            values = labels
            texts = [serialize(descs[k]) if descs.get(k) is not None else (None if di else k) for k in labels]
        elif t == "score":
            if not isinstance(crit, list):
                raise LayoutError("question %r: score criteria must be a list of levels" % qid)
            values = [str(i) for i in range(len(crit))]
            texts = [serialize(x) for x in crit]
        else:
            raise LayoutError("question %r: unknown type %r" % (qid, t))
        if not 1 <= len(values) <= max_options:
            raise LayoutError("question %r: %d options; this model takes 1 to %d" % (qid, len(values), max_options))
        fields.append((qid, desc, values, texts))
        meta.append({"qid": qid, "type": t, "k": len(values)})
    return fields, meta


class NimbleLayout:
    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        self.templates = decision["templates"]
        self.codes = decision["codes"]["strings"]
        self.code_ids = decision["codes"]["ids"]
        self.legacy = decision["codes"]["letter_codes"]
        self.max_tokens = decision["max_prompt_tokens"]
        self.mapping = decision.get("mapping", "openjev")
        self.noul_defaults = decision.get("noul_defaults")

    def prompts(self, state, questions):
        """-> (prompt strings, meta)."""
        fields, meta = fields_of(questions, len(self.codes), self.mapping, self.noul_defaults)
        if self.mapping == "decision-index" and isinstance(state, str) and not state.strip():
            state = json.dumps(state)
        context = serialize(state)
        if not context.strip():
            raise LayoutError("state must not be blank for this model")
        schema = [{"name": qid, "description": desc,
                   "choices": [{"code": code, "value": value, **({} if text is None else {"description": text})}
                               for code, value, text in zip(self.codes, values, texts)]}
                  for qid, desc, values, texts in fields]
        t = self.templates["letter" if max(m["k"] for m in meta) <= self.legacy else "short"]
        body = safe_json({"context": context, "schema": schema}) + "\n\nRequested field: "
        return [t["pre"] + body + safe_json(qid) + t["post"] for qid, _, _, _ in fields], meta

    def encode(self, state, questions):
        """-> (rows, meta): rows[q] = {"ids", "candidates"} (one row per question, request order)."""
        prompts, meta = self.prompts(state, questions)
        rows = []
        for p, m in zip(prompts, meta):
            ids = self.tok.encode(p, add_special_tokens=False).ids
            if len(ids) > self.max_tokens:
                raise LayoutError("the prompt for question %r has %d tokens; this model reads at most %d and "
                                  "never truncates" % (m["qid"], len(ids), self.max_tokens))
            rows.append({"ids": ids, "candidates": self.code_ids[: m["k"]]})
        return rows, meta


def option_logits(cand_row: List[float], k: int) -> List[float]:
    """The question's option logits from its row's 255 code logits."""
    return list(cand_row[:k])
