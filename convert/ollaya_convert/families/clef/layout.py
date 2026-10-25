"""`clef-joint-v1`: the request -> token layout of Cloudflare's Clef models (`encode_record` in the model
repository's `joint_schema_model.py`), written the way the Rust port is (HF `tokenizers` and `decision.json`
only). `check.py` and `goldens.py` compare it id for id with upstream.

One sequence per request holds every question:

    ids    = tok(prefix) + tok(render(state))[:budget] + schema + tok(suffix)
    schema = tok(schema_head) + for question i, in request order:
               tok(field(n=i+1, id=qid, type=type)) + tok(render(instructions or qid))   <- question span
               + tok(options_head)
               + for option j: tok(option(n=j+1)) + tok(render({"option_id": id, "description": d}))   <- option span
                               + tok(option_end)
               + tok(field_end)

Each piece is tokenized on its own, special tokens parsed (upstream escapes nothing). `render` is the text as
is for a string, else json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False); a description
of None is left out. Options: noul [true, false] (upstream's default descriptions, replaced key by key by the
question's criteria), a choice's labels sorted, score levels in order. The state is cut so the sequence fits
`max_tokens`; budget = max_tokens - len(everything else).

The head reads the mean hidden state over each span, so a question whose instructions render to no token
(an empty id and no instructions) is rejected; upstream would return NaN.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

PLACEHOLDER = re.compile(r"\{(n|id|type)\}")
# The head's type embedding rows (upstream QUESTION_TYPES).
TYPE_INDEX = {"noul": 0, "choice": 1, "score": 2}


class LayoutError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def fill(template: str, **values) -> str:
    """Substitute {n}, {id} and {type} in one pass over the template (values are never rescanned)."""
    return PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


def ollaya_rule(qid, q):
    """Why Ollaya's shared question parser (crates/ollaya-decision/src/question.rs) rejects a definition upstream
    may accept, or None. The runtime applies it before the layout."""
    if not isinstance(q, dict):
        return "definition must be an object"
    t = q.get("type")
    if t not in TYPE_INDEX:
        return "unknown type"
    if "instructions" not in q:
        return "no 'instructions'"
    if q.get("ordinal") not in (None, True, False) or (q.get("ordinal") is True and t != "choice"):
        return "'ordinal' is a boolean, on choice questions only"
    c = q.get("criteria")
    if t == "choice":
        if isinstance(c, dict) and c:
            return None
        if isinstance(c, list) and c and all(isinstance(x, str) for x in c):
            return None
        return "a choice question takes a non-empty object or list of labels"
    if t == "score":
        return None if isinstance(c, list) and c else "a score question takes a non-empty list of levels"
    return None if c is None or isinstance(c, dict) else "noul criteria must be an object"


class ClefLayout:
    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        self.p = decision["prompt"]
        self.noul = decision["noul_criteria"]
        self.max_tokens = decision["max_tokens"]

    def enc(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def options(self, qid, q):
        """upstream question_options, after systemone's checks: [(option_id, description or None)]."""
        if not isinstance(q, dict):
            raise LayoutError("question %r must be an object" % qid)
        t = q.get("type")
        if t not in TYPE_INDEX:
            raise LayoutError("question %r: type must be noul, choice, or score" % qid)
        c = q.get("criteria")
        if t == "noul":
            crit = dict(self.noul)
            if c:
                if not isinstance(c, dict):
                    raise LayoutError("question %r: noul criteria must be an object" % qid)
                crit.update(c)
            return t, [("true", crit["true"]), ("false", crit["false"])]
        if not c:
            raise LayoutError("question %r: criteria must not be empty" % qid)
        if t == "choice":
            if not isinstance(c, dict):
                raise LayoutError("question %r: choice criteria must be an object of label -> description" % qid)
            return t, sorted((str(k), v) for k, v in c.items())
        if not isinstance(c, list):
            raise LayoutError("question %r: score criteria must be a list of levels" % qid)
        return t, [(str(i), v) for i, v in enumerate(c)]

    def encode(self, state, questions):
        """-> {"ids", "questions": [{"qid", "type", "span", "options", "option_ids"}], "state_tokens", "state_truncated"}.
        Spans are [start, end) token positions in "ids"."""
        if not isinstance(questions, dict) or not questions:
            raise LayoutError("questions must be a non-empty object")
        p = self.p
        schema = self.enc(p["schema"])
        qs = []
        for i, (qid, q) in enumerate(questions.items()):
            t, opts = self.options(qid, q)
            schema += self.enc(fill(p["field"], n=i + 1, id=qid, type=t))
            start = len(schema)
            instructions = q.get("instructions")
            if instructions is None or instructions == "":
                instructions = str(qid)
            schema += self.enc(render(instructions))
            span = [start, len(schema)]
            if span[0] == span[1]:
                raise LayoutError("question %r: its instructions are empty" % qid)
            schema += self.enc(p["options"])
            spans, ids = [], []
            for j, (oid, desc) in enumerate(opts):
                schema += self.enc(fill(p["option"], n=j + 1))
                a = len(schema)
                semantics = {"option_id": oid}
                if desc is not None:
                    semantics["description"] = desc
                schema += self.enc(render(semantics))
                spans.append([a, len(schema)])
                ids.append(oid)
                schema += self.enc(p["option_end"])
            schema += self.enc(p["field_end"])
            qs.append({"qid": qid, "type": t, "span": span, "options": spans, "option_ids": ids})
        prefix, suffix = self.enc(p["prefix"]), self.enc(p["suffix"])
        state_ids = self.enc(render(state))
        fixed = len(prefix) + len(schema) + len(suffix)
        if fixed > self.max_tokens:
            raise LayoutError("the questions take %d tokens; this model reads at most %d" % (fixed, self.max_tokens))
        kept = state_ids[: self.max_tokens - fixed]
        off = len(prefix) + len(kept)
        for q in qs:
            q["span"] = [q["span"][0] + off, q["span"][1] + off]
            q["options"] = [[a + off, b + off] for a, b in q["options"]]
        return {"ids": prefix + kept + schema + suffix, "questions": qs, "state_tokens": len(state_ids),
                "state_truncated": len(kept) < len(state_ids)}


def graph_inputs(row, pad, seq_multiple):
    """The graph's inputs for one encoded request (numpy int64)."""
    import numpy as np

    n = len(row["ids"])
    seq = -(-n // seq_multiple) * seq_multiple
    ids = np.full((1, seq), pad, dtype=np.int64)
    ids[0, :n] = row["ids"]
    qs = row["questions"]
    return {
        "input_ids": ids,
        "token_positions": np.arange(n, dtype=np.int64),
        "question_spans": np.array([q["span"] for q in qs], dtype=np.int64),
        "question_types": np.array([TYPE_INDEX[q["type"]] for q in qs], dtype=np.int64),
        "option_spans": np.array([s for q in qs for s in q["options"]], dtype=np.int64),
        "option_question": np.array([i for i, q in enumerate(qs) for _ in q["options"]], dtype=np.int64),
    }
