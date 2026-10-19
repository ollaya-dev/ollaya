"""`pplx-joint-v1`: the sequence of perplexity-ai/pplx-decider (upstream `pplx_decider.protocol`, text only), from the
values in `decision.json`. The Rust port (`crates/ollaya-decision/src/pplx.rs`) mirrors this file.

    text    = pre + "State:\\n" + esc(describe(state))
                  + for question i: "\\n\\nQuestion {i}:\\n" + esc(describe(instructions or DEFAULT)) + "\\nOptions:\\n"
                                    + "\\n".join("{code_j}: " + esc(describe(option_j))) + "\\n" + MARKER
                  + post
    ids     = tokenizer(text), special tokens parsed (the chat template's), none added
    markers = the positions of MARKER, one per question, in order

`pre` and `post` are the checkpoint's chat template (system prompt, `add_generation_prompt`, thinking off) around
the user text. `describe` is the string itself or `json.dumps(ensure_ascii=False)`. `esc` writes every special token
(and the marker) with full-width brackets, so user text cannot form one. Options: a choice's labels in the given
order, `label` or `label: describe(description)`; a score's levels; noul `false`, `true`, described by the criteria
or "No / false" / "Yes / true" when missing or falsy. A sequence over `max_length` tokens is split by halving its
question list, recursively, every part keeping the full state; one question that does not fit is rejected.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


class LayoutError(ValueError):
    pass


def describe(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def falsy(value):
    """Python truthiness of a JSON value (`x or default`)."""
    return value is None or value is False or value == 0 and not isinstance(value, bool) or value in ("", [], {})


def options(q):
    """(option count, descriptions) in prompt order."""
    t, crit = q["type"], q.get("criteria")
    if t == "choice":
        return len(crit), [k if v is None else "%s: %s" % (k, describe(v)) for k, v in crit.items()]
    if t == "score":
        return len(crit), list(crit)
    c = crit if not falsy(crit) else {}
    if not isinstance(c, dict):   # upstream fails on `criteria.get`
        raise LayoutError("noul criteria must be an object")
    d = [c.get("false"), c.get("true")]
    return 2, [d[0] if not falsy(d[0]) else "No / false", d[1] if not falsy(d[1]) else "Yes / true"]


class PplxLayout:
    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        t = decision["templates"]
        self.pre, self.post = t["pre"], t["post"]
        self.default = t["default_instructions"]
        self.escapes = decision["escapes"]
        self.marker, self.marker_id = decision["marker"]["token"], decision["marker"]["id"]
        self.codes = decision["codes"]
        self.max_length = decision["max_length"]
        self.max_options = decision["max_options"]

    def esc(self, value):
        text = describe(value)
        for s in self.escapes:
            text = text.replace(s, s.replace("<", "＜").replace(">", "＞"))
        return text

    def validate(self, questions):
        if not isinstance(questions, dict) or not questions:
            raise LayoutError("A request needs at least one question")
        for key, q in questions.items():
            if not isinstance(key, str) or not key:
                raise LayoutError("Question IDs must be nonempty strings")
            kind = q.get("type") if isinstance(q, dict) else None
            if kind not in ("choice", "noul", "score"):
                raise LayoutError("Question %r: type must be 'choice', 'noul' or 'score'" % key)
            if kind == "choice" and not isinstance(q.get("criteria"), dict):
                raise LayoutError("Question %r: choice criteria must map option names to descriptions" % key)
            if kind == "score" and not isinstance(q.get("criteria"), list):
                raise LayoutError("Question %r: score criteria must be a list of level descriptions" % key)
            n = options(q)[0]
            if not 1 <= n <= min(self.max_options, len(self.codes)):
                raise LayoutError("Question %r: %d options; 1 to %d are supported" % (key, n, self.max_options))
            if kind == "score" and n < 2:
                raise LayoutError("Question %r: a score needs at least two ordered levels" % key)

    def text(self, state, items):
        parts = ["State:\n" + self.esc(state)]
        for i, (_, q) in enumerate(items):
            instructions = q.get("instructions")
            parts.append("Question %d:\n%s\nOptions:\n" % (i, self.esc(self.default if falsy(instructions) else instructions))
                         + "\n".join("%s: %s" % (c, self.esc(d)) for c, d in zip(self.codes, options(q)[1])) + "\n" + self.marker)
        return self.pre + "\n\n".join(parts) + self.post

    def sequence(self, state, items):
        ids = self.tok.encode(self.text(state, items), add_special_tokens=False).ids
        markers = [i for i, x in enumerate(ids) if x == self.marker_id]
        if len(markers) != len(items):
            raise LayoutError("Readout markers do not match questions after tokenization")
        return {"ids": ids, "markers": markers, "counts": [options(q)[0] for _, q in items], "keys": [k for k, _ in items]}

    def encode(self, state, questions) -> List[dict]:
        """The request's sequences: upstream Encoder.plan."""
        self.validate(questions)

        def split(items):
            s = self.sequence(state, items)
            if len(s["ids"]) <= self.max_length:
                return [s]
            if len(items) == 1:
                raise LayoutError("Question %r needs %d tokens with the state, over the %d-token limit"
                                  % (items[0][0], len(s["ids"]), self.max_length))
            mid = len(items) // 2
            return split(items[:mid]) + split(items[mid:])

        return split(list(questions.items()))
