"""`d3-code-readout-v1`: the sequences of vllm-sr's d3 (upstream `d3_format` and `d3_runtime.normalize_question`, text
only), from the values in `decision.json`. The Rust port (`crates/ollaya-decision/src/d3.rs`) mirrors this file.

One sequence per question, read at its last token:

    text = pre + "State:\\n" + (EMPTY if state is null or "" else describe(state))
               + "\\n\\nQuestion:\\n" + (describe(instructions) if truthy else DEFAULT)
               + "\\n\\nOptions:\\n" + "\\n".join("{code_j}: " + option_j)
               + "\\n\\n" + CLOSING
               + post
    ids  = tokenizer(text), special tokens parsed (the chat template's and any in user text), none added

`pre` and `post` are the checkpoint's chat template (system prompt, `add_generation_prompt`, thinking off) around the
user text. `describe` is the string itself or `json.dumps(ensure_ascii=False)`; nothing is escaped. Options: a
choice's labels in the given order, `label` or `label: describe(description)` (1 to 255); noul `false`, `true`,
described by the criteria (keys "false" and "true" only) or "No / false" / "Yes / true" when missing or falsy; a
score's levels as `describe(level)`, 2 to 10, distinct and nonempty. A question over `max_length` tokens is rejected.
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


class D3Layout:
    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        t = decision["templates"]
        self.pre, self.post = t["pre"], t["post"]
        self.default = t["default_instructions"]
        self.noul = [t["noul_defaults"]["false"], t["noul_defaults"]["true"]]
        self.empty, self.closing = t["empty_state"], t["closing_line"]
        self.codes = decision["codes"]
        self.max_length = decision["max_length"]
        self.max_options = decision["max_options"]
        self.score_levels = decision["score_levels"]

    def options(self, qid, q):
        """The option texts of one question in prompt order; LayoutError where upstream's normalize_question raises."""
        bad = lambda msg: LayoutError("Question %r: %s" % (qid, msg))  # noqa: E731
        kind, crit = q.get("type"), q.get("criteria")
        if kind == "choice":
            if not isinstance(crit, dict) or not 1 <= len(crit) <= self.max_options:
                raise bad("choice criteria must map 1 to %d option keys to descriptions" % self.max_options)
            if any(not k for k in crit):
                raise bad("choice option keys must be nonempty strings")
            return [k if v is None else "%s: %s" % (k, describe(v)) for k, v in crit.items()]
        if kind == "noul":
            if crit is not None and (not isinstance(crit, dict) or set(crit) - {"false", "true"}):
                raise bad('noul criteria may only describe "false" and "true"')
            given = crit or {}
            return [describe(given.get(k)) if not falsy(given.get(k)) else d for k, d in zip(("false", "true"), self.noul)]
        if kind == "score":
            low, high = self.score_levels
            if not isinstance(crit, list) or not low <= len(crit) <= high:
                raise bad("score criteria must be an ordered list of %d to %d levels" % (low, high))
            levels = [describe(x) for x in crit]
            if any(not x for x in levels) or len(set(levels)) != len(levels):
                raise bad("score levels must be distinct and nonempty")
            return levels
        raise bad("unsupported question type %r (choice, noul or score)" % kind)

    def text(self, state, q, options):
        instructions = q.get("instructions")
        lines = ["%s: %s" % (c, o) for c, o in zip(self.codes, options)]
        body = "\n".join(["State:", self.empty if state is None or state == "" else describe(state), "",
                          "Question:", self.default if falsy(instructions) else describe(instructions), "",
                          "Options:", *lines, "", self.closing])
        return self.pre + body + self.post

    def encode(self, state, questions) -> List[dict]:
        """One sequence per question, in request order: upstream `D3.prepare`."""
        if not isinstance(questions, dict) or not questions:
            raise LayoutError("questions must be a nonempty mapping of question IDs to questions")
        if any(not k for k in questions):
            raise LayoutError("question IDs must be nonempty strings")
        out = []
        for qid, q in questions.items():
            options = self.options(qid, q)
            ids = self.tok.encode(self.text(state, q, options), add_special_tokens=False).ids
            if len(ids) > self.max_length:
                raise LayoutError("Question %r: the question prompt has %d tokens, over the %d-token limit"
                                  % (qid, len(ids), self.max_length))
            out.append({"ids": ids, "markers": [len(ids) - 1], "counts": [len(options)], "keys": [qid]})
        return out
