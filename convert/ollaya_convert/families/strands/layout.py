"""`strands-pointer-v1`: the request -> token rows layout of Strands Decider, written the way the Rust port
is (HF `tokenizers` and `decision.json` only, no `strands_decider` package). `goldens.py` checks it id for id
against upstream (`render_state`, `render_question`, `SystemOneEngine._chunks` / `_fit` / `_option_idx`).

One causal row per question: the state, then the question.

    state    = "<state>\\n" + render(state) + "\\n</state>\\n"
    question = '<question type="{type}">\\n' + header[type] + "\\n" + render(instructions) + "\\n<options>\\n"
               + "\\n".join(lines) + "\\n</options>\\n</question>\\n<answer>"
    line_j   = "{j+1}. {name_j}" + (" \u2014 " + collapse(text_j) if collapse(text_j) else "")
    noul:   names false, true; texts {**noul_defaults, **criteria}, rendered (a null criterion is no text)
    choice: names = the keys, texts = render(description)
    score:  names "0".."n-1", texts = the levels as given
    render(x)   = "" for null, x.strip() for a string, json.dumps(x, indent=2, ensure_ascii=False) otherwise
    collapse(s) = " ".join(s.split())

The state is tokenized with the tokenizer's special tokens (Qwen's add none), each question without, with
character offsets. Questions are fitted in runs of `max_batch` (32) in request order. With L the longest
question of a run, reserve = min(L, int(max_length * max_question_fraction)); each question keeps its last
`reserve` tokens (cut from the front) and the state its first max(1, max_length - reserve):

    row     = state[:budget] + question[-reserve:]
    decide  = len(row) - 1                    (the `<answer>` token)
    opts[j] = len(state[:budget]) + the last question token whose offsets lie inside line j (empty ones skipped)

A request whose cut removes every token of an option line is rejected. The graph returns raw pointer
scores; the runtime divides them by the fitted temperature of the question's type and softmaxes.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

KINDS = ("choice", "score", "noul")


class LayoutError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


def render(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return json.dumps(v, indent=2, ensure_ascii=False)


def collapse(s: str) -> str:
    return " ".join(s.split())


def _content(v) -> bool:
    return isinstance(v, (str, dict, list))


def question_text(kind: str, header: str, instructions: str, pairs):
    """-> (text, spans): the question and each option line's (start, end) character span in it."""
    prefix = '<question type="%s">\n%s\n%s\n<options>\n' % (kind, header, instructions)
    lines, spans, cursor = [], [], len(prefix)
    for j, (name, desc) in enumerate(pairs):
        desc = collapse(desc)
        line = "%d. %s" % (j + 1, name) + (" \u2014 %s" % desc if desc else "")
        lines.append(line)
        spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1
    return prefix + "\n".join(lines) + "\n</options>\n</question>\n<answer>", spans


class StrandsLayout:
    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        self.max_length = decision["max_length"]
        self.reserve_cap = max(1, int(decision["max_length"] * decision["max_question_fraction"]))
        self.max_batch = decision["max_batch"]
        self.headers = decision["templates"]["headers"]
        self.noul_defaults = decision["templates"]["noul_defaults"]
        self.choice_options = decision["choice_options"]
        self.score_levels = decision["score_levels"]

    def pairs(self, qid, q):
        """-> (kind, rendered instructions, [(name, text)]). Mirrors upstream's pydantic schema."""
        bad = lambda msg: LayoutError("question %r: %s" % (qid, msg))  # noqa: E731
        if not isinstance(q, dict):
            raise bad("must be an object")
        t, crit = q.get("type"), q.get("criteria")
        if not _content(q.get("instructions")):
            raise bad("instructions must be a string, an object or an array")
        if t == "noul":
            if crit is not None and (not isinstance(crit, dict) or set(crit) - {"true", "false"}):
                raise bad("noul criteria take only \"true\" and \"false\"")
            if any(v is not None and not _content(v) for v in (crit or {}).values()):
                raise bad("criteria must be strings, objects, arrays or null")
            c = {**self.noul_defaults, **(crit or {})}
            pairs = [(k, render(c[k])) for k in ("false", "true")]
        elif t == "choice":
            lo, hi = self.choice_options
            if not isinstance(crit, dict) or not lo <= len(crit) <= hi:
                raise bad("choice criteria must be an object of %d to %d options" % (lo, hi))
            if any(v is not None and not _content(v) for v in crit.values()):
                raise bad("criteria must be strings, objects, arrays or null")
            pairs = [(k, render(v)) for k, v in crit.items()]
        elif t == "score":
            lo, hi = self.score_levels
            if not isinstance(crit, list) or not lo <= len(crit) <= hi or not all(isinstance(x, str) for x in crit):
                raise bad("score criteria must be a list of %d to %d strings" % (lo, hi))
            pairs = [(str(i), x) for i, x in enumerate(crit)]
        else:
            raise bad("unknown type %r" % (t,))
        return t, render(q["instructions"]), pairs

    def encode(self, state, questions):
        """-> (rows, meta): rows[q] = {"ids", "decide", "opts"} (one row per question, request order)."""
        if not _content(state):
            raise LayoutError("state must be a string, an object or an array for this model")
        if not isinstance(questions, dict) or not questions:
            raise LayoutError("questions must contain at least one question")
        qs, meta = [], []
        for qid, q in questions.items():
            kind, instructions, pairs = self.pairs(qid, q)
            qs.append(question_text(kind, self.headers[kind], instructions, pairs))
            meta.append({"qid": qid, "type": kind, "k": len(pairs)})
        state_text = "<state>\n%s\n</state>\n" % render(state)
        state_ids = self.tok.encode(state_text, add_special_tokens=True).ids
        rows = []
        for g in range(0, len(qs), self.max_batch):
            encs = [self.tok.encode(text, add_special_tokens=False) for text, _ in qs[g: g + self.max_batch]]
            reserve = min(max(len(e.ids) for e in encs), self.reserve_cap)
            s = state_ids[: max(1, self.max_length - reserve)]
            for (text, spans), e in zip(qs[g: g + self.max_batch], encs):
                cut = max(0, len(e.ids) - reserve)
                ids, offs = e.ids[cut:], e.offsets[cut:]
                opts = []
                for a, b in spans:
                    last = -1
                    for j, (lo, hi) in enumerate(offs):
                        if hi > lo and lo >= a and hi <= b:
                            last = j
                    if last < 0:
                        raise LayoutError("an option of this question was cut from the prompt; shorten the question")
                    opts.append(len(s) + last)
                row = list(s) + list(ids)
                rows.append({"ids": row, "decide": len(row) - 1, "opts": opts})
        return rows, meta
